import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel

from code_generator.agents.coding_agent import (
    CodeAgent,
    CodeAgentInput,
    CodeAgentOutput,
)
from code_generator.agents.human_agent import HumanAgent, HumanInput
from code_generator.agents.orchestrator import (
    OrchestratorAgent,
    OrchestratorInput,
)
from code_generator.llm_interface import LLMInterface
from code_generator.sandbox import DockerSandbox, ExecutionResult

# --- Configuration ---
MAX_ORCHESTRATOR_STEPS = 25
MAX_CODE_AGENT_ATTEMPTS = 5
EXECUTION_COMMAND = (
    "python3 -m venv .venv && "
    ". .venv/bin/activate && "
    "uv pip install --no-cache -r requirements.txt && "
    "uv pip install --no-cache -q pytest && "
    "pytest -p no:cacheprovider -v"
)
RUNS_DIR = Path("runs")


class Checkpoint(BaseModel):
    """Represents the state of the system at a given point."""

    objective: str
    history: List[str]
    latest_code: Optional[CodeAgentOutput]
    execution_feedback: Optional[str]
    orchestrator_step: int

    def save(self, path: Path) -> None:
        """Saves the checkpoint to a file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        """Loads a checkpoint from a file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def save_run_artifacts(
    run_dir: Path,
    iteration: int,
    agent_name: str,
    agent_input: Any,
    agent_output: Any,
    execution_result: Optional[ExecutionResult] = None,
) -> None:
    """Saves the artifacts for a single iteration of the main loop for debugging."""
    iter_dir = run_dir / f"iteration_{iteration:02d}_{agent_name}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(agent_input, BaseModel):
        (iter_dir / "agent_input.json").write_text(
            agent_input.model_dump_json(indent=4)
        )
    else:
        (iter_dir / "agent_input.txt").write_text(str(agent_input))

    if isinstance(agent_output, BaseModel):
        (iter_dir / "agent_output.json").write_text(
            agent_output.model_dump_json(indent=4)
        )
    else:
        (iter_dir / "agent_output.txt").write_text(str(agent_output))

    if isinstance(agent_output, CodeAgentOutput):
        code_dir = iter_dir / "code"
        code_dir.mkdir()
        for code_file in agent_output.files:
            file_path = code_dir / code_file.relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(code_file.content)

    if execution_result:
        report_content = f"""
--- EXECUTION REPORT ---
Timed Out: {execution_result.timed_out}
Exit Code: {execution_result.exit_code}
--- STDOUT ---
{execution_result.stdout or "No standard output."}
--- STDERR ---
{execution_result.stderr or "No standard error."}
--- END REPORT ---
        """
        (iter_dir / "execution_report.txt").write_text(report_content)

    logging.info(f"Saved artifacts for iteration {iteration} to {iter_dir}")


class Application:
    def __init__(
        self, objective: Optional[str] = None, resume_from: Optional[str] = None
    ):
        self.objective = objective
        self.resume_from = resume_from
        self.run_dir: Optional[Path] = None
        self.llm = LLMInterface()
        self.orchestrator: Optional[OrchestratorAgent] = None
        self.code_agent: Optional[CodeAgent] = None
        self.human_agent: Optional[HumanAgent] = None
        self.history: List[str] = []
        self.latest_code: Optional[CodeAgentOutput] = None
        self.execution_feedback: Optional[str] = None
        self.start_step = 1

    def _setup_run_dir(self):
        if self.resume_from:
            self.run_dir = Path(self.resume_from)
            if not self.run_dir.exists():
                raise FileNotFoundError(f"Resume directory not found: {self.run_dir}")
            logging.info(f"Resuming run from directory: {self.run_dir}")
        else:
            run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_dir = RUNS_DIR / f"run_{run_timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Starting new run in directory: {self.run_dir}")
            # Save the objective at the start of a new run
            if self.objective:
                (self.run_dir / "objective.txt").write_text(self.objective)

    def _load_checkpoint(self):
        if not self.resume_from:
            return

        checkpoint_path = self.run_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            logging.warning(
                f"No checkpoint.json found in {self.run_dir}. Cannot resume state."
            )
            # Try to load the objective from the original file if checkpoint is missing
            objective_path = self.run_dir / "objective.txt"
            if objective_path.exists():
                self.objective = objective_path.read_text()
                logging.info("Loaded objective from objective.txt")
            return

        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = Checkpoint.load(checkpoint_path)
        self.objective = checkpoint.objective
        self.history = checkpoint.history
        self.latest_code = checkpoint.latest_code
        self.execution_feedback = checkpoint.execution_feedback
        self.start_step = checkpoint.orchestrator_step + 1
        logging.info(f"Resuming from step {self.start_step}")

    def _initialize_agents(self):
        available_tools = {
            "code_agent": "Writes and refines code based on a prompt and execution feedback. Use this to write, test, and fix code.",
            "human_agent": "Asks the human user for clarification or input. Use this if the objective is unclear or you need guidance.",
            "finish": "Ends the process when the task is completed successfully or if you are stuck.",
        }
        self.orchestrator = OrchestratorAgent(self.llm, available_tools=available_tools)
        self.code_agent = CodeAgent(self.llm)
        self.human_agent = HumanAgent()

    def _handle_code_generation_action(
        self,
        prompt: str,
        command: str,
        orchestrator_step: int,
    ) -> Tuple[bool, CodeAgentOutput, Optional[str]]:
        """Handles the execution of the CodeAgent, including the retry loop."""
        logging.info("Delegating to CodeAgent...")
        execution_feedback = self.execution_feedback

        for attempt in range(1, MAX_CODE_AGENT_ATTEMPTS + 1):
            logging.info(
                f"--- Code Agent Attempt {attempt}/{MAX_CODE_AGENT_ATTEMPTS} ---"
            )

            agent_input = CodeAgentInput(
                prompt=prompt,
                command=command,
                previous_result=self.latest_code,
                execution_feedback=execution_feedback,
            )

            self.latest_code = self.code_agent.run(agent_input)

            with DockerSandbox(
                files=self.latest_code.files, command=agent_input.command
            ) as sandbox:
                execution_result = sandbox.run()

            save_run_artifacts(
                self.run_dir,
                orchestrator_step,
                f"code_agent_attempt_{attempt}",
                agent_input,
                self.latest_code,
                execution_result,
            )

            if execution_result.was_successful:
                logging.info("✅ Code execution was successful.")
                return True, self.latest_code, None
            else:
                logging.warning(f"❌ Code execution failed on attempt {attempt}.")
                execution_feedback = f"STDOUT:\n{execution_result.stdout}\n\nSTDERR:\n{execution_result.stderr}"
                logging.debug(f"Execution feedback:\n{execution_feedback}")

        logging.error("Code agent failed to produce working code after all attempts.")
        return False, self.latest_code, execution_feedback

    def run(self):
        """Main application loop."""
        self._setup_run_dir()
        self._initialize_agents()
        self._load_checkpoint()

        if not self.objective:
            logging.error(
                "Objective not set. "
                "Please provide one via --objective or --objective_file,"
                " or resume a run with a valid checkpoint."
            )
            return

        DockerSandbox.setup_image()

        try:
            for i in range(self.start_step, MAX_ORCHESTRATOR_STEPS + 1):
                logging.info(f"--- Orchestrator Step {i}/{MAX_ORCHESTRATOR_STEPS} ---")

                orchestrator_input = OrchestratorInput(
                    objective=self.objective, history=self.history
                )
                orchestrator_output = self.orchestrator.run(orchestrator_input)
                agent_name = orchestrator_output.agent_name
                agent_args = orchestrator_output.args

                save_run_artifacts(
                    self.run_dir,
                    i,
                    "orchestrator",
                    orchestrator_input,
                    orchestrator_output,
                )

                history_message = ""
                continue_loop = True

                if agent_name == "code_agent":
                    prompt = agent_args["prompt"]
                    command = (
                        EXECUTION_COMMAND + " && " + agent_args["command"]
                        if agent_args["command"]
                        else EXECUTION_COMMAND
                    )

                    was_successful, self.latest_code, self.execution_feedback = (
                        self._handle_code_generation_action(
                            prompt=prompt, command=command, orchestrator_step=i
                        )
                    )

                    files_detail = self.latest_code.model_dump_json(
                        include={"files"}, indent=2
                    )
                    if was_successful:
                        self.execution_feedback = None  # Reset on success
                        history_message = (
                            f"Action: code_agent. Result: Code executed successfully.\n"
                            f"Agent's Reasoning: {self.latest_code.reasoning}\n"
                            f"Generated Files:\n{files_detail}"
                        )
                    else:
                        history_message = (
                            f"Action: code_agent. Result: Execution failed after {MAX_CODE_AGENT_ATTEMPTS} attempts.\n"
                            f"Agent's Final Reasoning: {self.latest_code.reasoning}\n"
                            f"Final Generated Files:\n{files_detail}\n"
                            f"Execution Feedback:\n{self.execution_feedback}"
                        )

                elif agent_name == "human_agent":
                    question = agent_args.get(
                        "question", "I need help. What should I do next?"
                    )
                    human_input = HumanInput(question=question)
                    human_output = self.human_agent.run(human_input)
                    save_run_artifacts(
                        self.run_dir, i, "human_agent", human_input, human_output
                    )
                    history_message = f"Action: human_agent. Question: {question}. Answer: {human_output.answer}"

                elif agent_name == "finish":
                    reason = agent_args.get("reason", "Task completed.")
                    logging.info(f"🏁 Orchestrator decided to finish. Reason: {reason}")
                    history_message = f"Action: finish. Reason: {reason}"
                    continue_loop = False

                else:
                    logging.error(f"Unknown agent name received: {agent_name}")
                    history_message = (
                        "Action: unknown. Result: An internal error occurred."
                    )
                    continue_loop = False

                self.history.append(history_message)

                # --- Save Checkpoint on Successful Iteration ---
                checkpoint = Checkpoint(
                    objective=self.objective,
                    history=self.history,
                    latest_code=self.latest_code,
                    execution_feedback=self.execution_feedback,
                    orchestrator_step=i,  # The step that just finished
                )
                checkpoint.save(self.run_dir / "checkpoint.json")
                logging.info(f"Saved checkpoint for step {i}.")

                if not continue_loop:
                    break
            else:
                logging.warning(
                    "Reached max orchestrator steps (%d) without finishing.",
                    MAX_ORCHESTRATOR_STEPS,
                )
        except Exception as e:
            logging.error(f"💥 Unhandled exception caught: {e}", exc_info=True)
            logging.error(
                "The application has crashed. State from the last successful step is saved."
            )
            logging.error(
                f"To resume, run the script with: --resume_from {self.run_dir}"
            )
            raise  # Re-raise to terminate the program with a non-zero exit code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="An AI agent that can write and execute code."
    )
    parser.add_argument(
        "--objective",
        type=str,
        help="The main objective for the agent.",
    )
    parser.add_argument(
        "--objective_file",
        type=Path,
        help="Path to a text file containing the main objective.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Path to a previous run directory to resume from.",
    )
    args = parser.parse_args()

    objective = None
    if args.objective and args.objective_file:
        logging.warning(
            "Both --objective and --objective_file provided. Using --objective."
        )
        objective = args.objective
    elif args.objective:
        objective = args.objective
    elif args.objective_file:
        try:
            objective = args.objective_file.read_text()
        except FileNotFoundError:
            logging.error(f"Objective file not found: {args.objective_file}")
            exit(1)

    # If resuming, the objective will be loaded from the checkpoint.
    # If it's a new run, one of the objective args must be provided.
    if not args.resume_from and not objective:
        parser.error(
            "For a new run, please provide the objective using --objective or --objective_file."
        )

    try:
        load_dotenv()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.getLogger("google").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        app = Application(objective=objective, resume_from=args.resume_from)
        app.run()
    except Exception:
        raise
