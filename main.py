import logging
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

# Import agents and other components
from code_generator.sandbox import DockerSandbox, ExecutionResult
from code_generator.llm_interface import LLMInterface
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


def save_run_artifacts(
    run_dir: Path,
    iteration: int,
    agent_name: str,
    agent_input: Any,
    agent_output: Any,
    execution_result: ExecutionResult = None,
) -> None:
    """Saves the artifacts for a single iteration of the main loop for debugging."""
    iter_dir = run_dir / f"iteration_{iteration:02d}_{agent_name}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    (iter_dir / "agent_input.txt").write_text(str(agent_input))
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


def _handle_code_generation_action(
    prompt: str,
    command: str,
    code_agent: CodeAgent,
    initial_code: CodeAgentOutput,
    run_dir: Path,
    orchestrator_step: int,
) -> Tuple[bool, CodeAgentOutput]:
    """
    Handles the execution of the CodeAgent, including the retry loop.

    Returns a tuple containing:
    - A boolean indicating if the final execution was successful.
    - The last generated code output.
    """
    logging.info("Delegating to CodeAgent...")
    latest_code = initial_code
    execution_feedback = None

    for attempt in range(1, MAX_CODE_AGENT_ATTEMPTS + 1):
        logging.info(f"--- Code Agent Attempt {attempt}/{MAX_CODE_AGENT_ATTEMPTS} ---")

        agent_input = CodeAgentInput(
            prompt=prompt,
            command=command,
            previous_result=latest_code,
            execution_feedback=execution_feedback,
        )

        latest_code = code_agent.run(agent_input)

        with DockerSandbox(
            files=latest_code.files, command=agent_input.command
        ) as sandbox:
            execution_result = sandbox.run()

        save_run_artifacts(
            run_dir,
            orchestrator_step,
            f"code_agent_attempt_{attempt}",
            agent_input,
            latest_code,
            execution_result,
        )

        if execution_result.was_successful:
            logging.info("✅ Code execution was successful.")
            return True, latest_code
        else:
            logging.warning(f"❌ Code execution failed on attempt {attempt}.")
            execution_feedback = f"STDOUT:\n{execution_result.stdout}\n\nSTDERR:\n{execution_result.stderr}"
            logging.debug(f"Execution feedback:\n{execution_feedback}")

    logging.error("Code agent failed to produce working code after all attempts.")
    return False, latest_code


def main() -> None:
    """Main function to orchestrate the agentic workflow."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Setup ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_run_dir = RUNS_DIR / f"run_{run_timestamp}"
    current_run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Run directory: {current_run_dir}")

    try:
        # --- Agent Initialization ---
        llm = LLMInterface()
        available_tools = {
            "code_agent": "Writes and refines code based on a prompt and execution feedback. Use this to write, test, and fix code.",
            "human_agent": "Asks the human user for clarification or input. Use this if the objective is unclear or you need guidance.",
            "finish": "Ends the process when the task is completed successfully or if you are stuck and cannot proceed.",
        }
        orchestrator = OrchestratorAgent(llm, available_tools=available_tools)
        code_agent = CodeAgent(llm)
        human_agent = HumanAgent()

        # --- Initial State ---
        objective = """
        You are tasked with creating a Python chess engine. Your goal is to produce well-structured, class-based code (using type hints and docstrings!) that can determine the best move in a given chess position.

        **Expected Output & File Structure:**

        Your final solution can consist of something like the following files, please use this only as an initial point of thinking and structure the code
        as much as you can so that it is PROFESSIONAL.:
        1.  `requirements.txt`: This file must specify all necessary libraries. The `python-chess` library will be essential for this task.
        2.  `chess_engine.py`: This file should contain the primary logic, organized into classes. For instance, you might create an `Engine` class.
        3.  `test_chess_engine.py`: This file must contain a comprehensive suite of pytest tests for your engine.
        
        The expected output must have some kind of interface so that the user can test it and write moves done in a chessboard
        during a game and obtain a score for that position, input next moves, have a ranking of the best moves that can be done
        explore some of the variants. The idea is that the user wants to be able to understand why a move was good or bad (basically you can show the lines).
        My main frustration is when playing a game and the engine suggests the best move , then it is followed byh an opponent move but
        I do not understand why they can/cannot plain certain moves after it. For instance, the engine suggests for the opponen to play something
        but then I do not understand what happens if I play a certain move because the outcome if many moves in the future. I want to be able to interactively go 
        forward and backwards and overwrite the moves.
        
        You can add more files if you think it will improve the structure of the code, or change this structure as you think is better, 
        this is just a suggestion on how it could go well.


        **Testing Requirements (`test_chess_engine.py`):**

        Your tests must be robust and cover several key scenarios:
        -   **Checkmate-in-One:** Create a test case where the engine is presented with a position where it can deliver checkmate in a single move. The test must verify that the engine makes the correct winning move.
        -   **Capture of a Hanging Piece:** Set up a board where an opponent's piece is undefended. The test should confirm that the engine correctly identifies and executes the capture.
        -   **Move Legality:** Ensure that your engine does not produce illegal moves. You can test this by providing a position with very few legal moves and asserting that the engine's choice is one of them.
        
        Add any other test that is required to have comprehensive testing of the FULL application. We want to make sure that the user is able to have the best possible
        experience. Everything should be complete. Please make sure to add all the functions you think the user might expect from this app, do not leave something incomplete
        or working to the minimum possible. It must be comprehensive and production grade software 
        
        If you have any questions or want me to make some important choice please do not doubt in contacting me for making decisions, but try to be
        autonomous. YOU GOT THIS. THINK THINK AND NEVER STOP THINKING!!! Every decision must reflect your expertise as an incredible python developer
        with a lot of curiosity and desire to get things right.
        
        """
        history: List[str] = []
        latest_code: CodeAgentOutput = None

        DockerSandbox.setup_image()

        # --- Main Orchestration Loop ---
        for i in range(1, MAX_ORCHESTRATOR_STEPS + 1):
            logging.info(f"--- Orchestrator Step {i}/{MAX_ORCHESTRATOR_STEPS} ---")

            orchestrator_input = OrchestratorInput(objective=objective, history=history)
            orchestrator_output = orchestrator.run(orchestrator_input)
            agent_name = orchestrator_output.agent_name
            agent_args = orchestrator_output.args

            save_run_artifacts(
                current_run_dir,
                i,
                "orchestrator",
                orchestrator_input,
                orchestrator_output,
            )

            history_message = ""
            continue_loop = True

            if agent_name == "code_agent":
                prompt = agent_args["prompt"]

                was_successful, latest_code = _handle_code_generation_action(
                    prompt=prompt,
                    command=EXECUTION_COMMAND,  # Always use the fixed command
                    code_agent=code_agent,
                    initial_code=latest_code,
                    run_dir=current_run_dir,
                    orchestrator_step=i,
                )

                files_detail = latest_code.model_dump_json(include={"files"}, indent=2)
                if was_successful:
                    history_message = (
                        f"Action: code_agent. Result: Code executed successfully.\n"
                        f"Agent's Reasoning: {latest_code.reasoning}\n"
                        f"Generated Files:\n{files_detail}"
                    )
                else:
                    history_message = (
                        f"Action: code_agent. Result: Execution failed after {MAX_CODE_AGENT_ATTEMPTS} attempts.\n"
                        f"Agent's Final Reasoning: {latest_code.reasoning}\n"
                        f"Final Generated Files:\n{files_detail}"
                    )

            elif agent_name == "human_agent":
                question = agent_args.get(
                    "question", "I need help. What should I do next?"
                )
                human_input = HumanInput(question=question)
                human_output = human_agent.run(human_input)
                save_run_artifacts(
                    current_run_dir, i, "human_agent", human_input, human_output
                )
                history_message = f"Action: human_agent. Question: {question}. Answer: {human_output.answer}"

            elif agent_name == "finish":
                reason = agent_args.get("reason", "Task completed.")
                logging.info(f"🏁 Orchestrator decided to finish. Reason: {reason}")
                history_message = f"Action: finish. Reason: {reason}"
                continue_loop = False

            else:
                logging.error(f"Unknown agent name received: {agent_name}")
                history_message = "Action: unknown. Result: An internal error occurred."
                continue_loop = False

            history.append(history_message)
            if not continue_loop:
                break
        else:
            logging.warning(
                f"Reached max orchestrator steps ({MAX_ORCHESTRATOR_STEPS}) without finishing."
            )

    except (ValueError, KeyError) as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
