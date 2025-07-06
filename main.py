import logging
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple, Optional

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
    previous_execution_feedback: Optional[str] = None,
) -> Tuple[bool, CodeAgentOutput, Optional[str]]:
    """
    Handles the execution of the CodeAgent, including the retry loop.

    Returns a tuple containing:
    - A boolean indicating if the final execution was successful.
    - The last generated code output.
    - The execution feedback if the last attempt failed.
    """
    logging.info("Delegating to CodeAgent...")
    latest_code = initial_code
    execution_feedback = previous_execution_feedback

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
            return True, latest_code, None
        else:
            logging.warning(f"❌ Code execution failed on attempt {attempt}.")
            execution_feedback = f"STDOUT:\n{execution_result.stdout}\n\nSTDERR:\n{execution_result.stderr}"
            logging.debug(f"Execution feedback:\n{execution_feedback}")

    logging.error("Code agent failed to produce working code after all attempts.")
    return False, latest_code, execution_feedback


def main() -> None:
    """Main function to orchestrate the agentic workflow."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("google").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)

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
            "human_agent": "Asks the human user for clarification or input. Use this if the objective is unclear or you need guidance. Do not be shy!",
            "finish": "Ends the process when the task is completed successfully or if you are stuck and cannot proceed.",
        }
        orchestrator = OrchestratorAgent(llm, available_tools=available_tools)
        code_agent = CodeAgent(llm)
        human_agent = HumanAgent()

        # --- Initial State ---
        objective = """
        You are tasked with creating a comprehensive Streamlit dashboard to analyze and compare the financial implications of renting versus buying a house in Sweden. You are the expert; make all technical and financial modeling decisions to create a robust and insightful tool.

        **Core Objective:**
        The primary goal is to determine the "break-even point" – the number of years after which buying a home becomes more financially advantageous than renting. The application must be interactive, user-friendly, and provide clear, actionable conclusions.

        **Financial Model & Currency:**
        - All financial calculations and user inputs must be in **Swedish Crowns (SEK)**.
        - You must research and use realistic default values for the Swedish market (particularly the Stockholm area) for all financial parameters. However, the user must be able to override every single default value.
        - The model must account for all significant costs associated with both buying and renting.

        **Dashboard & Interactivity - User Inputs:**
        The Streamlit dashboard must have a clear sidebar for user inputs, including:
        1.  **Buy Scenario:**
            -   Total Property Price (SEK)
            -   Down Payment  (%)
            -   Mortgage Interest Rate  (%)
            -   Loan Term  (years)
            -   Monthly Housing Association Fee  (SEK)
            -   Annual Property Maintenance Costs (% of property price)
            -   Expected Annual Property Value Appreciation (%)
        2.  **Rent Scenario:**
            -   Monthly Rent  (SEK)
            -   Expected Annual Rent Increase (%)
        3.  **General Assumptions:**
            -   Analysis Timeframe (years)
            -   Expected Annual Return on Investment (for the down payment and other saved/invested capital) (%)
            -   Inflation Rate (%)

        **Dashboard & Visualizations - Outputs:**
        The main area of the dashboard must clearly present the results:
        1.  **The Verdict:** A clear, top-level summary stating which option is financially better over the specified timeframe and, most importantly, the calculated **break-even point in years**.
        2.  **Cumulative Cost Chart:** A line chart visualizing the total cumulative costs of buying vs. renting over the analysis timeframe. The point where the lines cross must be clearly marked as the break-even point.
        3.  **Detailed Cost Breakdowns:** Use tables or charts to show the detailed breakdown of costs for both scenarios, including total interest paid, total principal paid, total rent, maintenance costs, and the opportunity cost of the down payment.
        4.  **Assumptions Panel:** Clearly list all the key financial assumptions being used for the calculation (whether default or user-provided).

        **Code & Structure:**
        - The primary application file should be `app.py`.
        - Organize the code logically. The financial calculations should be separated into their own functions or a separate module for clarity and reusability.
        - The code must be well-commented, include type hints, and be easy to understand.
        - Ensure all necessary libraries are listed in `requirements.txt`.

        **Your Autonomy:**
        You are the expert. Research the necessary financial formulas (e.g., loan amortization, future value calculations). Make informed decisions to build a tool that is not just functional but genuinely insightful for a user in Sweden.
        """
        history: List[str] = []
        latest_code: CodeAgentOutput = None
        execution_feedback: Optional[str] = None

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
                if len(agent_args["command"]) > 0:
                    command = EXECUTION_COMMAND + " && " + agent_args["command"]
                else:
                    command = EXECUTION_COMMAND

                was_successful, latest_code, execution_feedback = (
                    _handle_code_generation_action(
                        prompt=prompt,
                        command=command,
                        code_agent=code_agent,
                        initial_code=latest_code,
                        run_dir=current_run_dir,
                        orchestrator_step=i,
                        previous_execution_feedback=execution_feedback,
                    )
                )

                files_detail = latest_code.model_dump_json(include={"files"}, indent=2)
                if was_successful:
                    execution_feedback = None  # Reset feedback on success
                    history_message = (
                        f"Action: code_agent. Result: Code executed successfully.\n"
                        f"Agent's Reasoning: {latest_code.reasoning}\n"
                        f"Generated Files:\n{files_detail}"
                    )
                else:
                    history_message = (
                        f"Action: code_agent. Result: Execution failed after {MAX_CODE_AGENT_ATTEMPTS} attempts.\n"
                        f"Agent's Final Reasoning: {latest_code.reasoning}\n"
                        f"Final Generated Files:\n{files_detail}\n"
                        f"Execution Feedback:\n{execution_feedback}"
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
