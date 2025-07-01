import logging
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# Import the necessary classes from our local modules
from code_generator.sandbox import DockerSandbox, ExecutionResult
from code_generator.llm_interface import LLMInterface
from code_generator.agents.coding_agent import (
    CodeAgent,
    CodeAgentInput,
    CodeAgentOutput,
)

# --- Configuration ---
MAX_ATTEMPTS = 3
EXECUTION_COMMAND = (
    "python3 -m venv .venv && "
    ". .venv/bin/activate && "
    "uv pip install --no-cache -q pytest && "
    "pytest -p no:cacheprovider -v"
)
RUNS_DIR = Path("runs")


def save_attempt_artifacts(
    run_dir: Path,
    attempt: int,
    generated_code: CodeAgentOutput,
    execution_result: ExecutionResult,
) -> None:
    """Saves the generated code and execution results for a specific attempt.

    Args:
        run_dir: The main directory for the current execution run.
        attempt: The current attempt number.
        generated_code: The CodeAgentOutput object from the agent.
        execution_result: The ExecutionResult from the sandbox.
    """
    attempt_dir = run_dir / f"attempt_{attempt}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    # Save the generated code files
    for code_file in generated_code.files:
        file_path = attempt_dir / code_file.relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(code_file.content)

    # Save the execution report
    report_content = f"""
--- EXECUTION REPORT (Attempt {attempt}) ---
Timed Out: {execution_result.timed_out}
Exit Code: {execution_result.exit_code}
--- STDOUT ---
{execution_result.stdout if execution_result.stdout.strip() else "No standard output."}
--- STDERR ---
{execution_result.stderr if execution_result.stderr.strip() else "No standard error."}
--- END REPORT ---
    """
    (attempt_dir / "execution_report.txt").write_text(report_content)
    logging.info(f"Saved artifacts for attempt {attempt} to {attempt_dir}")


def main() -> None:
    """Main function to orchestrate the code generation and testing process."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create a directory for the current run, timestamped
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_run_dir = RUNS_DIR / f"run_{run_timestamp}"
    current_run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created run directory: {current_run_dir}")

    try:
        # 1. Instantiate the LLM interface and the agent.
        llm = LLMInterface()
        agent = CodeAgent(llm_interface=llm)

        # 2. Ensure the Docker image is ready.
        DockerSandbox.setup_image()

        # 3. Define the initial prompt for the AI.
        prompt = "Create a python function that adds two numbers, and a test for it."

        generated_code: CodeAgentOutput = None
        execution_result: ExecutionResult = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            logging.info(f"--- Attempt {attempt}/{MAX_ATTEMPTS} ---")

            # Prepare the input for the agent
            feedback = None
            if execution_result:
                feedback = f"STDOUT:\n{execution_result.stdout}\n\nSTDERR:\n{execution_result.stderr}"

            agent_input = CodeAgentInput(
                prompt=prompt,
                command=EXECUTION_COMMAND,
                previous_result=generated_code,
                execution_feedback=feedback,
            )

            # Run the agent to get the generated code
            generated_code = agent.run(agent_input)

            # Use the sandbox to execute the generated code
            with DockerSandbox(
                files=generated_code.files, command=EXECUTION_COMMAND
            ) as sandbox:
                execution_result = sandbox.run()

            # Save the artifacts for this attempt
            save_attempt_artifacts(
                current_run_dir, attempt, generated_code, execution_result
            )

            if execution_result.was_successful:
                logging.info(f"✅ Run was successful on attempt {attempt}!")
                break
            else:
                logging.error(
                    f"❌ Attempt {attempt} failed with exit code {execution_result.exit_code}."
                )

        else:  # This runs if the for loop completes without a 'break'
            logging.error(
                f"❌ Failed to generate correct code after {MAX_ATTEMPTS} attempts."
            )

    except (ValueError, KeyError) as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
