import logging
import textwrap
from pathlib import Path
from typing import List

# Import the necessary classes from the sandbox module
from code_generator.sandbox import DockerSandbox, CodeFile

# --- Placeholders for AI-Generated Inputs ---
SRC_FILES: List[CodeFile] = [
    CodeFile(relative_path=Path("src/__init__.py"), content=""),
    CodeFile(
        relative_path=Path("src/function_to_test.py"),
        content=textwrap.dedent("""
            def add(x: float, y: float) -> float:
                return x + y
            def subtract(x: float, y: float) -> float:
                return x - y
        """),
    ),
]
TEST_FILES: List[CodeFile] = [
    CodeFile(
        relative_path=Path("tests/test_functions.py"),
        content=textwrap.dedent("""
            from src.function_to_test import add, subtract
            import pytest
            def test_add_positive_numbers() -> None:
                assert add(2, 3) == 5
            def test_subtract() -> None:
                assert subtract(10, 5) == 5
        """),
    )
]
EXECUTION_COMMAND_PLACEHOLDER: str = (
    "python3 -m venv .venv && "
    ". .venv/bin/activate && "
    "uv pip install --no-cache -q pytest && "
    "pytest -p no:cacheprovider -v"
)

def main() -> None:
    """Main function to orchestrate the code testing process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Ensure the Docker image is ready before creating any sandboxes.
    DockerSandbox.setup_image()

    # 2. Define the project state.
    all_files = SRC_FILES + TEST_FILES
    command = EXECUTION_COMMAND_PLACEHOLDER

    # 3. Use the sandbox as a context manager to handle setup and cleanup.
    with DockerSandbox(files=all_files, command=command) as sandbox:
        execution_result = sandbox.run()

    # 4. Process and display the results.
    logging.info("--- Execution Finished ---")
    if execution_result.was_successful:
        logging.info("✅ Run was successful! All tests passed.")
    else:
        logging.error(f"❌ Run failed with exit code {execution_result.exit_code}.")

    logging.info(f"""
        --- EXECUTION REPORT ---
        Timed Out: {execution_result.timed_out}
        Exit Code: {execution_result.exit_code}
        --- STDOUT ---
        {execution_result.stdout if execution_result.stdout.strip() else "No standard output."}
        --- STDERR ---
        {execution_result.stderr if execution_result.stderr.strip() else "No standard error."}
        --- END REPORT ---
    """)

if __name__ == "__main__":
    main()
