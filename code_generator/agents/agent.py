import logging
from abc import ABC, abstractmethod

from typing import List, TypeVar, Generic
from pydantic import BaseModel, Field

from code_generator.llm_interface import LLMInterface

OutputType = TypeVar("OutputType", bound=BaseModel)
InputType = TypeVar("InputType", bound=BaseModel)


class Agent(Generic[OutputType], ABC):
    """
    An abstract agent that accepts a structured Pydantic model as input
    and returns a result as a specific Pydantic model.
    """

    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface

    @abstractmethod
    def run(self, prompt_input: InputType) -> OutputType:
        """Processes the structured input and returns a structured output."""
        pass


# --- Pydantic Models for Type Hinting ---
# This allows for generic type hinting for the response model.
OutputType = TypeVar("T", bound=BaseModel)


class TaskOutput(BaseModel):
    """Defines the complete, structured output for the code generation task."""

    files: List[CodeFile] = Field(
        ..., description="A list of all source and test files for the project."
    )


class LLMInterface:
    """
    Handles the application-specific logic for code generation and feedback,
    using the GeminiApiClient for communication.
    """

    def __init__(self):
        """Initializes the LLMInterface with a Gemini API client."""
        self.client = GeminiApiClient(model_name="gemini-1.5-pro")
        logging.info("LLMInterface initialized successfully.")

    def generate_code(self, prompt: str, command: str) -> TaskOutput:
        """Generates code by constructing a prompt and using the API client."""
        logging.info(f"Generating code for prompt: '{prompt}'")
        generation_prompt = f"""
        You are an expert Python developer. Your task is to generate a set of Python files based on the following prompt.
        The code will be executed in a sandboxed Docker container running a standard Linux environment.
        The following command will be run from the root of the workspace to test your code:
        --- COMMAND ---
        {command}
        --- END COMMAND ---

        Please adhere to the following project structure:
        - All main source code must be placed inside a `src/` directory.
        - All tests must be placed inside a `tests/` directory.
        - You MUST include an empty `src/__init__.py` file to ensure the `src` directory is treated as a package.

        Please provide a complete and correct set of source and test files to accomplish the task described in the prompt.
        Ensure your test files are compatible with pytest.

        Prompt: "{prompt}"

        The output MUST be a valid JSON object that conforms to the required schema.
        """
        return self.client.generate_structured_json(
            prompt_content=generation_prompt, response_model=TaskOutput
        )

    def provide_feedback(
        self,
        prompt: str,
        command: str,
        previous_result: TaskOutput,
        execution_feedback: str,
    ) -> TaskOutput:
        """Refines code by constructing a feedback prompt and using the API client."""
        logging.info("Providing feedback to the model for code refinement...")
        feedback_prompt = f"""
        You are an expert Python developer. Your task is to fix a set of Python files based on execution feedback.
        The original prompt was: "{prompt}"

        The code runs in a sandboxed Docker container. The command used for execution was:
        --- COMMAND ---
        {command}
        --- END COMMAND ---

        You previously generated the following code:
        --- FILES ---
        {[file.model_dump() for file in previous_result.files]}

        When the command was run, it failed with the following output:
        --- EXECUTION FEEDBACK ---
        {execution_feedback}
        --- END EXECUTION FEEDBACK ---

        Based on this feedback, please fix the code and provide a new, complete version of all the files.
        The output MUST be a valid JSON object that conforms to the required schema.
        """
        return self.client.generate_structured_json(
            prompt_content=feedback_prompt, response_model=TaskOutput
        )
