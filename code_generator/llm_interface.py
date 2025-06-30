import os
import logging
import json
from google import genai
from typing import List
from pydantic import BaseModel, Field

# --- Pydantic Models for Structured Output ---
# These models define the exact JSON structure we expect from the Gemini API.


class CodeFile(BaseModel):
    """Represents a single file to be written to the workspace."""

    relative_path: str = Field(
        ...,
        description="The path of the file relative to the workspace root, e.g., 'src/main.py'.",
    )
    content: str = Field(
        ..., description="The full source code or text content of the file."
    )


class TaskOutput(BaseModel):
    """Defines the complete, structured output from the code generation model."""

    # The 'command' field has been removed as it's now provided externally.
    files: List[CodeFile] = Field(
        ..., description="A list of all source and test files for the project."
    )


class LLMInterface:
    """Handles all communication with the Gemini API."""

    def __init__(self):
        """Initializes the Gemini client and generative model."""
        try:
            api_key = os.environ["GEMINI_API_KEY"]
            self.client = genai.Client(api_key=api_key)
        except KeyError:
            logging.error("GEMINI_API_KEY environment variable not set.")
            raise

        generation_config = {
            "temperature": 0.4,
            "top_p": 1.0,
            "top_k": 32,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json",
            # This line is corrected to pass the schema dictionary, not the class.
            "response_schema": TaskOutput,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
        ]

        self.model = "gemini-2.5-pro"
        logging.info("LLMInterface initialized successfully.")

    def generate_code(self, prompt: str, command: str) -> TaskOutput:
        """Generates code based on a prompt and execution context.

        Args:
            prompt: The user's prompt describing the desired functionality.
            command: The command that will be used to execute the code.

        Returns:
            A TaskOutput object containing the generated files.
        """
        logging.info(f"Generating code for prompt: '{prompt}'")
        generation_prompt = f"""
        You are an expert Python developer. Your task is to generate a set of Python files based on the following prompt.
        The code will be executed in a sandboxed Docker container running a standard Linux environment.
        The following command will be run from the root of the workspace to test your code:
        --- COMMAND ---
        {command}
        --- END COMMAND ---

        Please provide a complete and correct set of source and test files to accomplish the task described in the prompt.
        Ensure your test files are compatible with pytest.

        Prompt: "{prompt}"

        The output MUST be a valid JSON object that conforms to the required schema.
        """
        logging.debug(f"Sending generation prompt to LLM:\n{generation_prompt}")
        response = self.client.models.generate_content(
            model=self.model,
            contents=generation_prompt,
            config = {
                "response_mime_type": "application/json",
                "response_schema": TaskOutput
            }
        )
        logging.debug(f"Received response {response.text}")
        logging.info("Received response from LLM.")
        try:
            output_dict = json.loads(response.text)
            logging.info("Successfully parsed LLM response.")
            return TaskOutput(**output_dict)
        except (json.JSONDecodeError, TypeError) as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            logging.error(f"Raw LLM response: {response.text}")
            raise ValueError("LLM did not return a valid JSON object.") from e

    def provide_feedback(
        self,
        prompt: str,
        command: str,
        previous_result: TaskOutput,
        execution_feedback: str,
    ) -> TaskOutput:
        """Provides feedback to the model to refine the code.

        Args:
            prompt: The original user prompt.
            command: The command that was run.
            previous_result: The previous TaskOutput from the model.
            execution_feedback: A string containing the stdout/stderr from the last run.

        Returns:
            A new, refined TaskOutput object.
        """
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

        logging.debug(f"Sending feedback prompt to LLM:\n{feedback_prompt}")
        response = self.client.models.generate_content(
            model=self.model,
            contents=feedback_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": TaskOutput,
            }
        )
        logging.info("Received response from LLM after providing feedback.")
        try:
            output_dict = json.loads(response.text)
            logging.info("Successfully parsed refined LLM response.")
            return TaskOutput(**output_dict)
        except (json.JSONDecodeError, TypeError) as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            logging.error(f"Raw LLM response: {response.text}")
            raise ValueError("LLM did not return a valid JSON object.") from e

