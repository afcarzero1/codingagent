import os
import logging
import json
from google import genai
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMInterface:
    """Handles all communication with the Gemini API."""

    def __init__(self, model: str = "gemini-2.5-pro"):
        """Initializes the Gemini client and generative model."""
        try:
            api_key = os.environ["GEMINI_API_KEY"]
            self.client = genai.Client(api_key=api_key)
        except KeyError:
            logging.error("GEMINI_API_KEY environment variable not set.")
            raise

        self.model = model
        logging.info("LLMInterface initialized successfully.")

    def generate_json(self, prompt: str, response_model: Type[T]) -> T:
        """Generates code based on a prompt and execution context.

        Args:
            prompt: The user's prompt describing the desired functionality.
            command: The command that will be used to execute the code.

        Returns:
            A TaskOutput object containing the generated files.
        """
        logging.info(f"Sending prompt: '{prompt}'")
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_model,
            },
        )
        logging.debug(f"Received response {response.text}")
        logging.info("Received response from LLM.")
        try:
            output_dict = json.loads(response.text)
            logging.info("Successfully parsed LLM response.")
            return response_model(**output_dict)
        except (json.JSONDecodeError, TypeError) as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            logging.error(f"Raw LLM response: {response.text}")
            raise ValueError("LLM did not return a valid JSON object.") from e
