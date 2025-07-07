import os
import logging
import json
import time
from google import genai
import google.genai.errors as genai_errors
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class LLMInterface:
    """Handles all communication with the Gemini API."""

    def __init__(self, model: str = "gemini-2.5-pro"):
        """Initializes the Gemini client and generative model."""
        try:
            api_key = os.environ["GEMINI_API_KEY"]
            self.client = genai.Client(api_key=api_key)
        except KeyError:
            logger.error("GEMINI_API_KEY environment variable not set.")
            raise

        self.model = model
        self.last_request_time = 0  # Add timestamp for rate limiting
        logger.info("LLMInterface initialized successfully.")

    def generate_json(self, prompt: str, response_model: Type[T]) -> T:
        """
        Generates a JSON object from a prompt, with rate limiting and retry logic.

        Args:
            prompt: The user's prompt describing the desired functionality.
            response_model: The Pydantic model for the expected JSON response.

        Returns:
            A Pydantic model instance of the response.
        """
        # Simple rate limiting: ensure at least 10 seconds between requests
        current_time = time.time()
        if self.last_request_time > 0:
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < 10:
                sleep_duration = 10 - time_since_last_request
                logger.info(f"Rate limiting. Waiting for {sleep_duration:.2f} seconds.")
                time.sleep(sleep_duration)

        # Update the last request time before making the new request
        self.last_request_time = time.time()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending prompt (attempt {attempt + 1}/{max_retries}).")
                logger.debug(f"Prompt: '{prompt}'")

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": response_model,
                    },
                )

                logger.debug(f"Received raw response: {response.text}")
                logger.info("Received response from LLM.")

                try:
                    output_dict = json.loads(response.text)
                    logger.info("Successfully parsed LLM response.")
                    return response_model(**output_dict)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Failed to parse LLM response as JSON: {e}")
                    logger.error(f"Raw LLM response: {response.text}")
                    # This is not a server error, so we don't retry.
                    raise ValueError("LLM did not return a valid JSON object.") from e

            except genai_errors.ServerError as e:
                logger.warning(
                    f"Server error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    logger.info("Waiting 2 minutes before retrying...")
                    time.sleep(300)
                else:
                    logger.error(
                        "Max retries reached. Could not get a response from the server."
                    )
                    raise  # Re-raise the last exception

        # This line should not be reachable if the loop is correct.
        # It's a fallback in case the loop finishes without returning or raising.
        raise RuntimeError("Failed to get a response from the LLM after all retries.")
