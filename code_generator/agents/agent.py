from abc import ABC, abstractmethod

from typing import TypeVar, Generic
from pydantic import BaseModel

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
