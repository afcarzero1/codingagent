from pydantic import BaseModel
from .agent import Agent


class HumanInput(BaseModel):
    """The input for the HumanAgent, containing the question to ask."""

    question: str


class HumanOutput(BaseModel):
    """The output from the HumanAgent, containing the user's answer."""

    answer: str


class HumanAgent(Agent[HumanOutput]):
    """
    An agent that prompts the human for input and returns their response.
    This agent does not use an LLM. It's a simple tool for the orchestrator
    to get human feedback.
    """

    def __init__(self):
        """
        Initializes the agent. Since it doesn't use an LLM,
        we pass `None` to the parent constructor.
        """
        super().__init__(llm_interface=None)

    def run(self, prompt_input: HumanInput) -> HumanOutput:
        """
        Prints the question to the console and waits for the user to type
        an answer and press Enter.
        """
        print("\n--- HUMAN ASSISTANCE REQUIRED ---")
        print(f"Question: {prompt_input.question}")
        answer = input("> ")
        print("--- THANK YOU ---")
        return HumanOutput(answer=answer)
