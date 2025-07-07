import logging
from typing import List, Dict, Any, Type, Literal
from pydantic import BaseModel, Field

from .agent import Agent
from ..llm_interface import LLMInterface
from .coding_agent import CodeAgentInput
from .human_agent import HumanInput


class AgentSelection(BaseModel):
    """The initial decision of which agent to use and why."""

    agent_name: Literal["code_agent", "human_agent", "finish"] = Field(
        ..., description="The name of the agent to be called."
    )
    reasoning: str = Field(
        ...,
        description="A detailed reasoning for choosing this agent and what you want it to accomplish.",
    )


class FinishArgs(BaseModel):
    """Arguments for the 'finish' action."""

    reason: str = Field(..., description="The reason for finishing the task.")


class OrchestratorInput(BaseModel):
    """The input to the orchestrator, including the main goal and history."""

    objective: str
    history: List[str] = []


class OrchestratorOutput(BaseModel):
    """The final output of the orchestrator, specifying the next agent and its arguments."""

    agent_name: str
    args: Dict[str, Any]


class OrchestratorAgent(Agent[OrchestratorOutput]):
    AGENT_SELECTION_PROMPT_TEMPLATE = """
You are an orchestrator agent. Your goal is to solve a programming task by coordinating other agents.
Based on the user's objective and the history of actions, you will first decide which single agent to call next.

IMPORTANT: Your decision and reasoning will be passed to a separate argument-generating AI. Therefore, your 'reasoning' must be a clear and direct instruction for the task you want the selected agent to perform.

Your available agents (tools) are:
--- AVAILABLE TOOLS ---
{available_tools}
--- END AVAILABLE TOOLS ---

The user's objective is:
--- OBJECTIVE ---
{objective}
--- END OBJECTIVE ---

Here is the history of actions taken so far:
--- HISTORY ---
{history}
--- END HISTORY ---

First, decide which agent to call and provide a clear, actionable reasoning for your choice.
You MUST choose one of the available agents. Your output must be a JSON object matching the AgentSelection schema.
"""

    ARGUMENT_GENERATION_PROMPT_TEMPLATE = """
You are an argument generation assistant for a multi-agent system.
Your task is to create the JSON arguments for the {agent_name} agent based on the orchestrator's reasoning.

The original objective was: "{objective}"

Here is the history of actions taken so far:
--- HISTORY ---
{history}
--- END HISTORY ---

The orchestrator chose to call the {agent_name} agent for the following reason:
--- REASONING ---
{reasoning}
--- END REASONING ---

Special Instructions for code_agent:
You must give the agent instructions! You do not need to write explicitly code in the prompt
unless you have a particular request. But you MUST give detailed instructions! Understand the
coding agent wont have all the context you do so you must provide it for him. ALL it needs
to know you must give (general aim, and his task). He will have access to current code, if there
is one.

If you are generating arguments for the code_agent, understand that the command you provide is an optional, additional command. 
The system uses the following Python logic to construct the final command that gets executed:

# This is the base command that always runs. It includes dependency installation and testing.
EXECUTION_COMMAND = (
    "python3 -m venv .venv && "
    ". .venv/bin/activate && "
    "uv pip install --no-cache -r requirements.txt && "
    "uv pip install --no-cache -q pytest && "
    "pytest -p no:cacheprovider -v"
)

# Your generated command (from agent_args["command"]) is appended if you provide one.
if len(agent_args["command"]) > 0:
    final_command = EXECUTION_COMMAND + " && " + agent_args["command"]
else:
    final_command = EXECUTION_COMMAND

    Your task is to generate the prompt for the coding agent and, if needed, the additional command string to be appended.
    The command is useful for running a specific part of the code (like a CLI) after the main tests have already run as part of EXECUTION_COMMAND.
    If no additional command is needed, you must still generate the prompt but provide an empty string "" for the command.
    DO NOT add && to the start or end of your command string.

Based on the reasoning, generate the JSON arguments for the code_agent. Your output MUST be a valid JSON object.
"""

    def __init__(self, llm_interface: LLMInterface, available_tools: Dict[str, str]):
        super().__init__(llm_interface)
        if not available_tools:
            raise ValueError("OrchestratorAgent requires at least one available tool.")
        self.available_tools = available_tools
        self.tool_to_model_map: Dict[str, Type[BaseModel]] = {
            "code_agent": CodeAgentInput,
            "human_agent": HumanInput,
            "finish": FinishArgs,
        }

    def _generate_tools_list(self) -> str:
        """Dynamically generates the list of tools for the prompt."""
        tool_descriptions = []
        for name, description in self.available_tools.items():
            tool_descriptions.append(f"- `{name}`: {description}")
        return "\n".join(tool_descriptions)

    def run(self, prompt_input: OrchestratorInput) -> OrchestratorOutput:
        history_str = (
            "\n".join(prompt_input.history)
            if prompt_input.history
            else "No actions taken yet."
        )
        tools_list_str = self._generate_tools_list()

        # --- Step 1: Decide which agent to call ---
        logging.info("Orchestrator: Step 1 - Selecting agent...")
        selection_prompt = self.AGENT_SELECTION_PROMPT_TEMPLATE.format(
            available_tools=tools_list_str,
            objective=prompt_input.objective,
            history=history_str,
        )
        agent_selection = self.llm_interface.generate_json(
            prompt=selection_prompt, response_model=AgentSelection
        )
        selected_agent_name = agent_selection.agent_name
        reasoning = agent_selection.reasoning
        logging.info(
            f"Orchestrator selected agent: '{selected_agent_name}' with reasoning: '{reasoning}'"
        )

        # --- Step 2: Generate the arguments for the chosen agent ---
        logging.info(
            f"Orchestrator: Step 2 - Generating arguments for '{selected_agent_name}'..."
        )
        argument_model = self.tool_to_model_map.get(selected_agent_name)
        if not argument_model:
            raise ValueError(
                f"No argument model found for agent: {selected_agent_name}"
            )

        args_prompt = self.ARGUMENT_GENERATION_PROMPT_TEMPLATE.format(
            agent_name=selected_agent_name,
            objective=prompt_input.objective,
            reasoning=reasoning,
            history=history_str,
        )

        generated_args_model = self.llm_interface.generate_json(
            prompt=args_prompt, response_model=argument_model
        )
        generated_args = generated_args_model.model_dump()
        logging.info(f"Orchestrator generated arguments: {generated_args}")

        return OrchestratorOutput(agent_name=selected_agent_name, args=generated_args)
