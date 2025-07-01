from typing import List, Optional
from pydantic import BaseModel, Field
from code_generator.agents.agent import Agent


class CodeFile(BaseModel):
    relative_path: str = Field(
        ..., description="The path of the file relative to the workspace root."
    )
    content: str = Field(
        ..., description="The full source code or text content of the file."
    )


class CodeAgentOutput(BaseModel):
    files: List[CodeFile] = Field(
        ..., description="A list of all generated source files."
    )


class CodeAgentInput(BaseModel):
    prompt: str = Field(
        ...,
        description="The primary user request describing the desired functionality.",
    )
    command: str = Field(
        ..., description="The command that will be used to execute the generated code."
    )
    previous_result: Optional[CodeAgentOutput] = Field(
        None, description="The output from the previous run."
    )
    execution_feedback: Optional[str] = Field(
        None, description="The stdout/stderr from the last failed run."
    )


class CodeAgent(Agent[CodeAgentOutput]):
    """
    An agent that generates or refines code by creating a prompt and calling an LLM.
    """

    INITIAL_PROMPT_TEMPLATE = """
    You are an expert Python developer. Your task is to generate a set of Python files based on the following prompt.
    The aim of the program you are writing is: "{prompt}"

    The code will be executed in a sandboxed environment, and the following command will be run from the root of the workspace to test your code:
    --- COMMAND ---
    {command}
    --- END COMMAND ---
    """

    REFINEMENT_PROMPT_TEMPLATE = """
    You are an expert Python developer. Your previous attempt to write code had issues.
    Your original aim was: "{prompt}"
    The command used for execution was: "{command}"

    You previously generated the following files:
    --- PREVIOUS FILES ---
    {previous_files_json}
    --- END PREVIOUS FILES ---

    When the command was run, it failed with the following output:
    --- EXECUTION FEEDBACK ---
    {execution_feedback}
    --- END EXECUTION FEEDBACK ---

    Based on this feedback, please fix the code and provide a new, complete version of all the files.
    """

    def get_prompt(self, is_refinement: bool, **kwargs) -> str:
        """Selects and formats the appropriate prompt template."""
        if is_refinement:
            return self.REFINEMENT_PROMPT_TEMPLATE.format(**kwargs)
        else:
            return self.INITIAL_PROMPT_TEMPLATE.format(**kwargs)

    def run(self, prompt_input: CodeAgentInput) -> CodeAgentOutput:
        """
        Generates or refines code by processing the input, creating a prompt,
        and calling the LLMInterface.
        """
        is_refinement = (
            prompt_input.previous_result is not None
            and prompt_input.execution_feedback is not None
        )
        prompt_args = prompt_input.model_dump()

        if is_refinement:
            prompt_args["previous_files_json"] = (
                prompt_input.previous_result.model_dump()
            )

        final_prompt = self.get_prompt(is_refinement, **prompt_args)

        # Use the LLM interface to get a structured response
        return self.llm_interface.generate_json(
            prompt=final_prompt, response_model=CodeAgentOutput
        )
