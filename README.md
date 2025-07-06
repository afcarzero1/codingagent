# System Overview: AI Coding Agent

This document outlines the architecture and workflow of the AI Coding Agent, a system designed to autonomously generate and execute code in a secure, sandboxed environment to accomplish a given task.

---

## 🏛️ System Architecture

The system is composed of three primary components that work in concert:

1.  **Orchestrator**: The "brain" of the operation. It's a high-level agent that oversees the entire task-solving process. It communicates with the user, breaks down complex problems, and directs the other components.

2.  **Code Generator**: A specialized language model responsible for writing Python code. It takes instructions from the Orchestrator and produces executable scripts tailored to the task.

3.  **Docker Sandbox (`DockerSandbox`)**: A secure, isolated environment for code execution. It prevents the agent from accessing the host filesystem or network, ensuring that even buggy or malicious code cannot cause harm. It dynamically builds a Docker image and runs the code inside a container.

---

## ⚙️ Workflow

The agent operates in a loop, moving from task definition to execution and refinement.

### Step 1: Task Definition

The process begins when the **Orchestrator** receives a high-level task from the user. For example: "Create a web scraper to get the titles of the top 10 articles from Hacker News."

### Step 2: Code Generation

The **Orchestrator** analyzes the task and instructs the **Code Generator** to write the necessary Python script. It provides context, requirements, and any relevant constraints. The Code Generator produces the Python code.

### Step 3: Sandbox Preparation

The **Orchestrator** takes the generated code and passes it to the `DockerSandbox`. The sandbox:
* Creates a temporary, empty workspace directory on the host machine.
* Writes the generated Python script(s) and any other required files into this workspace.
* Ensures the correct Docker image (`python-uv-image`) is available, building it if necessary using the dynamically generated `Dockerfile`. The `Dockerfile` is configured to set the `PYTHONPATH`, ensuring modules are correctly resolved.

### Step 4: Secure Execution

The `DockerSandbox` runs the code within a new Docker container.
* It mounts the temporary workspace into the `/app` directory inside the container.
* It executes the user-defined command (e.g., `python main.py`).
* It captures all outputs from the execution, including **`stdout`**, **`stderr`**, and the **`exit_code`**.
* A timeout is enforced to prevent infinite loops or long-running processes.

### Step 5: Analysis and Iteration

The **Orchestrator** receives the `ExecutionResult` from the sandbox.
* **If the code ran successfully** (`exit_code == 0`), the Orchestrator analyzes the `stdout` to determine if the task was completed successfully.
* **If the code failed** (non-zero `exit_code` or `stderr` has content), the Orchestrator analyzes the error messages to understand what went wrong.

Based on the analysis, the Orchestrator decides on the next action:
* **Task Complete**: If the output indicates success, the Orchestrator reports the result to the user.
* **Refine and Retry**: If the code failed or the result is incorrect, the Orchestrator goes back to **Step 2**, providing the error feedback to the Code Generator to produce a corrected version of the script.

This iterative loop of **Generate -> Execute -> Analyze** continues until the task is successfully completed or a maximum number of attempts is reached.

### Step 6: Cleanup

Once the `DockerSandbox` context is exited (at the end of each execution run), it automatically performs cleanup:
* The Docker container is stopped and removed.
* The temporary workspace directory on the host machine is deleted.

This ensures that no artifacts are left behind, maintaining a clean state for every execution.