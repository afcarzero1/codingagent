import os
import subprocess
import tempfile
import shutil
import textwrap
import logging
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

from pydantic import BaseModel, Field


# Import the CodeFile class from the llm_interface module
class CodeFile(BaseModel):
    relative_path: str = Field(
        ..., description="The path of the file relative to the workspace root."
    )
    content: str = Field(
        ..., description="The full source code or text content of the file."
    )


# --- Data Structures ---
@dataclass
class ExecutionResult:
    """Represents the outcome of a command executed in Docker."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def was_successful(self) -> bool:
        """Returns True if the command executed successfully."""
        return self.exit_code == 0


# --- Docker Sandbox Class ---
class DockerSandbox:
    """Manages the lifecycle of a sandboxed Docker execution environment."""

    DOCKER_IMAGE_NAME: str = "python-uv-image"
    WORKSPACE_HOST_PREFIX: str = "gemini_workspace_"

    def __init__(self, files: List[CodeFile], command: str):
        """Initializes the sandbox with its state.

        Args:
            files: A list of CodeFile objects to create in the workspace.
            command: The command to execute within the sandbox.
        """
        self.files = files
        self.command = command
        self.workspace_path: Optional[Path] = None

    def __enter__(self) -> "DockerSandbox":
        """Sets up the workspace when entering the 'with' context."""
        self.workspace_path = Path(tempfile.mkdtemp(prefix=self.WORKSPACE_HOST_PREFIX))
        logging.info(f"Created temporary workspace: {self.workspace_path}")

        for code_file in self.files:
            full_path = self.workspace_path / code_file.relative_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(code_file.content)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up the workspace when exiting the 'with' context."""
        if self.workspace_path and self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
            logging.info(f"Cleaned up temporary workspace: {self.workspace_path}")

    def run(self, timeout: int = 30) -> ExecutionResult:
        """Runs the command inside the Docker container.

        This function runs the container with the same user ID as the host user
        to prevent file permission errors in the mounted volume.

        Args:
            timeout: The maximum execution time in seconds.

        Returns:
            An ExecutionResult object with the outcome of the command.
        """
        if not self.workspace_path:
            raise TypeError("run() must be called within a 'with' block.")

        logging.info("--- Starting Execution in Docker Container ---")
        logging.info(f"Command: {self.command}")
        host_path = str(self.workspace_path.resolve())
        container_path = "/app"

        docker_command = [
            "docker",
            "run",
            "--rm",
            f"--volume={host_path}:{container_path}",
        ]

        if os.name == "posix":
            user_id = os.getuid()
            group_id = os.getgid()
            docker_command.extend(["--user", f"{user_id}:{group_id}"])

        docker_command.extend([self.DOCKER_IMAGE_NAME, "bash", "-c", self.command])

        try:
            result = subprocess.run(
                docker_command, capture_output=True, text=True, timeout=timeout
            )
            return ExecutionResult(
                exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr
            )
        except subprocess.TimeoutExpired as e:
            logging.error("--- DOCKER EXECUTION FAILED: TIMEOUT ---")
            return ExecutionResult(
                exit_code=-1,
                stdout=e.stdout or "",
                stderr=e.stderr or f"Execution timed out after {timeout} seconds.",
                timed_out=True,
            )
        except Exception as e:
            logging.error(f"--- DOCKER EXECUTION FAILED: UNEXPECTED ERROR --- \n{e}")
            return ExecutionResult(
                exit_code=-1, stdout="", stderr=str(e), timed_out=False
            )

    @staticmethod
    def setup_image() -> None:
        """Checks if the Docker image exists and builds it if not."""
        try:
            subprocess.run(
                ["docker", "image", "inspect", DockerSandbox.DOCKER_IMAGE_NAME],
                check=True,
                capture_output=True,
            )
            logging.info(
                f"Docker image '{DockerSandbox.DOCKER_IMAGE_NAME}' already exists."
            )
        except subprocess.CalledProcessError:
            logging.warning(
                f"Docker image '{DockerSandbox.DOCKER_IMAGE_NAME}' not found."
            )
            DockerSandbox._create_dockerfile_if_not_exists()
            DockerSandbox._build_docker_image()

    @staticmethod
    def _create_dockerfile_if_not_exists() -> None:
        """Creates the Dockerfile."""
        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            logging.info("Dockerfile not found. Creating one...")
            dockerfile_path.write_text(
                textwrap.dedent("""
                FROM ghcr.io/astral-sh/uv:latest as uv-installer
                FROM python:3.12-slim
                ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
                # Set the PYTHONPATH to include the working directory
                ENV PYTHONPATH="/app"
                COPY --from=uv-installer /uv /usr/local/bin/uv
                COPY --from=uv-installer /uvx /usr/local/bin/uvx
                RUN python -V && uv --version
                WORKDIR /app
                CMD [ "bash" ]
            """)
            )

    @staticmethod
    def _build_docker_image() -> None:
        """Builds the Docker image."""
        logging.info(f"Building Docker image '{DockerSandbox.DOCKER_IMAGE_NAME}'...")
        try:
            subprocess.run(
                ["docker", "build", "-t", DockerSandbox.DOCKER_IMAGE_NAME, "."],
                check=True,
                capture_output=True,
                text=True,
            )
            logging.info("Docker image built successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"--- DOCKER BUILD FAILED ---\n{e.stderr}")
            raise
