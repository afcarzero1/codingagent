# ---- Stage 1: Get the uv binary ----
# Use a minimal image that has curl to download the uv installer
# The official uv Docker images are hosted on ghcr.io
FROM ghcr.io/astral-sh/uv:latest as uv-installer

# ---- Stage 2: Build the final image ----
# Start from an official, minimal Python image.
# python:3.12-slim is a good choice for a balance of size and functionality.
FROM python:3.12-slim

# Set environment variables for best practices with Python in Docker
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy the uv and uvx binaries from the first stage into our final image
# This is the most efficient way to get uv into the image without
# needing to run installers or leaving build tools behind.
COPY --from=uv-installer /uv /usr/local/bin/uv
COPY --from=uv-installer /uvx /usr/local/bin/uvx

# --- Verification (Optional but Recommended) ---
# Run a command to verify that python and uv are installed correctly.
# This command will be executed during the build process.
RUN python -V && uv --version

# Set a default working directory
WORKDIR /code

# The image is now ready. The default command can be to open a shell.
CMD [ "bash" ]