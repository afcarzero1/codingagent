
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
