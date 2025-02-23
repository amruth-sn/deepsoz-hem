# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for building pyEDFlib
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libglib2.0-dev \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*  # Cleanup

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=deepsoz-hem/,target=deepsoz-hem/ \
    python -m pip install ./deepsoz-hem

# Switch to the non-privileged user to run the application.
USER appuser

# Create a data volume
VOLUME ["/data"]
VOLUME ["/output"]

# Define environment variables
ENV INPUT=""
ENV OUTPUT=""

# Run the application
CMD python3 -m deepsoz-hem "/data/$INPUT" "/output/$OUTPUT"
