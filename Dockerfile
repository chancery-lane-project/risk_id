# Use an official Python runtime as a base image
FROM python:3.10.15

# Install system packages needed to compile hdbscan and similar libs
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gfortran \
    libatlas-base-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock to the working directory
COPY pyproject.toml poetry.lock README.md /app/

# Install Poetry
RUN pip install poetry

# Install dependencies (disable virtualenv so Poetry installs system-wide)
RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-root

# Copy the application code
COPY tclp/ /app/tclp/

# Set environment variables for FastAPI
ENV HOST=0.0.0.0
ENV OPENBLAS_NUM_THREADS=1

# Expose FastAPI ports
EXPOSE 8000
EXPOSE 8080

# Default command (will be overridden by docker-compose)
CMD ["echo", "Base image built!"]
