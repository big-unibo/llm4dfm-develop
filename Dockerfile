FROM python:3.13-slim

# Set environment variables (avoids poetry creating virtualenvs inside container)
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files (first pyproject to leverage caching)
COPY pyproject.toml poetry.lock* ./

# Install deps
RUN poetry install --no-root

# Copy rest of the project
COPY . .

# Make sure the script is executable
RUN chmod +x llm4dfm/run-all.sh
RUN chmod +x llm4dfm/resources/automatic-run.sh
RUN chmod +x llm4dfm/resources/automatic-metrics.sh

# Set default entrypoint to your script
ENTRYPOINT ["/bin/bash", "poetry poe run_all"]