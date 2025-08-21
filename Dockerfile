FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    ca-certificates \
    lsb-release \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-venv python3.12-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt install -Y git \

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    ln -sfn /usr/bin/python3.12 /usr/bin/python && \
    ln -sfn /usr/local/bin/pip3 /usr/bin/pip

ENV POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

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
RUN chmod +x setup-container.sh
RUN chmod +x llm4dfm/run-all.sh
RUN chmod +x llm4dfm/resources/automatic-run.sh
RUN chmod +x llm4dfm/resources/automatic-metrics.sh
