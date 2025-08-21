#!/bin/bash

apt install tmux
poetry install
pip install -U "huggingface_hub[cli]"
pip install yq
HUGGINGFACE_TOKEN=$(yq -r '.hf.key.import' ./llm4dfm/resources/credentials.yml)
hf auth login --token "$HUGGINGFACE_TOKEN"
poetry run poe run_all