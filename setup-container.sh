#!/bin/bash

apt-get update && apt-get install -y jq python3-pip
pip install yq
apt install -y tmux
poetry install
pip install -U "huggingface_hub[cli]"
HUGGINGFACE_TOKEN=$(yq -r '.hf.key.import' ./llm4dfm/resources/credentials.yml)
hf auth login --token "$HUGGINGFACE_TOKEN"