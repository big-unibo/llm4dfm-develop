#!/bin/bash

apt install tmux
poetry install
pip install -U "huggingface_hub[cli]"
apt-get install jq
HUGGINGFACE_TOKEN=$(yq -r '.hf.key.import' ./llm4dfm/resources/credentials.yml)
hf auth login --token "$HUGGINGFACE_TOKEN"