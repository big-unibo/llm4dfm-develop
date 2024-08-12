#!/bin/bash

ARGS=$#
n_runs=0
prompt_version=v2
FIRST_STEP="./pipeline/first-step.py"
SECOND_STEP="./pipeline/second-step.py"

# Define the components of the regex pattern as variables
ex_dir="../../../datasets/"
ex_prefix="exercise-"
ex_version="sql"

# parse configuration args
if [ "$ARGS" -ge 1 ]; then
  n_runs=$1
  if [ "$ARGS" -ge 2 ]; then
    ex_version=$2
    if [ "$ARGS" -ge 3 ]; then
      prompt_version=$3
    fi
  fi
fi

echo "Runs: $n_runs, Exercise version: $ex_version, Prompt version: $prompt_version"



# Combine the variables to form the full pattern
regex="$ex_dir$ex_prefix*$ex_version*"

# If no file given, look for the match ones
if [ "$ARGS" -lt 4 ]; then
  for ex in $regex; do
    if [ -f "$ex" ]; then
      for ((i=1; i<=n_runs; i++)); do
        echo "Execution $i on $ex"
        python3 -W ignore "$FIRST_STEP" --exercise "$ex" --p_version "$prompt_version"
        python3 -W ignore "$SECOND_STEP" --exercise "$ex" --p_version "$prompt_version"
      done
    fi
  done
else
  # Else shift argument and read from them
  shift
  shift
  shift
  for part_file in "$@"; do
      for ((i=1; i<=n_runs; i++)); do
        echo "Execution $i:"
        python3 -W ignore "$FIRST_STEP" --exercise "$ex_dir$ex_prefix$part_file-$ex_version-text.yml"
        python3 -W ignore "$SECOND_STEP" --exercise "$ex_dir$ex_prefix$part_file-$ex_version-text.yml"
      done
  done
fi

