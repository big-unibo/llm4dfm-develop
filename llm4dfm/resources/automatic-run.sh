#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=$#
prompt_version=v4
model="gpt"
model_label="gpt4o"
dir_label=""
PY_PROG="$SCRIPT_DIR/../pipeline/pipeline.py"
GRAPH_PROG="$SCRIPT_DIR/../pipeline/csv_graph.py"
t_sleep=0
n_runs=1

# Define the components of the regex pattern as variables
ex_dir="../datasets/"
ex_prefix="exercise-"
ex_version="sql"

# parse configuration args
if [ -n "$1" ]; then
  n_runs=$1
fi
if [ -n "$2" ]; then
  ex_version=$2
fi
if [ -n "$3" ]; then
  prompt_version=$3
fi
if [ -n "$4" ]; then
  model=$4
fi
if [ -n "$6" ]; then
  model_label=$6
fi
if [ -n "$7" ]; then
  dir_label=$7
else
  dir_label=$(date +"%Y-%m-%d_%H-%M-%S")
fi

echo "Runs: $n_runs, Exercise version: $ex_version, Prompt version: $prompt_version, Model: $model, Model label: $model_label, Label directory: $dir_label"

# Combine the variables to form the full pattern
regex="$ex_dir$ex_prefix*$ex_version*"

# If no file given, look for the match ones
if [ "$ARGS" -lt 5 ] || [ -z "$5" ]; then
  for ex in $regex; do
    if [ -f "$ex" ]; then
      for ((i=1; i<=n_runs; i++)); do
        echo "Execution $i on $ex"
        python -W ignore "$PY_PROG" --exercise "$ex" --p_version "$prompt_version" --exercise_version "$ex_version" --model "$model" --model_label "$model_label" --dir_label "$dir_label"
        if [ "$i" != "$n_runs" ]; then
          sleep $t_sleep
        fi
      done
    fi
  done
  python -W ignore "$GRAPH_PROG" --prompt_version "$prompt_version" --exercise_v "$ex_version" --model_label "$model_label" --dir_label "$dir_label"
else
  exercises="$5"
  IFS=' ' read -r -a ex_nums <<< "$exercises"
  for ex_num in "${ex_nums[@]}"; do
      for ((i=1; i<=n_runs; i++)); do
        echo "Execution $i on $ex_num:"
        python -W ignore "$PY_PROG" --exercise "$ex_dir$ex_prefix$ex_num-$ex_version-text.yml" --p_version "$prompt_version" --exercise_version "$ex_version" --model "$model" --model_label "$model_label" --dir_label "$dir_label"
        if [ "$i" != "$n_runs" ]; then
          sleep $t_sleep
        fi
      done
  done
  python -W ignore "$GRAPH_PROG" --prompt_version "$prompt_version" --exercise_v "$ex_version" --model_label "$model_label" --dir_label "$dir_label"
fi

