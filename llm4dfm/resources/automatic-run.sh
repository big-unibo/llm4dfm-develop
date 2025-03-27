#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=$#
prompt_version="rq5"
model="gpt"
model_label="gpt4o"
dir_label=$(date +"%Y-%m-%d_%H-%M-%S")
PY_PROG="$SCRIPT_DIR/../pipeline/pipeline.py"
GRAPH_PROG="$SCRIPT_DIR/../pipeline/csv_graph.py"
n_runs=1
exercises=""

# Define the components of the regex pattern as variables
ex_dir="datasets/"
ex_prefix="exercise-"
ex_version="sql"

# Parse JSON config if `-f file.json` is passed
while getopts "f:" opt; do
  case ${opt} in
    f ) FILE_PATH="$OPTARG" ;;
    \? ) echo "Usage: $0 [-f file_path] [n_runs] [ex_version] [prompt_version] [model] [model_label] [dir_label]"
         exit 1 ;;
  esac
done
shift $((OPTIND -1))  # Shift to remove processed options

# Load JSON values if a file is provided
if [[ -n "$FILE_PATH" && -f "$FILE_PATH" ]]; then
  echo "Loading configuration from JSON: $FILE_PATH"

  n_runs=$(jq -r '.n_runs // "'"$n_runs"'"' "$FILE_PATH")
  ex_version=$(jq -r '.ex_version // "'"$ex_version"'"' "$FILE_PATH")
  prompt_version=$(jq -r '.prompt_version // "'"$prompt_version"'"' "$FILE_PATH")
  model=$(jq -r '.model // "'"$model"'"' "$FILE_PATH")
  model_label=$(jq -r '.model_label // "'"$model_label"'"' "$FILE_PATH")
  dir_label=$(jq -r '.dir_label | select(. != "") // "'"$dir_label"'"' "$FILE_PATH")
  exercises=$(jq -r 'if (.exercises | length) > 0 then .exercises | join(" ") else "" end' "$FILE_PATH")
fi

# Override with positional arguments if provided
if [ -n "$1" ]; then n_runs=$1; fi
if [ -n "$2" ]; then ex_version=$2; fi
if [ -n "$3" ]; then prompt_version=$3; fi
if [ -n "$4" ]; then model=$4; fi
if [ -n "$5" ]; then exercises="$5"; fi
if [ -n "$6" ]; then model_label=$6; fi
if [ -n "$7" ]; then dir_label=$7; fi

echo "Runs: $n_runs, Exercise version: $ex_version, Prompt version: $prompt_version, Model: $model, Model label: $model_label, Label directory: $dir_label"

# If no file given, look for the match ones
# shellcheck disable=SC2235
if [ -z "$exercises" ] && ([ "$ARGS" -lt 5 ] || [ -z "$5" ]); then
  pwd
  regex="$ex_dir$ex_prefix*$ex_version*"
  for ex in $regex; do
    if [ -f "$ex" ]; then
      python -W ignore "$PY_PROG" --n_runs "$n_runs" --exercise "$ex" --p_version "$prompt_version" --exercise_version "$ex_version" --model "$model" --model_label "$model_label" --dir_label "$dir_label"
    fi
  done
  python -W ignore "$GRAPH_PROG" --prompt_version "$prompt_version" --exercise_v "$ex_version" --model_label "$model_label" --dir_label "$dir_label"
else
  IFS=' ' read -r -a ex_nums <<< "$exercises"
  for ex_num in "${ex_nums[@]}"; do
      python -W ignore "$PY_PROG" --n_runs "$n_runs" --exercise "$ex_dir$ex_prefix$ex_num-$ex_version-text.yml" --exercise_num "$ex_num" --p_version "$prompt_version" --exercise_version "$ex_version" --model "$model" --model_label "$model_label" --dir_label "$dir_label"
  done
  python -W ignore "$GRAPH_PROG" --prompt_version "$prompt_version" --exercise_v "$ex_version" --model_label "$model_label" --dir_label "$dir_label"
fi

