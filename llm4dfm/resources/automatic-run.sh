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
model_loading="import"
device="cpu"

# Define the components of the regex pattern as variables
ex_dir="datasets/"
ex_prefix="exercise-"
ex_version="sql"

# Parse JSON config if `-f file.json` is passed
while getopts "f:" opt; do
  case ${opt} in
    f ) FILE_PATH="$OPTARG" ;;
    \? ) echo "Usage: $0 [-f file_path] [n_runs] [ex_version] [prompt_version] [model] [model_loading] [model_label] [exercises] [dir_label] [device] [debug_print]"
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
  model_loading=$(jq -r '.model_loading // "'"$model_loading"'"' "$FILE_PATH")
  model_label=$(jq -r '.model_label // "'"$model_label"'"' "$FILE_PATH")
  device=$(jq -r '.device // "'"device"'"' "$FILE_PATH")
  dir_label=$(jq -r '.dir_label | select(. != "") // "'"$dir_label"'"' "$FILE_PATH")
  exercises=$(jq -r 'if (.exercises | length) > 0 then .exercises | join(" ") else "" end' "$FILE_PATH")
else
  # Override with positional arguments if provided
  if [ -n "$1" ]; then n_runs=$1; fi
  if [ -n "$2" ]; then ex_version=$2; fi
  if [ -n "$3" ]; then prompt_version=$3; fi
  if [ -n "$4" ]; then model=$4; fi
  if [ -n "$5" ]; then model_loading=$5; fi
  if [ -n "$6" ]; then model_label=$6; fi
  if [ -n "$7" ]; then exercises="$7"; fi
  if [ -n "$8" ]; then dir_label=$8; fi
  if [ -n "$9" ]; then device=$9; fi
fi

DEBUG=false

# Loop through all arguments
for arg in "$@"
do
  if [ "$arg" == "--debug_print" ]; then
    DEBUG=true
    break
  fi
done

ex_list=()

if [ -z "$exercises" ]; then
  regex="$ex_dir$ex_prefix*$ex_version*"
  for ex in $regex; do
    if [ -f "$ex" ]; then
      ex_list+=("$ex")  # Add matching files to the list
    fi
  done
else
  IFS=' ' read -r -a ex_nums <<< "$exercises"
  for ex_num in "${ex_nums[@]}"; do
    ex_file="$ex_dir$ex_prefix$ex_num-$ex_version-text.yml"
    ex_list+=("$ex_file")  # Collect exercise filenames
  done
fi

echo "Runs: $n_runs, Prompt version: $prompt_version, Model: $model, Model label: $model_label, Label directory: $dir_label, Exercises: [${ex_list[@]}], Device: $device"

if [ ${#ex_list[@]} -gt 0 ]; then
    CMD="python -W ignore \"$PY_PROG\" \
    --n_runs \"$n_runs\" \
    --exercises \"${ex_list[@]}\" \
    --p_version \"$prompt_version\" \
    --exercise_version \"$ex_version\" \
    --model \"$model\" \
    --model_loading \"$model_loading\" \
    --model_label \"$model_label\" \
    --dir_label \"$dir_label\"
    --device \"$device\""

    # Append --debug_print if DEBUG is true
    if [ "$DEBUG" = true ]; then
      CMD="$CMD --debug_print"
    fi

    # Execute the command
    eval $CMD

    python -W ignore "$GRAPH_PROG" --prompt_version "$prompt_version" --exercise_v "$ex_version" --model_label "$model_label" --dir_label "$dir_label" --model_loading "$model_loading" --device "$device"
fi