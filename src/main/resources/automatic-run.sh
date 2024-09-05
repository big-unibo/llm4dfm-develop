#!/bin/bash

ARGS=$#
n_runs=0
prompt_version=v4
label=""
PY_PROG="../python/pipeline/pipeline.py"
t_sleep=6
# VIS_PROG="../python/pipeline/visualisation.py"

# Define the components of the regex pattern as variables
ex_dir="../../../datasets/"
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
if [ -n "$5" ]; then
  label=$5
fi


echo "Runs: $n_runs, Exercise version: $ex_version, Prompt version: $prompt_version"

# Combine the variables to form the full pattern
regex="$ex_dir$ex_prefix*$ex_version*"

# If no file given, look for the match ones
if [ "$ARGS" -lt 4 ] || [ -z "$4" ]; then
  for ex in $regex; do
    if [ -f "$ex" ]; then
      for ((i=1; i<=n_runs; i++)); do
        echo "Execution $i on $ex"
        python -W ignore "$PY_PROG" --exercise "$ex" --p_version "$prompt_version" --exercise_version "$ex_version" --label "$label" --runs "$n_runs"
        if [ "$i" != "$n_runs" ]; then
          sleep $t_sleep
        fi
      done
    fi
  done
else
  exercises="$4"
  IFS=' ' read -r -a ex_nums <<< "$exercises"
  for ex_num in "${ex_nums[@]}"; do
      for ((i=1; i<=n_runs; i++)); do
        echo "Execution $i:"
        python -W ignore "$PY_PROG" --exercise "$ex_dir$ex_prefix$ex_num-$ex_version-text.yml" --p_version "$prompt_version" --exercise_version "$ex_version" --label "$label" --runs "$n_runs"
        if [ "$i" != "$n_runs" ]; then
          sleep $t_sleep
        fi
      done
  done
fi

