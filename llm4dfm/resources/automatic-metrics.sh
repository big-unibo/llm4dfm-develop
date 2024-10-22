#!/bin/bash

PY_PROG="../pipeline/metrics.py"
dir='LLM-rizzi'
demand=true

for ((i=4; i<=5; i++)); do
  for ((j=1; j<=9; j++)); do
    ex="rq$i-tc$j-ste"
    gt="exercise-$j"
    echo "Execution on $ex with gt $gt"
    python -W ignore "$PY_PROG" --exercise "$ex" --exercise_gt "$gt" --dir "$dir" --demand "$demand"
  done
done