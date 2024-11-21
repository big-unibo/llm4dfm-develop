#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_PROG="$SCRIPT_DIR/../pipeline/metrics.py"
GRAPH_PROG="$SCRIPT_DIR/../pipeline/csv_graph.py"
OUTPUT="$SCRIPT_DIR/../../outputs/"
dir=''
version="sql"

if [ -n "$1" ]; then
  dir=$1
fi
if [ -n "$2" ]; then
  version=$2
fi

# Iterate over files matching the pattern
for file in "$OUTPUT$dir"/*.yml; do
    # Check if it's a file (skip if it's not)
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        ex_name=${filename%.*}
        # Depending on file name convention, first one is for first occurrence to understand exercise number
        ex_number=$(echo "$ex_name" | grep -o -E '[0-9]+' | head -n 1)
        # Second one is for last occurrence for ex. number
        # ex_number=$(echo "$ex_name" | grep -o -E '[0-9]+' | tail -n 1)
        gt="exercise-$ex_number"

        echo "Execution on $ex_name ex_num $ex_number"

        python -W ignore "$PY_PROG" --exercise "$ex_name" --exercise_num "$ex_number" --exercise_gt "$gt" --dir "$dir" --version "$version"
    fi
done

python -W ignore "$GRAPH_PROG" --dir "$dir"