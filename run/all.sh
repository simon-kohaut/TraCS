#!/usr/bin/env bash

[ -d "run/out" ] || mkdir "run/out"

# Iterate over all scenario/*.txt files and run tracs_gen on them
for file in scenarios/4_*.txt; do
    output="run/out/$(basename "${file%.txt}").resin"
    tracs_gen "$file" --output "$output" --backend local --coder-model google/gemma-4-E4B-it --judge-model google/gemma-4-E4B-it --max-iterations 5
done
