#!/usr/bin/env bash

# set -euo pipefail

BACKBONES=(DeMo Polaris QCNet)
SPLITS=(val test)
NUM_WORKERS=80
SUBSAMPLE="1.0"

for LOGIC in logic_programs/*.resin; do
	LOGIC_NAME=$(basename "$LOGIC" .resin)
	for BACKBONE in "${BACKBONES[@]}"; do
		for SPLIT in "${SPLITS[@]}"; do
			echo "Running TraCS prediction with logic: $LOGIC, backbone: $BACKBONE, split: $SPLIT (#1)"
			timeout -k 30 20m python examples/tracs_predict.py --logic "$LOGIC" --backbone "$BACKBONE" --split "$SPLIT" \
				--workers "$NUM_WORKERS"  --subsample "$SUBSAMPLE"

			# Run again because it sometimes hangs once (will resume automatically)
			echo "Running TraCS prediction with logic: $LOGIC, backbone: $BACKBONE, split: $SPLIT (#2)"
			timeout -k 30 20m python examples/tracs_predict.py --logic "$LOGIC" --backbone "$BACKBONE" --split "$SPLIT" \
				--workers "$NUM_WORKERS"  --subsample "$SUBSAMPLE"
			echo "Running TraCS prediction with logic: $LOGIC, backbone: $BACKBONE, split: $SPLIT (#3)"
			timeout -k 30 20m python examples/tracs_predict.py --logic "$LOGIC" --backbone "$BACKBONE" --split "$SPLIT" \
				--workers "$NUM_WORKERS"  --subsample "$SUBSAMPLE"
			echo "Running TraCS prediction with logic: $LOGIC, backbone: $BACKBONE, split: $SPLIT (#4)"
			timeout -k 30 20m python examples/tracs_predict.py --logic "$LOGIC" --backbone "$BACKBONE" --split "$SPLIT" \
				--workers "$NUM_WORKERS"  --subsample "$SUBSAMPLE"
		done
	done
done
