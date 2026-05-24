#!/usr/bin/env bash

set -euo pipefail

BACKBONES=(DeMo Polaris QCNet)
SPLITS=(val test)
NUM_WORKERS=16

for BACKBONE in "${BACKBONES[@]}"; do
	for SPLIT in "${SPLITS[@]}"; do
		echo "Running TraCS prediction with backbone: $BACKBONE, split: $SPLIT"
		python examples/tracs_predict.py --backbone "$BACKBONE" --split "$SPLIT" --workers "$NUM_WORKERS"
	done
done
