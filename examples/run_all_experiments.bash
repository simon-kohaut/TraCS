#!/usr/bin/env bash

set -euo pipefail

BACKBONES=(DeMo Polaris QCNet)
SPLITS=(val test)
NUM_WORKERS=180
SUBSAMPLE="0.2"

for LOGIC in logic_programs/*.resin; do
	LOGIC_NAME=$(basename "$LOGIC" .resin)
	for BACKBONE in "${BACKBONES[@]}"; do
		for SPLIT in "${SPLITS[@]}"; do
			echo "Running TraCS prediction with logic: $LOGIC, backbone: $BACKBONE, split: $SPLIT, subsample: $SUBSAMPLE (#1)"
			python examples/tracs_predict.py --logic "$LOGIC" --backbone "$BACKBONE" --split "$SPLIT" \
				--workers "$NUM_WORKERS"  --subsample "$SUBSAMPLE"

			# Run again because it sometimes hangs once (will resume automatically)
			echo "Running TraCS prediction with logic: $LOGIC, backbone: $BACKBONE, split: $SPLIT, subsample: $SUBSAMPLE (#2)"
			python examples/tracs_predict.py --logic "$LOGIC" --backbone "$BACKBONE" --split "$SPLIT" \
				--workers "$NUM_WORKERS"  --subsample "$SUBSAMPLE"
		done

		TARGET="data/results/$LOGIC_NAME/$BACKBONE.ipynb"
		if [ -f "$TARGET" ]; then
			echo "Skipping creating $TARGET, already exists."
		else
			mkdir -p "data/results/$LOGIC_NAME"
			papermill examples/tracs_train.ipynb "$TARGET" \
			    -p backbone "$BACKBONE" \
			    -p program_name "$LOGIC_NAME" \
				--cwd examples
		fi
	done
done
