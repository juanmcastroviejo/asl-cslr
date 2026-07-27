#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-data/raw}"
HOLDOUT="${2:-B}"
EPOCHS="${EPOCHS:-60}"

echo "=== data=$DATA  holdout-session=$HOLDOUT  epochs=$EPOCHS ==="

for group in hands hands_pose full; do
  for head in bilstm transformer; do
    run="${group}_${head}"
    echo ""
    echo "--- training $run ---"
    python -m src.train \
      --data "$DATA" \
      --group "$group" \
      --head "$head" \
      --run "$run" \
      --epochs "$EPOCHS" \
      --holdout-session "$HOLDOUT"
  done
done

echo ""
echo "--- benchmarking latency ---"
python -m src.benchmark --with-mediapipe --runs \
  results/hands_bilstm results/hands_transformer \
  results/hands_pose_bilstm results/hands_pose_transformer \
  results/full_bilstm results/full_transformer

echo ""
echo "--- generating figures and tables ---"
python -m src.evaluate --runs \
  results/hands_bilstm results/hands_transformer \
  results/hands_pose_bilstm results/hands_pose_transformer \
  results/full_bilstm results/full_transformer

echo ""
echo "Done. See results/figures/ and results/tables/."
