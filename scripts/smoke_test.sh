#!/usr/bin/env sh
set -e

run_train="smoke_train_$(date +%Y%m%d_%H%M%S)"
run_pipeline="smoke_run_$(date +%Y%m%d_%H%M%S)"

python main.py --mode train --dry-run --run-name "$run_train" >/dev/null
python main.py --mode run --dry-run --run-name "$run_pipeline" >/dev/null

for run in "$run_train" "$run_pipeline"; do
  test -d "runs/$run" 
  test -f "runs/$run/logs/pipeline.log"
  test -f "runs/$run/manifest.json"
  test -f "runs/$run/config_used.yml"
done

echo "Smoke test OK"
