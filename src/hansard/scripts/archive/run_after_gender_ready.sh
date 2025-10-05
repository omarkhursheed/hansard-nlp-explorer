#!/usr/bin/env bash
set -euo pipefail

# Waits for the regenerated gender dataset to be ready, then runs
# corpus and milestone analyses in sequence with logging.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/src/hansard/gender_analysis_data_FULL"
LOG_DIR="$PROJECT_ROOT/analysis_output"
mkdir -p "$LOG_DIR"

RUN_CORPUS=1
RUN_MILESTONES=1
SLEEP_SECS=60

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-corpus) RUN_CORPUS=0; shift ;;
    --no-milestones) RUN_MILESTONES=0; shift ;;
    --interval) SLEEP_SECS=${2:-60}; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--no-corpus] [--no-milestones] [--interval SECONDS]";
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "Watching for gender dataset readiness in: $DATA_DIR"

READY_FLAG=0
while true; do
  if [[ -f "$DATA_DIR/ALL_debates_with_confirmed_mps.parquet" && -f "$DATA_DIR/dataset_metadata.json" ]]; then
    READY_FLAG=1
    break
  fi
  echo "[\$(date)] Not ready yet; sleeping $SLEEP_SECS s..."
  sleep "$SLEEP_SECS"
done

echo "Gender dataset detected. Starting downstream analyses..."

if [[ "$RUN_CORPUS" -eq 1 ]]; then
  CORPUS_LOG="$LOG_DIR/corpus_analysis_post_gender.log"
  echo "[\$(date)] Running corpus analysis (sample 10000)... logs: $CORPUS_LOG"
  (cd "$PROJECT_ROOT/src/hansard" && python analysis/comprehensive_corpus_analysis.py --full --sample 10000) \
    > "$CORPUS_LOG" 2>&1 || { echo "Corpus analysis failed; see $CORPUS_LOG"; exit 1; }
  echo "[\$(date)] Corpus analysis complete."
fi

if [[ "$RUN_MILESTONES" -eq 1 ]]; then
  MILES_LOG="$LOG_DIR/milestones_post_gender.log"
  echo "[\$(date)] Running milestone analysis (all, aggressive)... logs: $MILES_LOG"
  (cd "$PROJECT_ROOT/src/hansard" && python analysis/comprehensive_milestone_analysis.py --all --filtering aggressive) \
    > "$MILES_LOG" 2>&1 || { echo "Milestone analysis failed; see $MILES_LOG"; exit 1; }
  echo "[\$(date)] Milestone analysis complete."
fi

echo "All scheduled analyses finished. Outputs under analysis/ directories."

