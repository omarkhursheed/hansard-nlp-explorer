#!/bin/bash

# Launch full classification run in background
# Job runs on Modal's servers - you can disconnect and it will continue

echo "================================================================================"
echo "LAUNCHING FULL CLASSIFICATION RUN"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - Dataset: 2,808 suffrage speeches"
echo "  - Prompt: v5 (active context use)"
echo "  - Context: 3 speeches before/after"
echo "  - Model: gpt-4o-mini"
echo "  - Expected cost: ~\$2.12"
echo "  - Expected time: ~45 minutes"
echo ""
echo "This will run on Modal's servers. You can:"
echo "  - Close this terminal (job continues)"
echo "  - Disconnect/travel (job continues)"
echo "  - Check status: modal app logs suffrage-classification"
echo "  - Download results when done: modal volume get suffrage-results full_results_v5_context_3.parquet ./outputs/llm_classification/"
echo ""
echo "================================================================================"
echo ""

read -p "Press ENTER to launch job (or Ctrl+C to cancel)... "

# Launch Modal job
modal run modal_suffrage_classification_v5.py

echo ""
echo "================================================================================"
echo "JOB LAUNCHED"
echo "================================================================================"
echo ""
echo "Check status:"
echo "  modal app logs suffrage-classification"
echo ""
echo "Download results when complete:"
echo "  modal volume get suffrage-results full_results_v5_context_3.parquet ./outputs/llm_classification/"
echo ""
echo "================================================================================"
