#!/bin/bash

# Check status of classification job running on Modal

echo "Checking Modal app status..."
echo ""

modal app list | grep suffrage

echo ""
echo "To see detailed logs:"
echo "  modal app logs suffrage-classification"
echo ""
echo "To download results (if complete):"
echo "  modal volume get suffrage-results full_results_v5_context_3.parquet ./outputs/llm_classification/"
