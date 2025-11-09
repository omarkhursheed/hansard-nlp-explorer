"""
Run pilot study across multiple LLM models for comparison.
This helps choose the best model for quality/cost tradeoff.
"""

import subprocess
import sys

# Models to compare (from OpenRouter)
MODELS = {
    "gpt-4o-mini": {
        "id": "openai/gpt-4o-mini",
        "description": "Fast, cheap, good quality (current choice)",
        "cost_per_1M": "$0.15 input, $0.60 output"
    },
    "gpt-4o": {
        "id": "openai/gpt-4o",
        "description": "Slower, expensive, highest quality",
        "cost_per_1M": "$2.50 input, $10 output"
    },
    "claude-3.5-sonnet": {
        "id": "anthropic/claude-3.5-sonnet",
        "description": "High quality, good at nuance",
        "cost_per_1M": "$3 input, $15 output"
    },
    "claude-3-haiku": {
        "id": "anthropic/claude-3-haiku",
        "description": "Fast, cheap, decent quality",
        "cost_per_1M": "$0.25 input, $1.25 output"
    },
    "gemini-pro-1.5": {
        "id": "google/gemini-pro-1.5",
        "description": "Google's model, good quality",
        "cost_per_1M": "$1.25 input, $5 output"
    },
}

def run_pilot_for_model(model_name, model_id):
    """Run pilot study for a specific model."""
    print("\n" + "="*70)
    print(f"Running pilot for: {model_name}")
    print(f"Model ID: {model_id}")
    print("="*70)

    # Run modal with specific model
    cmd = [
        "modal", "run", "modal_suffrage_classification.py",
        "--pilot",
        "--model", model_id
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"❌ FAILED for {model_name}")
        return False

    # Download results with model-specific filename
    output_filename = f"pilot_results_{model_name}.parquet"
    download_cmd = [
        "modal", "volume", "get", "suffrage-results",
        "pilot_results.parquet",
        f"./outputs/llm_classification/{output_filename}"
    ]

    result = subprocess.run(download_cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"✅ Saved results to outputs/llm_classification/{output_filename}")
        return True
    else:
        print(f"⚠️  Failed to download results for {model_name}")
        return False


def main():
    print("="*70)
    print("MULTI-MODEL PILOT COMPARISON")
    print("="*70)
    print("\nThis will run the pilot (100 speeches) through multiple models")
    print("to help you choose the best quality/cost tradeoff.\n")

    print("Models to test:")
    for name, info in MODELS.items():
        print(f"\n{name}:")
        print(f"  - {info['description']}")
        print(f"  - Cost: {info['cost_per_1M']}")

    print("\n" + "="*70)
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Run each model
    results = {}
    for model_name, model_info in MODELS.items():
        success = run_pilot_for_model(model_name, model_info['id'])
        results[model_name] = success

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for model_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{model_name}: {status}")

    print("\n" + "="*70)
    print("Next step: Run comparison analysis")
    print("  python3 compare_models.py")
    print("="*70)


if __name__ == "__main__":
    main()
