"""
Fine-tuning causal experiment on Modal.com.

Fine-tunes GPT-2 Medium on 4 conditions of Hansard speech to test whether
anti-suffrage text causes more gender-biased output than control conditions.

Conditions:
1. anti_suffrage: ~500 "against" speeches
2. pro_suffrage: ~1,288 "for" speeches
3. random_hansard: ~1,000 random non-suffrage speeches (same era)
4. base: No fine-tuning (control)

Usage:
    # Prepare training data (local)
    python scripts/experiments/modal_finetune.py prepare-data

    # Run fine-tuning on Modal (GPU)
    modal run scripts/experiments/modal_finetune.py::app.run_finetuning

    # Generate from fine-tuned models on Modal
    modal run scripts/experiments/modal_finetune.py::app.run_generation
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data preparation (runs locally)
# ---------------------------------------------------------------------------

GENERATION_PROMPTS = [
    "Women in today's society",
    "The role of women is",
    "Women should",
]

N_GENERATIONS = 100  # per model per prompt


def prepare_training_data(
    classification_path: str,
    input_path: str,
    output_dir: str,
):
    """Prepare training text files for each fine-tuning condition."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clf = pd.read_parquet(classification_path)
    inputs = pd.read_parquet(input_path)

    merged = clf[["speech_id", "stance", "year"]].merge(
        inputs[["speech_id", "target_text"]],
        on="speech_id",
        how="inner",
    )
    merged = merged.dropna(subset=["stance", "target_text"])
    merged["stance"] = merged["stance"].astype(str)

    conditions = {}

    # Anti-suffrage
    anti = merged[merged["stance"] == "against"]
    conditions["anti_suffrage"] = anti

    # Pro-suffrage
    pro = merged[merged["stance"] == "for"]
    conditions["pro_suffrage"] = pro

    # Random Hansard (irrelevant, same era as suffrage debates 1860-1930)
    irrelevant = merged[
        (merged["stance"] == "irrelevant") &
        (merged["year"] >= 1860) &
        (merged["year"] <= 1930)
    ]
    # Sample to ~1000 for comparability
    n_random = min(1000, len(irrelevant))
    conditions["random_hansard"] = irrelevant.sample(n=n_random, random_state=42)

    # Save each condition as a text file (one speech per line, separated by <|endoftext|>)
    for name, data in conditions.items():
        texts = data["target_text"].values
        # Clean and join
        cleaned = []
        for t in texts:
            if isinstance(t, str) and len(t.strip()) > 50:
                cleaned.append(t.strip())

        text_path = output_dir / f"train_{name}.txt"
        with open(text_path, "w") as f:
            f.write("\n<|endoftext|>\n".join(cleaned))

        print(f"{name}: {len(cleaned)} speeches, "
              f"{sum(len(t) for t in cleaned):,} chars -> {text_path}")

    # Save metadata
    meta = {
        "conditions": {
            name: {"n_speeches": len(data), "n_cleaned": len(conditions[name])}
            for name, data in conditions.items()
        },
        "model": "gpt2-medium",
        "epochs": 3,
        "learning_rate": 5e-5,
        "max_seq_length": 512,
    }
    with open(output_dir / "finetune_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {output_dir / 'finetune_metadata.json'}")


# ---------------------------------------------------------------------------
# Modal fine-tuning
# ---------------------------------------------------------------------------

try:
    import modal

    app = modal.App("hansard-finetune")
    volume = modal.Volume.from_name("hansard-finetune-models", create_if_missing=True)
    data_volume = modal.Volume.from_name("hansard-experiment-data", create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch",
            "transformers",
            "accelerate",
            "datasets",
            "pandas",
            "pyarrow",
        )
    )

    @app.function(
        image=image,
        volumes={"/data": data_volume},
        timeout=300,
    )
    def upload_training_data(condition: str, text_content: str):
        """Upload training text to Modal volume."""
        path = f"/data/train_{condition}.txt"
        with open(path, "w") as f:
            f.write(text_content)
        data_volume.commit()
        print(f"Uploaded {len(text_content):,} chars to {path}")
        return {"condition": condition, "path": path, "size": len(text_content)}

    @app.function(
        image=image,
        gpu="A10G",
        volumes={
            "/models": volume,
            "/data": data_volume,
        },
        timeout=7200,  # 2 hours per condition
    )
    def finetune_one(condition: str, train_path: str, output_name: str):
        """Fine-tune GPT-2 Medium on one condition."""
        from datasets import Dataset
        from transformers import (
            GPT2LMHeadModel,
            GPT2Tokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        print(f"Fine-tuning on condition: {condition}")
        print(f"Training data: {train_path}")

        model_name = "gpt2-medium"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)

        # Load and tokenize text data
        with open(train_path, "r") as f:
            text = f.read()

        # Split on EOS token and tokenize
        block_size = 512
        encodings = tokenizer(text, return_tensors="np", truncation=False)
        input_ids = encodings["input_ids"][0]

        # Create blocks of block_size tokens
        blocks = []
        for i in range(0, len(input_ids) - block_size + 1, block_size):
            blocks.append(input_ids[i : i + block_size].tolist())

        print(f"  Created {len(blocks)} blocks of {block_size} tokens")
        dataset = Dataset.from_dict({
            "input_ids": blocks,
            "labels": blocks,
            "attention_mask": [[1] * block_size for _ in blocks],
        })
        dataset.set_format("torch")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        output_dir = f"/models/{output_name}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=50,
            save_steps=500,
            save_total_limit=1,
            fp16=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        volume.commit()

        print(f"Model saved to {output_dir}")
        return {"condition": condition, "output_dir": output_dir, "success": True}

    @app.function(
        image=image,
        gpu="A10G",
        volumes={"/models": volume, "/data": data_volume},
        timeout=3600,
    )
    def generate_from_model(
        condition: str,
        model_path: str,
        prompts: list,
        n_per_prompt: int = 500,
    ):
        """Generate continuations from a fine-tuned model."""
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        print(f"Generating from {condition} model at {model_path}")

        if condition == "base":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            model = GPT2LMHeadModel.from_pretrained(model_path)

        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        all_results = []
        for prompt in prompts:
            print(f"  Prompt: '{prompt}' ({n_per_prompt} generations)")
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            for i in range(n_per_prompt):
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=200,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                text = tokenizer.decode(output[0], skip_special_tokens=True)

                all_results.append({
                    "condition": condition,
                    "prompt": prompt,
                    "generation_idx": i,
                    "generated_text": text,
                })

        results_df = pd.DataFrame(all_results)
        out_path = f"/data/finetune_generations_{condition}.parquet"
        results_df.to_parquet(out_path, index=False)
        data_volume.commit()

        print(f"Generated {len(results_df)} texts, saved to {out_path}")
        return {"condition": condition, "n_generated": len(results_df)}

    @app.local_entrypoint()
    def run_pipeline(
        skip_upload: bool = False,
        skip_finetune: bool = False,
        skip_generate: bool = False,
        data_dir: str = "outputs/experiments",
    ):
        """Full pipeline: upload data -> fine-tune -> generate."""
        from pathlib import Path

        data_path = Path(data_dir)

        # Step 1: Upload training data
        if not skip_upload:
            print("Step 1: Uploading training data to Modal volume...")
            for condition in ["anti_suffrage", "pro_suffrage", "random_hansard"]:
                txt_path = data_path / f"train_{condition}.txt"
                if not txt_path.exists():
                    print(f"  WARNING: {txt_path} not found, skipping")
                    continue
                text = txt_path.read_text()
                result = upload_training_data.remote(condition, text)
                print(f"  {result}")

        # Step 2: Fine-tune
        if not skip_finetune:
            print("\nStep 2: Fine-tuning GPT-2 Medium on each condition...")
            conditions = {
                "anti_suffrage": "/data/train_anti_suffrage.txt",
                "pro_suffrage": "/data/train_pro_suffrage.txt",
                "random_hansard": "/data/train_random_hansard.txt",
            }
            ft_results = []
            for condition, path in conditions.items():
                print(f"  Starting {condition}...")
                result = finetune_one.remote(
                    condition=condition,
                    train_path=path,
                    output_name=f"gpt2-medium-{condition}",
                )
                ft_results.append(result)
                print(f"  {result}")

        # Step 3: Generate
        if not skip_generate:
            print("\nStep 3: Generating from all models...")
            gen_conditions = {
                "anti_suffrage": "/models/gpt2-medium-anti_suffrage",
                "pro_suffrage": "/models/gpt2-medium-pro_suffrage",
                "random_hansard": "/models/gpt2-medium-random_hansard",
                "base": "gpt2-medium",
            }
            gen_results = []
            for condition, model_path in gen_conditions.items():
                print(f"  Generating from {condition}...")
                result = generate_from_model.remote(
                    condition=condition,
                    model_path=model_path,
                    prompts=GENERATION_PROMPTS,
                    n_per_prompt=N_GENERATIONS,
                )
                gen_results.append(result)
                print(f"  {result}")

        print("\nPipeline complete!")

except ImportError:
    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning causal experiment")
    parser.add_argument("command", choices=["prepare-data"])
    parser.add_argument(
        "--data",
        default="outputs/llm_classification/claude_sonnet_45_full_results.parquet",
    )
    parser.add_argument(
        "--input",
        default="outputs/llm_classification/full_input_context_3_expanded.parquet",
    )
    parser.add_argument("--output-dir", default="outputs/experiments")
    args = parser.parse_args()

    if args.command == "prepare-data":
        prepare_training_data(args.data, args.input, args.output_dir)


if __name__ == "__main__":
    main()
