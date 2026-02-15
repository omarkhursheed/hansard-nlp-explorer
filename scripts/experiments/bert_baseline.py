"""
DistilBERT baseline for suffrage stance classification.

Fine-tunes distilbert-base-uncased with stratified 5-fold CV using LLM
pseudo-labels, then compares against the LLM classifier. Provides a strong
supervised baseline to justify the need for LLM-based classification.

Usage:
    python scripts/experiments/bert_baseline.py
    python scripts/experiments/bert_baseline.py --epochs 3 --batch-size 16
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


LABELS = ["for", "against", "both", "irrelevant"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 512


class SpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def run_fold(fold, train_texts, train_labels, val_texts, val_labels,
             tokenizer, device, epochs, batch_size, lr):
    """Train and evaluate one fold."""
    train_ds = SpeechDataset(train_texts, train_labels, tokenizer)
    val_ds = SpeechDataset(val_texts, val_labels, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    for epoch in range(epochs):
        loss = train_epoch(model, train_dl, optimizer, scheduler, device)
        print(f"  Fold {fold+1} epoch {epoch+1}/{epochs}: loss={loss:.4f}")

    preds, labels = predict(model, val_dl, device)

    # Free memory
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return preds, labels


def main():
    parser = argparse.ArgumentParser(description="DistilBERT baseline classifier")
    parser.add_argument(
        "--data",
        default="outputs/llm_classification/claude_sonnet_45_full_results.parquet",
    )
    parser.add_argument(
        "--input",
        default="outputs/llm_classification/full_input_context_3_expanded.parquet",
    )
    parser.add_argument("--output-dir", default="outputs/experiments")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit samples (0 = all data)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # Load data
    llm = pd.read_parquet(args.data)
    inputs = pd.read_parquet(args.input)

    merged = llm[["speech_id", "stance", "confidence"]].merge(
        inputs[["speech_id", "target_text"]],
        on="speech_id",
        how="inner",
    )
    merged = merged.dropna(subset=["stance", "target_text"])
    merged["stance"] = merged["stance"].astype(str).replace("neutral", "irrelevant")
    merged = merged[merged["stance"].isin(LABELS)]

    if args.max_samples > 0:
        merged = merged.sample(n=min(args.max_samples, len(merged)), random_state=42)

    print(f"Data: {len(merged)} speeches")
    print(f"Labels: {merged['stance'].value_counts().to_dict()}")

    texts = merged["target_text"].tolist()
    labels = [LABEL2ID[s] for s in merged["stance"]]
    labels = np.array(labels)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Stratified K-fold CV
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    all_preds = np.zeros(len(labels), dtype=int)

    start = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\nFold {fold+1}/{args.n_folds} ({len(train_idx)} train, {len(val_idx)} val)")
        train_texts = [texts[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = labels[val_idx]

        preds, _ = run_fold(
            fold, train_texts, train_labels, val_texts, val_labels,
            tokenizer, device, args.epochs, args.batch_size, args.lr,
        )
        all_preds[val_idx] = preds

    elapsed = time.time() - start
    print(f"\nTotal training time: {elapsed/60:.1f} min")

    # Convert predictions back to labels
    y_true = [ID2LABEL[l] for l in labels]
    y_pred = [ID2LABEL[p] for p in all_preds]

    # Metrics
    kappa = cohen_kappa_score(y_true, y_pred)
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    report = classification_report(y_true, y_pred, labels=LABELS,
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print(f"\n{'='*60}")
    print(f"DISTILBERT BASELINE ({args.n_folds}-FOLD CV)")
    print(f"{'='*60}")
    print(f"Accuracy: {100*accuracy:.1f}%")
    print(f"Cohen's kappa: {kappa:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")

    for label in LABELS:
        r = report.get(label, {})
        print(f"  {label:12s}: P={r.get('precision',0):.3f} R={r.get('recall',0):.3f} "
              f"F1={r.get('f1-score',0):.3f} n={r.get('support',0)}")

    print(f"\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    print(cm_df.to_string())

    # Save
    results = {
        "method": "distilbert_baseline",
        "model": MODEL_NAME,
        "n_samples": len(merged),
        "n_folds": args.n_folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_len": MAX_LEN,
        "training_time_minutes": round(elapsed / 60, 1),
        "accuracy": round(accuracy, 4),
        "kappa": round(kappa, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": {k: v for k, v in report.items() if k in LABELS},
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
    }

    json_path = output_dir / "bert_baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
