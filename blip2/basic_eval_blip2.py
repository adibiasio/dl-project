import os
import glob
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from tqdm import tqdm
import argparse
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from classifier_model_blip2 import VLMClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def decode_labels_from_batch(batch_labels, processor):
    # batch_labels: tensor with -100 for padding as in training
    labels_for_decode = batch_labels.clone()
    labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
    decoded = processor.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
    # binary class: 1 if 'yes' in the decoded text else 0
    y = [1 if "yes" in txt.lower() else 0 for txt in decoded]
    return torch.tensor(y, dtype=torch.long)


def evaluate(
    run_dir,
    datapath="./data_preprocessed_blip2",
    batch_size=64,
    output_root="outputs",
    vlm_path="blip2-finetuned",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_from_disk(datapath)
    # ensure tensors
    data.set_format("torch")
    val_dataset = data["validation"]

    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Loading processor and model to device={device}...")

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    # load the VLM (decoder) - prefer local finetuned path if provided
    print(f"Attempting to load VLM from {vlm_path} ...")
    model = Blip2ForConditionalGeneration.from_pretrained(vlm_path).to(device)
    print(f"Loaded VLM from {vlm_path}")

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        
        for batch in tqdm(val_loader, desc="Eval", leave=False):
            labels = decode_labels_from_batch(batch["labels"], processor).to(device)

            outputs = model.generate(
                pixel_values=batch['pixel_values'].to(device),
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                max_new_tokens=3
            )

            generated_tokens = outputs[:, batch["input_ids"].shape[1]:]

            preds = processor.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )

            y_pred.extend([1 if p.strip().lower().startswith("yes") else 0 for p in preds])
            y_true.extend(labels.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    conf = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": conf,
        "n_samples": int(len(y_true)),
    }

    # save metrics into a timestamped eval subfolder inside the provided run dir
    out_json = os.path.join(run_dir, "eval_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # generate and save a confusion matrix plot next to the ROC curve
    try:
        cm = np.array(conf)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # tick labels for binary classes 0 and 1
        classes = ["0", "1"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix")

        # annotate cells with counts
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(int(cm[i, j]), 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        cm_path = os.path.join(run_dir, "confusion_matrix.png")
        fig.savefig(cm_path)
        plt.close(fig)
        print(f"Saved confusion matrix to {cm_path}")
    except Exception as e:
        print(f"Warning: failed to compute/save confusion matrix: {e}")

    # print concise summary
    print("\nEvaluation summary:")
    print(f"  samples: {metrics['n_samples']}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall: {metrics['recall']:.4f}")
    print(f"  f1: {metrics['f1']:.4f}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")
    print(f"Saved metrics to {out_json}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="./data_preprocessed_blip2", help="Path to preprocessed dataset")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--run-dir", default="outputs/blip2_basic", help="Folder containing training artifacts (e.g. outputs/blip2_<timestamp>) where classifier.pt is located and where eval outputs will be saved")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--vlm-path", default="blip2-finetuned", help="Path or HF id for the Blip2 VLM to load (local finetuned model or HF repo id). Defaults to local 'blip2-finetuned'.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(
        datapath=args.data_path,
        batch_size=args.batch_size,
        run_dir=args.run_dir,
        output_root=args.output_root,
        vlm_path=args.vlm_path,
    )
