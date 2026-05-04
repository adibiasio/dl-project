import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from peft import PeftModel
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
    labels_for_decode = batch_labels.clone()
    labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
    decoded = processor.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
    y = [1 if "yes" in txt.lower() else 0 for txt in decoded]
    return torch.tensor(y, dtype=torch.long)


def evaluate(
    run_dir,
    datapath="./data_preprocessed_blip2",
    batch_size=64,
    output_root="outputs",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_from_disk(datapath)
    data.set_format("torch")
    val_loader = DataLoader(data["validation"], batch_size=batch_size)

    # Resolve paths
    full_run_dir = os.path.join(output_root, run_dir)
    classifier_path = os.path.join(full_run_dir, "classifier.pt")
    lora_adapter_path = os.path.join(full_run_dir, "lora_adapter")

    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")
    if not os.path.exists(lora_adapter_path):
        raise FileNotFoundError(f"LoRA adapter not found: {lora_adapter_path}")

    print(f"Loading processor and base VLM to device={device}...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    base_vlm = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.bfloat16
    ).to(device)

    # Re-attach LoRA adapter
    print(f"Loading LoRA adapter from {lora_adapter_path} ...")
    vlm = PeftModel.from_pretrained(base_vlm, lora_adapter_path).to(device)

    # Wrap in classifier and load classifier head weights
    model = VLMClassifier(vlm).to(device)
    print(f"Loading classifier head from {classifier_path} ...")
    state = torch.load(classifier_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)

    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = decode_labels_from_batch(batch["labels"], processor).to(device)

            logits = model(pixel_values, input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    conf = confusion_matrix(y_true, y_pred).tolist()

    roc_auc = None
    if len(np.unique(y_true)) == 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": conf,
        "roc_auc": roc_auc,
        "n_samples": int(len(y_true)),
    }

    os.makedirs(full_run_dir, exist_ok=True)
    out_json = os.path.join(full_run_dir, "eval_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig = plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})" if roc_auc else "ROC curve")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(full_run_dir, "roc_curve.png")
        fig.savefig(roc_path)
        plt.close(fig)
        print(f"Saved ROC curve to {roc_path}")
    except Exception as e:
        print(f"Warning: failed to save ROC curve: {e}")

    # Confusion matrix
    try:
        cm = np.array(conf)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        classes = ["0", "1"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix")
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(int(cm[i, j]), 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        cm_path = os.path.join(full_run_dir, "confusion_matrix.png")
        fig.savefig(cm_path)
        plt.close(fig)
        print(f"Saved confusion matrix to {cm_path}")
    except Exception as e:
        print(f"Warning: failed to save confusion matrix: {e}")

    # Score distribution
    try:
        fig = plt.figure(figsize=(6, 4))
        bins = np.linspace(0.0, 1.0, 21)
        plt.hist(y_prob[y_true == 0], bins=bins, density=True, alpha=0.6, label="neg (0)", color="C0", edgecolor="black")
        plt.hist(y_prob[y_true == 1], bins=bins, density=True, alpha=0.6, label="pos (1)", color="C1", edgecolor="black")
        plt.xlabel("Predicted probability (positive class)")
        plt.ylabel("Density")
        plt.title("Score Distribution by True Label")
        plt.legend(loc="upper center")
        plt.tight_layout()
        sd_path = os.path.join(full_run_dir, "score_distribution.png")
        fig.savefig(sd_path)
        plt.close(fig)
        print(f"Saved score distribution to {sd_path}")
    except Exception as e:
        print(f"Warning: failed to save score distribution: {e}")

    print("\nEvaluation summary:")
    print(f"  samples:          {metrics['n_samples']}")
    print(f"  accuracy:         {metrics['accuracy']:.4f}")
    print(f"  precision:        {metrics['precision']:.4f}")
    print(f"  recall:           {metrics['recall']:.4f}")
    print(f"  f1:               {metrics['f1']:.4f}")
    if roc_auc is not None:
        print(f"  roc_auc:          {roc_auc:.4f}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")
    print(f"Saved metrics to {out_json}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="./data_preprocessed_blip2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--run-dir", default="blip2_20260503_221248", help="Timestamped subfolder inside --output-root containing classifier.pt and lora_adapter/")
    p.add_argument("--output-root", default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        datapath=args.data_path,
        batch_size=args.batch_size,
        run_dir=args.run_dir,
        output_root=args.output_root,
    )