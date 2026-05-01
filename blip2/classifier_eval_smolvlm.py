import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText
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

from distill_blip2_to_tinyvlm import SmolVLMClassifier
from combined_preprocessing import get_distillation_dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def evaluate(
    run_dir,
    batch_size=64,
    output_root="outputs",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    split_data = get_distillation_dataset()
    val_dataset = split_data["test"]
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    student_vlm = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    model = SmolVLMClassifier(student_vlm).to(device)

    model_path = os.path.join(output_root, run_dir, "student_classifier.pt")
    full_run_dir = os.path.dirname(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading checkpoint {model_path} ...")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval", leave=False):
            pixel_values = batch["smol_pixel_values"].to(device)
            input_ids = batch["smol_input_ids"].to(device)
            attention_mask = batch["smol_attention_mask"].to(device)
            labels = batch["labels"].long().to(device)

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

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": conf,
        "roc_auc": roc_auc,
        "n_samples": int(len(y_true)),
    }

    out_json = os.path.join(full_run_dir, "eval_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig = plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})" if roc_auc is not None else "ROC curve")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (SmolVLM Student)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(os.path.join(full_run_dir, "roc_curve.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Warning: ROC plot failed: {e}")

    # Plot CM
    try:
        cm = np.array(conf)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        classes = ["0", "1"]
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title="Confusion Matrix", ylabel="True label", xlabel="Predicted label")
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(int(cm[i, j]), 'd'), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        fig.savefig(os.path.join(full_run_dir, "confusion_matrix.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Warning: CM plot failed: {e}")

    print("\nEvaluation summary:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"Saved metrics to {out_json}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--run-dir", default="distill_smolvlm_20260430_142048", help="Subfolder in outputs/ containing student_classifier.pt")
    p.add_argument("--output-root", default="outputs")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(run_dir=args.run_dir, batch_size=args.batch_size, output_root=args.output_root)