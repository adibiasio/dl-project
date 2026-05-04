import os
import math
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    BitsAndBytesConfig,
    LlavaForConditionalGeneration, 
    AutoModelForImageTextToText,
)

from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from classifier import LLaVAClassifier
from combined_preprocessing import get_distillation_dataset


# Config
BATCH_SIZE = 4  # Reduced batch size to prevent OOM with LLaVA-7B + SmolVLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01


class SmolVLMClassifier(nn.Module):
    """Classifier head for SmolVLM-Instruct student.
    SmolVLM outputs image-text multimodal features; we extract the last hidden state
    from the language model output and apply a classifier head.
    """

    def __init__(self, vlm_model):
        super().__init__()
        self.vlm = vlm_model
        
        # Determine hidden size from the model config
        hidden_size = vlm_model.config.text_config.hidden_size
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1024, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract last hidden state from language model outputs
        hidden_states = outputs.hidden_states[-1]

        # Get last non-padding token
        last_indices = attention_mask.sum(dim=1) - 1
        last_token = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            last_indices
        ].float()

        logits = self.classifier(last_token).squeeze(-1)
        return logits


def load_teacher(vlm_path=None):
    # Load base LLaVA and wrap with LLaVAClassifier
    print("Loading teacher LLaVA model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    vlm = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        quantization_config=bnb_config,
        device_map="auto"
    ).to(DEVICE)

    teacher = LLaVAClassifier(vlm).to(DEVICE)

    # If a checkpoint path is provided (finetuned classifier), load weights
    if vlm_path and os.path.exists(vlm_path):
        print(f"Loading teacher weights from {vlm_path}...")
        teacher = torch.load(vlm_path, map_location=DEVICE, weights_only=False)

    # Put teacher in eval and freeze
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    return teacher


def main(teacher_ckpt=None):
    # Load dataset with RAM-based transform from combined_preprocessing
    split_data = get_distillation_dataset()

    train_loader = DataLoader(split_data["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(split_data["test"], batch_size=BATCH_SIZE)

    if not teacher_ckpt:
        raise ValueError("--teacher-ckpt is required. Specify path to classifier.pt")

    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_ckpt}")

    teacher = load_teacher(teacher_ckpt)

    # Build student using SmolVLM-Instruct and attach a classifier head
    print("Loading student model (SmolVLM-256M-Instruct)...")
    student_vlm = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)

    student = SmolVLMClassifier(student_vlm).to(DEVICE)

    # Freeze backbone
    for name, p in student.named_parameters():
        if not name.startswith('classifier'):
            p.requires_grad = False

    # Losses
    from combined_preprocessing import count_labels_json, DATAPATH
    
    train_json = os.path.join(DATAPATH, "train.jsonl")
    print(f"Calculating class weights from {train_json}...")
    counts = count_labels_json(train_json)
    
    if not counts:
        raise FileNotFoundError(f"Could not calculate label counts from {train_json}")

    pos_count = counts["hateful"]
    neg_count = counts["not_hateful"]

    pos_weight = torch.tensor(neg_count / max(1, pos_count)).to(DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    kd_loss_fn = nn.KLDivLoss(reduction='batchmean')

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # metrics
    train_losses = []
    val_losses = []
    val_accs = []

    T = 2.0  # temperature for KD
    alpha = 0.7  # weight for KD loss
    print("Starting distillation training...")

    for epoch in range(NUM_EPOCHS):
        student.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False):
            # Move batch tensors to device
            pixel_values_llava = batch["llava_pixel_values"].to(DEVICE).half()
            input_ids_llava = batch["llava_input_ids"].to(DEVICE)
            attention_mask_llava = batch["llava_attention_mask"].to(DEVICE)
            
            pixel_values_smol = batch["smol_pixel_values"].to(DEVICE)
            input_ids_smol = batch["smol_input_ids"].to(DEVICE)
            attention_mask_smol = batch["smol_attention_mask"].to(DEVICE)
            
            labels = batch["labels"].float().to(DEVICE)

            optimizer.zero_grad()

            # Teacher logits (detached)
            with torch.no_grad():
                teacher_logits = teacher(pixel_values_llava, input_ids_llava, attention_mask_llava)

            student_logits = student(pixel_values_smol, input_ids_smol, attention_mask_smol)

            # BCE loss
            loss_bce = bce_loss_fn(student_logits, labels)

            # KD loss
            def two_class_probs(logits, temp):
                p = torch.sigmoid(logits / temp)
                return torch.stack([p, 1 - p], dim=1)

            s_probs = two_class_probs(student_logits, T)
            t_probs = two_class_probs(teacher_logits.to(student_logits.dtype), T)

            s_log = torch.log(torch.clamp(s_probs, 1e-8, 1.0))
            kd_loss = kd_loss_fn(s_log, t_probs) * (T * T)

            loss = alpha * kd_loss + (1 - alpha) * loss_bce

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_train_loss = total_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        print(f"Epoch {epoch} Train Loss: {epoch_train_loss}")

        # Validation
        student.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False):
                pixel_values_llava = batch["llava_pixel_values"].to(DEVICE).half()
                input_ids_llava = batch["llava_input_ids"].to(DEVICE)
                attention_mask_llava = batch["llava_attention_mask"].to(DEVICE)
                
                pixel_values_smol = batch["smol_pixel_values"].to(DEVICE)
                input_ids_smol = batch["smol_input_ids"].to(DEVICE)
                attention_mask_smol = batch["smol_attention_mask"].to(DEVICE)
                
                labels = batch["labels"].float().to(DEVICE)

                teacher_logits = teacher(pixel_values_llava, input_ids_llava, attention_mask_llava)
                student_logits = student(pixel_values_smol, input_ids_smol, attention_mask_smol)

                loss_bce = bce_loss_fn(student_logits, labels)
                
                s_probs = two_class_probs(student_logits, T)
                t_probs = two_class_probs(teacher_logits.to(student_logits.dtype), T)
                s_log = torch.log(torch.clamp(s_probs, 1e-8, 1.0))
                kd_loss = kd_loss_fn(s_log, t_probs) * (T * T)

                loss = alpha * kd_loss + (1 - alpha) * loss_bce

                val_loss += loss.item()

                preds = (torch.sigmoid(student_logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_acc = correct / total if total > 0 else 0.0
        val_accs.append(val_acc)

        print(f"Epoch {epoch} Validation Loss: {epoch_val_loss}")
        print(f"Epoch {epoch} Validation Accuracy: {val_acc}")

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"distill_llava_smol_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    metrics = {
        "val_losses": val_losses,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "epochs": list(range(1, len(val_losses) + 1)),
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LR,
    }
    with open(os.path.join(run_dir, "distill_metrics.json"), "w") as mf:
        json.dump(metrics, mf, indent=2)

    # plots
    epochs = metrics["epochs"]
    fig_loss = plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'loss.png'))
    plt.close(fig_loss)

    fig_acc = plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_accs, marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'val_acc.png'))
    plt.close(fig_acc)

    # Save student model
    torch.save(student.state_dict(), os.path.join(run_dir, 'student_classifier.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distill LLaVA teacher into SmolVLM student.")
    parser.add_argument(
        '--teacher-ckpt',
        type=str,
        default="outputs/llava_20260317_014817/classifier_full.pt",
        help='Path to the finetuned LLaVA classifier.pt'
    )
    args = parser.parse_args()
    main(teacher_ckpt=args.teacher_ckpt)
