import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import math
import json
import os

from classifier_model_blip2 import VLMClassifier

DATAPATH = "./data_preprocessed_blip2"
BATCH_SIZE = 16
NUM_EPOCHS = 15
LR = 5e-5
WEIGHT_DECAY = 0.01
device = "cuda"

# -----------------------------
# Load dataset
# -----------------------------

data = load_from_disk(DATAPATH)
data.set_format("torch")

train_loader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(data["validation"], batch_size=BATCH_SIZE)

# -----------------------------
# Load BLIP2 + LoRA
# -----------------------------

vlm = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.bfloat16
).to(device)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Freeze all VLM parameters first
for param in vlm.parameters():
    param.requires_grad = False

# Apply LoRA to QFormer
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "crossattention.attention.query",
        "crossattention.attention.value",
        "attention.attention.query",
        "attention.attention.value",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)
vlm = get_peft_model(vlm, lora_config)
print("LoRA trainable parameters:")
vlm.print_trainable_parameters()

# Unfreeze last 5% of OPT decoder layers
decoder_layers = vlm.base_model.model.language_model.model.decoder.layers
num_layers = len(decoder_layers)
n_unfreeze = max(1, math.ceil(0.05 * num_layers))
for layer in decoder_layers[-n_unfreeze:]:
    for p in layer.parameters():
        p.requires_grad = True

print(f"Also unfroze last {n_unfreeze}/{num_layers} OPT decoder layers")

# -----------------------------
# Classifier
# -----------------------------

model = VLMClassifier(vlm).to(device)

# Compute pos/neg counts for class-balanced loss
pos_count = 0
neg_count = 0
for example in data["train"]:
    labels_for_decode = example["labels"].clone()
    labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
    decoded = processor.tokenizer.decode(labels_for_decode, skip_special_tokens=True)
    if "yes" in decoded.lower():
        pos_count += 1
    else:
        neg_count += 1

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg_count / pos_count).to(device))

# Optimise classifier head + LoRA params + unfrozen decoder layers
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

# -----------------------------
# Training loop
# -----------------------------

val_losses = []
train_losses = []
val_accs = []

for epoch in range(NUM_EPOCHS):

    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False):

        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels_for_decode = batch["labels"].clone()
        labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
        decoded_labels = processor.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
        labels = torch.tensor([1 if "yes" in l.lower() else 0 for l in decoded_labels]).float().to(device)

        optimizer.zero_grad()
        logits = model(pixel_values, input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Train Loss: {total_loss / len(train_loader)}")
    train_losses.append(total_loss / len(train_loader))

    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False):

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels_for_decode = batch["labels"].clone()
            labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
            decoded_labels = processor.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            labels = torch.tensor([1 if "yes" in l.lower() else 0 for l in decoded_labels]).float().to(device)

            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch {epoch} Validation Loss: {val_loss / len(val_loader)}")
    print(f"Epoch {epoch} Validation Accuracy: {correct / total}")

    val_losses.append(val_loss / len(val_loader))
    val_accs.append(correct / total)

# -----------------------------
# Save outputs
# -----------------------------

epochs = range(1, len(val_losses) + 1)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("outputs", f"blip2_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

metrics = {
    "val_losses": val_losses,
    "train_losses": train_losses,
    "val_accs": val_accs,
    "epochs": list(epochs),
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
}
with open(os.path.join(run_dir, "finetune_metrics.json"), "w") as mf:
    json.dump(metrics, mf, indent=2)

fig_loss = plt.figure(figsize=(6, 4))
plt.plot(epochs, train_losses, marker='o', label='Train Loss')
plt.plot(epochs, val_losses, marker='o', label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "loss.png"))
plt.close(fig_loss)

fig_acc = plt.figure(figsize=(6, 4))
plt.plot(epochs, val_accs, marker='o')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "val_acc.png"))
plt.close(fig_acc)

# Save classifier head weights
torch.save(model.state_dict(), os.path.join(run_dir, "classifier.pt"))

# Save LoRA adapter separately so it can be reloaded with PeftModel
vlm.save_pretrained(os.path.join(run_dir, "lora_adapter"))
print(f"Saved all outputs to {run_dir}")