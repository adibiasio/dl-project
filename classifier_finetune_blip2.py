import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import Blip2ForConditionalGeneration, Blip2Processor
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

train_loader = DataLoader(
    data["train"],
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    data["validation"],
    batch_size=BATCH_SIZE
)

# Load BLIP2 model
vlm = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.bfloat16
).to(device)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Freeze all VLM parameters
for param in vlm.parameters():
    param.requires_grad = False

# Access OPT decoder layers
decoder_layers = vlm.language_model.model.decoder.layers
num_layers = len(decoder_layers)

# Number of layers to unfreeze (5%)
n_unfreeze = max(1, math.ceil(0.05 * num_layers))

# Unfreeze last 5%
for layer in decoder_layers[-n_unfreeze:]:
    for p in layer.parameters():
        p.requires_grad = True

model = VLMClassifier(vlm).to(device)
# Compute positive / negative label counts on the training set to handle class imbalance.
# Labels are tokenized sequences; replace -100 with pad token id before decoding.
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
optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# storage for validation metrics for plotting
val_losses = []
train_losses = []
val_accs = []

# Training
for epoch in range(NUM_EPOCHS):

    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False):

        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels_for_decode = batch["labels"].clone()
        labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
        decoded_labels = processor.tokenizer.batch_decode(
            labels_for_decode,
            skip_special_tokens=True
        )
        labels = torch.tensor([1 if "yes" in l.lower() else 0 for l in decoded_labels]).float().to(device)

        optimizer.zero_grad()

        logits = model(pixel_values, input_ids, attention_mask)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Train Loss: {total_loss / len(train_loader)}")
    # store training loss per epoch
    train_losses.append(total_loss / len(train_loader))

    # Validation
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
            decoded_labels = processor.tokenizer.batch_decode(
                labels_for_decode,
                skip_special_tokens=True
            )
            labels = torch.tensor([1 if "yes" in l.lower() else 0 for l in decoded_labels]).float().to(device)

            logits = model(pixel_values, input_ids, attention_mask)

            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch {epoch} Validation Loss: {val_loss / len(val_loader)}")
    print(f"Epoch {epoch} Validation Accuracy: {correct / total}")

    # store metrics for plotting
    val_losses.append(val_loss / len(val_loader))
    val_accs.append(correct / total)

# Generate outputs
epochs = range(1, len(val_losses) + 1)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Create an outputs directory and a blip2-specific subfolder for this timestamp
output_root = "outputs"
run_dir = os.path.join(output_root, f"blip2_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# Save numeric metrics to JSON so they are available separately from the plots
metrics = {
    "val_losses": val_losses,
    "train_losses": train_losses,
    "val_accs": val_accs,
    "epochs": list(epochs),
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY
}
metrics_filename = os.path.join(run_dir, f"finetune_metrics.json")
with open(metrics_filename, "w") as mf:
    json.dump(metrics, mf, indent=2)

# Loss figure (train & validation)
fig_loss = plt.figure(figsize=(6, 4))
plt.plot(epochs, train_losses, marker='o', label='Train Loss')
plt.plot(epochs, val_losses, marker='o', label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
loss_filename = os.path.join(run_dir, f"loss.png")
plt.savefig(loss_filename)
plt.close(fig_loss)

# Validation accuracy figure
fig_acc = plt.figure(figsize=(6, 4))
plt.plot(epochs, val_accs, marker='o')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
acc_filename = os.path.join(run_dir, f"val_acc.png")
plt.savefig(acc_filename)
plt.close(fig_acc)

# Save model
torch.save(model.state_dict(), os.path.join(run_dir, f"classifier.pt"))