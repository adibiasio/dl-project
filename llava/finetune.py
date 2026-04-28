import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import os

from classifier import LLaVAClassifier


# -----------------------------
# Config
# -----------------------------
DATAPATH = "./data_preprocessed_llava"
BATCH_SIZE = 8
NUM_EPOCHS = 15
LR = 5e-5
WEIGHT_DECAY = 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load dataset
# -----------------------------
print("loading from disk")
data = load_from_disk(DATAPATH)
data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "pixel_values", "labels"]
)

train_loader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(data["validation"], batch_size=BATCH_SIZE, num_workers=0)

# -----------------------------
# Load LLaVA model + processor
# -----------------------------
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
llava = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# -----------------------------
# Freeze all backbone parameters
# -----------------------------
for param in llava.parameters():
    param.requires_grad = False

# -----------------------------
# Unfreeze last 5% of decoder layers
# -----------------------------
decoder_layers = llava.language_model.model.layers
num_layers = len(decoder_layers)
n_unfreeze = max(1, int(0.05 * num_layers))  # last 5%
for layer in decoder_layers[-n_unfreeze:]:
    for p in layer.parameters():
        p.requires_grad = True
print(f"Unfroze last {n_unfreeze} decoder layers out of {num_layers}")

# Wrap with classifier
model = LLaVAClassifier(llava, hidden_size=llava.config.text_config.hidden_size).to(device)

# -----------------------------
# Compute class imbalance for pos_weight
# -----------------------------
labels = data["train"]["labels"]
pos_count = sum(labels)
neg_count = len(labels) - pos_count

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg_count / pos_count).to(device))
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# -----------------------------
# Training loop
# -----------------------------
train_losses, val_losses, val_accs = [], [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)

        optimizer.zero_grad()
        logits = model(pixel_values, input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} Train Loss: {train_loss}")
    train_losses.append(train_loss)

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)

            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total
    print(f"Epoch {epoch} Validation Loss: {val_loss}")
    print(f"Epoch {epoch} Validation Accuracy: {val_acc}")
    val_losses.append(val_loss)
    val_accs.append(val_acc)

# -----------------------------
# Save outputs
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_root = "outputs"
run_dir = os.path.join(output_root, f"llava_{timestamp}")
os.makedirs(run_dir, exist_ok=True)
epochs = range(1, len(val_losses) + 1)


# Save metrics
metrics = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_accs": val_accs,
    "epochs": list(epochs),
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY
}
with open(os.path.join(run_dir, "finetune_metrics.json"), "w") as mf:
    import json
    json.dump(metrics, mf, indent=2)

# Save model
torch.save(model.state_dict(), os.path.join(run_dir, "classifier.pt"))

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

print("Training complete, model saved at", run_dir)
