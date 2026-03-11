import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import Blip2ForConditionalGeneration, Blip2Processor

DATAPATH = "./data_preprocessed_blip2"
BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-4
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
    "Salesforce/blip2-opt-2.7b"
).to(device)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Freeze all VLM parameters
for param in vlm.parameters():
    param.requires_grad = False


# Classifier Model
class VLMClassifier(nn.Module):

    def __init__(self, vlm_model):
        super().__init__()

        self.vlm = vlm_model
        hidden_size = vlm_model.language_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, pixel_values, input_ids, attention_mask):

        outputs = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.language_model_outputs.hidden_states[-1]
        last_token = hidden_states[:, -1, :]
        logits = self.classifier(last_token).squeeze(-1)

        return logits


model = VLMClassifier(vlm).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LR
)

# Training
for epoch in range(NUM_EPOCHS):

    model.train()
    total_loss = 0

    for batch in train_loader:

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

    # Validation
    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in val_loader:

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

# Save model
torch.save(model.state_dict(), "vlm_hateful_classifier.pt")