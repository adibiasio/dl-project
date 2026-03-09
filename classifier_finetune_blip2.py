import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import Blip2Processor, Blip2Model

DATAPATH = "./data_preprocessed_blip2"
BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-4
device = "cuda"

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

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip = Blip2Model.from_pretrained(
    "Salesforce/blip2-opt-2.7b"
).to(device)

for param in blip.parameters():
    param.requires_grad = False

for param in blip.qformer.parameters():
    param.requires_grad = True


class BLIP2Classifier(nn.Module):

    def __init__(self, blip_model):
        super().__init__()

        self.blip = blip_model
        hidden_size = blip_model.qformer.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, pixel_values, input_ids, attention_mask):

        outputs = self.blip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        qformer_output = outputs.qformer_outputs.last_hidden_state
        pooled = qformer_output.mean(dim=1)
        logits = self.classifier(pooled)

        return logits


model = BLIP2Classifier(blip).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

model.train()

yes_token = processor.tokenizer("Yes", add_special_tokens=False).input_ids[0]

for epoch in range(NUM_EPOCHS):

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

        labels = torch.tensor(
            [1 if "yes" in l.lower() else 0 for l in decoded_labels]
        ).to(device)

        optimizer.zero_grad()

        logits = model(pixel_values, input_ids, attention_mask)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Train Loss: {total_loss / len(train_loader)}")

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

            labels = torch.tensor(
                [1 if "yes" in l.lower() else 0 for l in decoded_labels]
            ).to(device)

            logits = model(pixel_values, input_ids, attention_mask)

            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch {epoch} Validation Loss: {val_loss / len(val_loader)}")
    print(f"Epoch {epoch} Validation Accuracy: {correct / total}")

    model.train()

torch.save(model.state_dict(), "blip2_classifier.pt")