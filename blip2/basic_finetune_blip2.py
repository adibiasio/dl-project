from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from transformers import Blip2ForConditionalGeneration, Blip2Processor

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
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b"
).to(device)

for param in model.parameters():
    param.requires_grad = False 

trainable_parameters = []
for param in model.qformer.parameters():
    param.requires_grad = True
    trainable_parameters.append(param)

optimizer = torch.optim.AdamW(trainable_parameters, lr=LR)

model.train()

yes_token = processor.tokenizer("Yes", add_special_tokens=False).input_ids[0]

for epoch in range(NUM_EPOCHS):

    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        forward = model(
            pixel_values=batch['pixel_values'].to(device),
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device)
        )

        loss = forward.loss
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

            forward = model(
                pixel_values=batch['pixel_values'].to(device),
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )

            val_loss += forward.loss.item()

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

            preds = [1 if p.strip().lower().startswith("yes") else 0 for p in preds]
            labels_for_decode = batch["labels"].clone()
            labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id

            decoded_labels = processor.tokenizer.batch_decode(
                labels_for_decode,
                skip_special_tokens=True
            )

            labels = torch.tensor(
                [1 if "yes" in l.lower() else 0 for l in decoded_labels]
            )

            correct += (torch.tensor(preds) == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch {epoch} Validation Loss: {val_loss / len(val_loader)}")
    print(f"Epoch {epoch} Validation Accuracy: {correct / total}")

    model.train()

model.save_pretrained("blip2-finetuned")