from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from transformers import Blip2ForConditionalGeneration, Blip2Processor

DATAPATH = "./data_preprocessed_blip2"
NUM_EPOCHS = 5
LEARNING_RATE = 1e-5
BATCH_SIZE = 32

data = load_from_disk(DATAPATH)
data.set_format("torch")

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b").to("cuda")

for param in model.parameters():
    param.requires_grad = False 

trainable_parameters = []
for param in model.qformer.parameters():
    param.requires_grad = True
    trainable_parameters.append(param)


train = DataLoader(data["train"], 
                   batch_size=BATCH_SIZE, 
                   shuffle=True)
optimizer = torch.optim.AdamW(trainable_parameters,
                              lr=LEARNING_RATE)

model.train()
for epoch in range(NUM_EPOCHS):
    for batch in train:
        optimizer.zero_grad()
        forward = model(
            pixel_values= batch['pixel_values'].to("cuda"),
            input_ids = batch['input_ids'].to("cuda"),
            attention_mask = batch['attention_mask'].to('cuda'),
            labels=batch['labels'].to('cuda')
        )

        loss = forward.loss
        loss.backward() 
        optimizer.step() 

    print(f"Loss in Epoch {epoch}: {loss.item()}")

model.save_pretrained("blip2-finetuned")
