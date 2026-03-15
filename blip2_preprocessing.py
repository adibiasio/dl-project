from datasets import load_dataset, DatasetDict
import os
from PIL import Image 
from transformers import Blip2Processor

DATAPATH = "data"
PREPROCESSED_PATH ="./data_preprocessed_blip2"
BATCH_SIZE = 32

def preprocess_batch(batch):
    images = [Image.open(os.path.join(DATAPATH, image_path)).convert("RGB") for image_path in batch["img"]]
    text = [f"Is this meme hateful, 'Yes' or 'No'? Text in meme: '{text}'." for text in batch["text"]]
    inputs = processor(images=images,
                       text=text, 
                       return_tensors="pt", 
                       padding="max_length", 
                       max_length=128,
                       truncation=True)
    
    targets = ["Yes" if label == 1 else "No" for label in batch["label"]]

    labels = processor.tokenizer(
        targets,
        return_tensors="pt",
        padding="max_length",
        add_special_tokens=True,
        max_length=8
    ).input_ids
    
    labels[labels == processor.tokenizer.pad_token_id] = -100 
    return {**inputs, "labels":labels}

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

print("Loading dataset files")
dataset = load_dataset(
    "json",
    data_files={"train": f"{DATAPATH}/train.jsonl"}
)

dataset = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42
)

dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})
print("Preprocessing dataset")

preprocessed = dataset.map(
    preprocess_batch,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=4
)

print("Saving to ", PREPROCESSED_PATH)
preprocessed.save_to_disk(PREPROCESSED_PATH)