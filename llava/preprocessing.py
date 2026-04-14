from datasets import load_dataset, DatasetDict
import os
from PIL import Image
from transformers import AutoProcessor

print("running")

# --- Paths ---
DATAPATH = "../data"
PREPROCESSED_PATH = "./data_preprocessed" # _tinyllava
BATCH_SIZE = 32

# --- Initialize processor ---
print("downloading HF processor")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
# processor = AutoProcessor.from_pretrained("YouLiXiya/tinyllava-v1.0-1.1b-hf")

# --- Preprocessing function ---
def preprocess_batch(batch):
    # Load images
    images = [
        Image.open(os.path.join(DATAPATH, img_path)).convert("RGB")
        for img_path in batch["img"]
    ]
    
    # Build prompts with <image> token
    prompts = [
        f"<image>\nIs this meme hateful? Answer Yes or No. Text in meme: '{text}'."
        for text in batch["text"]
    ]
    
    # Processor call
    inputs = processor(
        text=prompts,
        images=images,
        padding="max_length",
        truncation=True,
        max_length=1024,
    )
    
    inputs["labels"] = batch["label"]
    
    return inputs

# --- Load dataset ---
print("Loading dataset files")
dataset = load_dataset(
    "json",
    data_files={"train": f"{DATAPATH}/train.jsonl"}
)

# --- Split dataset ---
dataset = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42
)

dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

# --- Preprocess dataset ---
print("Preprocessing dataset")
preprocessed = dataset.map(
    preprocess_batch,
    batched=True,
    batch_size=BATCH_SIZE,
    writer_batch_size=BATCH_SIZE,
    remove_columns=dataset["train"].column_names,
)

# --- Save preprocessed dataset ---
print("Saving to", PREPROCESSED_PATH)
preprocessed.save_to_disk(PREPROCESSED_PATH)

print("Done!")