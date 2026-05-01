from datasets import load_dataset, DatasetDict
import os
import json
import torch
from PIL import Image 
from transformers import Blip2Processor, AutoProcessor

# Configuration
DATAPATH = "../data"
CACHE_DIR = "/home/hice1/dwu359/scratch/hf-cache"
os.environ["HF_HOME"] = CACHE_DIR

def count_labels_json(file_path):
    """
    Directly counts 0 and 1 labels from a jsonl file without loading into HF datasets.
    """
    ones = 0
    zeros = 0
    total = 0
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
        
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            label = data.get("label", data.get("labels"))
            if label == 1:
                ones += 1
            elif label == 0:
                zeros += 1
            total += 1
            
    return {"total": total, "hateful": ones, "not_hateful": zeros}

# Global processors for use in transform
print("Loading processors...")
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=CACHE_DIR)
smol_processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", cache_dir=CACHE_DIR)

def preprocess_transform(batch):
    """
    On-the-fly transformation that happens in RAM during DataLoader iteration.
    This avoids writing massive preprocessed tensors to disk.
    """
    images = [Image.open(os.path.join(DATAPATH, img_path)).convert("RGB") for img_path in batch["img"]]
    
    # Text prompts
    blip2_texts = [f"Is this meme hateful, 'Yes' or 'No'? Text in meme: '{t}'." for t in batch["text"]]
    smol_texts = [f"<image>Is this meme hateful, 'Yes' or 'No'? Text in meme: '{t}'." for t in batch["text"]]
    
    # 1. BLIP-2 Processing
    blip2_inputs = blip2_processor(
        images=images,
        text=blip2_texts, 
        return_tensors="pt", 
        padding="max_length", 
        max_length=128,
        truncation=True
    )
    
    # 2. SmolVLM Processing
    smol_inputs = smol_processor(
        images=images,
        text=smol_texts,
        return_tensors="pt",
        padding="max_length",
        max_length=2048,
        truncation=True
    )
    
    # Package into a single dict with prefixes
    transformed = {}
    for k, v in blip2_inputs.items():
        transformed[f"blip2_{k}"] = v
    for k, v in smol_inputs.items():
        transformed[f"smol_{k}"] = v
        
    # Map 'label' from the batch to 'labels' for the training loop
    if "label" in batch:
        transformed["labels"] = torch.tensor(batch["label"])
    elif "labels" in batch:
        transformed["labels"] = torch.tensor(batch["labels"])
    else:
        # Debugging fallback to see available keys if we still get a KeyError
        raise KeyError(f"Neither 'label' nor 'labels' found in batch. Available keys: {list(batch.keys())}")
    
    return transformed

def get_distillation_dataset():
    """
    Returns the dataset with set_transform applied.
    """
    print("Loading raw dataset...")
    dataset = load_dataset(
        "json",
        data_files={"train": f"{DATAPATH}/train.jsonl"},
        cache_dir=CACHE_DIR
    )["train"]

    # Split into train/val
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Apply RAM-based transformation
    split.set_transform(preprocess_transform)
    
    return split

if __name__ == "__main__":
    # 1. Count labels directly from all JSONL files
    for filename in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        file_path = os.path.join(DATAPATH, filename)
        if os.path.exists(file_path):
            print(f"\n--- {filename} Distribution (Direct JSONL) ---")
            counts = count_labels_json(file_path)
            if counts:
                print(f"Total entries: {counts['total']}")
                print(f"Hateful (1): {counts['hateful']} ({counts['hateful']/counts['total']*100:.2f}%)")
                print(f"Not Hateful (0): {counts['not_hateful']} ({counts['not_hateful']/counts['total']*100:.2f}%)")

    # 2. Test the transformation
    split = get_distillation_dataset()
    
    # Test the transformation on a small batch
    sample = split["train"][:2]
    print("\n--- Transformation Test ---")
    print("Keys generated:", sample.keys())
    print("BLIP-2 pixels shape:", sample["blip2_pixel_values"].shape)
    print("SmolVLM pixels shape:", sample["smol_pixel_values"].shape)
