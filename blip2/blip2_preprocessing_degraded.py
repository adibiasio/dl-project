from datasets import load_dataset, DatasetDict
import os
from PIL import Image
from transformers import Blip2Processor
import argparse

print("running")

# -----------------------
# args
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--level", choices=["clean", "medium", "heavy"], required=True)
args = parser.parse_args()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAPATH = "/home/hice1/dkwon70/scratch/dl-project/data"
PREPROCESSED_PATH = os.path.join(BASE, "data_preprocessed_blip2")

if args.level == "clean":
    IMG_DIR = os.path.join(DATAPATH, "img")
else:
    IMG_DIR = os.path.join(DATAPATH, f"img_{args.level}")

TRAIN_OUT = os.path.join(PREPROCESSED_PATH, f"train_{args.level}")
VAL_OUT = os.path.join(PREPROCESSED_PATH, f"validation_{args.level}")

os.makedirs(TRAIN_OUT, exist_ok=True)
os.makedirs(VAL_OUT, exist_ok=True)

# -----------------------
# preprocessing function
# -----------------------
def preprocess_batch(batch):
    images = []
    texts = []
    targets = []

    for img_name, text, label in zip(batch["img"], batch["text"], batch["label"]):
        img_path = os.path.join(IMG_DIR, os.path.basename(img_name))

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            img = Image.new("RGB", (224, 224))

        images.append(img)

        prompt = f"Is this meme hateful, 'Yes' or 'No'? Text in meme: '{text}'."
        texts.append(prompt)

        targets.append("Yes" if label == 1 else "No")

    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True
    )

    labels = processor.tokenizer(
        targets,
        return_tensors="pt",
        padding="max_length",
        max_length=8,
        add_special_tokens=True
    ).input_ids

    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {**inputs, "labels": labels}


# -----------------------
# load dataset
# -----------------------
print("downloading hf processor")
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

# -----------------------
# preprocess
# -----------------------
print(f"Preprocessing dataset ({args.level})")

preprocessed = dataset.map(
    preprocess_batch,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=4
)

# -----------------------
# save splits separately
# -----------------------
print("Saving train split...")
preprocessed["train"].save_to_disk(TRAIN_OUT)

print("Saving validation split...")
preprocessed["validation"].save_to_disk(VAL_OUT)

print("Done!")
print(f"Saved to:\n{TRAIN_OUT}\n{VAL_OUT}")