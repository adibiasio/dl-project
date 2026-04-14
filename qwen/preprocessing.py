from datasets import load_dataset, DatasetDict
import os
from transformers import AutoProcessor

print("running")

# --- Paths ---
DATAPATH = "../data"
PREPROCESSED_PATH = "./data_preprocessed_qwen"
BATCH_SIZE = 32

# --- Initialize Qwen processor ---
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Thinking")

# --- Preprocessing function ---
def preprocess_sample(sample):
    """
    Converts dataset rows to Qwen3-VL chat-style messages
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(DATAPATH, sample["img"])},
                {
                    "type": "text",
                    "text": f"Is this meme hateful? Answer Yes or No. Text in meme: '{sample['text']}'. Answer: {'Yes' if sample['label']==1 else 'No'}"
                }
            ],
        }
    ]

    # Use processor to convert messages into model inputs
    processed = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        padding=True,
        truncation=True,
        max_length=1024,
    )

    # Add label tensor for classification
    processed["labels"] = [sample["label"]]
    return processed


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
    preprocess_sample,
    batched=False,
    remove_columns=dataset["train"].column_names,
)

# --- Save preprocessed dataset ---
print("Saving to", PREPROCESSED_PATH)
preprocessed.save_to_disk(PREPROCESSED_PATH)

print("Done!")