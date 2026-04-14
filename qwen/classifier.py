import torch
import torch.nn as nn

class QwenClassifier(nn.Module):
    def __init__(self, qwen_model, hidden_size=4096):
        super().__init__()
        self.qwen = qwen_model

        # Add a linear classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1)   # binary classification
        )

    def forward(self, batch):
        # Forward pass through Qwen
        outputs = self.qwen(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            return_dict=True
        )
        # Use the last hidden state of the first token
        # shape: (batch_size, seq_len, hidden_dim)
        last_hidden = outputs.last_hidden_state[:, 0, :]  # pick first token
        logits = self.classifier(last_hidden).squeeze(-1)
        return logits
