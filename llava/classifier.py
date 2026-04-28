import torch
import torch.nn as nn


class LLaVAClassifier(nn.Module):
    def __init__(self, llava_model, hidden_size=4096):
        super().__init__()
        self.llava = llava_model

        # Add a linear classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1)   # binary classification
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # Forward pass through LLaVA
        outputs = self.llava(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True
        )
        # Use the last hidden state of the first token (<image> token usually)
        # shape: (batch_size, seq_len, hidden_dim)
        last_hidden = outputs.last_hidden_state[:, 0, :]  # pick first token
        logits = self.classifier(last_hidden).squeeze(-1)
        return logits
