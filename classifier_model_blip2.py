import torch
import torch.nn as nn

class VLMClassifier(nn.Module):

    def __init__(self, vlm_model):
        super().__init__()

        self.vlm = vlm_model
        hidden_size = vlm_model.language_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1280, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1280, 1)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.language_model_outputs.hidden_states[-1]

        # index of last non-padding token
        last_indices = attention_mask.sum(dim=1) - 1
        last_token = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            last_indices
        ].float()

        logits = self.classifier(last_token).squeeze(-1)

        return logits
