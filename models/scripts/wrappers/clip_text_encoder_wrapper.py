import torch
import torch.nn as nn


class CLIPTextEncoderWrapper(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model

    def forward(self, text_tokens):
        x = self.model.token_embedding(text_tokens).type(self.model.dtype)  # [B,77,512]
        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)
        eos_idx = text_tokens.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_idx] @ self.model.text_projection
        return x
