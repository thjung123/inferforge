import os
from pathlib import Path

import torch
from transformers import BertModel
import clip

BASE_DIR = Path(__file__).parent
WEIGHT_DIR = BASE_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("[BERT] Downloading pretrained 'bert-base-uncased' weights...")
bert = BertModel.from_pretrained("bert-base-uncased")
bert_path = WEIGHT_DIR / "bert-base-uncased.pt"
torch.save(bert.state_dict(), bert_path)
print(
    f"[BERT] Saved to {bert_path} ({sum(p.numel() for p in bert.parameters())/1e6:.1f}M parameters)"
)

print("[CLIP] Downloading pretrained 'ViT-B/32' weights...")
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)

text_encoder_path = WEIGHT_DIR / "clip_text_encoder.pt"
image_encoder_path = WEIGHT_DIR / "clip_image_encoder.pt"

torch.save(clip_model.transformer.state_dict(), text_encoder_path)
torch.save(clip_model.visual.state_dict(), image_encoder_path)

print(f"[CLIP] Saved text encoder → {text_encoder_path}")
print(f"[CLIP] Saved image encoder → {image_encoder_path}")

print("\nDownload complete. Files under:", WEIGHT_DIR)
for f in os.listdir(WEIGHT_DIR):
    print(" -", f)
