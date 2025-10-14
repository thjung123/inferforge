import os
import torch
from transformers import BertModel
import clip


os.makedirs("weights", exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[BERT] Downloading pretrained 'bert-base-uncased' weights...")

bert = BertModel.from_pretrained("bert-base-uncased")
bert_path = os.path.join("weights", "bert-base-uncased.pt")
torch.save(bert.state_dict(), bert_path)

print(
    f"[BERT] Saved to {bert_path} ({sum(p.numel() for p in bert.parameters())/1e6:.1f}M parameters)"
)
print("[CLIP] Downloading pretrained 'ViT-B/32' weights...")

clip_model, _ = clip.load("ViT-B/32", device=DEVICE)

text_encoder_path = os.path.join("weights", "clip_text_encoder.pt")
image_encoder_path = os.path.join("weights", "clip_image_encoder.pt")

# save transformer (text encoder)
torch.save(clip_model.transformer.state_dict(), text_encoder_path)
# save visual (image encoder)
torch.save(clip_model.visual.state_dict(), image_encoder_path)

print(f"[CLIP] Saved text encoder → {text_encoder_path}")
print(f"[CLIP] Saved image encoder → {image_encoder_path}")

print("\nDownload complete. Files under ./weights/:")
for f in os.listdir("weights"):
    print(" -", f)
