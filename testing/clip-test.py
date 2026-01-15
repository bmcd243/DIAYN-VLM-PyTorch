import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Load images
img_stand_1 = preprocess(Image.open("stand_1.png")).unsqueeze(0).to(device)
img_stand_2 = preprocess(Image.open("stand_2.png")).unsqueeze(0).to(device)
img_fall    = preprocess(Image.open("fall.png")).unsqueeze(0).to(device)

with torch.no_grad():
    emb_stand_1 = model.encode_image(img_stand_1)
    emb_stand_2 = model.encode_image(img_stand_2)
    emb_fall    = model.encode_image(img_fall)

    # Calculate Cosine Similarity (Higher is closer)
    # Normalize vectors first
    emb_stand_1 /= emb_stand_1.norm(dim=-1, keepdim=True)
    emb_stand_2 /= emb_stand_2.norm(dim=-1, keepdim=True)
    emb_fall    /= emb_fall.norm(dim=-1, keepdim=True)

    sim_stand_stand = (emb_stand_1 @ emb_stand_2.T).item()
    sim_stand_fall  = (emb_stand_1 @ emb_fall.T).item()

print(f"Similarity (Stand vs Stand): {sim_stand_stand:.4f}")
print(f"Similarity (Stand vs Fall):  {sim_stand_fall:.4f}")

# PASS CRITERIA: Stand-Stand should be > 0.9. Stand-Fall should be < 0.8.