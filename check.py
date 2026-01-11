import torch
import clip
from PIL import Image
import time

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load all images
images = []
for i in range(1, 6):
    images.append(Image.open(f"standing_{i}.png"))
    images.append(Image.open(f"kneeling_{i}.png"))

# Warm-up
for img in images:
    _ = model.encode_image(preprocess(img).unsqueeze(0).to(device))

# Timing
start = time.time()
with torch.no_grad():
    for img in images:
        _ = model.encode_image(preprocess(img).unsqueeze(0).to(device))
end = time.time()

print(f"Processed {len(images)} images in {end - start:.4f} seconds.")
print(f"Average time per image: {(end - start)/len(images):.4f} seconds.")