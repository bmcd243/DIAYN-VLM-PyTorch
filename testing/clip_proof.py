import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean, cosine
from transformers import CLIPProcessor, CLIPModel
import torch
import os

print("Loading CLIP model...")
model_id = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# Check if images exist before loading
required_files = ["arm_right.png", "arm_left.png", "lying_down.png"]
for f in required_files:
    if not os.path.exists(f):
        print(f"⚠️ Error: '{f}' not found. Please ensure you ran the image generation script first.")
        exit()

# Load images
print("\nLoading images...")
img_right = Image.open("arm_right.png")  # State A
img_left = Image.open("arm_left.png")    # State B
img_lying = Image.open("lying_down.png") # State C (Failure)

# Text descriptions describing the *intent* of the states
descriptions = [
    "A humanoid robot standing upright with its right arm raised", # State A
    "A humanoid robot standing upright with its left arm raised",  # State B
    "A humanoid robot lying on the floor"                          # State C
]

print("\n" + "="*70)
print("SEMANTIC GAP ANALYSIS: Standing Skills vs. Falling Failure")
print("="*70)

# 1. PIXEL SPACE: Euclidean Distance
print("\n--- PIXEL SPACE (Raw RGB values) ---")
# Flatten images to 1D arrays for standard Euclidean calculation
arr_right = np.array(img_right).flatten().astype(float)
arr_left = np.array(img_left).flatten().astype(float)
arr_lying = np.array(img_lying).flatten().astype(float)

pixel_dist_RL = euclidean(arr_right, arr_left)
pixel_dist_R_Lying = euclidean(arr_right, arr_lying)

print(f"Pixel Dist (Right Arm <-> Left Arm):   {pixel_dist_RL:.2f}")
print(f"Pixel Dist (Right Arm <-> Lying Down): {pixel_dist_R_Lying:.2f}")

if pixel_dist_RL > pixel_dist_R_Lying * 0.8: # Threshold check
    print("⚠️  Observation: The pixel distance between Left & Right arms is HUGE.")
    print("    Standard RL might think switching arms is as big a change as falling down.")

# 2. SEMANTIC SPACE: CLIP Embeddings
print("\n--- SEMANTIC SPACE (CLIP Embeddings) ---")

# Get text embeddings
inputs_text = processor(text=descriptions, return_tensors="pt", padding=True)
with torch.no_grad():
    text_embeds = model.get_text_features(**inputs_text)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

# Get image embeddings
inputs_images = processor(images=[img_right, img_left, img_lying], return_tensors="pt")
with torch.no_grad():
    image_embeds = model.get_image_features(**inputs_images)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

# Extract individual embeddings
img_embed_right = image_embeds[0].cpu().numpy()
img_embed_left = image_embeds[1].cpu().numpy()
img_embed_lying = image_embeds[2].cpu().numpy()

# Calculate Cosine Distances (Lower is more similar)
sem_dist_RL = cosine(img_embed_right, img_embed_left)
sem_dist_R_Lying = cosine(img_embed_right, img_embed_lying)

print(f"Semantic Dist (Right Arm <-> Left Arm):   {sem_dist_RL:.4f}")
print(f"Semantic Dist (Right Arm <-> Lying Down): {sem_dist_R_Lying:.4f}")

print("\n--- CONCLUSION ---")
if sem_dist_RL < sem_dist_R_Lying:
    print("✅ CLIP successfully groups the standing states!")
    print(f"   The semantic distance between arm raises ({sem_dist_RL:.4f}) is")
    print(f"   SMALLER than the distance to lying down ({sem_dist_R_Lying:.4f}).")
    print("   This proves the VLM captures the 'standing' concept despite pixel differences.")
else:
    print("❌ Unexpected result. Check image clarity.")

# 3. TEXT-TO-IMAGE ALIGNMENT CHECK
print("\n--- DOES THE VLM UNDERSTAND THE POSES? ---")
text_right_emb = text_embeds[0].cpu().numpy()
text_left_emb = text_embeds[1].cpu().numpy()

# Check if the "Right Arm" text actually matches the "Right Arm" image better than the "Left Arm" image
score_right_img_right_text = 1 - cosine(img_embed_right, text_right_emb)
score_left_img_right_text = 1 - cosine(img_embed_left, text_right_emb)

print(f"Similarity: Image(Right) vs Text('Right Arm'): {score_right_img_right_text:.4f}")
print(f"Similarity: Image(Left)  vs Text('Right Arm'): {score_left_img_right_text:.4f}")

if score_right_img_right_text > score_left_img_right_text:
    print("✅ The model correctly distinguishes Left from Right semantically.")