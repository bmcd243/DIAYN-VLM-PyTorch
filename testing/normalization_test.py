import gymnasium as gym
import torch
import clip
import numpy as np
import sys
import os
os.environ["MUJOCO_GL"] = "egl"

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Common.preprocessing import VLMRMPreprocessor
import envs

device = "cuda"
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()
preprocessor = VLMRMPreprocessor(clip_preprocess, augment=True)

env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
env.reset()

# Test normalization
def get_semantic_embedding(env, clip_model, preprocessor, device):
    frame = env.render()
    image_input = preprocessor.preprocess_frame(frame).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.cpu().numpy()[0]

# Get 10 embeddings
print("Testing embedding normalization...\n")
for i in range(10):
    env.step(env.action_space.sample())
    embedding = get_semantic_embedding(env, clip_model, preprocessor, device)
    norm = np.linalg.norm(embedding)
    print(f"Frame {i}: ||e|| = {norm:.6f} (should be ~1.0)")

env.close()