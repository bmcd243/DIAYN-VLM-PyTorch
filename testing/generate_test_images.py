import gymnasium as gym
import numpy as np
from PIL import Image
import sys
import os
os.environ["MUJOCO_GL"] = "egl"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom environments
import envs

def generate_humanoid_test_images():
    """Generate standing and falling humanoid images for CLIP testing"""
    
    env = gym.make("TexturedHumanoid-v5", render_mode="rgb_array")
    
    print("Generating test images...")
    
    # 1. Generate standing pose (initial reset)
    print("1. Capturing standing pose 1...")
    obs, _ = env.reset()
    frame_stand_1 = env.render()
    img_stand_1 = Image.fromarray(frame_stand_1)
    img_stand_1.save("stand_1.png")
    print("   Saved stand_1.png")
    
    # 2. Take a few small steps to get slightly different standing pose
    print("2. Capturing standing pose 2...")
    for _ in range(5):
        # Small random actions to move slightly but stay standing
        action = env.action_space.sample() * 0.1  # Small actions
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    frame_stand_2 = env.render()
    img_stand_2 = Image.fromarray(frame_stand_2)
    img_stand_2.save("stand_2.png")
    print("   Saved stand_2.png")
    
    # 3. Generate falling pose (take large random actions until it falls)
    print("3. Capturing falling pose...")
    obs, _ = env.reset()
    fallen = False
    
    for step in range(100):
        # Large random actions to make it fall
        action = env.action_space.sample() * 2.0
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Check if humanoid has fallen (z-coordinate is low)
        if terminated:  # Unhealthy = fallen
            frame_fall = env.render()
            img_fall = Image.fromarray(frame_fall)
            img_fall.save("fall.png")
            print(f"   Saved fall.png (fell at step {step})")
            fallen = True
            break
    
    if not fallen:
        print("   Warning: Humanoid didn't fall, using last frame anyway")
        frame_fall = env.render()
        img_fall = Image.fromarray(frame_fall)
        img_fall.save("fall.png")
    
    env.close()
    
    print("\n✅ All test images generated!")
    print("   - stand_1.png: Initial standing pose")
    print("   - stand_2.png: Standing pose after small movements")
    print("   - fall.png: Fallen/unhealthy pose")
    print("\nNow run: python clip-test.py")

if __name__ == "__main__":
    generate_humanoid_test_images()