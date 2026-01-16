import gymnasium as gym
import numpy as np
from PIL import Image
import os
import sys

# Set MuJoCo backend
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

# Add project root (optional, depending on your folder structure)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Attempt to import custom envs, fallback to standard if missing
try:
    import envs
    env_name = "TexturedHumanoid-v5"
except ImportError:
    env_name = "Humanoid-v4"

def generate_humanoid_test_images():
    try:
        env = gym.make(env_name, render_mode="rgb_array", width=480, height=480)
    except:
        env = gym.make("Humanoid-v4", render_mode="rgb_array", width=480, height=480)

    print(f"Generating images using {env.spec.id}...")
    
    # --- Joint Indices ---
    def get_idx(name):
        try:
            return env.unwrapped.model.joint(name).qposadr
        except:
            return None

    # Get indices safely
    idx_r_shoulder = get_idx("right_shoulder1")
    idx_r_elbow    = get_idx("right_elbow")
    idx_l_shoulder = get_idx("left_shoulder1")
    idx_l_elbow    = get_idx("left_elbow")

    # --- Helper: Standing Pose ---
    def set_standing_pose(arm="none"):
        env.reset()
        qpos = env.unwrapped.model.qpos0.copy()
        qvel = np.zeros_like(env.unwrapped.data.qvel)

        # 1. Height: Lowered to 1.28 so feet touch the floor (fixes floating)
        qpos[2] = 1.28 
        
        # 2. Rotation: Rotate 90 deg around Z to face camera
        # Quaternion [w, x, y, z] for 90 deg Z-rotation
        qpos[3] = 0.7071
        qpos[4] = 0.0
        qpos[5] = 0.0
        qpos[6] = 0.7071

        # 3. Arms
        if arm == "right" and idx_r_shoulder:
            qpos[idx_r_shoulder] = -1.5  # Less extreme angle
            qpos[idx_r_elbow]    = 0.0
        elif arm == "left" and idx_l_shoulder:
            qpos[idx_l_shoulder] = 1.2
            qpos[idx_l_elbow]    = 0.0

        env.unwrapped.set_state(qpos, qvel)
        # Note: We do NOT step physics here. Stepping without a policy 
        # causes the robot to collapse instantly.
        
        return env.render()

    # --- Helper: Lying Down Pose ---
    def set_lying_down_pose():
        env.reset()
        qpos = env.unwrapped.model.qpos0.copy()
        qvel = np.zeros_like(env.unwrapped.data.qvel)

        # 1. Height: Very low
        qpos[2] = 0.5

        # 2. Rotation: 90 deg Pitch (Lying on back) + 90 deg Yaw (Face Camera)
        # This complex quaternion orients the body flat on the ground
        # approximating "lying on back"
        qpos[3] = 0.5
        qpos[4] = 0.5
        qpos[5] = 0.5
        qpos[6] = 0.5

        env.unwrapped.set_state(qpos, qvel)
        
        # 3. Settle: Step physics to let gravity pull it flat to floor
        for _ in range(50):
            env.step(np.zeros(env.action_space.shape))
            
        return env.render()

    # --- GENERATE IMAGES ---

    # 1. Right Arm Raised
    print("1. Generating Right Arm...")
    img = set_standing_pose(arm="right")
    Image.fromarray(img).save("arm_right.png")
    
    # 2. Left Arm Raised
    print("2. Generating Left Arm...")
    img = set_standing_pose(arm="left")
    Image.fromarray(img).save("arm_left.png")

    # 3. Lying Down (Replaces Crouch)
    print("3. Generating Lying Down...")
    img = set_lying_down_pose()
    Image.fromarray(img).save("lying_down.png")
    
    env.close()
    print("\n✅ Done! Check 'lying_down.png' and fixed arm images.")

if __name__ == "__main__":
    generate_humanoid_test_images()