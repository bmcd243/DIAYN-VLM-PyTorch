# import gym
import gymnasium as gym
from Brain import SACAgent
from Common import Play, Logger, get_params
from Common.preprocessing import VLMRMPreprocessor
import numpy as np
from tqdm import tqdm
import os
import torch
import clip
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torch.amp import autocast
from torch.amp.grad_scaler import GradScaler
import wandb
import sys
import envs


scaler = GradScaler('cuda')
os.environ["MUJOCO_GL"] = "egl"

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])



def get_semantic_embedding(env, clip_model, clip_processor, device):
    """
    Captures a frame, preprocesses it for CLIP with VLM-RM style augmentation, and generates the embedding (e_t).
    Ref: DIAYN_VLM_Algorithm.pdf [Source 7, 14]
    """
    frame = env.render()
    image = Image.fromarray(frame)
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_embeds = clip_model.get_image_features(**inputs)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds.cpu().numpy()[0]



if __name__ == "__main__":
    params = get_params()

    # Handle checkpoint loading
    if (params.get("checkpoint_path") or params.get("checkpoint_dir")) and "do_train" not in sys.argv:
        params["do_train"] = False
        print("Checkpoint specified without --do_train flag, switching to evaluation mode")

    if not params["do_train"] and (params.get("checkpoint_dir") or params.get("checkpoint_path")):
        if params.get("checkpoint_dir"):
            ckpt_path = os.path.join(params["checkpoint_dir"], "params.pth")
        else:
            ckpt_path = params["checkpoint_path"]
        
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        if "hyperparameters" in checkpoint:
            print("Loading hyperparameters from checkpoint...")
            for key, value in checkpoint["hyperparameters"].items():
                params[key] = value
                print(f"  {key} = {value}")
        else:
            print("Warning: Old checkpoint format, trying to load from config.txt...")
    
    # Setup CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading CLIP Model...")
    # clip_model, clip_preprocess = clip.load(params["clip_model"], device=device)
    clip_model = CLIPModel.from_pretrained(params["clip_model"]).to(device)
    clip_processor = CLIPProcessor.from_pretrained(params["clip_model"])
    print(f"Loaded{params["clip_model"]}")
    clip_model.eval()
    
    # preprocessor = VLMRMPreprocessor(clip_processor, augment=params.get("use_augmentation", True))
    # print(f"Using VLMRMPreprocessor with augmentation={preprocessor.augment}")
    preprocessor = clip_processor  # Use clip_processor directly
    print(f"Using CLIPProcessor directly (no augmentation)")
    params["embedding_dim"] = 768

    # Create environment (SINGLE INSTANCE)
    if params["env_name"] == "TexturedHumanoid-v5":
        env = gym.make(
            params["env_name"],
            render_mode="rgb_array",
            terminate_when_unhealthy=params.get("terminate_when_unhealthy", True),
            healthy_z_range=tuple(params.get("healthy_z_range", [1.0, 2.0])),
            reset_noise_scale=params.get("reset_noise_scale", 0.005)
        )
        print(f"TexturedHumanoid params: terminate_when_unhealthy={params.get('terminate_when_unhealthy')}, "
              f"healthy_z_range={params.get('healthy_z_range')}, "
              f"reset_noise_scale={params.get('reset_noise_scale')}")
    else:
        env = gym.make(params["env_name"], render_mode="rgb_array")

    # Extract environment specs
    params.update({
        "n_states": env.observation_space.shape[0],
        "n_actions": env.action_space.shape[0],
        "action_bounds": [env.action_space.low[0], env.action_space.high[0]]
    })
    print("params:", params)

    target_embedding = None
    if params["reward_mode"] == "vlm_rm":
        print(f"\n[VLM-RM MODE ACTIVE]")
        print(f"Target Prompt: '{params['target_prompt']}'")
        
        # 1. Encode Goal
        goal_inputs = clip_processor(text=[params["target_prompt"]], return_tensors="pt", padding=True)
        goal_inputs = {k: v.to(device) for k, v in goal_inputs.items()}
        with torch.no_grad():
            goal_emb = clip_model.get_text_features(**goal_inputs)
            goal_emb = goal_emb / goal_emb.norm(dim=-1, keepdim=True)
        
        # 2. Check if baseline prompt exists, if so compute direction vector
        if "baseline_prompt" in params and params["baseline_prompt"]:
            print(f"Baseline Prompt: '{params['baseline_prompt']}'")
            print("[Using Direction Vector: Baseline -> Goal]")
            
            # Encode Baseline
            base_inputs = clip_processor(text=[params["baseline_prompt"]], return_tensors="pt", padding=True)
            base_inputs = {k: v.to(device) for k, v in base_inputs.items()}
            with torch.no_grad():
                base_emb = clip_model.get_text_features(**base_inputs)
                base_emb = base_emb / base_emb.norm(dim=-1, keepdim=True)
            
            # 3. Compute Direction (The "Delta")
            # VLM-RM Idea: Reward movement along the vector from Baseline -> Goal
            direction_vector = goal_emb - base_emb
            direction_vector = direction_vector / direction_vector.norm(dim=-1, keepdim=True)
            
            target_embedding = direction_vector.cpu().numpy()[0]
            
            print(f"Direction vector norm: {np.linalg.norm(target_embedding):.4f}")
        else:
            # Use goal embedding directly (original behavior)
            print("[Using Goal Embedding directly (no baseline)]")
            target_embedding = goal_emb.cpu().numpy()[0]

    # Initialize agent and logger
    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, target_embedding=target_embedding, **params)
    logger = Logger(agent, **params)

    if params["do_train"]:

        # Initialize wandb
        wandb.init(
            project="DIAYN-VLM",
            name=f"{params['config_name']}",
            tags=[params['reward_mode'], params['env_name']]
        )

        if not params["train_from_scratch"]:
            episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            np.random.set_state(np_rng_state)
            env.np_random.set_state(env_rng_states[0])
            env.observation_space.np_random.set_state(env_rng_states[1])
            env.action_space.np_random.set_state(env_rng_states[2])
            agent.set_rng_states(torch_rng_state, random_rng_state)
            print("Keep training from previous run.")

        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            env.reset(seed=params["seed"])
            env.action_space.seed(params["seed"])
            env.observation_space.seed(params["seed"])
            print("Training from scratch.")

        logger.on()
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            state, _ = env.reset()
            embedding = get_semantic_embedding(env, clip_model, preprocessor, device)
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            embedding_freq = params.get("embedding_freq", 1)

            max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
            for step in range(1, 1 + max_n_steps):

                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # SPEEDUP: Only compute embedding every N steps or on done
                if step % embedding_freq == 0 or done:
                    next_embedding = get_semantic_embedding(env, clip_model, preprocessor, device)
                else:
                    next_embedding = embedding  # Reuse previous embedding
                    
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action, next_state, next_embedding)
                # with autocast('cuda'):
                logq_zs = agent.train()
                if logq_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(logq_zs)
                episode_reward += reward
                state = next_state
                embedding = next_embedding
                if done:
                    break

            avg_logq_zs = sum(logq_zses) / len(logq_zses)
            
            # Log to wandb
            wandb.log({
                "episode": episode,
                "episode_reward": episode_reward,
                "skill_z": z,
                "avg_logq_z": avg_logq_zs,
                "episode_length": step,
            })
            
            logger.log(episode,
                       episode_reward,
                       z,
                       sum(logq_zses) / len(logq_zses),
                       step,
                       np.random.get_state(),
                       *agent.get_rng_states(),
                       )
            last_logq_zs = avg_logq_zs
        
        # Finish wandb run
        wandb.finish()

    else:
        # Evaluation mode
        print("="*50)
        print("EVALUATION MODE")
        print("="*50)
        
        # Load weights with optional specific checkpoint
        logger.load_weights(
            checkpoint_path=params.get("checkpoint_path"),
            checkpoint_dir=params.get("checkpoint_dir")
        )
        
        # Print which checkpoint was loaded
        print(f"Loaded checkpoint from: {logger.log_dir}")
        print("="*50)
        
        # Pass log_dir to Play for better video naming
        player = Play(env, agent, n_skills=params["n_skills"], 
                     log_dir=logger.log_dir,
                     config_name=params.get("config"))
        
        # Evaluate with multiple episodes per skill for better statistics
        skill_rewards = player.evaluate(
            num_episodes_per_skill=params.get("eval_episodes", 3),
            save_video=params.get("save_eval_video", True)
        )