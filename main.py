# import gym
import gymnasium as gym
from Brain import SACAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
import os
import torch
import clip
from PIL import Image
os.environ["MUJOCO_GL"] = "egl" 
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])



def get_semantic_embedding(env, clip_model, clip_preprocess, device):
    """
    Captures a frame, preprocesses it for CLIP, and generates the embedding (e_t).
    Ref: DIAYN_VLM_Algorithm.pdf [Source 7, 14]
    """
    # 1. Render frame (Pixel Observation)
    frame = env.render(mode='rgb_array') 
    
    # 2. Resize to 224x224 for CLIP
    image = Image.fromarray(frame).resize((224, 224))
    
    # 3. Encode using Frozen VLM
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        
    # Return 512-dim embedding as numpy array
    return embedding.cpu().numpy()[0]



if __name__ == "__main__":
    params = get_params()

    # 1. CHANGE: Set Environment to HalfCheetah
    params["env_name"] = "HalfCheetah-v3"

    # 2. ADD: Initialize Frozen CLIP Model (ViT-B/32)
    # Ref: DIAYN_VLM_Algorithm.pdf [Source 12]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading CLIP Model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    test_env = gym.make(params["env_name"])
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    # env = gym.make(params["env_name"])
    env = gym.make(params["env_name"], render_mode="rgb_array")

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)

    if params["do_train"]:

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
            embedding = get_semantic_embedding(env, clip_model, clip_preprocess, device)
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
            for step in range(1, 1 + max_n_steps):

                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_embedding = get_semantic_embedding(env, clip_model, clip_preprocess, device)
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action, next_state, next_embedding)
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

            logger.log(episode,
                       episode_reward,
                       z,
                       sum(logq_zses) / len(logq_zses),
                       step,
                       np.random.get_state(),
                    #    env.np_random.get_state(),
                    #    env.observation_space.np_random.get_state(),
                    #    env.action_space.np_random.get_state(),
                       *agent.get_rng_states(),
                       )

    else:
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()
