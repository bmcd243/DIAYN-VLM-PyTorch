import argparse
import yaml
import os
from pathlib import Path


def load_config_from_yaml(config_path):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {config_path}")
    return config


def get_params():
    parser = argparse.ArgumentParser(
        description="DIAYN-VLM: Diversity is All You Need with Vision-Language Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Config file argument (highest priority after CLI args)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (e.g., configs/halfcheetah_vit_l14.yaml)")

    # Environment
    parser.add_argument("--env_name", default=None, type=str, help="Name of the environment.")
    parser.add_argument("--seed", default=None, type=int, help="Random seed.")
    
    # Training
    parser.add_argument("--interval", default=None, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by episodes.")
    parser.add_argument("--do_train", action="store_true",
                        help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--train_from_scratch", action="store_true",
                        help="The flag determines whether to train from scratch or continue previous tries.")
    parser.add_argument("--max_n_episodes", default=None, type=int, help="Maximum number of training episodes.")
    parser.add_argument("--max_episode_len", default=None, type=int, help="Maximum steps per episode.")
    
    # DIAYN
    parser.add_argument("--mem_size", default=None, type=int, help="The memory size.")
    parser.add_argument("--n_skills", default=None, type=int, help="The number of skills to learn.")
    
    # SAC Hyperparameters
    parser.add_argument("--lr", default=None, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size for training.")
    parser.add_argument("--gamma", default=None, type=float, help="Discount factor.")
    parser.add_argument("--alpha", default=None, type=float, help="SAC temperature parameter.")
    parser.add_argument("--tau", default=None, type=float, help="Soft update coefficient.")
    parser.add_argument("--reward_scale", default=None, type=float, help="The reward scaling factor introduced in SAC.")
    
    # Network Architecture
    parser.add_argument("--n_hiddens", default=None, type=int, help="Number of hidden units in networks.")
    
    # CLIP Model
    parser.add_argument("--clip_model", default=None, type=str, 
                        help="CLIP model name (ViT-B/32, ViT-B/16, ViT-L/14, etc.)")
    parser.add_argument("--embedding_dim", default=None, type=int, 
                        help="CLIP embedding dimension (512 for ViT-B, 768 for ViT-L)")
    parser.add_argument("--embedding_freq", default=None, type=int,
                        help="Compute CLIP embedding every N steps (higher = faster)")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default=None, type=str, help="W&B project name.")
    parser.add_argument("--wandb_entity", default=None, type=str, help="W&B entity (username/team).")

    args = parser.parse_args()

    # Step 1: Load base config (default values)
    base_config_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    if base_config_path.exists():
        params = load_config_from_yaml(str(base_config_path))
    else:
        # Fallback to hardcoded defaults if base.yaml doesn't exist
        params = {
            "env_name": "HalfCheetah-v5",
            "seed": 123,
            "do_train": False,
            "train_from_scratch": False,
            "max_n_episodes": 5000,
            "max_episode_len": 1000,
            "interval": 100,
            "n_skills": 20,
            "lr": 0.0003,
            "batch_size": 256,
            "mem_size": 1000000,
            "gamma": 0.99,
            "alpha": 0.1,
            "tau": 0.005,
            "reward_scale": 1,
            "n_hiddens": 300,
            "clip_model": "ViT-B/32",
            "embedding_dim": 512,
            "embedding_freq": 1,
            "use_wandb": False,
            "wandb_project": "DIAYN-VLM",
            "wandb_entity": None,
        }
    
    # Step 2: Override with specific config file if provided
    if args.config is not None:
        config_params = load_config_from_yaml(args.config)
        params.update(config_params)
    
    # Step 3: Override with command-line arguments (highest priority)
    cli_args = vars(args)
    for key, value in cli_args.items():
        if key != 'config' and value is not None:
            # Special handling for boolean flags
            if key in ['do_train', 'train_from_scratch', 'use_wandb']:
                if value:  # Only set to True if flag is present
                    params[key] = True
            else:
                params[key] = value
    
    # Print final configuration
    print("\n" + "="*50)
    print("CONFIGURATION")
    print("="*50)
    for key, value in sorted(params.items()):
        print(f"{key:20s}: {value}")
    print("="*50 + "\n")
    
    return params
