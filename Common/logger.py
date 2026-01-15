import time
import numpy as np
import psutil
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import datetime
import glob


class Logger:
    def __init__(self, agent, **config):
        self.config = config
        self.agent = agent
        
        # Extract config name if using a config file
        config_name = self._get_config_name()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        # Format: <env_name>_<config_name>_<timestamp>
        self.log_dir = f"{self.config['env_name'][:-3]}/{config_name}_{timestamp}"
        
        self.start_time = 0
        self.duration = 0
        self.running_logq_zs = 0
        self.max_episode_reward = -np.inf
        self._turn_on = False
        self.to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

        if self.config["do_train"] and self.config["train_from_scratch"]:
            self._create_wights_folder(self.log_dir)
            self._log_params()
    
    def _get_config_name(self):
        """Extract config name from config path or generate descriptive name"""
        if "config" in self.config and self.config["config"]:
            # Extract filename without extension: "configs/halfcheetah_vit_l14.yaml" -> "halfcheetah_vit_l14"
            config_path = self.config["config"]
            config_name = os.path.splitext(os.path.basename(config_path))[0]
        else:
            # Generate descriptive name from key parameters
            clip_model = self.config.get("clip_model", "ViT-B32").replace("/", "")
            n_skills = self.config.get("n_skills", 20)
            config_name = f"skills{n_skills}_{clip_model}"
        
        return config_name

    @staticmethod
    def _create_wights_folder(dir):
        if not os.path.exists("Checkpoints"):
            os.mkdir("Checkpoints")
        os.makedirs("Checkpoints/" + dir, exist_ok=True)

    def _log_params(self):
        """Log all parameters and save to file"""
        log_path = "Logs/" + self.log_dir
        os.makedirs(log_path, exist_ok=True)
        
        with SummaryWriter(log_path) as writer:
            for k, v in self.config.items():
                writer.add_text(k, str(v))
        
        # Also save params as a readable text file
        with open(os.path.join(log_path, "config.txt"), "w") as f:
            f.write(f"Run started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n")
            for k, v in sorted(self.config.items()):
                f.write(f"{k}: {v}\n")

    def on(self):
        self.start_time = time.time()
        self._turn_on = True

    def _off(self):
        self.duration = time.time() - self.start_time

    def log(self, *args):
        if not self._turn_on:
            print("First you should turn the logger on once, via on() method to be able to log parameters.")
            return
        self._off()

        episode, episode_reward, skill, logq_zs, step, *rng_states = args

        self.max_episode_reward = max(self.max_episode_reward, episode_reward)

        if self.running_logq_zs == 0:
            self.running_logq_zs = logq_zs
        else:
            self.running_logq_zs = 0.99 * self.running_logq_zs + 0.01 * logq_zs

        ram = psutil.virtual_memory()
        assert self.to_gb(ram.used) < 0.98 * self.to_gb(ram.total), "RAM usage exceeded permitted limit!"

        if episode % (self.config["interval"] // 3) == 0:
            self._save_weights(episode, *rng_states)

        if episode % self.config["interval"] == 0:
            print("E: {}| "
                  "Skill: {}| "
                  "E_Reward: {:.1f}| "
                  "EP_Duration: {:.2f}| "
                  "Memory_Length: {}| "
                  "Mean_steps_time: {:.3f}| "
                  "{:.1f}/{:.1f} GB RAM| "
                  "Time: {} ".format(episode,
                                     skill,
                                     episode_reward,
                                     self.duration,
                                     len(self.agent.memory),
                                     self.duration / step,
                                     self.to_gb(ram.used),
                                     self.to_gb(ram.total),
                                     datetime.datetime.now().strftime("%H:%M:%S"),
                                     ))

        with SummaryWriter("Logs/" + self.log_dir) as writer:
            writer.add_scalar("Max episode reward", self.max_episode_reward, episode)
            writer.add_scalar("Running logq(z|s)", self.running_logq_zs, episode)
            writer.add_histogram(str(skill), episode_reward)
            writer.add_histogram("Total Rewards", episode_reward)

        self.on()

    def _save_weights(self, episode, *rng_states):
        checkpoint_path = "Checkpoints/" + self.log_dir + "/params.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        torch.save({"policy_network_state_dict": self.agent.policy_network.state_dict(),
                    "q_value_network1_state_dict": self.agent.q_value_network1.state_dict(),
                    "q_value_network2_state_dict": self.agent.q_value_network2.state_dict(),
                    "value_network_state_dict": self.agent.value_network.state_dict(),
                    "discriminator_state_dict": self.agent.discriminator.state_dict(),
                    "q_value1_opt_state_dict": self.agent.q_value1_opt.state_dict(),
                    "q_value2_opt_state_dict": self.agent.q_value2_opt.state_dict(),
                    "policy_opt_state_dict": self.agent.policy_opt.state_dict(),
                    "value_opt_state_dict": self.agent.value_opt.state_dict(),
                    "discriminator_opt_state_dict": self.agent.discriminator_opt.state_dict(),
                    "episode": episode,
                    "rng_states": rng_states,
                    "max_episode_reward": self.max_episode_reward,
                    "running_logq_zs": self.running_logq_zs,
                    # ADD: Save hyperparameters in checkpoint
                    "hyperparameters": {
                        "n_skills": self.config["n_skills"],
                        "n_hiddens": self.config["n_hiddens"],
                        "n_states": self.config["n_states"],
                        "n_actions": self.config["n_actions"],
                        "embedding_dim": self.config.get("embedding_dim", 768),
                        "env_name": self.config["env_name"],
                    }
                    },
                checkpoint_path)

    def load_weights(self, checkpoint_path=None, checkpoint_dir=None):
        """
        Load checkpoint with priority:
        1. Specific checkpoint_path (if provided)
        2. Latest from checkpoint_dir (if provided)
        3. Latest from current environment
        """
        # Priority 1: Specific checkpoint path
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            print(f"Loading specific checkpoint: {checkpoint_path}")
        
        # Priority 2: Latest from specific directory
        elif checkpoint_dir:
            if not os.path.exists(checkpoint_dir):
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            checkpoint_path = os.path.join(checkpoint_dir, "params.pth")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"No params.pth found in {checkpoint_dir}")
            print(f"Loading checkpoint from directory: {checkpoint_path}")
            # Update log_dir to match the loaded checkpoint
            self.log_dir = os.path.relpath(checkpoint_dir, "Checkpoints")
        
        # Priority 3: Latest from current environment (default behavior)
        else:
            base_dir = "Checkpoints/" + self.config["env_name"][:-3] + "/"
            
            if not os.path.exists(base_dir):
                raise FileNotFoundError(f"No checkpoints found in {base_dir}")
            
            # Find all subdirectories (config_name_timestamp runs)
            subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            
            if not subdirs:
                raise FileNotFoundError(f"No checkpoint directories found in {base_dir}")
            
            # Sort by modification time (latest last)
            subdirs.sort(key=os.path.getmtime)
            latest_dir = subdirs[-1]
            checkpoint_path = os.path.join(latest_dir, "params.pth")
            
            print(f"Loading latest checkpoint: {checkpoint_path}")
            
            # Update log_dir to continue logging in the same directory
            self.log_dir = os.path.relpath(latest_dir, "Checkpoints")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Load config from checkpoint directory
        config_txt_path = checkpoint_path.replace("params.pth", "").replace("Checkpoints", "Logs") + "config.txt"
        if os.path.exists(config_txt_path):
            print(f"Found config file: {config_txt_path}")
            print("Loading hyperparameters from checkpoint...")
            # You could parse this if needed
        
        self.agent.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        self.agent.q_value_network1.load_state_dict(checkpoint["q_value_network1_state_dict"])
        self.agent.q_value_network2.load_state_dict(checkpoint["q_value_network2_state_dict"])
        self.agent.value_network.load_state_dict(checkpoint["value_network_state_dict"])
        self.agent.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.agent.q_value1_opt.load_state_dict(checkpoint["q_value1_opt_state_dict"])
        self.agent.q_value2_opt.load_state_dict(checkpoint["q_value2_opt_state_dict"])
        self.agent.policy_opt.load_state_dict(checkpoint["policy_opt_state_dict"])
        self.agent.value_opt.load_state_dict(checkpoint["value_opt_state_dict"])
        self.agent.discriminator_opt.load_state_dict(checkpoint["discriminator_opt_state_dict"])