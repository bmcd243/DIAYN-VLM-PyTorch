import cv2
import numpy as np
import os
import datetime
from PIL import Image


class Play:
    def __init__(self, env, agent, n_skills, log_dir=None, config_name=None):
        self.env = env
        self.agent = agent
        self.n_skills = n_skills
        self.log_dir = log_dir
        self.config_name = config_name
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create video directory with better naming
        self.video_dir = self._create_video_directory()
        
        # ADD: Storage for collage frames
        self.skill_frame_sequences = {}

    def _create_video_directory(self):
        """Create directory for evaluation videos with timestamp and config name"""
        if self.log_dir:
            video_dir = f"Vid/{self.log_dir}"
        elif self.config_name:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            env_name = self.env.spec.id.replace("-v5", "").replace("-v4", "")
            video_dir = f"Vid/{env_name}/{self.config_name}_{timestamp}"
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            env_name = self.env.spec.id.replace("-v5", "").replace("-v4", "")
            video_dir = f"Vid/{env_name}/eval_{timestamp}"
        
        os.makedirs(video_dir, exist_ok=True)
        print(f"Saving evaluation videos to: {video_dir}")
        return video_dir

    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def evaluate(self, num_episodes_per_skill=1, save_video=True, create_collage=True, 
                 video_size=(480, 480), collage_size=(720, 720)):
        """
        Evaluate all skills and optionally save videos
        
        Args:
            num_episodes_per_skill: Number of episodes to run per skill
            save_video: Whether to save individual MP4s
            create_collage: Whether to create a collage video
            video_size: Resolution for individual skill videos (width, height)
            collage_size: Resolution per skill in collage grid (width, height)
        """
        skill_rewards = {}
        
        for z in range(self.n_skills):
            episode_rewards = []
            episode_lengths = []
            
            # Initialize frame storage for this skill
            if create_collage:
                self.skill_frame_sequences[z] = []
            
            for episode in range(num_episodes_per_skill):
                video_writer = None
                if save_video and episode == 0:
                    video_path = os.path.join(self.video_dir, f"skill_{z:02d}.mp4")
                    video_writer = cv2.VideoWriter(video_path, self.fourcc, 50.0, video_size)
                
                s, _ = self.env.reset()
                s = self.concat_state_latent(s, z, self.n_skills)
                episode_reward = 0
                step_count = 0
                
                for _ in range(self.env.spec.max_episode_steps):
                    action = self.agent.choose_action(s)
                    s_, r, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    s_ = self.concat_state_latent(s_, z, self.n_skills)
                    episode_reward += r
                    step_count += 1
                    
                    # Render frame
                    frame = self.env.render()
                    
                    # Store for collage (use configurable collage_size)
                    if create_collage and episode == 0 and len(self.skill_frame_sequences[z]) < 100:
                        # CHANGED: Use collage_size parameter
                        frame_small = cv2.resize(frame, collage_size)
                        self.skill_frame_sequences[z].append(frame_small)
                    
                    # Save to MP4 if recording
                    if video_writer is not None:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        # CHANGED: Resize to video_size
                        frame_resized = cv2.resize(frame_bgr, video_size)
                        video_writer.write(frame_resized)
                    
                    if done:
                        print(f"  Skill {z} Episode {episode}: Terminated at step {step_count} (terminated={terminated}, truncated={truncated})")
                        break
                    s = s_
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(step_count)
                
                if video_writer is not None:
                    video_writer.release()
            
            # Store statistics including episode length
            skill_rewards[z] = {
                'mean': np.mean(episode_rewards),
                'std': np.std(episode_rewards),
                'min': np.min(episode_rewards),
                'max': np.max(episode_rewards),
                'episodes': episode_rewards,
                'mean_length': np.mean(episode_lengths),
                'lengths': episode_lengths
            }
            
            print(f"Skill {z:2d}: Mean Reward: {skill_rewards[z]['mean']:8.1f} "
                f"± {skill_rewards[z]['std']:6.1f} "
                f"| Mean Length: {skill_rewards[z]['mean_length']:.0f} steps")
        
        # Create collage video
        if create_collage:
            print("\nCreating skill collages...")
            self._create_skill_collage_video(max_frames=50, fps=10)
        
        # Save evaluation results
        self._save_evaluation_results(skill_rewards)
        
        self.env.close()
        cv2.destroyAllWindows()
        
        return skill_rewards
    
    def _save_evaluation_results(self, skill_rewards):
        """Save evaluation results to a text file"""
        results_path = os.path.join(self.video_dir, "evaluation_results.txt")
        
        with open(results_path, 'w') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"{'='*70}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Environment: {self.env.spec.id}\n")
            f.write(f"Number of skills: {self.n_skills}\n")
            f.write(f"{'='*70}\n\n")
            
            f.write(f"{'Skill':<10}{'Mean':<15}{'Std':<15}{'Min':<15}{'Max':<15}{'Mean Length':<15}\n")
            f.write(f"{'-'*85}\n")
            
            for z in range(self.n_skills):
                stats = skill_rewards[z]
                f.write(f"{z:<10}{stats['mean']:<15.1f}{stats['std']:<15.1f}"
                       f"{stats['min']:<15.1f}{stats['max']:<15.1f}{stats['mean_length']:<15.1f}\n")
            
            f.write(f"\n{'='*85}\n")
            f.write(f"Overall Statistics:\n")
            all_means = [skill_rewards[z]['mean'] for z in range(self.n_skills)]
            f.write(f"Mean of means: {np.mean(all_means):.1f}\n")
            f.write(f"Std of means: {np.std(all_means):.1f}\n")
            f.write(f"Best skill: {np.argmax(all_means)} (reward: {np.max(all_means):.1f})\n")
            f.write(f"Worst skill: {np.argmin(all_means)} (reward: {np.min(all_means):.1f})\n")
        
        print(f"\nEvaluation results saved to: {results_path}")

    def _create_skill_collage_video(self, max_frames=100, fps=30):
        """
        Create an MP4 video showing all skills in a grid layout
        """
        # Check if we have any frames
        if not self.skill_frame_sequences:
            print("Warning: No frames captured for collage")
            return
        
        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(self.n_skills)))
        n_rows = int(np.ceil(self.n_skills / n_cols))
        
        print(f"Creating {n_rows}x{n_cols} grid video for {self.n_skills} skills")
        
        # Find minimum frame count across all skills
        min_frames = min(len(frames) for frames in self.skill_frame_sequences.values())
        min_frames = min(min_frames, max_frames)
        
        if min_frames == 0:
            print("Warning: No frames captured for collage (episodes too short)")
            return
        
        print(f"Using {min_frames} frames for collage")
        
        # Get frame dimensions (assuming all frames are same size)
        frame_h, frame_w = self.skill_frame_sequences[0][0].shape[:2]
        collage_h = n_rows * frame_h
        collage_w = n_cols * frame_w
        
        # Initialize video writer
        video_path = os.path.join(self.video_dir, "skills_collage.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (collage_w, collage_h))
        
        # Create and write frames
        for frame_idx in range(min_frames):
            collage = np.zeros((collage_h, collage_w, 3), dtype=np.uint8)
            
            for skill_idx in range(self.n_skills):
                row = skill_idx // n_cols
                col = skill_idx % n_cols
                
                if frame_idx < len(self.skill_frame_sequences[skill_idx]):
                    frame = self.skill_frame_sequences[skill_idx][frame_idx]
                    
                    # Add skill label
                    frame_labeled = frame.copy()
                    cv2.putText(frame_labeled, f"Skill {skill_idx}", 
                               (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Place in grid
                    y_start = row * frame_h
                    y_end = y_start + frame_h
                    x_start = col * frame_w
                    x_end = x_start + frame_w
                    collage[y_start:y_end, x_start:x_end] = frame_labeled
            
            # Convert RGB to BGR for cv2
            collage_bgr = cv2.cvtColor(collage, cv2.COLOR_RGB2BGR)
            video_writer.write(collage_bgr)
        
        video_writer.release()
        print(f"✅ Collage video saved to: {video_path}")