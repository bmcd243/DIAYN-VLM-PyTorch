import cv2
import numpy as np
import os

# No longer importing mujoco_py
# Using gymnasium's built-in rendering logic

class Play:
    def __init__(self, env, agent, n_skills):
        self.env = env
        self.agent = agent
        self.n_skills = n_skills
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if not os.path.exists("Vid/"):
            os.mkdir("Vid/")

    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def evaluate(self):
        for z in range(self.n_skills):
            # 50.0 is the framerate; (250, 250) is size
            video_writer = cv2.VideoWriter(f"Vid/skill{z}.mp4", self.fourcc, 50.0, (250, 250))
            
            # Gymnasium reset returns (state, info)
            s, _ = self.env.reset()
            s = self.concat_state_latent(s, z, self.n_skills)
            episode_reward = 0
            
            for _ in range(self.env.spec.max_episode_steps):
                action = self.agent.choose_action(s)
                
                # Gymnasium step returns 5 values
                s_, r, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                s_ = self.concat_state_latent(s_, z, self.n_skills)
                episode_reward += r
                
                # Render the frame
                # Ensure your env was created with render_mode="rgb_array"
                frame = self.env.render() 
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (250, 250))
                video_writer.write(frame)
                
                if done:
                    break
                s = s_
                
            print(f"skill: {z}, episode reward:{episode_reward:.1f}")
            video_writer.release()
            
        self.env.close()
        cv2.destroyAllWindows()