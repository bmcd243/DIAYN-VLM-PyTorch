import os
import numpy as np
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


class TexturedHumanoidEnv(HumanoidEnv):
    def __init__(self, 
                 terminate_when_unhealthy=True,
                 healthy_z_range=(1.0, 2.0),
                 reset_noise_scale=0.005,
                 **kwargs):
        """
        Textured humanoid with configurable termination conditions
        
        Args:
            terminate_when_unhealthy: Whether to terminate on unhealthy states
            healthy_z_range: Acceptable z-coordinate range for torso
            reset_noise_scale: Scale of noise added to initial state
        """
        # Store custom parameters
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        
        # Call parent with custom XML
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_file = os.path.join(current_dir, "..", "assets", "mujoco", "humanoid_textured.xml")
        super().__init__(
            xml_file=xml_file,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            reset_noise_scale=reset_noise_scale,
            **kwargs
        )
    
    def reset_model(self):
        """Override to use custom reset_noise_scale"""
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(
            self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation