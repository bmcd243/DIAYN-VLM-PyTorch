from .textured_humanoid import TexturedHumanoidEnv
from gymnasium.envs.registration import register

# Register textured humanoid environment
register(
    id='TexturedHumanoid-v5',
    entry_point='envs.textured_humanoid:TexturedHumanoidEnv',
    max_episode_steps=1000,
)


__all__ = ['TexturedHumanoidEnv']
