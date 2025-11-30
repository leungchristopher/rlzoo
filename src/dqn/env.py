import jax
import numpy as np
import gymnasium as gym
import ale_py


class PongActionWrapper(gym.ActionWrapper):
    """
    Reducing the action space to NOOP, UP, DOWN
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action):
        if action == 0:
            return 0 # NOOP
        elif action == 1:
            return 2 # RIGHT (UP)
        elif action == 2:
            return 3 # LEFT (DOWN)
        return 0


def crop_pong(obs):
    """
    Crop the pong observation.
    Args:
        obs: The observation.
    Returns:
        obs: The cropped observation.
    """
    obs = obs[34:195, :, 0] # Crop height to 160, take one channel
    obs = jax.image.resize(obs, (84, 84), method='bilinear')
    obs = obs.at[obs == 144].set(0)     # Remove background
    obs = obs.at[obs == 109].set(0)     # Remove background
    obs = obs.at[obs > 0].set(1)        # Make everything the same colour
    return obs


class PongObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to crop and resize the Pong observation.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84), dtype=np.uint8)

    def observation(self, obs):
        return np.array(crop_pong(obs), dtype=np.uint8)


def make_env(render: bool = False):
    """
    Make the environment.
    """
    gym.register_envs(ale_py)
    if render:
        env = gym.make("Pong-v0", render_mode="human")
    else:
        env = gym.make("Pong-v0")

    env = PongActionWrapper(env)
    env = PongObservationWrapper(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    return env


if __name__ == "__main__":
    # Visualise a random game for setup testing.
    env = make_env(render=True)
    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
