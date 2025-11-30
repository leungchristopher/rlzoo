import numpy as np
import jax
import jax.numpy as jnp

class ReplayBuffer:
    """
    Store 4 successive frames, actions, rewards, next_obs, and dones. 
    Store as integers and convert to jax arrays when sampling.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.next_buffer = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.next_idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """
        Add a new experience to the buffer.
        Args:
            obs: The observation.
            action: The action.
            reward: The reward.
            next_obs: The next observation.
            done: Whether the episode is done.
        """
        self.buffer[self.next_idx] = obs
        self.next_buffer[self.next_idx] = next_obs
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = reward
        self.dones[self.next_idx] = done
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, rng_key, batch_size: int):
        """
        Return a batch of transitions as JAX arrays.
        Args:
            rng_key: JAX PRNG key for sampling.
            batch_size: Number of samples to return.
        """
        indices = jax.random.randint(rng_key, (batch_size,), 0, self.size)
        obs = jnp.array(self.buffer[indices], dtype=jnp.float32)
        next_obs = jnp.array(self.next_buffer[indices], dtype=jnp.float32)
        actions = jnp.array(self.actions[indices])
        rewards = jnp.array(self.rewards[indices])
        dones = jnp.array(self.dones[indices])
        return obs, actions, rewards, next_obs, dones

    def save(self, filename):
        np.savez_compressed(
            filename,
            buffer=self.buffer,
            next_buffer=self.next_buffer,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            next_idx=self.next_idx,
            size=self.size
        )
        print(f"Buffer saved to {filename}")

    def load(self, filename):
        data = np.load(filename)
        self.buffer = data['buffer']
        self.next_buffer = data['next_buffer']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.dones = data['dones']
        self.next_idx = int(data['next_idx'])
        self.size = int(data['size'])
        print(f"Buffer loaded from {filename}")
