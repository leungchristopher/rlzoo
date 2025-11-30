"""
Play with a trained DQN agent on the Pong environment using the latest checkpoint.
"""

import jax
from agent import create_train_state, select_action
from env import make_env
from utils_replacement import CheckpointManager


LEARNING_RATE = 0.00025 # Needed for state initialization, though not used for playing
EPSILON_PLAY = 0.01     # Low epsilon for evaluation/play

def main():
    rng_key = jax.random.PRNGKey(16)

    ckpt_manager = CheckpointManager("checkpoints")
    env = make_env(render=True)
    rng_key, init_key = jax.random.split(rng_key)
    state = create_train_state(init_key, LEARNING_RATE, (1, 4, 84, 84))

    print("Restoring checkpoint...")
    restored_state = ckpt_manager.restore(state)

    if restored_state is state:
        print("WARNING: No checkpoint found! Playing with random initialisation.")
    else:
        state = restored_state
        latest_ckpt = ckpt_manager.latest_step()
        print(f"Successfully restored from {latest_ckpt}")

    try:
        while True:
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0

            while not (done or truncated):
                rng_key, action_key = jax.random.split(rng_key)
                action = select_action(state, state.params, obs, EPSILON_PLAY, action_key)

                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward

            print(f"Game Over. Total Reward: {total_reward}")

    except KeyboardInterrupt:
        print("\nStopping play...")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
