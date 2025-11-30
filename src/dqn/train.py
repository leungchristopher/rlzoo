"""
Train a DQN agent on the Pong environment
We also provide an option to render and observe a checkpointed agent.
"""
import os
import jax

from agent import create_train_state, select_action, train_step, sync_target_network
from buffer import ReplayBuffer
from env import make_env
from utils_replacement import CheckpointManager, MetricLogger

LEARNING_RATE = 0.0001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000 # Frames
BUFFER_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 100
TOTAL_STEPS = 1000000
START_TRAINING_STEP = 500
CHECKPOINT_FREQ = 2000

def main():
    rng_key = jax.random.PRNGKey(16)

    # Setup functions
    ckpt_manager = CheckpointManager("checkpoints")
    logger = MetricLogger()
    env = make_env(render=False)
    buffer = ReplayBuffer(BUFFER_SIZE)

    rng_key, init_key = jax.random.split(rng_key)
    state = create_train_state(init_key, LEARNING_RATE, (1, 4, 84, 84))

    restored_state = ckpt_manager.restore(state)
    start_step = 0
    if restored_state is not state:
        state = restored_state
        latest_ckpt = ckpt_manager.latest_step()
        if latest_ckpt:
            try:
                start_step = int(latest_ckpt.split('_')[-1])
                print(f"Resuming from step {start_step}")
                buffer_path = os.path.join(ckpt_manager.directory, f"buffer_{start_step}.npz")
                if os.path.exists(buffer_path):
                    buffer.load(buffer_path)
                else:
                    print(f"Warning: Buffer backup not found at {buffer_path}")
            except ValueError:
                print("Could not parse step from checkpoint path.")

    obs, _ = env.reset()
    epsilon = EPSILON_START
    loss = 0.0

    if start_step > 0:
        epsilon = max(EPSILON_END, EPSILON_START - start_step / EPSILON_DECAY)

    for step in range(start_step, TOTAL_STEPS):
        epsilon = max(EPSILON_END, EPSILON_START - step / EPSILON_DECAY)
        rng_key, action_key = jax.random.split(rng_key)
        action = select_action(state, state.params, obs, epsilon, action_key)
        next_obs, reward, done, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()

        if buffer.size > START_TRAINING_STEP:
            # Sample batch
            rng_key, batch_key = jax.random.split(rng_key)
            batch = buffer.sample(batch_key, BATCH_SIZE)
            state, loss = train_step(state, batch, GAMMA, rng_key)
            logger.log(step, loss)
            if step % TARGET_UPDATE_FREQ == 0:
                state = sync_target_network(state)

        if step % 100 == 0:
            print(f"Step {step}, Epsilon {epsilon:.3f}, Loss {loss:.3f}")

        if step % CHECKPOINT_FREQ == 0 and step >= 0:
            ckpt_manager.save(step, state)
            # Save buffer
            buffer_path = os.path.join(ckpt_manager.directory, f"buffer_{step}.npz")
            buffer.save(buffer_path)

if __name__ == "__main__":
    main()
