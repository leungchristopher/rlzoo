"""Initalise the training state and steps for DQN."""

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
from DQNConvNet import DQNConvNet

class TrainState(train_state.TrainState):
    target_params: flax.core.FrozenDict

def create_train_state(rng_key, learning_rate, input_shape):
    """
    Initialize the TrainState with the network and optimizer.
    """
    convnet = DQNConvNet()
    params = convnet.init(rng_key, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=convnet.apply,
        params=params,
        target_params=params,
        tx=tx,
    )

def select_action(state: TrainState, params, obs, epsilon, rng_key):
    """
    Select an action using epsilon-greedy policy.
    """
    def random_action(key):
        return jax.random.randint(key, shape=(), minval=0, maxval=3)

    def greedy_action(key):
        q_values = state.apply_fn({'params': params}, 
                                  obs[None, ...]) # Add batch dim: (1, 4, 84, 84)
        return jnp.argmax(q_values)

    rng_key, subkey = jax.random.split(rng_key)
    should_explore = jax.random.uniform(subkey) < epsilon

    action = jax.lax.cond(should_explore, random_action, greedy_action, rng_key)
    return action

def compute_loss(params, target_params, batch, gamma, apply_fn):
    """
    Compute the DQN loss (Bellman error).
    """
    obs, actions, rewards, next_obs, dones = batch

    # Q(s, a) - Predicted Q-values for the actions taken
    q_values = apply_fn({'params': params}, obs)
    q_action = jnp.take_along_axis(q_values, actions[:, None], axis=1).squeeze()

    # Q_target(s', a') - Max Q-values from the target network for next states
    q_next = apply_fn({'params': target_params}, next_obs)
    next_q_max = jnp.max(q_next, axis=1)

    # Compute the Bellman target for the loss.
    target = rewards + gamma * next_q_max * (1 - dones)

    loss = optax.huber_loss(q_action, target).mean()
    return loss

@jax.jit
def train_step(state: TrainState, batch, gamma, rng_key):
    """
    Perform a single training step.
    """
    def loss_fn(params):
        return compute_loss(params, state.target_params, batch, gamma, state.apply_fn)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def sync_target_network(state: TrainState):
    """
    Update target_params with current params.
    """
    return state.replace(target_params=state.params)
