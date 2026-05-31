from typing import Callable, cast

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import orthogonal


class CriticNet(nn.Module):
    hidden_sizes: tuple[int, ...]
    activation: Callable

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        hidden_init = orthogonal(np.sqrt(2))
        critic_head_init = orthogonal(1.0)

        value = nn.Dense(
            self.hidden_sizes[0],
            kernel_init=hidden_init,
            bias_init=nn.initializers.zeros,
        )(obs)

        value = self.activation(value)

        value = nn.Dense(
            self.hidden_sizes[1],
            kernel_init=hidden_init,
            bias_init=nn.initializers.zeros,
        )(value)

        value = self.activation(value)

        value = nn.Dense(
            1, kernel_init=critic_head_init, bias_init=nn.initializers.zeros
        )(value)

        return jnp.squeeze(value, axis=-1)


class ActorNet(nn.Module):
    hidden_sizes: tuple[int, ...]
    action_dim: int = 2
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        hidden_init = orthogonal(np.sqrt(2))
        actor_head_init = orthogonal(0.01)

        x = nn.Dense(
            self.hidden_sizes[0],
            kernel_init=hidden_init,
            bias_init=nn.initializers.zeros,
        )(obs)

        x = self.activation(x)

        x = nn.Dense(
            self.hidden_sizes[1],
            kernel_init=hidden_init,
            bias_init=nn.initializers.zeros,
        )(x)

        x = self.activation(x)

        logits = nn.Dense(
            self.action_dim,
            kernel_init=actor_head_init,
            bias_init=nn.initializers.zeros,
        )(x)

        return logits


class PPOAgent(nn.Module):
    hidden_sizes: tuple[int, ...]
    action_dim: int = 2
    activation: Callable = nn.tanh

    def setup(self):
        self.actor_net = ActorNet(
            hidden_sizes=self.hidden_sizes,
            action_dim=self.action_dim,
            activation=self.activation,
        )
        self.critic_net = CriticNet(
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        )

    def critic(self, obs: jnp.ndarray):
        return self.critic_net(obs)

    def actor(self, obs: jnp.ndarray):
        return self.actor_net(obs)

    def act(self, params, obs: jnp.ndarray, rng: jax.Array):
        logits = cast(jnp.ndarray, self.apply(params, obs, method=self.actor))

        dist = distrax.Categorical(logits=logits)

        act = dist.sample(seed=rng)
        logp_act = dist.log_prob(act)

        deterministic_act = jnp.argmax(logits, axis=-1)

        return act, deterministic_act, logp_act

    def get_action_log_prob_and_entropy(
        self, params, obs: jnp.ndarray, action: jnp.ndarray
    ):
        logits = cast(jnp.ndarray, self.apply(params, obs, method=self.actor))

        dist = distrax.Categorical(logits=logits)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy

    def __call__(self, obs: jnp.ndarray):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value
