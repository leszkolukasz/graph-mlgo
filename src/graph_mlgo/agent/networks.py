from typing import Callable
import jax.numpy as jnp
import numpy as np
import distrax
import jax
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
            self.hidden_sizes[0], kernel_init=hidden_init, bias_init=nn.initializers.zeros
            )(obs)

        value = self.activation(value)

        value = nn.Dense(
            self.hidden_sizes[1], kernel_init=hidden_init, bias_init=nn.initializers.zeros
        )(value)

        value = self.activation(value)

        value = nn.Dense(
            1, kernel_init=critic_head_init, bias_init=nn.initializers.zeros
        )(value)

        return jnp.squeeze(value, axis=-1)


class ActorNet(nn.Module):
    hidden_sizes: tuple[int, ...]
    action_dim: int = 1
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        hidden_init = orthogonal(np.sqrt(2))
        actor_head_init = orthogonal(0.01)

        x = nn.Dense(
            self.hidden_sizes[0], kernel_init=hidden_init, bias_init=nn.initializers.zeros
            )(obs)

        x = self.activation(x)

        x = nn.Dense(
            self.hidden_sizes[1], kernel_init=hidden_init, bias_init=nn.initializers.zeros
        )(x)

        x = self.activation(x)

        mean = nn.Dense(
            self.action_dim, kernel_init=actor_head_init, bias_init=nn.initializers.zeros
        )(x)

        log_std = nn.Dense(
            self.action_dim, kernel_init=actor_head_init, bias_init=nn.initializers.zeros
        )(x)

        log_std = jnp.clip(log_std, -20, 2)

        return mean, log_std


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

    def _apply_squashing_function(self, mean, action, logp_action):
        tanh_logp_action = logp_action - jnp.sum(
            2 * (jnp.log(2) - action - nn.softplus(-2 * action)), axis=-1
        )
        tanh_mean = jnp.tanh(mean)
        tanh_action = jnp.tanh(action)
        return tanh_action, tanh_mean, tanh_logp_action

    def act(self, params, obs: jnp.ndarray, rng: jax.Array):
        mean, log_std = self.apply(params, obs, method=self.actor)

        std = jnp.exp(log_std) # ty: ignore
        dist = distrax.MultivariateNormalDiag(mean, std)

        act = dist.sample(seed=rng)
        logp_act = dist.log_prob(act)

        squashed_action, deterministic_squashed_action, logp_squashed_action = self._apply_squashing_function(
            mean, act, logp_act
        )

        return squashed_action, deterministic_squashed_action, logp_squashed_action

    def get_action_log_prob_and_entropy(
        self, params, obs: jnp.ndarray, squashed_action: jnp.ndarray
    ):
        mean, log_std = self.apply(params, obs, method=self.actor)

        std = jnp.exp(log_std) # ty: ignore
        dist = distrax.MultivariateNormalDiag(mean, std)

        def unsquash(act):
            return jnp.arctanh(jnp.clip(act, -1+1e-6, 1-1e-6))

        act = unsquash(squashed_action)
        log_prob = dist.log_prob(act)
        entropy = dist.entropy()

        _, _, squashed_log_prob = self._apply_squashing_function(
            mean, act, log_prob
        )

        return squashed_log_prob, entropy

    def __call__(self, obs: jnp.ndarray):
        mean, log_std = self.actor(obs)
        value = self.critic(obs)
        return mean, log_std, value