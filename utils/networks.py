from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            #print("x_input.shape, size, layer_id)", x.shape, size, i)
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            #print("x_output.shape", x.shape)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
                
        return x
    
class MLPResNet(nn.Module):
    """Multi-layer perceptron with residual connections.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            #print("action x_input.shape, size, layer_id)", x.shape, size, i)
            x_res = x
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activations(x)
            if (i + 1 < len(self.hidden_dims) and i > 0) or self.activate_final:
                x += x_res
            #print("action x_output.shape", x.shape)
        
        
        return x
    
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    embed_dim: int
    scale: float = 30.0

    @nn.compact
    def __call__(self, x):
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        W = self.param(
            "W",
            nn.initializers.normal(stddev=self.scale),
            (self.embed_dim // 2,),
        )
        
        #import pdb
        #pdb.set_trace()

        # Compute the Gaussian random features
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    input_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=self.output_dim)(x)

class SiLU(nn.Module):
    """SiLU activation function."""
    
    @nn.compact
    def __call__(self, x):
        return x * nn.sigmoid(x)

class ResidualBlock(nn.Module):
    """Residual block with dense layers and SiLU activation."""
    input_dim: int
    output_dim: int
    t_dim: int = 128
    last: bool = False

    @nn.compact
    def __call__(self, x, t, deterministic: bool):
        # Time MLP
        time_mlp = nn.Sequential([
            SiLU(),
            nn.Dense(features=self.output_dim)
        ])
        
        # Dense layers
        dense1 = nn.Sequential([
            nn.Dense(features=self.output_dim),
            nn.Dropout(rate=0.1, deterministic=deterministic),  # Dropout with dynamic PRNGKey
            SiLU()
        ])
        
        dense2 = nn.Sequential([
            nn.Dense(features=self.output_dim),
            nn.Dropout(rate=0.1, deterministic=deterministic),  # Dropout with dynamic PRNGKey
            SiLU()
        ])
        
        # Modify x if dimensions do not match
        modify_x = nn.Dense(features=self.output_dim) if self.input_dim != self.output_dim else lambda x: x

        h1 = dense1(x) + time_mlp(t)
        h2 = dense2(h1)
        return h2 + modify_x(x)

class ScoreNet(nn.Module):
    """Score-based neural network."""
    input_dim: int
    output_dim: int
    time_dim: int
    embed_dim: int = 32

    @nn.compact
    def __call__(self, x, t, condition: Optional[jnp.ndarray] = None, deterministic: bool = False):
        # Embed time steps
        embed = nn.Sequential([
            GaussianFourierProjection(embed_dim=self.embed_dim),
            nn.Dense(features=self.embed_dim)
        ])(t)

        # Pre-sort condition
        pre_sort_condition = nn.Sequential([
            Dense(input_dim=self.input_dim, output_dim=32),
            SiLU()
        ])

        # Sort time embedding
        sort_t = nn.Sequential([
            nn.Dense(features=128),
            SiLU(),
            nn.Dense(features=128)
        ])

        # Down blocks
        down_block1 = ResidualBlock(input_dim=self.input_dim, output_dim=512)
        down_block2 = ResidualBlock(input_dim=512, output_dim=256)
        down_block3 = ResidualBlock(input_dim=256, output_dim=128)

        # Middle block
        middle1 = ResidualBlock(input_dim=128, output_dim=128)

        # Up blocks
        up_block3 = ResidualBlock(input_dim=256, output_dim=256)
        up_block2 = ResidualBlock(input_dim=512, output_dim=512)

        # Last layer
        last = nn.Dense(features=self.output_dim)

        # Process condition
        if condition is not None:
            condition_embed = pre_sort_condition(condition)
        else:
            condition_embed = pre_sort_condition(jnp.ones((x.shape[0], self.input_dim)))
        
        embed = embed.reshape(-1, self.embed_dim)
        embed = jnp.concatenate([condition_embed, embed], axis=-1)
        embed = sort_t(embed)

        # Forward pass
        d1 = down_block1(x, embed, deterministic)
        d2 = down_block2(d1, embed, deterministic)
        d3 = down_block3(d2, embed, deterministic)
        u3 = middle1(d3, embed, deterministic)
        u2 = up_block3(jnp.concatenate([d3, u3], axis=-1), embed, deterministic)
        u1 = up_block2(jnp.concatenate([d2, u2], axis=-1), embed, deterministic)
        u0 = jnp.concatenate([d1, u1], axis=-1)
        h = last(u0)

        # Normalize output
        return h

class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v

class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    model_arch: str = 'mlp'
    encoder: nn.Module = None
    observation_dim: int = 0
    time_dim: int = 1

    def setup(self) -> None:
        if self.model_arch == 'mlp':
            self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        elif self.model_arch == 'mlp_resnet':
            self.mlp = MLPResNet(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
                kernel_init=default_init(1e-2),
            )
        elif self.model_arch == 'score_net':
            self.mlp = ScoreNet(
                input_dim=self.observation_dim,
                output_dim=self.action_dim,
                time_dim=self.time_dim,
                embed_dim=32,
            )
        else:
            raise ValueError(f"Unknown model architecture: {self.model_arch}")

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False, deterministic=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        if self.model_arch == 'score_net':
            if times is None:
                times = jnp.zeros((observations.shape[0], self.time_dim))
            v = self.mlp(observations, times, actions, deterministic=deterministic)
        else:
            if times is None:
                inputs = jnp.concatenate([observations, actions], axis=-1)
            else:
                #output_times = jnp.ones_like(times)
                #key = jax.random.PRNGKey(0)  # seed 可以是任意整数
                #output_times = jax.random.uniform(key, shape=times.shape, minval=0.0, maxval=1.0)
                inputs = jnp.concatenate([observations, actions, times], axis=-1)
            v = self.mlp(inputs)

        return v

class ActorVectorMeanField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    model_arch: str = 'mlp'
    encoder: nn.Module = None
    observation_dim: int = 0
    time_dim: int = 1

    def setup(self) -> None:
        if self.model_arch == 'mlp':
            self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        elif self.model_arch == 'mlp_resnet':
            self.mlp = MLPResNet(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
                kernel_init=default_init(1e-2),
            )
        elif self.model_arch == 'score_net':
            self.mlp = ScoreNet(
                input_dim=self.observation_dim,
                output_dim=self.action_dim,
                time_dim=self.time_dim,
                embed_dim=32,
            )
        else:
            raise ValueError(f"Unknown model architecture: {self.model_arch}")

    @nn.compact
    def __call__(self, observations, actions, input_times=None, output_times=None, is_encoded=False, deterministic=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        if self.model_arch == 'score_net':
            if input_times is None:
                input_times = jnp.zeros((observations.shape[0], self.time_dim))
            if output_times is None:
                output_times = jnp.ones((observations.shape[0], self.time_dim))
            # Ensure input_times and output_times have the correct shape
            #v = self.mlp(observations, times, actions, deterministic=deterministic)
            v = self.mlp(observations, input_times, output_times, actions, deterministic=deterministic)
        else:
            if input_times is None:
                input_times = jnp.zeros((observations.shape[0], self.time_dim))
            if output_times is None:
                output_times = jnp.ones((observations.shape[0], self.time_dim))    
            inputs = jnp.concatenate([observations, actions, input_times, output_times], axis=-1)
            v = self.mlp(inputs)

        return v


class ActorVectorField_old(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    model_arch: str = 'mlp'
    encoder: nn.Module = None
    observation_dim: int = 0
    time_dim: int = 1

    def setup(self) -> None:
        if self.model_arch == 'mlp':
            self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        elif self.model_arch == 'mlp_resnet':
            self.mlp = MLPResNet(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
                kernel_init=default_init(1e-2),
            )
        elif self.model_arch == 'score_net':
            self.mlp = ScoreNet(
                input_dim=self.observation_dim,
                output_dim=self.action_dim,
                time_dim=self.time_dim,
                embed_dim=32,
            )
        else:
            raise ValueError(f"Unknown model architecture: {self.model_arch}")
            
    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False, deterministic=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if(self.model_arch == 'score_net'):
            if times is None:
                times = jnp.ones((observations.shape[0]))
            variables = self.mlp.init(jax.random.PRNGKey(0), actions, times, observations, deterministic)
            rng = jax.random.PRNGKey(1)
            rngs = {'dropout': rng}
            v = self.mlp.apply(variables, actions, times, observations, deterministic, rngs=rngs)
            #v = self.mlp(actions, times, observations, deterministic, rngs=rngs)
        else:
            if times is None:
                inputs = jnp.concatenate([observations, actions], axis=-1)
            else:
                inputs = jnp.concatenate([observations, actions, times], axis=-1)

            v = self.mlp(inputs)

        return v
