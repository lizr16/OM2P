import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
import tensorflow_probability.substrates.jax.distributions as tfd


class FQLAgent_dc(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        rng, sample_rng = jax.random.split(rng)
        
        if(self.config['q_algo'] == 'iql'):
            next_qs = self.network['target_critic'](batch['next_observations'], actions=batch['next_actions'])
        else:
            next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
            next_actions = jnp.clip(next_actions, -1, 1)

            next_qs = self.network['target_critic'](batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * (1.0 - batch['dones']) * next_q

        q = self.network['critic'](batch['observations'], actions=batch['actions'], params=grad_params)
        
        #critic_loss = jnp.square(q - target_q).mean()
        critic_loss = self.expectile_loss(diff = target_q - q, expectile = self.config['expectile_term']).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }
        
    def expectile_loss(self, diff, expectile=0.8):
        weight = jnp.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def compute_softmax_acs(self, q_vals, acs):
        
        # 计算每个样本的最大 Q 值
        max_q_vals = jnp.max(q_vals, axis=1, keepdims=True)
        
        # 归一化 Q 值
        norm_q_vals = q_vals - max_q_vals
        
        # 计算 e_beta_normQ
        e_beta_normQ = jnp.exp(norm_q_vals)
        
        # 假设 action: (256, 10, 2), q_values: (256, 10, 1)
        # 第一步：归一化 q 值作为权重（防止数值不稳定）
        weights = jnp.squeeze(e_beta_normQ, axis=-1)  # 变成 (256, 10)
        weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + 1e-8)  # 避免除0，归一化权重

        # 第二步：加权平均
        # weights: (256, 10) -> (256, 10, 1)，广播乘法
        weighted_action = acs * weights[..., None]  # (256, 10, 2)

        # 第三步：沿 axis=1 求和
        mean_action = jnp.sum(weighted_action, axis=1)  # (256, 2)
        
        return mean_action

    def softmax_q_loss(self, batch, grad_params, rng):
        
        batch_size, action_dim = batch['actions'].shape
        
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        #target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network['actor_onestep_flow'](batch['observations'], noises, params=grad_params)
        actor_actions = jnp.clip(actor_actions, -1, 1)
        q_values = self.network['critic'](batch['observations'], actions=actor_actions)
        
        init_omar_mu = self.config['omar_mu']
        init_omar_sigma = self.config['omar_sigma']
        omar_num_samples = self.config['omar_num_samples']
        omar_iters = self.config['omar_iters']
        omar_mu = jnp.full((batch_size, action_dim), init_omar_mu)
        omar_sigma = jnp.full((batch_size, action_dim), init_omar_sigma)
        
        max_action = 1.0
        
        # 格式化观测值
        formatted_obs = jnp.tile(batch['observations'][:, None, :], (1, omar_num_samples, 1)).reshape(-1, batch['observations'].shape[1])

        for iter_idx in range(omar_iters):
            eps = 1e-10
            dist = tfd.Normal(loc=omar_mu, scale=omar_sigma + eps)

            # 从随机数生成器中拆分一个子键
            sample_rng, _ = jax.random.split(rng)
            # 采样并调整形状
            cem_sampled_acs = dist.sample(sample_shape=(omar_num_samples,), seed=sample_rng).transpose((1, 0, 2)).clip(-max_action, max_action)
            formatted_cem_sampled_acs = cem_sampled_acs.reshape(-1, cem_sampled_acs.shape[-1])

            # 计算 Q 值
            vf_in = jnp.concatenate((formatted_obs, formatted_cem_sampled_acs), axis=1)
            #all_pred_qvals = curr_agent.critic.q1(formatted_obs, formatted_cem_sampled_acs)
            all_pred_qvals = self.network['critic'](formatted_obs, actions=formatted_cem_sampled_acs)
            all_pred_qvals = all_pred_qvals[0].reshape(batch_size, omar_num_samples, -1)

            # 更新 mu 和 sigma
            updated_mu = self.compute_softmax_acs(all_pred_qvals, cem_sampled_acs)
            omar_mu = updated_mu

            updated_sigma = jnp.sqrt(jnp.mean((cem_sampled_acs - updated_mu[:, None, :]) ** 2, axis=1))
            omar_sigma = updated_sigma

        # 获取 top Q 值和动作
        top_qvals, top_inds = jax.lax.top_k(all_pred_qvals, k=1)
        top_ac_inds = jnp.tile(top_inds, (1, 1, action_dim))
        top_acs = jnp.take_along_axis(cem_sampled_acs, top_ac_inds, axis=1)

        # 计算候选 Q 值和动作
        cem_qvals = top_qvals
        pol_qvals = q_values[0][:,None, None]
        cem_acs = top_acs
        pol_acs = actor_actions[:, None]

        candidate_qvals = jnp.concatenate([pol_qvals, cem_qvals], axis=1)
        candidate_acs = jnp.concatenate([pol_acs, cem_acs], axis=1)

        # q_values: (256, 11, 1) -> (256, 11)
        q_values_squeezed = jnp.squeeze(candidate_qvals, axis=-1)

        # 找到最大 q 值的位置索引（axis=1）
        max_indices = jnp.argmax(q_values_squeezed, axis=1)  # shape: (256,)

        # 构造 batch 的索引
        batch_indices = jnp.arange(candidate_acs.shape[0])  # shape: (256,)

        # 从 action 中取出最大 q 值对应的动作
        best_actions = candidate_acs[batch_indices, max_indices]  # shape: (256, 2)

        """ # 获取最大 Q 值和动作
        max_qvals, max_inds = jax.lax.top_k(candidate_qvals, k=1)
        max_ac_inds = jnp.tile(max_inds, (1, 1, action_dim))
        max_acs = jnp.take_along_axis(candidate_acs, max_ac_inds, axis=1).squeeze(axis=1)

        # 计算模仿损失
        mimic_acs = max_acs """
        mimic_term = jnp.mean((actor_actions - best_actions) ** 2)

        return mimic_term
        

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network['actor_bc_flow'](batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network['actor_onestep_flow'](batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network['critic'](batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)
        
        # Softmax loss.
        if(self.config['is_softmax_q'] == 1):
            softmax_q_loss = self.softmax_q_loss(batch, grad_params, rng)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        if(self.config['training_task'] == 3):
            actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss + self.config['omar_coe'] * softmax_q_loss
        elif(self.config['training_task'] == 2):
            actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss
        elif(self.config['training_task'] == 1):
            actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss
        elif(self.config['training_task'] == 0):
            actor_loss = bc_flow_loss + q_loss
        else:
            actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }

    def actor_loss_bc(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network['actor_bc_flow'](batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        return bc_flow_loss, {
            'bc_flow_loss': bc_flow_loss,
        }

    def actor_loss_distill(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network['actor_onestep_flow'](batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network['critic'](batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }



    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network[f'{module_name}'].params,
            network[f'target_{module_name}'].params,
        )
        # 创建一个新的 TrainState 对象来更新 target network
        new_target_network = network[f'target_{module_name}'].replace(params=new_target_params)
        # 更新 network 字典
        network[f'target_{module_name}'] = new_target_network

    """ def create_trainstate(module_def, inputs, rng, lr):
        params = module_def.init(rng, *inputs)['params']
        tx = optax.adam(lr)
        return TrainState.create(
            apply_fn=module_def.apply,
            params=params,
            tx=tx,
        ) """

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, *init_rngs = jax.random.split(rng, 5)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Encoders
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define modules
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        lr = config['lr']
        tx = optax.adam(learning_rate=lr)

        # Independent TrainStates
        critic = TrainState.create(
            model_def=critic_def,
            params=critic_def.init(init_rngs[0], ex_observations, ex_actions)['params'],
            tx=tx
        )
        target_critic = TrainState.create(
            model_def=critic_def,
            params=critic_def.init(init_rngs[1], ex_observations, ex_actions)['params'],
            tx=tx
        )
        actor_bc_flow = TrainState.create(
            model_def=actor_bc_flow_def,
            params=actor_bc_flow_def.init(init_rngs[2], ex_observations, ex_actions, ex_times)['params'],
            tx=tx
        )
        actor_onestep_flow = TrainState.create(
            model_def=actor_onestep_flow_def,
            params=actor_onestep_flow_def.init(init_rngs[3], ex_observations, ex_actions)['params'],
            tx=tx
        )

        # Wrap as dict
        network = {
            'critic': critic,
            'target_critic': target_critic,
            'actor_bc_flow': actor_bc_flow,
            'actor_onestep_flow': actor_onestep_flow,
        }

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng=rng, network=network, config=flax.core.FrozenDict(config))


    # 替换 update 函数中的梯度更新部分如下：
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def critic_loss_fn(params):
            loss, info = self.critic_loss(batch, params, rng)
            return loss, info

        def actor_loss_bc_fn(params):
            loss, info = self.actor_loss_bc(batch, params, rng)
            return loss, info
        
        def actor_loss_distill_fn(params):
            loss, info = self.actor_loss_distill(batch, params, rng)
            return loss, info

        # Critic 更新
        grads, info_c = jax.grad(critic_loss_fn, has_aux=True)(self.network['critic'].params)
        critic_state = self.network['critic'].apply_gradients(grads=grads)

        # Target 更新
        self.target_update(self.network, 'critic')  # 仍使用你原来的方式

        # Actor 更新
        grads, info_a = jax.grad(actor_loss_bc_fn, has_aux=True)(self.network['actor_bc_flow'].params)
        actor_bc_state = self.network['actor_bc_flow'].apply_gradients(grads=grads)

        grads, info_o = jax.grad(actor_loss_distill_fn, has_aux=True)(self.network['actor_onestep_flow'].params)
        actor_onestep_state = self.network['actor_onestep_flow'].apply_gradients(grads=grads)

        # Replace agent
        """ new_network = self.network.copy(
            add_or_replace={
                'critic': critic_state,
                'actor_bc_flow': actor_bc_state,
                'actor_onestep_flow': actor_onestep_state,
            }
        ) """
        new_network = {
            'critic': critic_state,
            'actor_bc_flow': actor_bc_state,
            'actor_onestep_flow': actor_onestep_state,
            'target_critic': self.network['target_critic'],  # 确保目标网络也被包含在内
        }

        # Merge info
        info = {**{f"critic/{k}": v for k, v in info_c.items()},
                **{f"actor_multistep/{k}": v for k, v in info_a.items()},
                **{f"actor_onestep/{k}": v for k, v in info_o.items()}}

        return self.replace(network=new_network, rng=new_rng), info


    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network['actor_onestep_flow'](observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def sample_flow_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        
        if self.config['encoder'] is not None:
            observations = self.network['actor_bc_flow_encoder'](observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network['actor_bc_flow'](observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions
    
    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network['actor_bc_flow_encoder'](observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network['actor_bc_flow'](observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fql_decompose',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            q_algo='ql',
            expectile_term=0.5,
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            training_task=-1,
            is_softmax_q=0,
            omar_coe=1.0,
            omar_iters=3,
            omar_mu=0.0,
            omar_sigma=1.0,
            omar_num_samples=10,
            omar_num_elites=10,
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
