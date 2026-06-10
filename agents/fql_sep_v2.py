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


class FQLAgent_sep_v2(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        rng, sample_rng = jax.random.split(rng)
        
        if(self.config['q_algo'] == 'iql'):
            next_qs = self.network['critic'].select('target_critic')(batch['next_observations'], actions=batch['next_actions'])
        else:
            next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
            next_actions = jnp.clip(next_actions, -1, 1)

            next_qs = self.network['critic'].select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        if('episodic' in self.config['q_algo']):
            accumulated_rewards = batch['rewards'].reshape(-1,25)
            for i in range(accumulated_rewards.shape[1]-2, -1, -1):
                accumulated_rewards = accumulated_rewards.at[:, i].add(self.config['discount'] * accumulated_rewards[:, i+1])
                #accumulated_rewards[:, i] = accumulated_rewards[:, i] + self.config['discount'] *  accumulated_rewards[:, i+1]
            target_q = accumulated_rewards.reshape(-1)
        else:
            target_q = batch['rewards'] + self.config['discount'] * (1.0 - batch['dones']) * next_q

        q = self.network['critic'].select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        
        if(self.config['q_algo'] == 'iql'):
            critic_loss = self.expectile_loss(diff = target_q - q, expectile = self.config['expectile_term']).mean()
        else:
            critic_loss = jnp.square(q - target_q).mean()
        
        if(self.config['q_algo'] == 'cql'):
            conservative_loss = self.conservative_q_loss(batch, grad_params, target_q, rng)
        else:
            conservative_loss = 0.0
            
        total_loss = critic_loss * self.config['critic_loss_weight'] + conservative_loss

        return total_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'conservative_loss': conservative_loss,
        }
        
    def conservative_q_loss(self, batch, grad_params, target_q, rng):
        
        num_sampled_action = self.config['num_sampled_actions']
        lse_temp = self.config['lse_temp']
        # 格式化观测值
        rng, sample_rng = jax.random.split(rng)
        
        formatted_obs = jnp.tile(batch['observations'].reshape(batch['observations'].shape[0], 1, batch['observations'].shape[1]),
                                (1, num_sampled_action, 1)).reshape(-1, batch['observations'].shape[1])

        # 生成随机动作
        random_acs = jax.random.uniform(sample_rng, shape=(batch['actions'].shape[0] * num_sampled_action, batch['actions'].shape[1]),
                            minval=-1, maxval=1)
        random_acs_log_pi = jnp.log(0.5 ** random_acs.shape[-1])

        # 计算 critic 的输出
        random_qvals = self.network.select('critic')(formatted_obs, actions=random_acs, params=grad_params)
        
        
        # 重塑 Q 值
        random_qvals1 = random_qvals[0].reshape(batch['observations'].shape[0], num_sampled_action)
        random_qvals2 = random_qvals[1].reshape(batch['observations'].shape[0], num_sampled_action)

        # 计算策略 Q 值
        policy_qvals1 = jax.nn.logsumexp((random_qvals1 - random_acs_log_pi) / lse_temp, axis=1, keepdims=True) * lse_temp
        policy_qvals2 = jax.nn.logsumexp((random_qvals2 - random_acs_log_pi) / lse_temp, axis=1, keepdims=True) * lse_temp

        # 计算 CQL 项
        cql_term = (policy_qvals1.reshape(-1) - target_q).mean() + (policy_qvals2.reshape(-1) - target_q).mean()

        return cql_term
        
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
        
        rng, noise_rng, dropout_rng = jax.random.split(rng,3)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        #target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        if(self.config['model_arch'] == 'score_net'):
            actor_actions = self.network['actor_onestep'].select('actor_onestep_flow')(batch['observations'], noises, rngs={'dropout': dropout_rng}, params=grad_params)
        else:
            actor_actions = self.network['actor_onestep'].select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        actor_actions = jnp.clip(actor_actions, -1, 1)
        q_values = self.network['critic'].select('critic')(batch['observations'], actions=actor_actions)
        
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
            all_pred_qvals = self.network['critic'].select('critic')(formatted_obs, actions=formatted_cem_sampled_acs)
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
        
    def actor_bc_loss_scorenet(self, batch, grad_params, rng):
        
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, dropout_rng = jax.random.split(rng, 4)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network['actor_bc'].select('actor_bc_flow')(batch['observations'], x_t, t, rngs={'dropout': dropout_rng}, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)
        
        # Softmax Q for bc flow.
        if(self.config['is_softmax_bc'] == 1):
            qs = self.network['actor'].select('critic')(batch['observations'], actions=target_flow_actions).mean(axis=0).reshape(-1,1)
            #weight_qs = jnp.exp(self.config['q_val_temperature'] * qs) / jnp.sum(jnp.exp(self.config['q_val_temperature'] * qs))
            weight_qs = jax.nn.softmax(self.config['q_val_temperature'] * qs)
            bc_flow_loss = jnp.mean(weight_qs * (pred - vel) ** 2) * weight_qs.shape[0]
        elif(self.config['is_softmax_bc'] == 2):
            qs = self.network['actor'].select('critic')(batch['observations'], actions=batch['actions']).mean(axis=0).reshape(-1,1)
            repeated_observations = jnp.repeat(batch['observations'][:, jnp.newaxis, :], 10, axis=1).reshape(batch_size * 10, -1)
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size * 10, action_dim))
            flow_actions = self.compute_flow_actions(repeated_observations, noises=noises)
            qs_tilde = self.network['critic'].select('critic')(repeated_observations, actions=flow_actions).mean(axis=0).reshape(-1,1)
            new_qs = 1 / jnp.exp(self.config['q_val_temperature'] * (qs_tilde - jnp.repeat(qs[:, jnp.newaxis, :], 10, axis=1).reshape(batch_size * 10, -1)))
            weight_qs = new_qs.reshape(batch_size, 10).mean(axis=1).reshape(-1,1)
            bc_flow_loss = jnp.mean(weight_qs * (pred - vel) ** 2) * weight_qs.shape[0]
            
        return bc_flow_loss, {
            'bc_flow_loss': bc_flow_loss,
        }
        
    def actor_onestep_loss_scorenet(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        
        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises,seed=rng)
        rng, dropout_rng = jax.random.split(rng)
        actor_actions = self.network['actor_onestep'].select('actor_onestep_flow')(batch['observations'], noises, rngs={'dropout': dropout_rng}, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network['critic'].select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)
        
        # Softmax loss.
        if(self.config['is_softmax_q'] == 1 or self.config['training_task'] == 3):
            softmax_q_loss = self.softmax_q_loss(batch, grad_params, rng)
        else:
            softmax_q_loss = 0

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss
            
        # Total loss.
        if(self.config['training_task'] == 3):
            actor_onestep_loss = self.config['alpha'] * distill_loss + self.config['q_loss_weight'] * q_loss + self.config['omar_coe'] * softmax_q_loss
        elif(self.config['training_task'] == 2):
            actor_onestep_loss = self.config['alpha'] * distill_loss + self.config['q_loss_weight'] * q_loss
        elif(self.config['training_task'] == 1):
            actor_onestep_loss = self.config['alpha'] * distill_loss
        elif(self.config['training_task'] == 0):
            actor_onestep_loss = self.config['q_loss_weight'] * q_loss
        else:
            actor_onestep_loss = self.config['alpha'] * distill_loss + self.config['q_loss_weight'] * q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_onestep_loss, {
            'actor_onestep_loss': actor_onestep_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
            'softmax_q_loss': softmax_q_loss,
        }
        
        
    def actor_loss_scorenet(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, dropout_rng = jax.random.split(rng, 4)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network['actor_bc'].select('actor_bc_flow')(batch['observations'], x_t, t, rngs={'dropout': dropout_rng}, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises,seed=rng)
        rng, dropout_rng = jax.random.split(rng)
        actor_actions = self.network['actor_onestep'].select('actor_onestep_flow')(batch['observations'], noises, rngs={'dropout': dropout_rng}, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network['critic'].select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)
        
        # Softmax loss.
        if(self.config['is_softmax_q'] == 1 or self.config['training_task'] == 3):
            softmax_q_loss = self.softmax_q_loss(batch, grad_params, rng)
        else:
            softmax_q_loss = 0

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss
            
        # Softmax Q for bc flow.
        if(self.config['is_softmax_bc'] == 1):
            qs = self.network['actor'].select('critic')(batch['observations'], actions=target_flow_actions).mean(axis=0).reshape(-1,1)
            #weight_qs = jnp.exp(self.config['q_val_temperature'] * qs) / jnp.sum(jnp.exp(self.config['q_val_temperature'] * qs))
            weight_qs = jax.nn.softmax(self.config['q_val_temperature'] * qs)
            bc_flow_loss = jnp.mean(weight_qs * (pred - vel) ** 2) * weight_qs.shape[0]
        elif(self.config['is_softmax_bc'] == 2):
            qs = self.network['actor'].select('critic')(batch['observations'], actions=batch['actions']).mean(axis=0).reshape(-1,1)
            repeated_observations = jnp.repeat(batch['observations'][:, jnp.newaxis, :], 10, axis=1).reshape(batch_size * 10, -1)
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size * 10, action_dim))
            flow_actions = self.compute_flow_actions(repeated_observations, noises=noises)
            qs_tilde = self.network['critic'].select('critic')(repeated_observations, actions=flow_actions).mean(axis=0).reshape(-1,1)
            new_qs = 1 / jnp.exp(self.config['q_val_temperature'] * (qs_tilde - jnp.repeat(qs[:, jnp.newaxis, :], 10, axis=1).reshape(batch_size * 10, -1)))
            weight_qs = new_qs.reshape(batch_size, 10).mean(axis=1).reshape(-1,1)
            bc_flow_loss = jnp.mean(weight_qs * (pred - vel) ** 2) * weight_qs.shape[0]
            

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
            'softmax_q_loss': softmax_q_loss,
        }

    def actor_bc_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network['actor_bc'].select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)
        
        # Softmax Q for bc flow.
        if(self.config['is_softmax_bc'] == 1):
            qs = self.network['critic'].select('critic')(batch['observations'], actions=target_flow_actions).mean(axis=0).reshape(-1,1)
            #weight_qs = jnp.exp(self.config['q_val_temperature'] * qs) / jnp.sum(jnp.exp(self.config['q_val_temperature'] * qs))
            weight_qs = jax.nn.softmax(self.config['q_val_temperature'] * qs)
            bc_flow_loss = jnp.mean(weight_qs * (pred - vel) ** 2) 
        elif(self.config['is_softmax_bc'] == 2):
            qs = self.network['critic'].select('critic')(batch['observations'], actions=batch['actions']).mean(axis=0).reshape(-1,1)
            repeated_observations = jnp.repeat(batch['observations'][:, jnp.newaxis, :], 10, axis=1).reshape(batch_size * 10, -1)
            rng, noise_rng, dropout_rng = jax.random.split(rng,3)
            noises = jax.random.normal(noise_rng, (batch_size * 10, action_dim))
            flow_actions = self.compute_flow_actions(repeated_observations, noises=noises, seed=dropout_rng)
            qs_tilde = self.network['critic'].select('critic')(repeated_observations, actions=flow_actions).mean(axis=0).reshape(-1,1)
            new_qs = 1 / jnp.exp(self.config['q_val_temperature'] * (qs_tilde - jnp.repeat(qs[:, jnp.newaxis, :], 10, axis=1).reshape(batch_size * 10, -1)))
            weight_qs = new_qs.reshape(batch_size, 10).mean(axis=1).reshape(-1,1)
            bc_flow_loss = jnp.mean(weight_qs * (pred - vel) ** 2) 
        
        return bc_flow_loss, {
            'bc_flow_loss': bc_flow_loss,
        }

    def actor_onestep_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises,seed=rng)
        rng, dropout_rng = jax.random.split(rng)
        actor_actions = self.network['actor_onestep'].select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network['critic'].select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)
        
        # Softmax loss.
        if(self.config['is_softmax_q'] == 1 or self.config['training_task'] == 3):
            softmax_q_loss = self.softmax_q_loss(batch, grad_params, rng)
        else:
            softmax_q_loss = 0

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss
            
        # Total loss.
        if(self.config['training_task'] == 3):
            actor_onestep_loss = self.config['alpha'] * distill_loss + self.config['q_loss_weight'] * q_loss + self.config['omar_coe'] * softmax_q_loss
        elif(self.config['training_task'] == 2):
            actor_onestep_loss = self.config['alpha'] * distill_loss + self.config['q_loss_weight'] * q_loss
        elif(self.config['training_task'] == 1):
            actor_onestep_loss = self.config['alpha'] * distill_loss
        elif(self.config['training_task'] == 0):
            actor_onestep_loss = self.config['q_loss_weight'] * q_loss
        else:
            actor_onestep_loss = self.config['alpha'] * distill_loss + self.config['q_loss_weight'] * q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_onestep_loss, {
            'actor_onestep_loss': actor_onestep_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
            'softmax_q_loss': softmax_q_loss,
        }


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

        pred = self.network['actor_bc'].select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises,seed=rng)
        rng, dropout_rng = jax.random.split(rng)
        actor_actions = self.network['actor_onestep'].select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network['critic'].select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)
        
        # Softmax loss.
        if(self.config['is_softmax_q'] == 1 or self.config['training_task'] == 3):
            softmax_q_loss = self.softmax_q_loss(batch, grad_params, rng)
        else:
            softmax_q_loss = 0

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss
            
        # Softmax Q for bc flow.
        if(self.config['is_softmax_bc'] == 1):
            qs = self.network['critic'].select('critic')(batch['observations'], actions=target_flow_actions).mean(axis=0).reshape(-1,1)
            #weight_qs = jnp.exp(self.config['q_val_temperature'] * qs) / jnp.sum(jnp.exp(self.config['q_val_temperature'] * qs))
            weight_qs = jax.nn.softmax(self.config['q_val_temperature'] * qs)
            bc_flow_loss = jnp.mean(weight_qs * (pred - vel) ** 2) * weight_qs.shape[0]
        elif(self.config['is_softmax_bc'] == 2):
            qs = self.network['critic'].select('critic')(batch['observations'], actions=batch['actions']).mean(axis=0).reshape(-1,1)
            repeated_observations = jnp.repeat(batch['observations'][:, jnp.newaxis, :], 10, axis=1).reshape(batch_size * 10, -1)
            rng, noise_rng, dropout_rng = jax.random.split(rng,3)
            noises = jax.random.normal(noise_rng, (batch_size * 10, action_dim))
            flow_actions = self.compute_flow_actions(repeated_observations, noises=noises, seed=dropout_rng)
            qs_tilde = self.network['critic'].select('critic')(repeated_observations, actions=flow_actions).mean(axis=0).reshape(-1,1)
            new_qs = 1 / jnp.exp(self.config['q_val_temperature'] * (qs_tilde - jnp.repeat(qs[:, jnp.newaxis, :], 10, axis=1).reshape(batch_size * 10, -1)))
            weight_qs = new_qs.reshape(batch_size, 10).mean(axis=1).reshape(-1,1)
            bc_flow_loss = jnp.mean(weight_qs * (pred - vel) ** 2) * weight_qs.shape[0]
            

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
            'softmax_q_loss': softmax_q_loss,
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

        if(self.config['model_arch'] == 'score_net'):
            actor_loss, actor_info = self.actor_loss_scorenet(batch, grad_params, actor_rng)
        else:
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network['critic'].params[f'modules_{module_name}'],
            self.network['critic'].params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn_actor(grad_params):
            if(self.config['model_arch'] == 'score_net'):
                return self.actor_loss_scorenet(batch, grad_params, rng=rng)
            else:
                return self.actor_loss(batch, grad_params, rng=rng)
            
        def loss_fn_actor_bc(grad_params):
            if(self.config['model_arch'] == 'score_net'):
                return self.actor_bc_loss_scorenet(batch, grad_params, rng=rng)
            else:
                return self.actor_bc_loss(batch, grad_params, rng=rng)
            
        def loss_fn_actor_onestep(grad_params):
            if(self.config['model_arch'] == 'score_net'):
                return self.actor_onestep_loss_scorenet(batch, grad_params, rng=rng)
            else:
                return self.actor_onestep_loss(batch, grad_params, rng=rng)
        
        def loss_fn_critic(grad_params):
            return self.critic_loss(batch, grad_params, rng=rng)

        new_network_critic, critic_info = self.network['critic'].apply_loss_fn(loss_fn=loss_fn_critic)
        new_network_actor_bc, actor_bc_info = self.network['actor_bc'].apply_loss_fn(loss_fn=loss_fn_actor_bc)
        new_network_actor_onestep, actor_onestep_info = self.network['actor_onestep'].apply_loss_fn(loss_fn=loss_fn_actor_onestep)
        self.target_update(new_network_critic, 'critic')
        
        new_network = {
            'critic': new_network_critic,
            'actor_bc': new_network_actor_bc,
            'actor_onestep': new_network_actor_onestep,
        }
        info = {}
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        for k, v in actor_bc_info.items():
            info[f'actor/{k}'] = v
        for k, v in actor_onestep_info.items():
            info[f'actor/{k}'] = v

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
        deterministic=False,
    ):
        """Sample actions from the one-step policy."""
        action_seed, dropout_key, seed = jax.random.split(seed,3)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        if(self.config['model_arch'] == 'score_net'):
            actions = self.network['actor_onestep'].select('actor_onestep_flow')(observations, noises, deterministic=deterministic, rngs={'dropout': dropout_key})
        else:
            actions = self.network['actor_onestep'].select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions
    
    @jax.jit
    def sample_actions_multiq(
        self,
        observations,
        seed=None,
        temperature=1.0,
        sample_q=50,
        deterministic=False,
    ):
        """Sample actions from the one-step policy."""
        action_seed, dropout_key, seed = jax.random.split(seed,3)
        observations_multiq_2d = jnp.tile(observations[:, None], (1, sample_q)) 
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
                sample_q,
            ),
        )
        if(self.config['model_arch'] == 'score_net'):
            actions = self.network['actor_onestep'].select('actor_onestep_flow')(observations_multiq_2d.T, noises.T, deterministic=deterministic, rngs={'dropout': dropout_key})
        else:
            actions = self.network['actor_onestep'].select('actor_onestep_flow')(observations_multiq_2d.T, noises.T)
        actions = jnp.clip(actions, -1, 1)
        qs = self.network['critic'].select('critic')(observations_multiq_2d.T, actions=actions)
        if self.config['q_agg'] == 'min':
            qs = qs.min(axis=0)
        else:
            qs = qs.mean(axis=0)
        max_index = jnp.argmax(qs)
        return actions[max_index]

    @jax.jit
    def sample_flow_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
        deterministic=False,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        
        action_seed, dropout_key, seed = jax.random.split(seed,3)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        
        if self.config['encoder'] is not None:
            observations = self.network['actor'].select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            if(self.config['model_arch'] == 'score_net'):
                vels = self.network['actor_bc'].select('actor_bc_flow')(observations, actions, t, is_encoded=True, deterministic=deterministic, rngs={'dropout': dropout_key})
            else:
                vels = self.network['actor_bc'].select('actor_bc_flow')(observations, actions, t, is_encoded=True, deterministic=deterministic)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions
    
    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
        seed=None,
        deterministic=False,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        action_seed, dropout_key, seed = jax.random.split(seed,3)
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            if(self.config['model_arch'] == 'score_net'):
                vels = self.network['actor_bc'].select('actor_bc_flow')(observations, actions, t, is_encoded=True, deterministic=deterministic, rngs={'dropout': dropout_key})
            else:
                vels = self.network['actor_bc'].select('actor_bc_flow')(observations, actions, t, is_encoded=True, deterministic=deterministic)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_actor_bc_rng, init_actor_onestep_rng, init_critic_rng = jax.random.split(rng, 4)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
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
            model_arch=config['model_arch'],
            observation_dim=ob_dims[0],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
            model_arch=config['model_arch'],
            observation_dim=ob_dims[0],
        )
        
        actor_bc_network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        actor_onestep_network_info = dict(
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        
        critic_network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
        )

        """ network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        ) """
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            actor_network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        
        actor_bc_networks = {k: v[0] for k, v in actor_bc_network_info.items()}
        actor_bc_network_args = {k: v[1] for k, v in actor_bc_network_info.items()}
        actor_onestep_networks = {k: v[0] for k, v in actor_onestep_network_info.items()}
        actor_onestep_network_args = {k: v[1] for k, v in actor_onestep_network_info.items()}
        critic_networks = {k: v[0] for k, v in critic_network_info.items()}
        critic_network_args = {k: v[1] for k, v in critic_network_info.items()}

        # Create separate optimizers for critic and actor
        critic_lr = config['critic_lr']  # Learning rate for critic
        actor_bc_lr = config['actor_bc_lr']   # Learning rate for actor
        actor_onestep_lr = config['actor_onestep_lr']   # Learning rate for actor

        # Define the optimizer for critic
        critic_tx = optax.adam(learning_rate=critic_lr)
        # Define the optimizer for actor
        actor_bc_tx = optax.adam(learning_rate=actor_bc_lr)
        actor_onestep_tx = optax.adam(learning_rate=actor_onestep_lr)

        # Initialize the network parameters
        actor_bc_network_def = ModuleDict(actor_bc_networks)
        actor_onestep_network_def = ModuleDict(actor_onestep_networks)
        critic_network_def = ModuleDict(critic_networks)
        
        actor_bc_network_params = actor_bc_network_def.init(init_actor_bc_rng, **actor_bc_network_args)['params']
        actor_onestep_network_params = actor_onestep_network_def.init(init_actor_onestep_rng, **actor_onestep_network_args)['params']
        critic_network_params = critic_network_def.init(init_critic_rng, **critic_network_args)['params']
        

        """ # Create separate TrainState for critic and actor
        critic_params = {k: v for k, v in network_params.items() if k.startswith('modules_critic')}
        actor_params = {k: v for k, v in network_params.items() if k.startswith('modules_actor')} """

        critic_state = TrainState.create(critic_network_def, critic_network_params, tx=critic_tx)
        actor_bc_state = TrainState.create(actor_bc_network_def, actor_bc_network_params, tx=actor_bc_tx)
        actor_onestep_state = TrainState.create(actor_onestep_network_def, actor_onestep_network_params, tx=actor_onestep_tx)

        # Combine the states into a single TrainState for convenience
        network_state = {
            'critic': critic_state,
            'actor_bc': actor_bc_state,
            'actor_onestep': actor_onestep_state,
        }

        params = network_state['critic'].params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network_state, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fql_sep_v2',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            critic_lr=3e-4,  # Learning rate.
            actor_bc_lr=3e-4,  # Learning rate for BC flow.
            actor_onestep_lr=3e-4,  # Learning rate for one-step flow.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            q_algo='ql',
            model_arch='mlp',  # Model architecture (mlp, cnn, etc.).
            expectile_term=0.5,
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            training_task=-1,
            is_softmax_q=0,
            is_softmax_bc=0,
            q_val_temperature=1.0,
            omar_coe=1.0,
            omar_iters=3,
            omar_mu=0.0,
            omar_sigma=1.0,
            omar_num_samples=10,
            omar_num_elites=10,
            num_sampled_actions=10,
            lse_temp=1.0,
            cql_alpha=1.0,
            critic_loss_weight=1.0,
            q_loss_weight=1.0,
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
