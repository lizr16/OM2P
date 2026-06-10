import os
import platform

import json
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, evaluate_bc, flatten, supply_rng
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.agents import DDPGAgent

import torch

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 10, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)

flags.DEFINE_integer('episode_length', 25, 'Episode Length.')
flags.DEFINE_integer('n_training_threads', 100, 'Number of n_training_threads.')
flags.DEFINE_integer('steps_per_update', 100, 'steps_per_update.')
flags.DEFINE_string('data_type', None, 'Data Type of the dataset.')
flags.DEFINE_integer('action_clip', 1, 'action_clip.')
flags.DEFINE_integer('is_sample_multiq', 0, 'is_sample_multiq.')
flags.DEFINE_integer('is_data_aug', 0, 'is_data_aug, 1 is DOM2, 2 is Diffusion synthsization.')
flags.DEFINE_integer('replicate_style', 0, 'replicate_style, 0 is no replicate, 1 is replicate by action, 2 is replicate by observation.')
flags.DEFINE_float('syn_ratio', 0.5, 'syn_ratio, ratio of synthetic data in the replay buffer.')

#softmax qlearning
flags.DEFINE_float('online_training_temperature', 0.2, 'online_training_temperature.')
""" flags.DEFINE_integer('is_softmax_q', 0, 'is_softmax_q.')
flags.DEFINE_float('omar_coe', 1.0, 'OMAR coefficient.')
flags.DEFINE_integer('omar_iters', 3, 'Number of OMAR iterations.')
flags.DEFINE_float('omar_mu', 0.0, 'Initial value for OMAR mu.')
flags.DEFINE_float('omar_sigma', 1.0, 'Initial value for OMAR sigma.')
flags.DEFINE_integer('omar_num_samples', 10, 'Number of samples for OMAR.')
flags.DEFINE_integer('omar_num_elites', 10, 'Number of elite samples for OMAR.') """

#flags.DEFINE_integer('training_task', 2, '0 means only q-learning, 1 means only bc distill, 2 means fql, ql+bc.')




def main(_):
    # Set up logger.
    #exp_name = get_exp_name(FLAGS.seed)
    if(FLAGS.agent.is_softmax_q == 1):
        algo_name = 'OMAR'
    else:
        if(FLAGS.agent.q_algo == 'iql'):
            algo_name = "IQL"
        elif(FLAGS.agent.q_algo == 'cql'):
            algo_name = "CQL"
        elif(FLAGS.agent.q_algo == 'ql'):
            algo_name = "QL"
        else:
            algo_name = None
    if('episodic' in FLAGS.agent.q_algo):
        algo_name += '_episodic'        
    if(FLAGS.is_sample_multiq == 1):
        algo_name += '_multiq'
    if('resnet' in FLAGS.agent.model_arch):
        algo_name += '_resnet'
    elif('score_net' in FLAGS.agent.model_arch):
        algo_name += '_scorenet'
    if(FLAGS.agent.is_softmax_bc == 1):
        algo_name += '_softmax_bc_1'
    elif(FLAGS.agent.is_softmax_bc == 2):
        algo_name += '_softmax_bc_2'
    if(FLAGS.agent.agent_name == 'fql_sep'):
        algo_name += '_sep'
    elif(FLAGS.agent.agent_name == 'fql_sep_v2'):
        algo_name += '_sep_v2'
    elif(FLAGS.agent.agent_name == 'fql_meanfield'):
        algo_name += '_meanfield'
        if(FLAGS.agent.is_autograd == 0):
            algo_name += '_nograd_delta'
        elif(FLAGS.agent.is_autograd == 1):
            algo_name += '_grad_delta'
        elif(FLAGS.agent.is_autograd == 2):
            algo_name += '_zero'
        if(FLAGS.agent.time_selection == 'exp'):
            algo_name += '_exp_time'
        elif(FLAGS.agent.time_selection == 'quadratic'):
            algo_name += '_quad_time'
        elif(FLAGS.agent.time_selection == 'adaptive_beta'):
            algo_name += '_beta_time'
    if(FLAGS.agent.normalize_q_loss == True):
        algo_name += '_norm_q_loss'
    if(FLAGS.is_data_aug == 1):
        algo_name += '_data_aug'   
    elif(FLAGS.is_data_aug == 2):
        algo_name += '_diffusion_aug'
    elif(FLAGS.is_data_aug == 3):
        algo_name += '_syn_aug'
    if(FLAGS.agent.is_flexiable_alpha == 1):
        algo_name += '_flexible_alpha' 
    if(FLAGS.agent.training_task == 4):
        algo_name += '_q_only'
    elif(FLAGS.agent.training_task == 5):
        algo_name += '_bc_only'
        
    
    
    if(FLAGS.online_steps > 0 and FLAGS.offline_steps > 0):
        phase = 'offline+online'
    elif(FLAGS.online_steps > 0 and FLAGS.offline_steps == 0):
        phase = 'online'
    elif(FLAGS.online_steps == 0 and FLAGS.offline_steps > 0):
        phase = 'offline'
    else:
        raise ValueError('Invalid online/offline steps configuration.')
    
    """ exp_name = f"{FLAGS.env_name}_0625_{phase}_a6000_{algo_name}"
    #wandb.require("legacy-service")
    setup_wandb(project='fql_0619', group=FLAGS.run_group, name=exp_name, mode="online") """
    exp_name = f"{FLAGS.env_name}_1114_4agent_{phase}_a6000_{algo_name}"
    #wandb.require("legacy-service")
    setup_wandb(project='fql_1114', group=FLAGS.run_group, name=exp_name, mode="online")

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    #FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    env, eval_env, train_dataset, val_dataset, replay_buffer, replay_buffer_syn, online_replay_buffer = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack, data_type=FLAGS.data_type, seed=FLAGS.seed, is_data_aug=FLAGS.is_data_aug, replicate_style=FLAGS.replicate_style)
        

    """ if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.' """
    if FLAGS.online_steps > 0:
        assert 'visual' not in FLAGS.env_name, 'Online fine-tuning is currently not supported for visual environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    """ # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    if FLAGS.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        # Use the training dataset as the replay buffer.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] == 'rebrac':
                dataset.return_next_actions = True """

    # Create agent.
    #example_batch = train_dataset.sample(1)
    data = replay_buffer.sample(1)
    
    if(FLAGS.env_name == 'simple_tag' or FLAGS.env_name == 'simple_world'):
        num_agent = env.n - 1
    else:
        num_agent = env.n
    
    agent_class = [agents[config['agent_name']] for _ in range(num_agent)]
    
    agent = [agent_class[agent_id].create(
        FLAGS.seed,
        data['observations'][0],
        data['actions'][0],
        config,
    ) for agent_id in range(num_agent)]
    
    if(FLAGS.env_name == 'simple_tag' or FLAGS.env_name == 'simple_world'):
        pretrained_model_dir = '/home/lizhuoran/mydata/datasets/{}/pretrained_adv_model.pt'.format(FLAGS.env_name)
        prey_obs_dim = data['observations'][-1].shape[1]
        prey_act_dim = data['actions'][-1].shape[1]
        prey_obs_act_dim = prey_obs_dim + prey_act_dim
        prey = DDPGAgent(num_in_pol=prey_obs_dim, num_out_pol=prey_act_dim, num_in_critic=prey_obs_act_dim)
        save_dict = torch.load(pretrained_model_dir)
        prey_params = save_dict['agent_params'][-1]
        prey.load_params_without_optims(prey_params)
        prey.policy.eval()
        prey.target_policy.eval()
        #ma_agent.load_pretrained_preys(pretrained_model_dir)
    else:
        prey = None

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            # Offline RL.
            if(i % FLAGS.steps_per_update < FLAGS.n_training_threads):
                
                if(FLAGS.is_data_aug == 3):
                    if FLAGS.syn_ratio == 0:
                        batch = replay_buffer.sample(config['batch_size'])
                    elif FLAGS.syn_ratio == 1:
                        batch = replay_buffer_syn.sample(config['batch_size'])
                    else:
                        # Balanced sampling from the training dataset and the replay buffer.
                        # Here we assume that the replay buffer is already initialized with synthetic data.
                        # If not, we can sample from the training dataset and the replay buffer separately.
                        # For example, if FLAGS.syn_ratio = 0.5, we can sample half from the training dataset and half from the replay buffer.
                        init_batch_size = int((1 - FLAGS.syn_ratio) * config['batch_size'])
                        syn_batch_size = int(FLAGS.syn_ratio * config['batch_size'])
                        batch_init = replay_buffer.sample(init_batch_size)
                        batch_syn = replay_buffer_syn.sample(syn_batch_size)
                        batch = {}
                        for k in batch_init:
                            batch[k] = [np.concatenate([batch_init[k][index], batch_syn[k][index]], axis=0) for index in range(num_agent)]
                        #batch = {k: np.concatenate([batch_init[k], batch_syn[k]], axis=0) for k in batch_init}
                else:
                    if(config['q_algo'] == 'iql'):
                        batch = replay_buffer.sample_iql(config['batch_size'])
                    elif(config['q_algo'] == 'ql_episodic'):
                        batch = replay_buffer.sample_ql_episodic(config['batch_size'])
                    else:
                        batch = replay_buffer.sample(config['batch_size'])

                if config['agent_name'] == 'rebrac':
                    agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
                else:
                    for agent_id in range(num_agent):
                        agent_batch = {k: v[agent_id] for k, v in batch.items()}

                        agent[agent_id], update_info = agent[agent_id].update(agent_batch)
                    #agent, update_info = agent.update(batch)
        else:
            # Online fine-tuning.
            online_rng, key = jax.random.split(online_rng)
            
            actor_fn = [supply_rng(agent[agent_id].sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))) for agent_id in range(len(agent))]
            

            if step == 0:
                env.reset()
            
            observation = [env._get_obs(env.agents[agent_id]) for agent_id in range(len(env.agents))]
            actions = []
            for agent_id in range(len(agent)):
                action = actor_fn[agent_id](observations=observation[agent_id], temperature=FLAGS.online_training_temperature)
                if(FLAGS.action_clip==1):
                    action = np.clip(action, -1, 1)
                actions.append(action)
            
            if(FLAGS.env_name == "simple_tag" or FLAGS.env_name == "simple_world"):
                prey_action = prey.step(torch.tensor(observation[-1]).reshape(1,-1).float()).detach().numpy()[0]
                if(FLAGS.action_clip==1):
                    prey_action = np.clip(prey_action, -1, 1)
                actions.append(prey_action)

            next_observation, reward, terminated, info = env._step(actions)
            
            batch = [{
                'obs': observation[agent_id],
                'acs': actions[agent_id],
                'rew': reward[agent_id],
                'dones': False,
                'next_obs': next_observation[agent_id],
                } for agent_id in range(env.n)]
            
            online_replay_buffer.insert(batch)

            observation = next_observation

            step += 1
            
            episode_length = 25
            if(step == episode_length):
                step = 0
            
            # Update agent.
            if(online_replay_buffer.filled_i >= config['batch_size']):
                if FLAGS.balanced_sampling:
                    # Half-and-half sampling from the training dataset and the replay buffer.
                    dataset_batch = replay_buffer.sample(config['batch_size'] // 2)
                    replay_batch = online_replay_buffer.sample(config['batch_size'] // 2)
                    batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
                else:
                    batch = online_replay_buffer.sample(config['batch_size'])

                if config['agent_name'] == 'rebrac':
                    agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
                else:
                    for agent_id in range(num_agent):
                        agent_batch = {k: v[agent_id] for k, v in batch.items()}

                        agent[agent_id], update_info = agent[agent_id].update(agent_batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                prey=prey,
                env=eval_env,
                env_name=FLAGS.env_name,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                is_action_clip=FLAGS.action_clip,
                is_sample_multiq=FLAGS.is_sample_multiq,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v
                
            if('fql' == FLAGS.agent.agent_name):
                eval_info_bc, _, _ = evaluate_bc(
                    agent=agent,
                    prey=prey,
                    env=eval_env,
                    env_name=FLAGS.env_name,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    is_action_clip=FLAGS.action_clip,
                )
                for k, v in eval_info_bc.items():
                    eval_metrics[f'evaluation/{k}_bc'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
