import collections
import re
import time

import gymnasium
import numpy as np
import ogbench
from gymnasium.spaces import Box
import copy

from utils.datasets import Dataset
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
try:
    from multiagent_mujoco.src.multiagent_mujoco.mujoco_multi import MujocoMulti
except:
    print ('MujocoMulti not installed')

class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(env_name, frame_stack=None, data_type=None, seed=0, action_clip_eps=1e-5, is_data_aug=False, replicate_style=0):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if 'singletask' in env_name:
        # OGBench.
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        # D4RL AntMaze.
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
        # D4RL Adroit.
        import d4rl.hand_manipulation_suite  # noqa
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'HalfCheetah-v2' in env_name:
        env_args = {"scenario": env_name, "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}
        env = MujocoMulti(env_args=env_args)
        env.seed(seed)
        eval_env = MujocoMulti(env_args=env_args)
        eval_env.seed(seed)
        env_info = env.get_env_info()
        
        if(is_data_aug == 3):
            replay_buffer = ReplayBuffer(
                int(5.2 * 1000000), env_info['n_agents'],
                [env_info['obs_shape'] for _ in env.observation_space],
                [acsp.shape[0] for acsp in env.action_space],
                is_mamujoco=True,
                state_dims=[env_info['state_shape'] for _ in env.observation_space],
            )
            replay_buffer_syn = ReplayBuffer(
                int(5.2 * 1000000), env_info['n_agents'],
                [env_info['obs_shape'] for _ in env.observation_space],
                [acsp.shape[0] for acsp in env.action_space],
                is_mamujoco=True,
                state_dims=[env_info['state_shape'] for _ in env.observation_space],
            )
        elif(is_data_aug == 2):
            replay_buffer = ReplayBuffer(
                int(5.2 * 1000000), env_info['n_agents'],
                [env_info['obs_shape'] for _ in env.observation_space],
                [acsp.shape[0] for acsp in env.action_space],
                is_mamujoco=True,
                state_dims=[env_info['state_shape'] for _ in env.observation_space],
            )
            replay_buffer_syn = None
        elif(is_data_aug == 1):
            replay_buffer = ReplayBuffer(
                int(5.2 * 1000000), env_info['n_agents'],
                [env_info['obs_shape'] for _ in env.observation_space],
                [acsp.shape[0] for acsp in env.action_space],
                is_mamujoco=True,
                state_dims=[env_info['state_shape'] for _ in env.observation_space],
            )
            replay_buffer_syn = None
        else:
            replay_buffer = ReplayBuffer(
                1000000, env_info['n_agents'],
                [env_info['obs_shape'] for _ in env.observation_space],
                [acsp.shape[0] for acsp in env.action_space],
                is_mamujoco=True,
                state_dims=[env_info['state_shape'] for _ in env.observation_space],
            )
            replay_buffer_syn = None
            
        online_replay_buffer = copy.deepcopy(replay_buffer)
        
        if(is_data_aug == 3):
            if(data_type == 'medium-expert'):
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/medium/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer_syn.load_batch_data_nine(dataset_dirs)
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/expert/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'expert' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer_syn.load_batch_data_one(dataset_dirs)
            elif(data_type == 'random-medium'):
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/random/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'random' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer_syn.load_batch_data_nine(dataset_dirs)
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/medium/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer_syn.load_batch_data_one(dataset_dirs)
            else:
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/{data_type}/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + data_type + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct(dataset_dirs, config.aug_noise)
                replay_buffer_syn.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct_quantile(dataset_dirs, config.aug_noise)
                
            if(data_type == 'medium-expert'):
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_nine(dataset_dirs)
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'expert' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_one(dataset_dirs)
            elif(data_type == 'random-medium'):
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'random' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_nine(dataset_dirs)
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_one(dataset_dirs)
            else:
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + data_type + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct(dataset_dirs, config.aug_noise)
                replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct_quantile(dataset_dirs, config.aug_noise)
        elif(is_data_aug == 2):
            if(data_type == 'medium-expert'):
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/medium/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_nine(dataset_dirs)
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/expert/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'expert' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_one(dataset_dirs)
            elif(data_type == 'random-medium'):
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/random/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'random' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_nine(dataset_dirs)
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/medium/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_one(dataset_dirs)
            else:
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/{data_type}/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + data_type + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct(dataset_dirs, config.aug_noise)
                replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct_quantile(dataset_dirs, config.aug_noise)
        elif(is_data_aug == 1):
            if(data_type == 'medium-expert'):
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                #replay_buffer.load_batch_data_nine(dataset_dirs)
                replay_buffer.load_batch_data_adaptive_nine(dataset_dirs, env_name)
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + 'expert' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                #replay_buffer.load_batch_data_one(dataset_dirs)
                replay_buffer.load_batch_data_adaptive_one(dataset_dirs, env_name)
            else:
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + data_type + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct(dataset_dirs, config.aug_noise)
                #replay_buffer.load_batch_data(dataset_dirs)
                replay_buffer.load_batch_data_adaptive(dataset_dirs, env_name, data_type)
        else:
            if(data_type == 'medium-expert'):
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_nine(dataset_dirs)
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + 'expert' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_one(dataset_dirs)
            elif(data_type == 'random-medium'):
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + 'random' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_nine(dataset_dirs)
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_one(dataset_dirs)
            else:
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + data_type + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct(dataset_dirs, config.aug_noise)
                replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct_quantile(dataset_dirs, config.aug_noise)
        
        train_dataset, val_dataset = None, None
        
        action_clip_eps = None
        
    elif 'simple_spread' in env_name or 'simple_tag' in env_name or 'simple_world' in env_name:
        # Multi-agent particle environments.
        env = make_env(env_name)
        eval_env = make_env(env_name)
        
        if(is_data_aug == 3):
            replay_buffer_syn = ReplayBuffer(
                int(1000000 * 4), env.n,
                [obsp.shape[0] for obsp in env.observation_space],
                [acsp.shape[0] for acsp in env.action_space],
            )
            replay_buffer = ReplayBuffer(
                int(1000000), env.n,
                [obsp.shape[0] for obsp in env.observation_space],
                [acsp.shape[0] for acsp in env.action_space],
            )
        elif(is_data_aug == 2):
            replay_buffer = ReplayBuffer(
                int(1000000 * 4), env.n,
                [obsp.shape[0] for obsp in env.observation_space],
                [acsp.shape[0] for acsp in env.action_space],
            )
            replay_buffer_syn = None
        elif(is_data_aug == 1):
            replay_buffer = ReplayBuffer(
                int(1000000 * 3.2), env.n,
                [obsp.shape[0] for obsp in env.observation_space],
                [acsp.shape[0] for acsp in env.action_space],
            )
            replay_buffer_syn = None
        else:
            replay_buffer = ReplayBuffer(
                int(1000000), env.n,
                [obsp.shape[0] for obsp in env.observation_space],
                [acsp.shape[0] for acsp in env.action_space],
            )
            replay_buffer_syn = None
        
        online_replay_buffer = copy.deepcopy(replay_buffer)
        
        if(is_data_aug == 3):
            if(data_type == 'medium-expert'):
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/medium/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer_syn.load_batch_data_nine(dataset_dirs)
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/expert/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'expert' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer_syn.load_batch_data_one(dataset_dirs)
            elif(data_type == 'random-medium'):
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/random/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'random' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer_syn.load_batch_data_nine(dataset_dirs)
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/medium/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer_syn.load_batch_data_one(dataset_dirs)
            else:
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/{data_type}/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + data_type + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct(dataset_dirs, config.aug_noise)
                replay_buffer_syn.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct_quantile(dataset_dirs, config.aug_noise)
                
            if(data_type == 'medium-expert'):
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_nine(dataset_dirs)
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'expert' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_one(dataset_dirs)
            elif(data_type == 'random-medium'):
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'random' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_nine(dataset_dirs)
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_one(dataset_dirs)
            else:
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + data_type + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct(dataset_dirs, config.aug_noise)
                replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct_quantile(dataset_dirs, config.aug_noise)
        elif(is_data_aug == 2):
            if(data_type == 'medium-expert'):
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/medium/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_nine(dataset_dirs)
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/expert/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'expert' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_one(dataset_dirs)
            elif(data_type == 'random-medium'):
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/random/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'random' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_nine(dataset_dirs)
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/medium/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_one(dataset_dirs)
            else:
                dataset_dirs = f"/home/lizhuoran/mydata/flow_matching/GTA-MA-main/data/generated_data_numpy/{env_name}/{data_type}/seed_{seed}_data"
                #dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + data_type + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct(dataset_dirs, config.aug_noise)
                replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct_quantile(dataset_dirs, config.aug_noise)
        elif(is_data_aug == 1):
            if(data_type == 'medium-expert'):
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                #replay_buffer.load_batch_data_nine(dataset_dirs)
                replay_buffer.load_batch_data_adaptive_nine(dataset_dirs, env_name, replicate_style)
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + 'expert' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                #replay_buffer.load_batch_data_one(dataset_dirs)
                replay_buffer.load_batch_data_adaptive_one(dataset_dirs, env_name, replicate_style)
            else:
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '/' + data_type + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct(dataset_dirs, config.aug_noise)
                #replay_buffer.load_batch_data(dataset_dirs)
                replay_buffer.load_batch_data_adaptive(dataset_dirs, env_name, data_type, replicate_style)
        else:
            if(data_type == 'medium-expert'):
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_nine(dataset_dirs)
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + 'expert' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_one(dataset_dirs)
            elif(data_type == 'random-medium'):
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + 'random' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_nine(dataset_dirs)
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + 'medium' + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data_half_quantile(dataset_dirs)
                replay_buffer.load_batch_data_one(dataset_dirs)
            else:
                real_seed = seed % 5
                dataset_dirs = '/home/lizhuoran/mydata/datasets/' + env_name + '-4agent/' + data_type + '/' + 'seed_{}_data'.format(seed)
                #replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct(dataset_dirs, config.aug_noise)
                replay_buffer.load_batch_data(dataset_dirs)
                #replay_buffer.load_batch_data_aug_2_correct_quantile(dataset_dirs, config.aug_noise)
        
        train_dataset, val_dataset = None, None
        
        action_clip_eps = None
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset, replay_buffer, replay_buffer_syn, online_replay_buffer

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action, barrier=[]):
    env = make_env(env_id, discrete_action=discrete_action, barrier=barrier)
    env._seed(seed + n_rollout_threads * 1000)
    np.random.seed(seed + n_rollout_threads * 1000)
    return env