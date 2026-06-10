from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
import torch

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    prey,
    env,
    env_name,
    config=None,
    num_eval_episodes=10,
    episode_length=25,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    is_action_clip=0,
    is_sample_multiq=0,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    
    if(is_sample_multiq==1):
        actor_fn = [supply_rng(agent[agent_id].sample_actions_multiq, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))) for agent_id in range(len(agent))]
    else:
        actor_fn = [supply_rng(agent[agent_id].sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))) for agent_id in range(len(agent))]
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        env.reset()
        
        if(env_name == 'HalfCheetah-v2'):
            observation = env.get_obs()
        else:
            observation = [env._get_obs(env.agents[agent_id]) for agent_id in range(len(env.agents))]
        
        done = False
        truncated = False
        rewards = 0.
        step = 0
        render = []
        if(env_name == 'HalfCheetah-v2'):
            episode_length = 1000
        for et_i in range(episode_length):
            actions = []
            for agent_id in range(len(agent)):
                action = actor_fn[agent_id](observations=observation[agent_id], temperature=eval_temperature)
                if(is_action_clip==1):
                    action = np.clip(action, -1, 1)
                actions.append(action)
            
            if(env_name == "simple_tag" or env_name == "simple_world"):
                prey_action = prey.step(torch.tensor(observation[-1]).reshape(1,-1).float()).detach().numpy()[0]
                if(is_action_clip==1):
                    prey_action = np.clip(prey_action, -1, 1)
                actions.append(prey_action)

            """ action = actor_fn[](observations=observation, temperature=eval_temperature) """
            #action = np.array(action)
            #action = np.clip(action, -1, 1)

            #next_observation, reward, terminated, truncated, info = env._step(actions)
            if(env_name == 'HalfCheetah-v2'):
                reward, terminated, info = env.step(actions)
                next_observation = env.get_obs()
            else:
                next_observation, reward, terminated, info = env._step(actions)
            if(env_name == "simple_tag" or env_name == "simple_world"):
                step_reward = reward[0]
            else:
                step_reward = np.mean(reward)
            rewards += step_reward
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env._render(mode='rgb_array',close=False).copy()[0]
                render.append(frame)

            transition = dict(
                observation=observation[0],
                next_observation=next_observation[0],
                action=action[0],
                reward=step_reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, dict(reward=rewards))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def evaluate_bc(
    agent,
    prey,
    env,
    env_name,
    config=None,
    num_eval_episodes=10,
    episode_length=25,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    is_action_clip=0,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    
    
    actor_fn = [supply_rng(agent[agent_id].sample_flow_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))) for agent_id in range(len(agent))]
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        env.reset()
        
        if(env_name == 'HalfCheetah-v2'):
            observation = env.get_obs()
        else:
            observation = [env._get_obs(env.agents[agent_id]) for agent_id in range(len(env.agents))]
        
        done = False
        truncated = False
        rewards = 0.
        step = 0
        render = []
        if(env_name == 'HalfCheetah-v2'):
            episode_length = 1000
        for et_i in range(episode_length):
            actions = []
            for agent_id in range(len(agent)):
                action = actor_fn[agent_id](observations=observation[agent_id], temperature=eval_temperature)
                if(is_action_clip==1):
                    action = np.clip(action, -1, 1)
                actions.append(action)
            
            if(env_name == "simple_tag" or env_name == "simple_world"):
                prey_action = prey.step(torch.tensor(observation[-1]).reshape(1,-1).float()).detach().numpy()[0]
                if(is_action_clip==1):
                    prey_action = np.clip(prey_action, -1, 1)
                actions.append(prey_action)

            """ action = actor_fn[](observations=observation, temperature=eval_temperature) """
            #action = np.array(action)
            #action = np.clip(action, -1, 1)

            #next_observation, reward, terminated, truncated, info = env._step(actions)
            if(env_name == 'HalfCheetah-v2'):
                reward, terminated, info = env.step(actions)
                next_observation = env.get_obs()
            else:
                next_observation, reward, terminated, info = env._step(actions)
            if(env_name == "simple_tag" or env_name == "simple_world"):
                step_reward = reward[0]
            else:
                step_reward = np.mean(reward)
            rewards += step_reward
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env._render(mode='rgb_array',close=False).copy()[0]
                render.append(frame)

            transition = dict(
                observation=observation[0],
                next_observation=next_observation[0],
                action=action[0],
                reward=step_reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, dict(reward=rewards))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
