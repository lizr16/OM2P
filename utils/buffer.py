import numpy as np
import copy
import math

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims, is_mamujoco=False, state_dims=None):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))

        self.is_mamujoco = is_mamujoco
        if self.is_mamujoco:
            self.state_buffs = []
            self.next_state_buffs = []
            for sdim in state_dims:
                self.state_buffs.append(np.zeros((max_steps, sdim)))
                self.next_state_buffs.append(np.zeros((max_steps, sdim)))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i
    
    def insert(self, batch):

        for i in range(self.num_agents):
            curr_obs = batch[i]['obs']
            curr_acs = batch[i]['acs']
            curr_rews = batch[i]['rew']
            curr_next_obs = batch[i]['next_obs']
            curr_dones = batch[i]['dones']
            
            self.obs_buffs[i][self.curr_i] = curr_obs
            self.ac_buffs[i][self.curr_i] = curr_acs
            self.rew_buffs[i][self.curr_i] = curr_rews
            self.next_obs_buffs[i][self.curr_i] = curr_next_obs
            self.done_buffs[i][self.curr_i] = curr_dones
            
        if(self.filled_i >= self.max_steps):
            self.filled_i = self.max_steps
        else:
            self.filled_i += 1
        self.curr_i += 1 
        if self.curr_i >= self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=False)
        #inds = np.array([x - 1 if x % 25 == 24 else x for x in inds])
        """ if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)] """
        ret_rews = [self.rew_buffs[i][inds] for i in range(self.num_agents)]
        if self.is_mamujoco:
            return {
                "observations":    [self.obs_buffs[i][inds] for i in range(self.num_agents)],
                "actions":  [self.ac_buffs[i][inds] for i in range(self.num_agents)],
                "rewards":  ret_rews,
                "next_observations":  [self.next_obs_buffs[i][inds] for i in range(self.num_agents)],
                "dones":  [self.done_buffs[i][inds] for i in range(self.num_agents)]
            }
            """ return (
                [cast(self.state_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_state_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)]
            ) """
        else:
            """ return (
                [cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)]
            ) """
            return {
                "observations":    [self.obs_buffs[i][inds] for i in range(self.num_agents)],
                "actions":  [self.ac_buffs[i][inds] for i in range(self.num_agents)],
                "rewards":  ret_rews,
                "next_observations":  [self.next_obs_buffs[i][inds] for i in range(self.num_agents)],
                "dones":  [self.done_buffs[i][inds] for i in range(self.num_agents)]
            }
            
    def sample_ql_episodic(self, N, to_gpu=False):
        inds = np.random.choice(np.arange(int(self.filled_i/25)), size=N, replace=False)
        #inds = np.array([x - 1 if x % 25 == 24 else x for x in inds])
        """ if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)] """
        ret_rews = [self.rew_buffs[i].reshape(-1,25)[inds].reshape(-1) for i in range(self.num_agents)]
        obs = [self.obs_buffs[i].reshape(-1,25,self.obs_buffs[0].shape[-1])[inds].reshape(-1,self.obs_buffs[0].shape[-1]) for i in range(self.num_agents)]
        acs = [self.ac_buffs[i].reshape(-1,25,self.ac_buffs[0].shape[-1])[inds].reshape(-1,self.ac_buffs[0].shape[-1]) for i in range(self.num_agents)]
        next_obs = [self.next_obs_buffs[i].reshape(-1,25,self.next_obs_buffs[0].shape[-1])[inds].reshape(-1,self.next_obs_buffs[0].shape[-1]) for i in range(self.num_agents)]
        dones = [self.done_buffs[i].reshape(-1,25)[inds].reshape(-1) for i in range(self.num_agents)]
        if self.is_mamujoco:
            return (
                [cast(self.state_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_state_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)]
            )
        else:
            """ return (
                [cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)]
            ) """
            return {
                "observations":   obs,
                "actions":  acs,
                "rewards":  ret_rews,
                "next_observations":  next_obs,
                "dones":  dones
            }


    def sample_iql(self, N, to_gpu=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=False)
        inds = np.array([x - 1 if x % 25 == 24 else x for x in inds])
        """ if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)] """
        ret_rews = [self.rew_buffs[i][inds] for i in range(self.num_agents)]
        if self.is_mamujoco:
            return (
                [cast(self.state_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_state_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)]
            )
        else:
            """ return (
                [cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)]
            ) """
            return {
                "observations":    [self.obs_buffs[i][inds] for i in range(self.num_agents)],
                "actions":  [self.ac_buffs[i][inds] for i in range(self.num_agents)],
                "rewards":  ret_rews,
                "next_observations":  [self.next_obs_buffs[i][inds] for i in range(self.num_agents)],
                "next_actions":  [self.ac_buffs[i][inds+1] for i in range(self.num_agents)],
                "dones":  [self.done_buffs[i][inds] for i in range(self.num_agents)]
            }
            
            

    def load_batch_data(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            self.obs_buffs[i][:num_experiences] = curr_obs
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews.reshape(-1)
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs
            self.done_buffs[i][:num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i = num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences
        
    def load_batch_data_adaptive_nine(self, dir, env_id, replicate_style):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(0))
        data_num = int(curr_obs.shape[0]/10*9)
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))[0:data_num]
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))[0:data_num]
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))[0:data_num]
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))[0:data_num]
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))[0:data_num]
        
            new_curr_obs = curr_obs.reshape(int(data_num/25),25,curr_obs.shape[1])
            new_curr_acs = curr_acs.reshape(int(data_num/25),25,curr_acs.shape[1])
            new_curr_next_obs = curr_next_obs.reshape(int(data_num/25),25,curr_next_obs.shape[1])
            new_curr_rews = curr_rews.reshape(int(data_num/25),25)
            new_curr_dones = curr_dones.reshape(int(data_num/25),25)

            aug_curr_obs = copy.deepcopy(new_curr_obs)
            aug_curr_acs = copy.deepcopy(new_curr_acs)
            aug_curr_rews = copy.deepcopy(new_curr_rews)
            aug_curr_next_obs = copy.deepcopy(new_curr_next_obs)
            aug_curr_dones = copy.deepcopy(new_curr_dones)

            new_reward = new_curr_rews.sum(axis=1)
            reward_sorted = sorted(new_reward)

            """ if(env_id == 'simple_spread'):
                barrier_values = [200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0]
            elif(env_id == 'simple_tag'):
                barrier_values = [100.0,150.0,200.0,250.0,300.0]
            elif(env_id == 'simple_world'):
                barrier_values = [50.0,75.0,100.0,125.0,150.0]
            else:
                barrier_values = [0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0] """
                
            if(env_id == 'simple_spread'):
                barrier_values = [reward_sorted[int(new_reward.shape[0]/2)],reward_sorted[int(new_reward.shape[0]/4*3)],reward_sorted[int(new_reward.shape[0]/8*7)],reward_sorted[int(new_reward.shape[0]/8*7)]]
            elif(env_id == 'simple_tag'):
                if(replicate_style == 0):
                    barrier_values = [100.0,150.0,200.0,250.0,300.0]
                elif(replicate_style == 1):
                    barrier_values = [175.0] * 1 + [200.0] * 2 + [225.0] * 4 + [250.0] * 16 + [300.0] * 64
            elif(env_id == 'simple_world'):
                barrier_values = [50.0,75.0,100.0,125.0,150.0]
            elif(env_id == 'HalfCheetah-v2'):
                barrier_values = [1800.0,1850.0,1900.0,1950.0,2000.0]
            else:
                barrier_values = [0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0] 
            
            """ reward_sorted = sorted(new_reward)
            if(len(reward_sorted) > 1):
                barrier_values = [reward_sorted[int(new_reward.shape[0]/2)],reward_sorted[int(new_reward.shape[0]/4*3)],reward_sorted[int(new_reward.shape[0]/8*7)],reward_sorted[int(new_reward.shape[0]/8*7)]]
            print(barrier_values) """
            
            for barrier_value in barrier_values:
                aug_curr_obs = np.concatenate([aug_curr_obs,new_curr_obs[new_reward > barrier_value,:,:]],axis=0)
                aug_curr_acs = np.concatenate([aug_curr_acs,new_curr_acs[new_reward > barrier_value,:,:]],axis=0)
                aug_curr_rews = np.concatenate([aug_curr_rews,new_curr_rews[new_reward > barrier_value,:]])
                aug_curr_next_obs = np.concatenate([aug_curr_next_obs,new_curr_next_obs[new_reward > barrier_value,:,:]],axis=0)
                aug_curr_dones = np.concatenate([aug_curr_dones,new_curr_dones[new_reward > barrier_value,:]])

            num_experiences = int(aug_curr_obs.shape[0] * 25)

            self.obs_buffs[i][:num_experiences] = aug_curr_obs.reshape(num_experiences,aug_curr_obs.shape[-1])
            self.ac_buffs[i][:num_experiences] = aug_curr_acs.reshape(num_experiences,aug_curr_acs.shape[-1])
            self.rew_buffs[i][:num_experiences] = aug_curr_rews.reshape(-1)
            self.next_obs_buffs[i][:num_experiences] = aug_curr_next_obs.reshape(num_experiences,aug_curr_next_obs.shape[-1])
            self.done_buffs[i][:num_experiences] = aug_curr_dones.reshape(-1)

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i += num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences


    def load_batch_data_adaptive_one(self, dir, env_id, replicate_style):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(0))
        data_num = int(curr_obs.shape[0]/10)
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))[0:data_num]
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))[0:data_num]
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))[0:data_num]
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))[0:data_num]
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))[0:data_num]
        
            new_curr_obs = curr_obs.reshape(int(data_num/25),25,curr_obs.shape[1])
            new_curr_acs = curr_acs.reshape(int(data_num/25),25,curr_acs.shape[1])
            new_curr_next_obs = curr_next_obs.reshape(int(data_num/25),25,curr_next_obs.shape[1])
            new_curr_rews = curr_rews.reshape(int(data_num/25),25)
            new_curr_dones = curr_dones.reshape(int(data_num/25),25)

            aug_curr_obs = copy.deepcopy(new_curr_obs)
            aug_curr_acs = copy.deepcopy(new_curr_acs)
            aug_curr_rews = copy.deepcopy(new_curr_rews)
            aug_curr_next_obs = copy.deepcopy(new_curr_next_obs)
            aug_curr_dones = copy.deepcopy(new_curr_dones)

            new_reward = new_curr_rews.sum(axis=1)
            reward_sorted = sorted(new_reward)

            """ if(env_id == 'simple_spread'):
                barrier_values = [525.0,550.0,575.0,600.0,625.0]
            elif(env_id == 'simple_tag'):
                barrier_values = [200.0,250.0,300.0,350.0,400.0]
            elif(env_id == 'simple_world'):
                barrier_values = [75.0,100.0,125.0,150.0,175.0]
            else:
                barrier_values = [0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0] """
            """ reward_sorted = sorted(new_reward)
            if(len(reward_sorted) > 1):
                barrier_values = [reward_sorted[int(new_reward.shape[0]/2)],reward_sorted[int(new_reward.shape[0]/4*3)],reward_sorted[int(new_reward.shape[0]/8*7)],reward_sorted[int(new_reward.shape[0]/8*7)]]
            print(barrier_values) """
            if(env_id == 'simple_spread'):
                barrier_values = [reward_sorted[int(new_reward.shape[0]/2)],reward_sorted[int(new_reward.shape[0]/4*3)],reward_sorted[int(new_reward.shape[0]/8*7)],reward_sorted[int(new_reward.shape[0]/8*7)]]
            elif(env_id == 'simple_tag'):
                if(replicate_style == 0):
                    barrier_values = [200.0,250.0,300.0,350.0,400.0]
                elif(replicate_style == 1):
                    barrier_values = [250.0] * 1 + [275.0] * 2 + [300.0] * 4 + [325.0] * 16 + [350.0] * 64
            elif(env_id == 'simple_world'):
                barrier_values = [75.0,100.0,125.0,150.0,175.0]
            elif(env_id == 'HalfCheetah-v2'):
                barrier_values = [3800.0,3850.0,3900.0,3950.0,4000.0]
            else:
                barrier_values = [0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0]
            
            for barrier_value in barrier_values:
                aug_curr_obs = np.concatenate([aug_curr_obs,new_curr_obs[new_reward > barrier_value,:,:]],axis=0)
                aug_curr_acs = np.concatenate([aug_curr_acs,new_curr_acs[new_reward > barrier_value,:,:]],axis=0)
                aug_curr_rews = np.concatenate([aug_curr_rews,new_curr_rews[new_reward > barrier_value,:]])
                aug_curr_next_obs = np.concatenate([aug_curr_next_obs,new_curr_next_obs[new_reward > barrier_value,:,:]],axis=0)
                aug_curr_dones = np.concatenate([aug_curr_dones,new_curr_dones[new_reward > barrier_value,:]])

            num_experiences = int(aug_curr_obs.shape[0] * 25)

            self.obs_buffs[i][:num_experiences] = aug_curr_obs.reshape(num_experiences,aug_curr_obs.shape[-1])
            self.ac_buffs[i][:num_experiences] = aug_curr_acs.reshape(num_experiences,aug_curr_acs.shape[-1])
            self.rew_buffs[i][:num_experiences] = aug_curr_rews.reshape(-1)
            self.next_obs_buffs[i][:num_experiences] = aug_curr_next_obs.reshape(num_experiences,aug_curr_next_obs.shape[-1])
            self.done_buffs[i][:num_experiences] = aug_curr_dones.reshape(-1)

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i += num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences


    def load_batch_data_adaptive(self, dir, env_id, data_type, replicate_style):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(0))
        data_num = curr_obs.shape[0]
        final_data_num = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            new_curr_obs = curr_obs.reshape(int(data_num/25),25,curr_obs.shape[1])
            new_curr_acs = curr_acs.reshape(int(data_num/25),25,curr_acs.shape[1])
            new_curr_next_obs = curr_next_obs.reshape(int(data_num/25),25,curr_next_obs.shape[1])
            new_curr_rews = curr_rews.reshape(int(data_num/25),25)
            new_curr_dones = curr_dones.reshape(int(data_num/25),25)

            aug_curr_obs = copy.deepcopy(new_curr_obs)
            aug_curr_acs = copy.deepcopy(new_curr_acs)
            aug_curr_rews = copy.deepcopy(new_curr_rews)
            aug_curr_next_obs = copy.deepcopy(new_curr_next_obs)
            aug_curr_dones = copy.deepcopy(new_curr_dones)

            new_reward = new_curr_rews.sum(axis=1)
            reward_sorted = sorted(new_reward)

            if(env_id == 'simple_spread'):
                if(data_type == 'expert'):
                    barrier_values = [525.0,550.0,575.0,600.0,625.0]
                elif(data_type == 'medium'):
                    barrier_values = [200.0,250.0,300.0,350.0,400.0,450.0,500.0,550.0]
                elif(data_type == 'medium-replay'):
                    barrier_values = [reward_sorted[int(new_reward.shape[0]/2)],reward_sorted[int(new_reward.shape[0]/4*3)],reward_sorted[int(new_reward.shape[0]/8*7)],reward_sorted[int(new_reward.shape[0]/8*7)]]
                elif(data_type == 'random'):
                    barrier_values = []
            elif(env_id == 'simple_tag'):
                if(data_type == 'expert'):
                    if(replicate_style == 0):
                        barrier_values = [200.0,250.0,300.0,350.0,400.0]
                    elif(replicate_style == 1):
                        barrier_values = [250.0] * 1 + [275.0] * 2 + [300.0] * 4 + [325.0] * 16 + [350.0] * 64
                elif(data_type == 'medium'):
                    if(replicate_style == 0):
                        barrier_values = [100.0,150.0,200.0,250.0,300.0]
                    elif(replicate_style == 1):
                        barrier_values = [175.0] * 1 + [200.0] * 2 + [225.0] * 4 + [250.0] * 16 + [300.0] * 64
                elif(data_type == 'medium-replay'):
                    barrier_values = [reward_sorted[int(new_reward.shape[0]/2)],reward_sorted[int(new_reward.shape[0]/4*3)],reward_sorted[int(new_reward.shape[0]/8*7)],reward_sorted[int(new_reward.shape[0]/8*7)]]
                elif(data_type == 'random'):
                    barrier_values = []
            elif(env_id == 'simple_world'):
                if(data_type == 'expert'):
                    barrier_values = [75.0,100.0,125.0,150.0,175.0]
                elif(data_type == 'medium'):
                    barrier_values = [reward_sorted[int(new_reward.shape[0]/2)],reward_sorted[int(new_reward.shape[0]/4*3)],reward_sorted[int(new_reward.shape[0]/8*7)],reward_sorted[int(new_reward.shape[0]/8*7)]]
                elif(data_type == 'medium-replay'):
                    barrier_values = [reward_sorted[int(new_reward.shape[0]/2)],reward_sorted[int(new_reward.shape[0]/4*3)],reward_sorted[int(new_reward.shape[0]/8*7)],reward_sorted[int(new_reward.shape[0]/8*7)]]
                elif(data_type == 'random'):
                    barrier_values = []
                #barrier_values = [50.0,75.0,100.0,125.0,150.0]
            elif(env_id == 'HalfCheetah-v2'):
                if(data_type == 'expert'):
                    barrier_values = [3800.0,3850.0,3900.0,3950.0,4000.0]
                elif(data_type == 'medium'):
                    barrier_values = [1800.0,1850.0,1900.0,1950.0,2000.0]
                elif(data_type == 'medium-replay'):
                    barrier_values = [100.0,300.0,500.0,1000.0,1500.0]
                elif(data_type == 'random'):
                    barrier_values = []
            else:
                barrier_values = [0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0]

            
            for barrier_value in barrier_values:
                aug_curr_obs = np.concatenate([aug_curr_obs,new_curr_obs[new_reward > barrier_value,:,:]],axis=0)
                aug_curr_acs = np.concatenate([aug_curr_acs,new_curr_acs[new_reward > barrier_value,:,:]],axis=0)
                aug_curr_rews = np.concatenate([aug_curr_rews,new_curr_rews[new_reward > barrier_value,:]])
                aug_curr_next_obs = np.concatenate([aug_curr_next_obs,new_curr_next_obs[new_reward > barrier_value,:,:]],axis=0)
                aug_curr_dones = np.concatenate([aug_curr_dones,new_curr_dones[new_reward > barrier_value,:]])

            num_experiences = int(aug_curr_obs.shape[0] * 25)
            final_data_num.append(num_experiences)

            self.obs_buffs[i][:num_experiences] = aug_curr_obs.reshape(num_experiences,aug_curr_obs.shape[-1])
            self.ac_buffs[i][:num_experiences] = aug_curr_acs.reshape(num_experiences,aug_curr_acs.shape[-1])
            self.rew_buffs[i][:num_experiences] = aug_curr_rews.reshape(-1)
            self.next_obs_buffs[i][:num_experiences] = aug_curr_next_obs.reshape(num_experiences,aug_curr_next_obs.shape[-1])
            self.done_buffs[i][:num_experiences] = aug_curr_dones.reshape(-1)

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i = final_data_num[0]
        self.curr_i = 0 if self.curr_i == self.max_steps else final_data_num[0]


    def load_batch_data_quantile(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = int(curr_obs.shape[0]/4)

            self.obs_buffs[i][:num_experiences] = curr_obs[:num_experiences]
            self.ac_buffs[i][:num_experiences] = curr_acs[:num_experiences]
            self.rew_buffs[i][:num_experiences] = curr_rews[:num_experiences]
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs[:num_experiences]
            self.done_buffs[i][:num_experiences] = curr_dones[:num_experiences]

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i = num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences

    def load_batch_data_aug_acs(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            self.obs_buffs[i][:num_experiences] = curr_obs
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs
            self.done_buffs[i][:num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            #curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            #curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            self.obs_buffs[i][num_experiences:2*num_experiences] = curr_obs
            self.ac_buffs[i][num_experiences:2*num_experiences] = curr_acs
            self.rew_buffs[i][num_experiences:2*num_experiences] = curr_rews
            self.next_obs_buffs[i][num_experiences:2*num_experiences] = curr_next_obs
            self.done_buffs[i][num_experiences:2*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i = num_experiences * 2
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences * 2
    
    def load_batch_data_aug_obs(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            self.obs_buffs[i][:num_experiences] = curr_obs
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs
            self.done_buffs[i][:num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            #curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            self.obs_buffs[i][num_experiences:2*num_experiences] = curr_obs
            self.ac_buffs[i][num_experiences:2*num_experiences] = curr_acs
            self.rew_buffs[i][num_experiences:2*num_experiences] = curr_rews
            self.next_obs_buffs[i][num_experiences:2*num_experiences] = curr_next_obs
            self.done_buffs[i][num_experiences:2*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i = num_experiences * 2
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences * 2

    def load_batch_data_aug_both(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            self.obs_buffs[i][:num_experiences] = curr_obs
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs
            self.done_buffs[i][:num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            self.obs_buffs[i][num_experiences:2*num_experiences] = curr_obs
            self.ac_buffs[i][num_experiences:2*num_experiences] = curr_acs
            self.rew_buffs[i][num_experiences:2*num_experiences] = curr_rews
            self.next_obs_buffs[i][num_experiences:2*num_experiences] = curr_next_obs
            self.done_buffs[i][num_experiences:2*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i = num_experiences * 2
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences * 2

    def load_batch_data_aug_1(self, dir):
        #plan 1
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            self.obs_buffs[i][:num_experiences] = curr_obs
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs
            self.done_buffs[i][:num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            self.obs_buffs[i][num_experiences:2*num_experiences] = curr_obs
            self.ac_buffs[i][num_experiences:2*num_experiences] = curr_acs
            self.rew_buffs[i][num_experiences:2*num_experiences] = curr_rews
            self.next_obs_buffs[i][num_experiences:2*num_experiences] = curr_next_obs
            self.done_buffs[i][num_experiences:2*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i = num_experiences * 2
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences * 2

    def load_batch_data_aug_2(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            self.obs_buffs[i][:num_experiences] = curr_obs
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs
            self.done_buffs[i][:num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            #curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            self.obs_buffs[i][num_experiences:2*num_experiences] = curr_obs
            self.ac_buffs[i][num_experiences:2*num_experiences] = curr_acs
            self.rew_buffs[i][num_experiences:2*num_experiences] = curr_rews
            self.next_obs_buffs[i][num_experiences:2*num_experiences] = curr_next_obs
            self.done_buffs[i][num_experiences:2*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states
                
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            #curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            #curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            self.obs_buffs[i][2*num_experiences:3*num_experiences] = curr_obs
            self.ac_buffs[i][2*num_experiences:3*num_experiences] = curr_acs
            self.rew_buffs[i][2*num_experiences:3*num_experiences] = curr_rews
            self.next_obs_buffs[i][2*num_experiences:3*num_experiences] = curr_next_obs
            self.done_buffs[i][2*num_experiences:3*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i = num_experiences * 3
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences * 3

    def load_batch_data_aug_2_correct(self, dir, aug_noise):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            self.obs_buffs[i][:num_experiences] = curr_obs
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs
            self.done_buffs[i][:num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]
            
            #curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            
            next_state_ag1 = curr_next_obs[:,2:4]
            next_state_ag2 = curr_next_obs[:,2:4] + curr_next_obs[:,16:18]
            next_state_ag3 = curr_next_obs[:,2:4] + curr_next_obs[:,18:20]
            x_data = np.cos(np.pi * 2 * np.arange(6) / 6)
            y_data = np.sin(np.pi * 2 * np.arange(6) / 6)
            dist_ag1 = 1 / np.min(np.array([np.sqrt(np.sum(np.square(next_state_ag1 - np.array([x_data[j],y_data[j]])),axis=1)) for j in range(6)]),axis=0)
            #np.concatenate([dist_ag1.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1)
            new_reward = np.min(np.concatenate([dist_ag1.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1),axis=1)
            dist_ag2 = 1 / np.min(np.array([np.sqrt(np.sum(np.square(next_state_ag2 - np.array([x_data[j],y_data[j]])),axis=1)) for j in range(6)]),axis=0)
            #np.concatenate([dist_ag1.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1)
            new_reward += np.min(np.concatenate([dist_ag2.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1),axis=1)
            dist_ag3 = 1 / np.min(np.array([np.sqrt(np.sum(np.square(next_state_ag3 - np.array([x_data[j],y_data[j]])),axis=1)) for j in range(6)]),axis=0)
            #np.concatenate([dist_ag1.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1)
            new_reward += np.min(np.concatenate([dist_ag3.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1),axis=1)


            self.obs_buffs[i][num_experiences:2*num_experiences] = curr_obs
            self.ac_buffs[i][num_experiences:2*num_experiences] = curr_acs
            self.rew_buffs[i][num_experiences:2*num_experiences] = new_reward
            self.next_obs_buffs[i][num_experiences:2*num_experiences] = curr_next_obs
            self.done_buffs[i][num_experiences:2*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states
                
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            #curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            #curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            self.obs_buffs[i][2*num_experiences:3*num_experiences] = curr_obs
            self.ac_buffs[i][2*num_experiences:3*num_experiences] = curr_acs
            self.rew_buffs[i][2*num_experiences:3*num_experiences] = curr_rews
            self.next_obs_buffs[i][2*num_experiences:3*num_experiences] = curr_next_obs
            self.done_buffs[i][2*num_experiences:3*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i = num_experiences * 3
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences * 3

    def load_batch_data_aug_2_correct_quantile(self, dir, aug_noise):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))[0:250000]
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))[0:250000]
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))[0:250000]
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))[0:250000]
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))[0:250000]
        
            num_experiences = curr_obs.shape[0]

            self.obs_buffs[i][:num_experiences] = curr_obs
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs
            self.done_buffs[i][:num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))[0:250000]
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))[0:250000]
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))[0:250000]
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))[0:250000]
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))[0:250000]
        
            num_experiences = curr_obs.shape[0]
            
            #curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            
            next_state_ag1 = curr_next_obs[:,2:4]
            next_state_ag2 = curr_next_obs[:,2:4] + curr_next_obs[:,16:18]
            next_state_ag3 = curr_next_obs[:,2:4] + curr_next_obs[:,18:20]
            x_data = np.cos(np.pi * 2 * np.arange(6) / 6)
            y_data = np.sin(np.pi * 2 * np.arange(6) / 6)
            dist_ag1 = 1 / np.min(np.array([np.sqrt(np.sum(np.square(next_state_ag1 - np.array([x_data[j],y_data[j]])),axis=1)) for j in range(6)]),axis=0)
            #np.concatenate([dist_ag1.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1)
            new_reward = np.min(np.concatenate([dist_ag1.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1),axis=1)
            dist_ag2 = 1 / np.min(np.array([np.sqrt(np.sum(np.square(next_state_ag2 - np.array([x_data[j],y_data[j]])),axis=1)) for j in range(6)]),axis=0)
            #np.concatenate([dist_ag1.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1)
            new_reward += np.min(np.concatenate([dist_ag2.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1),axis=1)
            dist_ag3 = 1 / np.min(np.array([np.sqrt(np.sum(np.square(next_state_ag3 - np.array([x_data[j],y_data[j]])),axis=1)) for j in range(6)]),axis=0)
            #np.concatenate([dist_ag1.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1)
            new_reward += np.min(np.concatenate([dist_ag3.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1),axis=1)


            self.obs_buffs[i][num_experiences:2*num_experiences] = curr_obs
            self.ac_buffs[i][num_experiences:2*num_experiences] = curr_acs
            self.rew_buffs[i][num_experiences:2*num_experiences] = new_reward
            self.next_obs_buffs[i][num_experiences:2*num_experiences] = curr_next_obs
            self.done_buffs[i][num_experiences:2*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states
                
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))[0:250000]
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))[0:250000]
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))[0:250000]
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))[0:250000]
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))[0:250000]
        
            num_experiences = curr_obs.shape[0]

            curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            #curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            #curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            self.obs_buffs[i][2*num_experiences:3*num_experiences] = curr_obs
            self.ac_buffs[i][2*num_experiences:3*num_experiences] = curr_acs
            self.rew_buffs[i][2*num_experiences:3*num_experiences] = curr_rews
            self.next_obs_buffs[i][2*num_experiences:3*num_experiences] = curr_next_obs
            self.done_buffs[i][2*num_experiences:3*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states


        self.filled_i = num_experiences * 3
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences * 3

    def load_batch_data_aug_2_correct_half_quantile(self, dir, aug_noise):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))[0:125000]
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))[0:125000]
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))[0:125000]
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))[0:125000]
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))[0:125000]
        
            num_experiences = curr_obs.shape[0]

            self.obs_buffs[i][:num_experiences] = curr_obs
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs
            self.done_buffs[i][:num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))[0:125000]
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))[0:125000]
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))[0:125000]
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))[0:125000]
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))[0:125000]
        
            num_experiences = curr_obs.shape[0]
            
            #curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            
            next_state_ag1 = curr_next_obs[:,2:4]
            next_state_ag2 = curr_next_obs[:,2:4] + curr_next_obs[:,16:18]
            next_state_ag3 = curr_next_obs[:,2:4] + curr_next_obs[:,18:20]
            x_data = np.cos(np.pi * 2 * np.arange(6) / 6)
            y_data = np.sin(np.pi * 2 * np.arange(6) / 6)
            dist_ag1 = 1 / np.min(np.array([np.sqrt(np.sum(np.square(next_state_ag1 - np.array([x_data[j],y_data[j]])),axis=1)) for j in range(6)]),axis=0)
            #np.concatenate([dist_ag1.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1)
            new_reward = np.min(np.concatenate([dist_ag1.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1),axis=1)
            dist_ag2 = 1 / np.min(np.array([np.sqrt(np.sum(np.square(next_state_ag2 - np.array([x_data[j],y_data[j]])),axis=1)) for j in range(6)]),axis=0)
            #np.concatenate([dist_ag1.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1)
            new_reward += np.min(np.concatenate([dist_ag2.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1),axis=1)
            dist_ag3 = 1 / np.min(np.array([np.sqrt(np.sum(np.square(next_state_ag3 - np.array([x_data[j],y_data[j]])),axis=1)) for j in range(6)]),axis=0)
            #np.concatenate([dist_ag1.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1)
            new_reward += np.min(np.concatenate([dist_ag3.reshape(curr_obs.shape[0],1),10*np.ones((curr_obs.shape[0],1))],axis=1),axis=1)


            self.obs_buffs[i][num_experiences:2*num_experiences] = curr_obs
            self.ac_buffs[i][num_experiences:2*num_experiences] = curr_acs
            self.rew_buffs[i][num_experiences:2*num_experiences] = new_reward
            self.next_obs_buffs[i][num_experiences:2*num_experiences] = curr_next_obs
            self.done_buffs[i][num_experiences:2*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states
                
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))[0:125000]
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))[0:125000]
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))[0:125000]
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))[0:125000]
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))[0:125000]
        
            num_experiences = curr_obs.shape[0]

            curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            #curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            #curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            self.obs_buffs[i][2*num_experiences:3*num_experiences] = curr_obs
            self.ac_buffs[i][2*num_experiences:3*num_experiences] = curr_acs
            self.rew_buffs[i][2*num_experiences:3*num_experiences] = curr_rews
            self.next_obs_buffs[i][2*num_experiences:3*num_experiences] = curr_next_obs
            self.done_buffs[i][2*num_experiences:3*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states


        self.filled_i = num_experiences * 3
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences * 3

    def load_batch_data_aug_3(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            self.obs_buffs[i][:num_experiences] = curr_obs
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs
            self.done_buffs[i][:num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            #curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            self.obs_buffs[i][num_experiences:2*num_experiences] = curr_obs
            self.ac_buffs[i][num_experiences:2*num_experiences] = curr_acs
            self.rew_buffs[i][num_experiences:2*num_experiences] = curr_rews
            self.next_obs_buffs[i][num_experiences:2*num_experiences] = curr_next_obs
            self.done_buffs[i][num_experiences:2*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states
                
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            #curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            #curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])

            self.obs_buffs[i][2*num_experiences:3*num_experiences] = curr_obs
            self.ac_buffs[i][2*num_experiences:3*num_experiences] = curr_acs
            self.rew_buffs[i][2*num_experiences:3*num_experiences] = curr_rews
            self.next_obs_buffs[i][2*num_experiences:3*num_experiences] = curr_next_obs
            self.done_buffs[i][2*num_experiences:3*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states
                
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            curr_acs += aug_noise * np.random.randn(curr_acs.shape[0],curr_acs.shape[1])
            curr_obs += aug_noise * np.random.randn(curr_obs.shape[0],curr_obs.shape[1])
            curr_next_obs += aug_noise * np.random.randn(curr_next_obs.shape[0],curr_next_obs.shape[1])


            self.obs_buffs[i][3*num_experiences:4*num_experiences] = curr_obs
            self.ac_buffs[i][3*num_experiences:4*num_experiences] = curr_acs
            self.rew_buffs[i][3*num_experiences:4*num_experiences] = curr_rews
            self.next_obs_buffs[i][3*num_experiences:4*num_experiences] = curr_next_obs
            self.done_buffs[i][3*num_experiences:4*num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i = num_experiences * 4
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences * 4


    def load_batch_data_nocom(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = curr_obs.shape[0]

            self.obs_buffs[i][:num_experiences] = curr_obs[:,:-4]
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs[:,:-4]
            self.done_buffs[i][:num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i = num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences
        
    def load_batch_data_nocom_ind(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(0))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(0))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(0))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(0))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(0))
        
            num_experiences = curr_obs.shape[0]

            self.obs_buffs[i][:num_experiences] = curr_obs
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs
            self.done_buffs[i][:num_experiences] = curr_dones

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states

        self.filled_i = num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences

    def load_batch_data_half(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = int(curr_obs.shape[0]/2)

            self.obs_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_obs[:num_experiences]
            self.ac_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_acs[:num_experiences]
            self.rew_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_rews[:num_experiences]
            self.next_obs_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_next_obs[:num_experiences]
            self.done_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_dones[:num_experiences]

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states[:num_experiences]
                self.next_state_buffs[i][:num_experiences] = curr_next_states[:num_experiences]

        self.filled_i += num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else self.filled_i
    
    def load_batch_data_half_quantile(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))[0:250000]
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))[0:250000]
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))[0:250000]
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))[0:250000]
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))[0:250000]
        
            num_experiences = int(curr_obs.shape[0]/2)

            self.obs_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_obs[:num_experiences]
            self.ac_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_acs[:num_experiences]
            self.rew_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_rews[:num_experiences]
            self.next_obs_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_next_obs[:num_experiences]
            self.done_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_dones[:num_experiences]

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states[:num_experiences]
                self.next_state_buffs[i][:num_experiences] = curr_next_states[:num_experiences]

        self.filled_i += num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else self.filled_i

    def load_batch_data_nine(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = int(curr_obs.shape[0]/10*9)

            self.obs_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_obs[:num_experiences]
            self.ac_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_acs[:num_experiences]
            self.rew_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_rews[:num_experiences]
            self.next_obs_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_next_obs[:num_experiences]
            self.done_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_dones[:num_experiences]

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_states[:num_experiences]
                self.next_state_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_next_states[:num_experiences]

        self.filled_i += num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else self.filled_i

    def load_batch_data_nine_quantile(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))[0:250000]
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))[0:250000]
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))[0:250000]
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))[0:250000]
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))[0:250000]
        
            num_experiences = int(curr_obs.shape[0]/10*9)

            self.obs_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_obs[:num_experiences]
            self.ac_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_acs[:num_experiences]
            self.rew_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_rews[:num_experiences]
            self.next_obs_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_next_obs[:num_experiences]
            self.done_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_dones[:num_experiences]

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states[:num_experiences]
                self.next_state_buffs[i][:num_experiences] = curr_next_states[:num_experiences]

        self.filled_i += num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else self.filled_i

    def load_batch_data_one(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
            num_experiences = int(curr_obs.shape[0]/10)

            self.obs_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_obs[:num_experiences]
            self.ac_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_acs[:num_experiences]
            self.rew_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_rews[:num_experiences]
            self.next_obs_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_next_obs[:num_experiences]
            self.done_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_dones[:num_experiences]

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_states[:num_experiences]
                self.next_state_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_next_states[:num_experiences]

        self.filled_i += num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else self.filled_i

    def load_batch_data_one_quantile(self, dir):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))[0:250000]
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))[0:250000]
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))[0:250000]
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))[0:250000]
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))[0:250000]
        
            num_experiences = int(curr_obs.shape[0]/10)

            self.obs_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_obs[:num_experiences]
            self.ac_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_acs[:num_experiences]
            self.rew_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_rews[:num_experiences]
            self.next_obs_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_next_obs[:num_experiences]
            self.done_buffs[i][self.curr_i:self.curr_i+num_experiences] = curr_dones[:num_experiences]

            if self.is_mamujoco:
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states[:num_experiences]
                self.next_state_buffs[i][:num_experiences] = curr_next_states[:num_experiences]

        self.filled_i += num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else self.filled_i
