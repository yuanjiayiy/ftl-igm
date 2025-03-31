import numpy as np
import torch
import pickle
from collections import namedtuple
from ..utils.rendering import *
import gymnasium as gym
import einops
import copy
import wandb
from transformers import T5Tokenizer, T5EncoderModel


def to_tensor(x, dtype=torch.float, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)

Batch = namedtuple('Batch', 'trajectories agent_idx past_trajectory conditions_obs')
# trajectories: output traj
# agent_idx: agent index
# past_trajectory: past trajectories until current start
# conditions_obs: condition on curr obs (model needs to predict next steps, create batches based on curr obs any t from training demos)

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5EncoderModel.from_pretrained("google/flan-t5-base").to(device)


def plot_traj(traj, init_s, save_fig_path, n_vehicles, feat_dim, sample_state=None, cond_text=''):
    plt.figure()
    plt.xlim(-120,600)
    plt.ylim(-120,100)   
    plt.plot(traj[:,1],traj[:,2], color='blue', marker='^')
    plt.plot(traj[0,1], traj[0,2], color='white', marker='^') #indicate agent (cone) direction (init state)
    plt.plot(traj[-1,1], traj[-1,2], color='black', marker='^') #indicate agent (cone) direction (final state)
    if sample_state is not None:
        plt.plot(sample_state[1], sample_state[2], color='green', marker='^') #indicate s_t_1
    #other vehicles init pos
    init_s = init_s.reshape(n_vehicles, feat_dim)
    for i in range(1,n_vehicles):
        plt.scatter(init_s[i,1], init_s[i,2], color='red', marker='^') #(x,y)        
    plt.title(cond_text)
    plt.savefig(save_fig_path)

def get_acc(scenario_text, all_scenario_rew, all_scenario_info, all_scenario_done, all_scenario_obs, **kwargs):
    # check crashed
    # exit — reached exit
    # intersection — complete turn
    # merge — reach end of lane  
    SLOWER = 4
    rews = [np.sum(x)/len(x) for x in all_scenario_rew]
    print('mean reward ', np.mean(rews), '\pm ', np.std(rews))
    Hs = [len(x) for x in all_scenario_rew] 
    print('horizon ', np.mean(Hs), '\pm ', np.std(Hs))     
    all_demo_crashed = []
    all_demo_act = []
    for demo in all_scenario_info:
        all_demo_crashed.append([ts['crashed'] for ts in demo])
        all_demo_act.append([ts['action'] for ts in demo])
    crashes = [x[-1] for x in all_demo_crashed]
    print('crashed ', np.mean(crashes))
    slower_proportion = [np.sum(x.count(SLOWER)) /len(x) for x in all_demo_act]
    print('mean slower ', np.mean(slower_proportion), '\pm ', np.std(slower_proportion))

    logs = {
        'episodal reward': np.mean(np.array([np.sum(x) for x in all_scenario_rew])),
        'mean reward': np.mean(rews),
        'std': np.std(rews),
        'horizon': np.mean(Hs),
        'horizon std': np.std(Hs),
        'crashed': np.mean(crashes),
        'all_scenario_info vehicle_count=' + str(kwargs['vehicles_count']): wandb.Table(data=[[str(x)] for x in all_scenario_info], columns=["info"]),
        **kwargs
    }
    wandb.log(logs)
    if 'exit' in scenario_text: #complete exit
        all_demo_success = []
        for demo in all_scenario_info:
            all_demo_success.append([ts['is_success'] for ts in demo])
        succ = [x[-1] for x in all_demo_success]
        print('make exit ', np.mean(succ))
    elif 'intersection' in scenario_text: #complete turn (if not crashed or turned, another behavior is stay idle)  
        all_demo_arrive = []
        for demo in all_scenario_info:
            all_demo_arrive.append([ts['rewards']['arrived_reward'] for ts in demo])
        arrive = [x[-1] for x in all_demo_arrive]
        print('cross intersection ', np.mean(arrive))
    elif 'merge' in scenario_text: #reach end of merge lane
        reach = [x[-1] and not y for x,y in zip(all_scenario_done,crashes)]
        print('pass merge lane ', np.mean(reach))
    elif 'roundabout' in scenario_text: #exit straight ahead (last step in each traj reaches y pos)
        ego_idx = 0
        exit_roundabout = [x[-1][ego_idx][2]<-20 for x in all_scenario_obs]
        print('exit roundabout ', np.mean(exit_roundabout))         
    print('------')

def safe_deepcopy_env(obj):
    """Perform a deep copy of an environment but without copying its viewer."""
    cls = obj.__class__
    result = cls.__new__(cls)
    memo = {id(obj): result}
    for k, v in obj.__dict__.items():
        if k not in ['viewer', '_monitor', 'grid_render', 'video_recorder', '_record_video_wrapper']:
            if isinstance(v, gym.Env):
                setattr(result, k, safe_deepcopy_env(v))
            else:
                setattr(result, k, copy.deepcopy(v, memo=memo))
        else:
            setattr(result, k, None)
    return result


class HighwaySequenceDataset(torch.utils.data.Dataset):

    def __init__(self, horizon=150, max_path_length=1000, use_padding=True, dataset_path=None, sample_rate=1, history_horizon=8, agent_idx=0, *args, **kwargs):

        self.horizon = horizon #32 (adjusted for subsampling)
        self.max_path_length = max_path_length #len largest path
        self.use_padding = use_padding
        self.sample_rate = sample_rate #predict sparse trajectory, doesn't affect rendering (1 is all)

        with open(dataset_path, "rb") as input_file:
            # observations: list, each traj size H x ([presence, x, y, vx, vy, cos_h, sin_h] * n_vehicles=5), can have different horizons H.
            # im_obs: same but H x height x width x 3
            self.observations, self.im_obs, rewards, dones, truncated, infos, video_idxs, self.conds_text = pickle.load(input_file)
        
        self.observation_dim = self.observations[0][0].shape[-1] # predict full ego features [present,x,y,vx,vy,cos_h,sin_h]
        self.action_dim = 0

        # self.obs_cond_dim = ? need to use attention for current obs
        self.obs_cond_dim = np.prod(self.observations[0][0].shape) # state space N vehicles x 7 features
        self.history_horizon = history_horizon
        self.agent_idx = agent_idx
        self.cond_dim = self.obs_cond_dim

        self.n_episodes = len(self.observations)
        
        # every traj is HxS, different H. normalize the states (S dim). normalize based on 5x7 dim
        reshaped_obs = np.vstack(copy.deepcopy(self.observations))
        self.mins = reshaped_obs.min(axis=0)
        self.maxs = reshaped_obs.max(axis=0)
        self.feat_dim = 7

        eps=1e-4
        self.normalized = self.maxs[0,1]<1.0+eps and self.mins[0,1]>-1.0+eps
        self.normalized = False

        self.path_lengths = [len(obs) for obs in self.observations]
        self.indices = self.make_indices(self.path_lengths, self.horizon)
        # self.conditions = self.get_conditions()
        self.normalize()
    

    # def normalize(self, keys=['observations', 'actions']):
    def normalize(self, keys=['observations']):
        '''
            normalize fields that will be predicted by the diffusion model
            + image s_0 / init state (5 vehicles)
        '''
        if self.normalized: # states already normalized and clipped.
            self.normed_observations = copy.deepcopy(self.observations)
            return 

        self.normed_observations = np.vstack(copy.deepcopy(self.observations)) 
        self.normed_observations = (self.normed_observations - self.mins) / (self.maxs - self.mins + 1e-5) # [0,1]
        self.normed_observations = (self.normed_observations * 2) - 1 # [-1,1]
        self.normed_observations = [self.normed_observations[np.sum(self.path_lengths[:i]) if i>0 else 0 : np.sum(self.path_lengths[:i+1])] for i in range(len(self.path_lengths))]


    def normalize_init(self, init_states):
        """normalize init state (5 vehicles)"""
        if self.normalized: # states already normalized and clipped.
            normed_init_states = init_states
            return normed_init_states
        # Fill missing rows
        init_states = np.array(init_states)
        if init_states != self.mins:
            padded_init_states = np.zeros_like(self.mins)
            padded_init_states[:init_states.shape[0], :init_states.shape[1]] = init_states
            init_states = padded_init_states
        normed_init_states = (init_states - self.mins) / (self.maxs - self.mins + 1e-5) # [0,1]
        normed_init_states = (normed_init_states * 2) - 1 # [-1,1]
        return normed_init_states
    

    def unnormalize(self, x, eps=1e-4): 
        ''' the output
            x : [ 0, 1 ]
            x [ horizon x obs_dim ] 
        '''
        if self.normalized: # states already normalized and clipped.
            return x

        assert x.max() <= 1.0 + eps and x.min() >= -1.0 - eps, f'x range: ({x.min():.4f}, {x.max():.4f})'
        mins, maxs = self.mins.flatten()[:x.shape[-1]], self.maxs.flatten()[:x.shape[-1]] 
        ret = x + 1 #[-1,1]-->[0,2]
        ret /= 2 #[0,2]-->[0,1]
        return ret * (maxs - mins + 1e-5) + mins #[min,max]


    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices
    

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return observations[0]


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        # shift by `self.history_horizon`
        history_start = max(0, start - self.history_horizon) 

        #observations of ego (first vehicle).
        observations = np.array([x[0] for x in self.normed_observations[path_ind][start:end]]) #sub traj: (horizon=N x 7 features)

        agent_idx = np.array(self.agent_idx)
        
        # past trajectory, in reverse order
        unpadded_past_trajectory = self.normed_observations[path_ind][history_start:start][::-1, self.agent_idx, :]
        past_trajectory = self.pad_history(unpadded_past_trajectory=unpadded_past_trajectory)
        
        # past_trajectory = self.normed_observations[path_ind][history_start:start, self.agent_idx, :].transpose(1, 0, 2).reshape(len(self.agent_idx), -1)

        # obs_conditions - normalized s_0 image
        obs_conditions = self.normed_observations[path_ind][start].flatten() #init state s0: (N vehicles x 7 features)
        
        batch = Batch(observations, agent_idx, past_trajectory, obs_conditions)
        return batch


    def get_task(self, idx):
        path_ind, _, _ = self.indices[idx]
        return self.conds_text[path_ind]
    

    def get_item_render(self, idx=None):
        if idx is None:
            idx = np.random.choice(range(len(self.indices)))
        path_ind, start, end = self.indices[idx]
        # shift by `self.history_horizon`
        history_start = max(0, start - self.history_horizon)

        gt_observations = np.array([x[0] for x in self.normed_observations[path_ind][start:end]]) #sub traj: (horizon=5 x 7 features)
        obs_conditions = self.normed_observations[path_ind][start].flatten() #init state normalized

        agent_idx = np.array(self.agent_idx)

        # past trajectory
        unpadded_past_trajectory = self.normed_observations[path_ind][history_start:start][::-1, self.agent_idx, :]
        past_trajectory = self.pad_history(unpadded_past_trajectory=unpadded_past_trajectory)

        return gt_observations, agent_idx.reshape(1), past_trajectory.reshape(1,-1), obs_conditions.reshape(1,-1)
    
    def pad_history(self, unpadded_past_trajectory):

        padded_past_trajectory = np.zeros((self.history_horizon, self.feat_dim))
        padded_past_trajectory[:unpadded_past_trajectory.shape[0], :unpadded_past_trajectory.shape[1]] = unpadded_past_trajectory
        past_trajectory = padded_past_trajectory.flatten().astype(np.float32)
        return past_trajectory

