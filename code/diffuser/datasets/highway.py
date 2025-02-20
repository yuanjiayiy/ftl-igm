import numpy as np
import torch
import pickle
from collections import namedtuple
from ..utils.rendering import *
import gymnasium as gym
import copy

from transformers import T5Tokenizer, T5EncoderModel


def to_tensor(x, dtype=torch.float, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)

Batch = namedtuple('Batch', 'trajectories conditions dummy_cond conditions_obs') #trajectories: output traj, conditions: concept (text embedding), conditions_obs: condition on curr obs (model needs to predict next steps, create batches based on curr obs any t from training demos)

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

def get_acc(scenario_text, all_scenario_rew, all_scenario_info, all_scenario_done, all_scenario_obs):
    # check crashed
    # exit — reached exit
    # intersection — complete turn
    # merge — reach end of lane  
    rews = [np.sum(x)/len(x) for x in all_scenario_rew]
    print('mean reward ', np.mean(rews), '\pm ', np.std(rews))
    Hs = [len(x) for x in all_scenario_rew] 
    print('horizon ', np.mean(Hs), '\pm ', np.std(Hs))     
    all_demo_crashed = []
    for demo in all_scenario_info:
        all_demo_crashed.append([ts['crashed'] for ts in demo])
    crashes = [x[-1] for x in all_demo_crashed]
    print('crashed ', np.mean(crashes))
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

    def __init__(self, horizon=150, max_path_length=1000, use_padding=True, dataset_path=None, sample_rate=1, *args, **kwargs):

        self.horizon = horizon #32 (adjusted for subsampling)
        self.max_path_length = max_path_length #len largest path
        self.use_padding = use_padding
        self.sample_rate = sample_rate #predict sparse trajectory, doesn't affect rendering (1 is all)

        with open(dataset_path, "rb") as input_file:
            # observations: list, each traj size H x ([presence, x, y, vx, vy, cos_h, sin_h] * n_vehicles=5), can have different horizons H.
            # im_obs: same but H x height x width x 3
            self.observations, self.im_obs, rewards, dones, truncated, infos, video_idxs, self.conds_text = pickle.load(input_file)

        # I will only take the highway for now.
        def filter_lists(reference_list, *target_lists):
            indices = [i for i, element in enumerate(reference_list) if element == "highway"]
            return [[lst[i] for i in indices] for lst in target_lists]
        
        filtered_dataset = filter_lists(self.conds_text, self.observations, self.im_obs, rewards, dones, truncated, infos, video_idxs)
        self.observations, self.im_obs, rewards, dones, truncated, infos, video_idxs = filtered_dataset
        self.conds_text = [element for index, element in enumerate(self.conds_text) if element == "highway"]

        self.dummy_cond = self.generate_representation('')
        self.conditions = [self.generate_representation(cond_text) for cond_text in self.conds_text]
        
        self.observation_dim = self.observations[0][0].shape[-1] # predict full ego features [present,x,y,vx,vy,cos_h,sin_h]
        self.action_dim = 0 
        self.cond_dim = len(self.conditions[0]) # 768 T5
        self.obs_cond_dim = np.prod(self.observations[0][0].shape) # state space 5 vehicles x 7 features
        self.n_vehicles = 5
        self.feat_dim = 7
        self.ego_idx = 0

        self.n_episodes = len(self.observations)
        
        # every traj is HxS, different H. normalize the states (S dim). normalize based on 5x7 dim
        reshaped_obs = np.vstack(copy.deepcopy(self.observations))
        self.mins = reshaped_obs.min(axis=0)
        self.maxs = reshaped_obs.max(axis=0)


        eps=1e-4
        self.normalized = self.maxs[0,1]<1.0+eps and self.mins[0,1]>-1.0+eps
        self.normalized = False

        self.path_lengths = [len(obs) for obs in self.observations]
        self.indices = self.make_indices(self.path_lengths, self.horizon)
        self.normalize()


    def generate_representation(self,str):
        cond = tokenizer(str, return_tensors="pt").input_ids.to(device)
        cond = model(cond).last_hidden_state.mean(axis=1).detach().cpu().numpy()[0]
        return cond
    

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
        normed_init_states = (np.array(init_states) - self.mins) / (self.maxs - self.mins + 1e-5) # [0,1]
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
        #observations of ego (first vehicle).
        observations = np.array([x[0] for x in self.normed_observations[path_ind][start:end]]) #sub traj: (horizon=5 x 7 features)
        # obs_conditions - normalized s_0 image
        obs_conditions = self.normed_observations[path_ind][start].flatten() #init state s0: (5 vehicles x 7 features)
        batch = Batch(observations, self.conditions[path_ind], self.dummy_cond, obs_conditions)
        import pdb; pdb.set_trace()
        return batch


    def get_task(self, idx):
        path_ind, _, _ = self.indices[idx]
        return self.conds_text[path_ind]
    

    def get_item_render(self, idx=None):
        if idx is None:
            idx = np.random.choice(range(len(self.indices)))
        path_ind, start, end = self.indices[idx]
        gt_observations = np.array([x[0] for x in self.normed_observations[path_ind][start:end]]) #sub traj: (horizon=5 x 7 features)
        obs_conditions = self.normed_observations[path_ind][start].flatten() #init state normalized
        return gt_observations, self.conditions[path_ind].reshape(1,-1), self.dummy_cond.reshape(1,-1), \
            obs_conditions.reshape(1,-1), self.conds_text[path_ind]

class HighwaySequencePartialObservedDataset(HighwaySequenceDataset):
    def __init__(self, horizon=150, max_path_length=1000, use_padding=True, dataset_path=None, sample_rate=1, *args, **kwargs):
        super().__init__(horizon, max_path_length, use_padding, dataset_path, sample_rate, *args, **kwargs)
        path_ind, start, end = self.indices[0]
        self.cond_dim = self.feat_dim * self.horizon
        self.dummy_cond = self.get_conditions(path_ind=path_ind, start=start, end=end, car_idx=1, dummy_cond=True)
        self.car_idx = 1
        self.car_obs_buffer = []

    def get_conditions(self, path_ind, start, end, car_idx, dummy_cond = False):
        '''
            condition on current observation for planning
        '''
        partial_observed_conditions = np.array([x[car_idx] for x in self.normed_observations[path_ind][start:end]]).flatten()
        partial_observed_conditions = np.pad(partial_observed_conditions, (0, max(0, self.cond_dim - len(partial_observed_conditions))), mode='constant').astype(np.float32)
        if dummy_cond:
            partial_observed_conditions = np.zeros_like(partial_observed_conditions)
        return partial_observed_conditions
    
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        #observations of ego (first vehicle).
        observations = np.array([x[0] for x in self.normed_observations[path_ind][start:end]]) #sub traj: (horizon=5 x 7 features)
        # obs_conditions - normalized s_0 image
        obs_conditions = self.normed_observations[path_ind][start].flatten() #init state s0: (5 vehicles x 7 features)
        batch = Batch(observations, self.get_conditions(path_ind=path_ind, start=start, end=end, car_idx=self.car_idx), self.dummy_cond, obs_conditions)
        return batch
    
    def get_item_render(self, idx=None):
        if idx is None:
            idx = np.random.choice(range(len(self.indices)))
        path_ind, start, end = self.indices[idx]
        gt_observations = np.array([x[0] for x in self.normed_observations[path_ind][start:end]]) #sub traj: (horizon=5 x 7 features)
        obs_conditions = self.normed_observations[path_ind][start].flatten() #init state normalized
        return gt_observations, self.get_conditions(path_ind=path_ind, start=max(0, start-self.horizon), end=start, car_idx=self.car_idx).reshape(1,-1), self.dummy_cond.reshape(1,-1), \
            obs_conditions.reshape(1,-1), self.conds_text[path_ind]
        # return gt_observations, self.get_conditions(path_ind=path_ind, start=start, end=end, car_idx=self.car_idx).reshape(1,-1), self.dummy_cond.reshape(1,-1), \
        #             obs_conditions.reshape(1,-1), self.conds_text[path_ind]
            
            
    
    def generate_conditions(self, obs):
        # if buffer not full, append
        # if buffer full, remove the oldest, append to last
        if len(self.car_obs_buffer) == self.horizon:
            self.car_obs_buffer.pop(0)
        self.car_obs_buffer.append(obs[self.car_idx])
        cond = np.array(self.car_obs_buffer).flatten()
        padded_cond = np.pad(cond, (0, max(0, self.cond_dim - len(cond))), mode='constant').astype(np.float32)
        return padded_cond