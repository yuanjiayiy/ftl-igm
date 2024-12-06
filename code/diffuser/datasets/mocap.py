import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from collections import namedtuple

from ..utils.rendering import *
import copy
from transformers import T5Tokenizer, T5EncoderModel


def to_tensor(x, dtype=torch.float, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)

Batch = namedtuple('Batch', 'trajectories conditions dummy_cond conditions_obs') #trajectories: output traj, conditions: concept (text embedding), conditions_obs: condition on curr obs (model needs to predict next steps, create batches based on curr obs any t from training demos)


has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5EncoderModel.from_pretrained("google/flan-t5-base").to(device)
cond_dummy = tokenizer('', return_tensors="pt").input_ids.to(device)
cond_dummy = model(cond_dummy).last_hidden_state.mean(axis=1)
cond_dummy = cond_dummy.reshape(1,-1)


class mocapSequenceDataset(Dataset):

    def __init__(self, horizon=150, max_path_length=1000, use_padding=True, dataset_path=None, sample_rate=1, *args, **kwargs):

        self.horizon = horizon #32 (adjusted for subsampling)
        self.max_path_length = max_path_length #len largest path
        self.use_padding = use_padding
        self.sample_rate = sample_rate # predict sparse trajectory, doesn't affect rendering (1 is all)

        with open(dataset_path, "rb") as input_file:
            #observations: list, each traj size H x (n joints * 3) -- can have different horizon H. state dim n(=31/14) joints x 3D pos
            self.observations, self.conditions, self.dummy_cond, self.conds_text = pickle.load(input_file)
        
        #sample obs (rendering looks the same)
        self.observations = [obs[::self.sample_rate,:] for obs in self.observations]
        
        self.observation_dim = self.observations[0].shape[-1] # every time step predict the skeleton: n joints x 3D pos
        self.action_dim = 0 
        self.cond_dim = self.conditions.shape[1] # 768 T5
        self.obs_cond_dim = self.observations[0].shape[-1] # init state: n joints x 3D pos

        self.n_episodes = len(self.observations)
        
        # every traj is Hx42, different H. normalize the states (42 dim) 
        reshaped_obs = np.vstack(copy.deepcopy(self.observations))
        self.mins = reshaped_obs.min(axis=0)
        self.maxs = reshaped_obs.max(axis=0)
        
        self.path_lengths = [obs.shape[0] for obs in self.observations]
        self.indices = self.make_indices(self.path_lengths, self.horizon)
        self.normalize()


    # def normalize(self, keys=['observations', 'actions']):
    def normalize(self, keys=['observations']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        self.normed_observations = np.vstack(copy.deepcopy(self.observations))
        self.normed_observations = (self.normed_observations - self.mins) / (self.maxs - self.mins + 1e-5) # [ 0, 1 ]
        self.normed_observations = (self.normed_observations * 2) - 1 # [ -1, 1 ]
        self.normed_observations = [self.normed_observations[np.sum(self.path_lengths[:i]) if i>0 else 0 : np.sum(self.path_lengths[:i+1])] for i in range(len(self.path_lengths))]


    def normalize_init(self, init_states):
        normed_init_states = (init_states - self.mins) / (self.maxs - self.mins + 1e-5) # [ 0, 1 ]        
        normed_init_states = (normed_init_states * 2) - 1 # [ -1, 1 ]
        return normed_init_states


    def unnormalize(self, x, eps=1e-4): 
        '''
            x : [ 0, 1 ]
            x [ horizon x obs_dim ]
        '''
        assert x.max() <= 1.0 + eps and x.min() >= -1.0 - eps, f'x range: ({x.min():.4f}, {x.max():.4f})' 
        mins, maxs = self.mins[:x.shape[-1]], self.maxs[:x.shape[-1]]
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
        observations = self.normed_observations[path_ind][start:end, :] #sub traj
        obs_conditions = self.normed_observations[path_ind][start, :] #init state
        batch = Batch(observations, self.conditions[path_ind], self.dummy_cond, obs_conditions)
        return batch


    def get_task(self, idx):
        path_ind, _, _ = self.indices[idx]
        return self.conds_text[path_ind]


    def get_item_render(self):
        idx = np.random.choice(range(len(self.indices)))
        path_ind, start, end = self.indices[idx]
        gt_observations = self.observations[path_ind][start:end, :] #sub traj gt
        obs_conditions = self.normed_observations[path_ind][start, :] #init state normalized
        return gt_observations, self.conditions[path_ind].reshape(1,-1), \
                self.dummy_cond.reshape(1,-1), obs_conditions.reshape(1,-1), self.conds_text[path_ind]


    def get_init(self):
        #textual inversion
        idx = np.random.choice(range(len(self.indices)))
        path_ind, start, end = self.indices[idx]
        gt_observations = self.observations[path_ind][start:end, :] #sub traj gt
        obs_conditions = self.normed_observations[path_ind][start, :] #init state normalized
        return gt_observations, obs_conditions.reshape(1,-1)
     