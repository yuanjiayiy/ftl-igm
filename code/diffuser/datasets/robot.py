import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from collections import namedtuple
import copy
from scipy.stats import bernoulli
import h5py

from ..utils.rendering import *
from ..utils.arrays import to_np

TASKS = {'pick green circle and place on book': 'green_pick_and_place',
         'push green circle to orange triangle': 'green_push',
         'pick green circle and place on elevated white surface': 'table_pick_and_place',
         'push green circle to orange triangle around purple bowl': 'push_around_bowl',
         'push green circle to orange triangle on elevated surface': 'book_push'}
      
               
def plot(savepath, pred, gt, conds_text, init_states, init_ims, plot_type='3D'):
    """observations: batch x horizon x obs_dim
    plot (obs pred + init gt + final pred) + (init im)"""
    ims_per_row = 2
    n_cols = len(pred)
    fig, axs = plt.subplots(n_cols, ims_per_row, figsize=(15, 15))
    for col in range(n_cols):
        obs = pred[col]
        gt_obs = gt[col]
        init_s = init_states[col]
        init_im = init_ims[col]
        if plot_type == '3D':
            axs[col,0].remove()
            axs[col,0] = fig.add_subplot(n_cols,ims_per_row,(col*2)+1, projection='3d')                   
            axs[col,0].scatter3D(obs[:,0], obs[:,1], obs[:,2], alpha=0.5, color='red') #obs
            axs[col,0].scatter3D(init_s[0], init_s[1], init_s[2], color='black', marker='o') #init gt
            axs[col,0].scatter3D(obs[-1,0], obs[-1,1], obs[-1,2], color='black', marker='x') #final obs pred
            axs[col,0].scatter3D(gt_obs[:,0], gt_obs[:,1], gt_obs[:,2], alpha=0.5, color='blue') #gt        
            axs[col,0].scatter3D(gt_obs[-1,0], gt_obs[-1,1], gt_obs[-1,2], color='black', marker='^') #final obs gt
        elif plot_type == '2D':
            axs[col,0].scatter(obs[:,0], obs[:,1], alpha=0.5, color='red') #pred
            axs[col,0].scatter(init_s[0], init_s[1], color='black', marker='o') #init gt
            axs[col,0].scatter(obs[-1,0], obs[-1,1], color='black', marker='x') #final obs pred  
            axs[col,0].scatter(gt_obs[:,0], gt_obs[:,1], alpha=0.5, color='blue') #gt        
            axs[col,0].scatter(gt_obs[-1,0], gt_obs[-1,1], color='black', marker='^') #final obs gt
        axs[col,0].set_title(conds_text[col])  
        axs[col,1].imshow(init_im.astype(np.uint8))
    fig.tight_layout()
    plt.savefig(savepath)

def to_tensor(x, dtype=torch.float, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)

Batch = namedtuple('Batch', 'trajectories conditions dummy_cond conditions_obs conditions_obs_im') #trajectories: ee_pose+gripper, conditions: concept (text embedding), conditions_obs: init robot joint pos + init image


class RobotSequenceDataset(Dataset):

    def __init__(self, horizon=150, max_path_length=1000, use_padding=True, dataset_path=None, dataset_stats_path=None, mode='train', *args, **kwargs):

        with open(dataset_stats_path, "rb") as input_file:
            self.cond_init_mins, self.cond_init_maxs, self.cond_obs_imL_mean, self.cond_obs_imL_std, \
                self.conditions, self.dummy_cond = pickle.load(input_file) #ee_pose+gripper, left image
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.action_dim = 0
        self.cond_dim = 768 # input to model init, T5 self.conditions
        resnet_dim = 512 # resnet18 penultimate layer embedding of s_0 image
        ee_pose_dim = 7
        gripper_dim = 1
        self.observation_dim = ee_pose_dim + gripper_dim  
        self.obs_cond_dim = resnet_dim + ee_pose_dim + gripper_dim # input to model, size of non task rep conds (image embedding + robot init state: ee_pose+gripper)      
        self.obs_mins, self.obs_maxs = self.cond_init_mins[:ee_pose_dim+gripper_dim], self.cond_init_maxs[:ee_pose_dim+gripper_dim]
            
        if mode == 'train':
            with open(f'{dataset_path}_nonim.pkl', 'rb') as input_file:
                self.observations, self.conditions_obs_robot, self.conditions, self.dummy_cond, self.text_conds = pickle.load(input_file)
            with h5py.File(f'{dataset_path}_im.h5', 'r') as file:
                self.cond_obs_imL = np.array(file.get('imL')) #stacked
            sample_rate = 1 #training data. only images were sampled. test sample_rate=1, init should not be called for test.
            self.observations = [obs[::sample_rate] for obs in self.observations]
            self.conditions_obs_robot = [cond_obs[::sample_rate] for cond_obs in self.conditions_obs_robot]
            self.n_episodes = len(self.observations)
            self.reshaped_observations = np.vstack(copy.deepcopy(self.observations)) # ee_pose_dim + gripper_dim
            self.path_lengths = [len(obs) for obs in self.observations]
            self.indices = self.make_indices(self.path_lengths, self.horizon)
            self.normalize() #observations, conditions_obs_robot, conditions_obs_im_left
                            

    def normalize(self):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        self.normed_observations = (self.reshaped_observations - self.obs_mins) / (self.obs_maxs - self.obs_mins + 1e-5) # [ 0, 1 ]
        self.normed_observations = (self.normed_observations * 2) - 1 # [ -1, 1 ]
        self.normed_observations = [self.normed_observations[np.sum(self.path_lengths[:i]) if i>0 else 0 : np.sum(self.path_lengths[:i+1])] for i in range(len(self.path_lengths))]
        #normed imL
        self.cond_obs_imL = (self.cond_obs_imL - self.cond_obs_imL_mean) / self.cond_obs_imL_std
        self.cond_obs_imL = [self.cond_obs_imL[np.sum(self.path_lengths[:i]) if i>0 else 0 : np.sum(self.path_lengths[:i+1])] for i in range(len(self.path_lengths))]


    def normalize_init(self, init_states_robot, normed_init_im_L):
        normed_init_states = (init_states_robot - self.obs_mins) / (self.obs_maxs - self.obs_mins + 1e-5) # [ 0, 1 ]        
        normed_init_states = (normed_init_states * 2) - 1 # [ -1, 1 ]
        normed_init_im_L = (normed_init_im_L - self.cond_obs_imL_mean) / self.cond_obs_imL_std 
        return normed_init_states, normed_init_im_L#, normed_init_im_R


    def unnormalize(self, x, eps=1e-2):
        '''
            x : [ 0, 1 ]
            x [ horizon x obs_dim ]
        '''
        assert x.max() <= 1.0 + eps and x.min() >= -1.0 - eps, f'x range: ({x.min():.4f}, {x.max():.4f})'
        ret = x + 1 #[-1,1]-->[0,2]
        ret /= 2 #[0,2]-->[0,1]
        #ee_pose+gripper
        return ret * (self.obs_maxs - self.obs_mins + 1e-5) + self.obs_mins #[min,max]


    def unnormalize_im(self, im):
        im = np.transpose(to_np(im),(0,2,3,1)) #chw->hwc
        return (im * self.cond_obs_imL_std) + self.cond_obs_imL_mean


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


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx=None, proprioception_dropout=True):
        if idx is None: idx = np.random.choice(range(len(self.indices)-1))        
        path_ind, start, end = self.indices[idx]
        observations = self.normed_observations[path_ind][start:end,:] # ee_pose + gripper
        obs_cond = self.normed_observations[path_ind][start] # ee_pose + gripper
        if proprioception_dropout and bernoulli.rvs(0.4, size=1)[0]: # proprioception dropout during training
            obs_cond = np.zeros_like(obs_cond)
        obs_cond_im_L = np.transpose(self.cond_obs_imL[path_ind][start],(2,0,1)).astype(np.float32) #normalized
        batch = Batch(observations, self.conditions[TASKS[self.text_conds[path_ind]]].squeeze(), self.dummy_cond.squeeze(), obs_cond, obs_cond_im_L)
        return batch
