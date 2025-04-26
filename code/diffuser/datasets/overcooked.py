import random
import numpy as np
import torch
import pickle
from collections import namedtuple
from ..utils.rendering import *
import gymnasium as gym
import copy

from transformers import T5Tokenizer, T5EncoderModel

# player identity would be <env_name><policy_name><subpolicy_name><index>
# env_name: "counter_circuit_o_1order" -> 00
# policy_name: "mep" -> 00
# subpolicy_name: "init" "mid" "final" -> 00, 01, 02
# index: 0, 1, 2, ... -> 00, 01, 02, ...
# for example, "counter_circuit_o_1order_mep mep2_final" -> 00000202
policy_name_dict = {"counter_circuit_o_1order_mep": 
                    ['mep1_final', 'mep1_init', 'mep1_mid', 'mep2_final', 'mep2_init', 'mep2_mid', 'mep3_final', 'mep3_init', 'mep3_mid', 'mep4_final', 'mep4_init', 'mep4_mid', 'mep5_final', 'mep5_init', 'mep5_mid', 'mep6_final', 'mep6_init', 'mep6_mid', 'mep7_final', 'mep7_init', 'mep7_mid', 'mep8_final', 'mep8_init', 'mep8_mid']}


def to_tensor(x, dtype=torch.float, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)

Batch = namedtuple('Batch', 'trajectories conditions dummy_cond conditions_obs conditions_mask') #trajectories: output traj, conditions: concept (text embedding), conditions_obs: condition on curr obs (model needs to predict next steps, create batches based on curr obs any t from training demos)

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


class OvercookedSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, args, sample_rate=1):

        dataset_path = args.dataset_path

        if dataset_path.endswith("hdf5"):
            from ..utils.hdf5_dataset import HDF5Dataset
            self.dataset = HDF5Dataset(args, "test") # TODO: change to train later
            self.observations = np.array(self.dataset.dset["obs"]) # path_num * (path_length + 1) * num_agent * height * width * channels
            self.actions = np.array(self.dataset.dset["actions"]) # path_num * path_length * num_agent * action_dim (1)
            self.dones = np.array(self.dataset.dset["dones"]) # path_num * path_length * num_agent
            self.env_info = np.array(self.dataset.dset["env_info"])
            self.policy_id = np.array(self.dataset.dset["policy_id"]) # path_num * num_agent (agent1_policy_name, agent2_policy_name)
            self.rewards = np.array(self.dataset.dset["rewards"]) # path_num * path_length * num_agent * reward_dim (1)

        else:
            with open(dataset_path, "rb") as input_file:
                self.cond_init_mins, self.cond_init_maxs, self.cond_obs_imL_mean, self.cond_obs_imL_std, \
                    self.conditions, self.dummy_cond = pickle.load(input_file) #ee_pose+gripper, left image
        
        self.horizon = args.horizon
        self.max_path_length = args.max_path_length
        self.use_padding = args.use_padding

        self.dummy_cond = np.int64(0)
        self.policy_names = [policy_name_dict[self.dataset.dataset_name][agent2_policy_id]
                           for agent1_policy_id, agent2_policy_id in self.policy_id]
        self.num_embeddings, self.conditions = self.convert_to_indices(self.policy_names)
        
        
        self.action_dim = 1
        self.cond_dim = 8 # input to model init, T5 self.conditions

        
            
        self.observation_dim = np.prod(self.observations[0, 0, 0].shape) # every time step predict the skeleton: n joints x 3D pos
        # self.cond_dim = self.conditions.shape[1] # 768 T5
        self.obs_cond_dim = np.prod(self.observations[0, 0, 0].shape) # init state: n joints x 3D pos
        
        self.n_episodes = len(self.observations)
        
        # mins and max 0, 255
        self.mins = 0
        self.maxs = 255

        self.path_lengths = [obs.shape[0] for obs in self.observations]
        # self.indices = self.make_indices(self.path_lengths, self.horizon)
        self.normalize()
    
    def generate_representation(self,str):
        cond = tokenizer(str, return_tensors="pt").input_ids.to(device)
        cond = model(cond).last_hidden_state.mean(axis=1).detach().cpu().numpy()[0]
        return cond
    
    def convert_to_indices(self, tokens):
        # Build vocabulary (token -> index)
        token_to_id = {}
        for token in tokens:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

        # Convert tokens to indices
        indices = [token_to_id[token] for token in tokens]

        num_embeddings = len(token_to_id)
        return num_embeddings, indices

    def normalize(self):
        '''
            normalize fields that will be predicted by the diffusion model
            normalize from [0, 255] to [-1, 1]
        '''
        self.normed_observations = copy.deepcopy(self.observations)
        self.normed_observations = (self.normed_observations - self.mins) / (self.maxs - self.mins + 1e-5) # [ 0, 1 ]
        self.normed_observations = (self.normed_observations * 2) - 1 # [ -1, 1 ]
        # self.normed_observations = [self.normed_observations[np.sum(self.path_lengths[:i]) if i>0 else 0 : np.sum(self.path_lengths[:i+1])] for i in range(len(self.path_lengths))]


    def normalize_init(self, init_states):
        """normalize init state"""
        normed_init_states = (np.array(init_states) - self.mins) / (self.maxs - self.mins + 1e-5) # [0,1]
        normed_init_states = (normed_init_states * 2) - 1 # [-1,1]
        return normed_init_states.astype(np.float32)



    def unnormalize(self, x, eps=1e-2):
        ''' the output
            x : [ 0, 1 ]
            x [ horizon x obs_dim ] 
        '''

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


    def __len__(self):
        return self.dataset.__len__()
    
    
    def __getitem__(self, idx, condition_single_input=True):
        obs, actions, policy_id = self.dataset.__getitem__(idx)
        # obs: horizon x agent_num (2) x H x W x C
        # actions: horizon x 2 x action dim (1) 
        # policy : 2 (tuple)
    
        T, _, H, W, C = obs.shape # Time, Agent, Height, Width, Channel 
        obs = self.normalize_init(obs)

        # Get Ego Agent Observation (Agent ID  = 0)
        start = random.randint(1, T - self.horizon)
        end = start + self.horizon
        trajectories = obs[start:end, 0]
        
        # Condition on Past Trajectory or Previous Start State
        conditions_obs = obs[start-1, 0] if condition_single_input else obs[:start, 0]

        # Condition on Partner (Agent ID = 1)
        conditions = policy_id[1]

        # Create a Mask for Valid Condition Observations
        valid_len = 1 if condition_single_input else start
        cond_inputs = np.zeros((self.horizon, H, W, C), dtype=np.float32)
        cond_masks = np.zeros((self.horizon), dtype=np.float32)
        cond_inputs[-valid_len:] = conditions_obs
        cond_masks[-valid_len:] = 1.0

        # Trajectory Shape: (Horizon, H, W, C)
        # Conditions Shape : (1)
        # Condition Inputs: (valid_len, H, W, C)
        # Condition Masks : (valid_len,)

        Batch(trajectories, conditions, self.dummy_cond, cond_inputs, cond_masks)


    # def __getitem__(self, idx, eps=1e-4):
    #     obs, actions, policy_id = self.dataset.__getitem__(idx)
    #     # obs: horizon x 2 x obs_dim
    #     # actions: horizon x 2 x action_dim
    #     # policy_id: 2
    #     # import pdb; pdb.set_trace()
    #     H, W, C = obs.shape[2:]
    #     obs = self.normalize_init(obs)
        
    #     chunk_length, n_agents, action_dim = actions.shape
    #     start = random.randint(1, chunk_length - self.horizon)
    #     context_len = self.horizon
    #     end = start + self.horizon
    #     trajectories = np.concatenate((obs[start:end, 0].reshape(self.horizon, -1), actions[start:end, 0]), axis=-1).astype(np.float32)
    #     policy_pairs = (policy_name_dict[self.dataset.dataset_name][policy_id[0]], policy_name_dict[self.dataset.dataset_name][policy_id[1]])
    #     conditions = self.policy_id[idx][1]
    #     conditions_obs = obs[:start, 0]
    #     # conditions_obs: valid_len x obs_dim
    #     valid_len = start
    #     # put state_flatten here maybe?
    #     cond_inputs = np.zeros((context_len, H, W, C), dtype=np.float32)
    #     cond_masks = np.zeros((context_len), dtype=np.float32)
    #     cond_inputs[-valid_len:] = conditions_obs
    #     cond_masks[-valid_len:] = 1.0

    #     print(f"Lawrence; Utilizing the following policy_id {self.policy_id[idx]}")
    #     batch = Batch(trajectories, conditions, self.dummy_cond, cond_inputs, cond_masks)
    #     return batch
    
    def pad_history(self, unpadded_past_trajectory):

        unpadded_horizon = unpadded_past_trajectory.shape[0]
        unpadded_feature_dim = unpadded_past_trajectory.shape[1:]

        padded_past_trajectory = np.zeros((self.horizon, unpadded_feature_dim))
        padded_past_trajectory[:unpadded_past_trajectory.shape[0], :unpadded_past_trajectory.shape[1]] = unpadded_past_trajectory
        past_trajectory = padded_past_trajectory.flatten().astype(np.float32)
        return past_trajectory

