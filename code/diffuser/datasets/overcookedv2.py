import random
import numpy as np
import torch
import pickle
from collections import namedtuple

from diffuser.datasets.overcooked import PLAYER0_CHANNEL_INDEX, PLAYER0_ORIENT_CHANNELS, PLAYER1_CHANNEL_INDEX, PLAYER1_ORIENT_CHANNELS
from ..utils.rendering import *
import gymnasium as gym
import copy
import math
from scripts.overcooked_sample_renderer import OvercookedSampleRenderer

from transformers import T5Tokenizer, T5EncoderModel

# player identity would be <env_name><policy_name><subpolicy_name><index>
# env_name: "counter_circuit_o_1order" -> 00
# policy_name: "mep" -> 00
# subpolicy_name: "init" "mid" "final" -> 00, 01, 02
# index: 0, 1, 2, ... -> 00, 01, 02, ...
# for example, "counter_circuit_o_1order_mep mep2_final" -> 00000202
policy_name_dict = {"counter_circuit_o_1order_comedi": 
                    ['bc_test', 'bc_test']}


def to_tensor(x, dtype=torch.float, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)

Batch = namedtuple('Batch', 'trajectories conditions dummy_cond conditions_obs') #trajectories: output traj, conditions: concept (text embedding), conditions_obs: condition on curr obs (model needs to predict next steps, create batches based on curr obs any t from training demos)

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5EncoderModel.from_pretrained("google/flan-t5-base").to(device)

class OvercookedSequenceDatasetV2(torch.utils.data.Dataset):

    def __init__(self, args, sample_rate=1):

        dataset_path = args.dataset_path

        if dataset_path.endswith("hdf5"):
            from ..utils.hdf5_dataset import HDF5Dataset
            self.dataset = HDF5Dataset(args, "train") # TODO: change to train later
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
        
        
        self.action_dim = (0)
        self.cond_dim = 8 # Agent ID
        
            
        # D (96) is player_i_features (46), other_player_features (46), player_i_rel_pos (2), player_i_abs_pos (2)
        
        self.n_episodes = len(self.observations)
        
        B, T, N, D = self.observations.shape
        flat_features = [
            self.observations[b, t, 0]
            for b in range(B)
            for t in range(T)
        ]

        self.obs_cond_dim = self.observation_dim = len(flat_features[0]) # initial state dimension
        reshaped_obs = np.vstack(flat_features)
        self.mins = reshaped_obs.min(axis=0)
        self.maxs = reshaped_obs.max(axis=0)

        self.path_lengths = [obs.shape[0] for obs in self.observations[:10,...]]
        self.indices = self.make_indices(self.path_lengths, self.horizon)
        # import pdb; pdb.set_trace()
        # self.normalize()
    
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
        ret = ret * (maxs - mins + 1e-5) + mins #[min,max]
        return np.rint(ret).astype(int)



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
        path_idx, start, end = self.indices[idx]
        obs = self.observations[path_idx]
        actions = self.actions[path_idx]
        policy_id = self.policy_id[path_idx]

        # obs: horizon x H x W x C
        # actions: horizon x 2 x action_dim

        # Get Ego Agent Observation (Agent ID  = 0)
        trajectories = obs[start:end, 0]
        trajectories = self.normalize_init(trajectories)
        
        # Condition on Past Trajectory or Previous Start State
        conditions_obs = obs[start-1, 0] if condition_single_input else obs[:start, 0]

        # Condition on Partner (Agent ID = 1)
        conditions = policy_id[1]

        return Batch(trajectories, conditions, self.dummy_cond, conditions_obs)


    def reconstruct_spatial_tensor(self, flat_features, H=8, W=5, C=26):
        """
        Reconstructs the spatial tensor from the flat features.
        Args:
            flat_features: The flat features to reconstruct. 96 = 46 + 46 + 2 + 2
            H: Height of the spatial tensor.
            W: Width of the spatial tensor.
            C: Number of channels in the spatial tensor.
        Returns:
            The reconstructed spatial tensor.
        """
        state = np.zeros((H, W, C), dtype=np.float32)
        ego_feature_start = 0
        feature_length = 46

        def tuple_to_feature_dict(tup):
            keys_and_lengths = [
                ('p0_orientation', 4),
                ('p0_objs', 4), # "onion", "soup", "dish", "tomato"
                ('p0_closest_onion', 2),
                ('p0_closest_tomato', 2),
                ('p0_closest_dish', 2),
                ('p0_closest_soup', 2),
                ('p0_closest_soup_n_onions', 1),
                ('p0_closest_soup_n_tomatoes', 1),
                ('p0_closest_serving', 2),
                ('p0_closest_empty_counter', 2),
                ('p0_closest_pot_0_exists', 1),
                ('p0_closest_pot_0_is_empty', 1),
                ('p0_closest_pot_0_is_full', 1),
                ('p0_closest_pot_0_is_cooking', 1),
                ('p0_closest_pot_0_is_ready', 1),
                ('p0_closest_pot_0_num_onions', 1),
                ('p0_closest_pot_0_num_tomatoes', 1),
                ('p0_closest_pot_0_cook_time', 1),
                ('p0_closest_pot_0', 2),
                ('p0_closest_pot_1_exists', 1),
                ('p0_closest_pot_1_is_empty', 1),
                ('p0_closest_pot_1_is_full', 1),
                ('p0_closest_pot_1_is_cooking', 1),
                ('p0_closest_pot_1_is_ready', 1),
                ('p0_closest_pot_1_num_onions', 1),
                ('p0_closest_pot_1_num_tomatoes', 1),
                ('p0_closest_pot_1_cook_time', 1),
                ('p0_closest_pot_1', 2),
                ('p0_wall_0', 1),
                ('p0_wall_1', 1),
                ('p0_wall_2', 1),
                ('p0_wall_3', 1),
            ]

            assert len(tup) == 46, f"Expected a 46-tuple, got {len(tup)} elements"

            result = {}
            idx = 0
            for key, length in keys_and_lengths:
                result[key] = tup[idx:idx+length]
                idx += length

            return result
        
        # Extract from flat features
        p0_feature_dict = tuple_to_feature_dict(flat_features[ego_feature_start:ego_feature_start+feature_length])
        p1_feature_dict = tuple_to_feature_dict(flat_features[ego_feature_start+feature_length:ego_feature_start+feature_length*2])


        x0, y0 = map(lambda x: int(np.rint(x)), flat_features[-2:])
        orient0 = np.argmax(p0_feature_dict['p0_orientation'])
        x1_rel_x0, y1_rel_x0 = map(lambda x: int(np.rint(x)), flat_features[-4:-2])
        x1, y1 = x0 + x1_rel_x0, y0 + y1_rel_x0
        orient1 = np.argmax(p1_feature_dict['p0_orientation'])
        

        # Set player 0 location and orientation
        CHANNEL_FEATURE_MAP = OvercookedSampleRenderer.CHANNEL_FEATURE_MAP
        if 0 <= x0 < H and 0 <= y0 < W:
            state[x0, y0, CHANNEL_FEATURE_MAP["player_0_loc"]] = 1.0
            if 0 <= orient0 < 4:
                state[x0, y0, PLAYER0_ORIENT_CHANNELS[orient0]] = 1.0

        # Set player 1 location and orientation
        if 0 <= x1 < H and 0 <= y1 < W:
            state[x1, y1, CHANNEL_FEATURE_MAP["player_1_loc"]] = 1.0
            if 0 <= orient1 < 4:
                state[x1, y1, PLAYER1_ORIENT_CHANNELS[orient1]] = 1.0

        # Set held object for player i
        def set_held_object(feature_dict, x, y):
            held_obj0 = feature_dict['p0_objs']
            # assert sum(held_obj0) <= 1 # cannot held more than one object
            if sum(held_obj0) > 1:
                print("Error: more than one object held", held_obj0)
            if held_obj0[0] == 1:
                state[x, y, CHANNEL_FEATURE_MAP["onions"]] += 1.0
            elif held_obj0[1] == 1:
                state[x, y, CHANNEL_FEATURE_MAP["soup_done"]] += 1.0
            elif held_obj0[2] == 1:
                state[x, y, CHANNEL_FEATURE_MAP["dishes"]] += 1.0
            elif held_obj0[3] == 1:
                state[x, y, CHANNEL_FEATURE_MAP["tomatoes"]] += 1.0
        
        def set_closed_object(feature_dict, x, y):
            # Set closest objects for player 0
            # print(x,y)
            # print(feature_dict['p0_closest_onion'])
            # print(feature_dict['p0_closest_tomato'])
            # print(feature_dict['p0_closest_dish'])
            # print(feature_dict['p0_closest_soup'])
            closest_onion = feature_dict['p0_closest_onion']
            if closest_onion[0] != 0 and closest_onion[1] != 0:
                state[x+closest_onion[0], y+closest_onion[1], CHANNEL_FEATURE_MAP["onions"]] += 1.0
            closest_tomato = feature_dict['p0_closest_tomato']
            if closest_tomato[0] != 0 and closest_tomato[1] != 0:
                state[x+closest_tomato[0], y+closest_tomato[1], CHANNEL_FEATURE_MAP["tomatoes"]] += 1.0
            closest_dish = feature_dict['p0_closest_dish']
            if closest_dish[0] != 0 and closest_dish[1] != 0:
                state[x+closest_dish[0], y+closest_dish[1], CHANNEL_FEATURE_MAP["dishes"]] += 1.0
            closest_soup = feature_dict['p0_closest_soup']
            if closest_soup[0] != 0 and closest_soup[1] != 0:
                state[x+closest_soup[0], y+closest_soup[1], CHANNEL_FEATURE_MAP["soup_done"]] += 1.0
    
            
        set_held_object(p0_feature_dict, x0, y0)
        set_held_object(p1_feature_dict, x1, y1)
        set_closed_object(p0_feature_dict, x0, y0)
        set_closed_object(p1_feature_dict, x1, y1)


        return state
