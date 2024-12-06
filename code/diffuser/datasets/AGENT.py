import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from collections import namedtuple
import random

from ..utils.rendering import *
from transformers import T5Tokenizer, T5EncoderModel


def to_tensor(x, dtype=torch.float, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)

Batch = namedtuple('Batch', 'trajectories conditions dummy_cond conditions_obs') #trajectories: output traj, conditions: concept (text embedding), conditions_obs: condition on curr obs (model needs to predict next steps, create batches based on curr obs any t from training demos)

colors = [[255 / 255, 87 / 255, 34 / 255, 1.0],
        [217 / 255, 191 / 255, 119 / 255, 1.0],
        [149 / 255, 56 / 255, 158 / 255, 1.0],
        [148 / 255, 252 / 255, 19 / 255, 1.0]] #red, yellow, purple, green
shapes = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]] #cube, cone, sphere, bowl


has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5EncoderModel.from_pretrained("google/flan-t5-base").to(device)
cond_dummy = tokenizer('', return_tensors="pt").input_ids.to(device)
cond_dummy = model(cond_dummy).last_hidden_state.mean(axis=1)
cond_dummy = cond_dummy.reshape(1,-1)

def get_task(cond):
        cond_feats = cond.reshape(1,-1)
        conds_map_keys = np.array([-0.0582, -0.0518, -0.0756, -0.1059])
        conds_map_vals = ['go to yellow object', 'go to red object', 'go to bowl', 'go to cube']
        cond_text = conds_map_vals[np.abs(conds_map_keys-float(cond_feats[0][0])).argmin()]
        return cond_text

def language_embedding(language_concept):
    concept_tokens = tokenizer(language_concept, return_tensors="pt").input_ids.to(device)
    concept_embedding = model(concept_tokens).last_hidden_state.mean(axis=1)
    return concept_embedding.reshape(1,-1)

def generate_cond(init_s):
    cond_text = random.sample(['go to yellow object', 'go to red object', 'go to bowl', 'go to cube'], 1)[0] #training distribution
    # init_s = copy.deepcopy(dataset.observations[1][0])
    
    # scene_offset=(0, 0, -3.806)
    init_s[24] = random.uniform(1.66, 0) #target 1 pos
    init_s[26] = random.uniform(-3.806-0.549, -3.806+0.549) #target 1 pos
    init_s[38] = random.uniform(0, -1.66) #target 2 pos
    init_s[40] = random.uniform(-3.806-0.549, -3.806+0.549) #target 2 pos    
    if 'yellow' in cond_text or 'red' in cond_text:
        #one is yellow one is red
        c_idx = random.choice([0,1])
        init_s[30:34] = colors[c_idx]
        init_s[44:48] = colors[1-c_idx]
        #shapes can be any combo of cone-sphere #cube, cone, sphere, bowl
        init_s[34:38] = shapes[random.choice([1,2])]
        init_s[48:52] = shapes[random.choice([1,2])]
    #
    elif 'bowl' in cond_text or 'cube' in cond_text:
        #one is bowl one is cube #cube, cone, sphere, bowl
        s_idx = random.choice([0,3])
        init_s[34:38] = shapes[s_idx]
        init_s[48:52] = shapes[list(set([0,3])-set([s_idx]))[0]]
        #colors can be any combo of purple-green
        init_s[30:34] = colors[random.choice([2,3])]
        init_s[44:48] = colors[random.choice([2,3])]
    # t5
    cond_features = tokenizer(cond_text, return_tensors="pt").input_ids.to(device)
    cond_features = model(cond_features).last_hidden_state.mean(axis=1)
    return cond_features, cond_text, init_s.reshape(1,-1)

def generate_cond_eval_concepts(init_s, cond_text):
    # scene_offset=(0, 0, -3.806)
    init_s[24] = random.uniform(1.66, 0) #target 1 pos
    init_s[26] = random.uniform(-3.806-0.549, -3.806+0.549) #target 1 pos
    init_s[38] = random.uniform(0, -1.66) #target 2 pos
    init_s[40] = random.uniform(-3.806-0.549, -3.806+0.549) #target 2 pos    
    if 'yellow' in cond_text or 'red' in cond_text:
        #one is yellow one is red
        c_idx = random.choice([0,1])
        init_s[30:34] = colors[c_idx]
        init_s[44:48] = colors[1-c_idx]
        #shapes can be any combo of cone-sphere #cube, cone, sphere, bowl
        init_s[34:38] = shapes[random.choice([1,2])]
        init_s[48:52] = shapes[random.choice([1,2])]
    elif 'cone' in cond_text or 'sphere' in cond_text:
         #one is yellow one is red
        c_idx = random.choice([0,1])
        init_s[30:34] = colors[c_idx]
        init_s[44:48] = colors[1-c_idx]
        #one is cone one is sphere
        s_idx = random.choice([1,2])
        init_s[34:38] = shapes[s_idx]
        init_s[48:52] = shapes[list(set([1,2])-set([s_idx]))[0]]
    #
    elif 'bowl' in cond_text or 'cube' in cond_text:
        #one is bowl one is cube #cube, cone, sphere, bowl
        s_idx = random.choice([0,3])
        init_s[34:38] = shapes[s_idx]
        init_s[48:52] = shapes[list(set([0,3])-set([s_idx]))[0]]
        #colors can be any combo of purple-green
        init_s[30:34] = colors[random.choice([2,3])]
        init_s[44:48] = colors[random.choice([2,3])]
    elif 'purple' in cond_text or 'green' in cond_text:
        #one is bowl one is cube #cube, cone, sphere, bowl
        s_idx = random.choice([0,3])
        init_s[34:38] = shapes[s_idx]
        init_s[48:52] = shapes[list(set([0,3])-set([s_idx]))[0]]
        #one is purple one is green
        c_idx = random.choice([2,3])
        init_s[30:34] = colors[c_idx]
        init_s[44:48] = colors[list(set([2,3])-set([c_idx]))[0]]
    return cond_text, init_s.reshape(1,-1)

"""go to red bowl"""
def generate_cond_compos(init_s):
    conds_text = ['go to red object', 'go to bowl']
    selected_color = colors[0] #red
    selected_shape = shapes[3] #bowl
    selected_target_num = str(random.randint(1,2)) #which obj is target
    n_set = str(random.randint(1,2)) #what is the other obj going to be sampled from
    # scene_offset=(0, 0, -3.806)
    init_s[24] = random.uniform(1.66, 0) #target 1 pos
    init_s[26] = random.uniform(-3.806-0.549, -3.806+0.549) #target 1 pos
    init_s[38] = random.uniform(0, -1.66) #target 2 pos
    init_s[40] = random.uniform(-3.806-0.549, -3.806+0.549) #target 2 pos    
    if "1" in selected_target_num: #1 is red bowl
        init_s[30:34] = selected_color
        init_s[34:38] = selected_shape
        if "1" in n_set: #2 red cone/sphere
            init_s[44:48] = selected_color
            init_s[48:52] = shapes[random.choice([1,2])]
        elif "2" in n_set: #2 purple/green bowl
            init_s[44:48] = colors[random.choice([2,3])]
            init_s[48:52] = selected_shape
    #            
    else: #2 is red bowl
        init_s[44:48] = selected_color
        init_s[48:52] = selected_shape
        if "1" in n_set: #1 red cone/sphere
            init_s[30:34] = selected_color
            init_s[34:38] = shapes[random.choice([1,2])]
        elif "2" in n_set: #1 purple/green bowl
            init_s[30:34] = colors[random.choice([2,3])]
            init_s[34:38] = selected_shape
    # t5
    conds_features = None
    for cond_text in conds_text:
        cond_features = tokenizer(cond_text, return_tensors="pt").input_ids.to(device)
        cond_features = model(cond_features).last_hidden_state.mean(axis=1)
        conds_features = cond_features if conds_features is None else torch.cat((conds_features,cond_features),0)
    return conds_features, conds_text[0]+ ' and\n' +conds_text[1], init_s.reshape(1,-1)

def one_hot2shape(oh):
    # return {[1,0,0,0]: 'cube', [0,1,0,0]: 'cone', [0,0,1,0]: 'sphere', [0,0,0,1]: 'bowl'}[oh]
    return ['cube', 'cone', 'sphere', 'bowl'][np.argmax(oh)]

def list2color(rgba):
    colors = np.array([[255/255, 87/255, 34/255, 1.0], [217/255, 191/255, 119/255, 1.0], [149/255, 56/255, 158/255, 1.0], [148/255, 252/255, 19/255, 1.0]])
    color_idx = np.argmin(np.linalg.norm(np.array(rgba) - colors, axis=1)) 
    return ['red', 'yellow', 'purple', 'green'][color_idx]

def dist_ag_target(ag_pos, target_pos):
    ag_pos[1] = 0
    target_pos[1] = 0
    return np.linalg.norm(ag_pos - target_pos)

def get_train_acc(samps, inits, conds_text, thres=np.inf):
    n_succ = []
    for samp, init_s, cond_text in zip(samps, inits, conds_text):
        agent_pos_0 = samp[0][:3]
        agent_pos_H = samp[-1][:3]
        #predict only agent pos
        target_1_pos = init_s[24:27]
        target_1_color = list2color(init_s[30:34])
        target_1_shape = one_hot2shape(init_s[34:38])
        target_2_pos = init_s[38:41]
        target_2_color = list2color(init_s[44:48])
        target_2_shape = one_hot2shape(init_s[48:52])
        ag_t1_0 = dist_ag_target(agent_pos_0, target_1_pos)
        ag_t1_H = dist_ag_target(agent_pos_H, target_1_pos)
        ag_t2_0 = dist_ag_target(agent_pos_0, target_2_pos)
        ag_t2_H = dist_ag_target(agent_pos_H, target_2_pos)    
        #target color
        if 'red' in cond_text: chosen_color = 'red'
        elif 'yellow' in cond_text: chosen_color = 'yellow'    
        elif 'purple' in cond_text: chosen_color = 'purple'
        elif 'green' in cond_text: chosen_color = 'green'
        else: chosen_color = None
        #target shape
        if 'bowl' in cond_text: chosen_shape = 'bowl'
        elif 'cube' in cond_text: chosen_shape = 'cube'    
        elif 'cone' in cond_text: chosen_shape = 'cone'  
        elif 'sphere' in cond_text: chosen_shape = 'sphere'    
        else: chosen_shape = None
        #target num
        if target_1_color == chosen_color or target_1_shape == chosen_shape:
            n_succ.append(1) if (ag_t1_H<ag_t1_0 and ag_t1_H < thres) else n_succ.append(0)
        elif target_2_color == chosen_color or target_2_shape == chosen_shape:
            n_succ.append(1) if (ag_t2_H<ag_t2_0 and ag_t2_H < thres) else n_succ.append(0)
        else:
            print('TARGET ERROR')
            exit()
    print(f'train init acc: {np.sum(n_succ)/len(samps)}')
    return np.sum(n_succ)/len(samps)

def get_test_acc(samps, inits, cond_text, thres=np.inf):
    n_succ = []
    for samp, init_s in zip(samps, inits):
        agent_pos_0 = samp[0][:3]
        agent_pos_H = samp[-1][:3]
        #predict only agent pos
        target_1_pos = init_s[24:27]
        target_1_color = list2color(init_s[30:34])
        target_1_shape = one_hot2shape(init_s[34:38])
        target_2_pos = init_s[38:41]
        target_2_color = list2color(init_s[44:48])
        target_2_shape = one_hot2shape(init_s[48:52])
        ag_t1_0 = dist_ag_target(agent_pos_0, target_1_pos)
        ag_t1_H = dist_ag_target(agent_pos_H, target_1_pos)
        ag_t2_0 = dist_ag_target(agent_pos_0, target_2_pos)
        ag_t2_H = dist_ag_target(agent_pos_H, target_2_pos)    
        #target color
        if 'red' in cond_text: chosen_color = 'red'
        elif 'yellow' in cond_text: chosen_color = 'yellow'    
        elif 'purple' in cond_text: chosen_color = 'purple'
        elif 'green' in cond_text: chosen_color = 'green'
        else: chosen_color = None
        #target shape
        if 'bowl' in cond_text: chosen_shape = 'bowl'
        elif 'cube' in cond_text: chosen_shape = 'cube'    
        elif 'cone' in cond_text: chosen_shape = 'cone'    
        elif 'sphere' in cond_text: chosen_shape = 'sphere' 
        else: chosen_shape = None 
        #target num
        if target_1_color == chosen_color and target_1_shape == chosen_shape:
            n_succ.append(1) if (ag_t1_H<ag_t1_0 and ag_t1_H < thres) else n_succ.append(0)
        elif target_2_color == chosen_color and target_2_shape == chosen_shape:
            n_succ.append(1) if (ag_t2_H<ag_t2_0 and ag_t2_H < thres) else n_succ.append(0)
    print(f'test init acc: {np.sum(n_succ)/len(samps)}')
    return np.sum(n_succ)/len(samps)

def get_train_acc_optimal(samps, inits, conds_text, thres=0.365):
    n_succ = []
    for idx, (samp, init_s, cond_text) in enumerate(zip(samps, inits, conds_text)):
        agent_pos_0 = samp[0][:3]
        agent_pos_H = samp[-1][:3]
        #predict only agent pos
        target_1_pos = init_s[24:27]
        target_1_color = list2color(init_s[30:34])
        target_1_shape = one_hot2shape(init_s[34:38])
        target_2_pos = init_s[38:41]
        target_2_color = list2color(init_s[44:48])
        target_2_shape = one_hot2shape(init_s[48:52])
        ag_t1_0 = dist_ag_target(agent_pos_0, target_1_pos)
        ag_t1_H = dist_ag_target(agent_pos_H, target_1_pos)
        ag_t2_0 = dist_ag_target(agent_pos_0, target_2_pos)
        ag_t2_H = dist_ag_target(agent_pos_H, target_2_pos)    
        #target color
        if 'red' in cond_text: chosen_color = 'red'
        elif 'yellow' in cond_text: chosen_color = 'yellow'    
        elif 'purple' in cond_text: chosen_color = 'purple'
        elif 'green' in cond_text: chosen_color = 'green'
        else: chosen_color = None
        #target shape
        if 'bowl' in cond_text: chosen_shape = 'bowl'
        elif 'cube' in cond_text: chosen_shape = 'cube'    
        elif 'cone' in cond_text: chosen_shape = 'cone'  
        elif 'sphere' in cond_text: chosen_shape = 'sphere'    
        else: chosen_shape = None
        #target num
        samp = np.array(samp)
        if target_1_color == chosen_color or target_1_shape == chosen_shape:
            #there wasn't a point where agent was getting closer to other target first 
            optimal = (~((np.linalg.norm(samp[:,[0,2]]-target_2_pos[[0,2]], axis=1) < thres) * \
                         ~(np.linalg.norm(samp[:,[0,2]]-target_1_pos[[0,2]], axis=1) < thres))).all()
            n_succ.append(1) if (ag_t1_H<ag_t1_0 and ag_t1_H < thres and optimal) else n_succ.append(0)
        elif target_2_color == chosen_color or target_2_shape == chosen_shape:
            optimal = (~((np.linalg.norm(samp[:,[0,2]]-target_1_pos[[0,2]], axis=1) < thres) * \
                         ~(np.linalg.norm(samp[:,[0,2]]-target_2_pos[[0,2]], axis=1) < thres))).all()
            n_succ.append(1) if (ag_t2_H<ag_t2_0 and ag_t2_H < thres and optimal) else n_succ.append(0)
    print(f'train init acc: {np.sum(n_succ)/len(samps)}')
    return np.sum(n_succ)/len(samps)

def get_test_acc_optimal(samps, inits, cond_text, thres=0.365):
    n_succ = []
    for idx, (samp, init_s) in enumerate(zip(samps, inits)):
        agent_pos_0 = samp[0][:3]
        agent_pos_H = samp[-1][:3]
        #predict only agent pos
        target_1_pos = init_s[24:27]
        target_1_color = list2color(init_s[30:34])
        target_1_shape = one_hot2shape(init_s[34:38])
        target_2_pos = init_s[38:41]
        target_2_color = list2color(init_s[44:48])
        target_2_shape = one_hot2shape(init_s[48:52])
        ag_t1_0 = dist_ag_target(agent_pos_0, target_1_pos)
        ag_t1_H = dist_ag_target(agent_pos_H, target_1_pos)
        ag_t2_0 = dist_ag_target(agent_pos_0, target_2_pos)
        ag_t2_H = dist_ag_target(agent_pos_H, target_2_pos)    
        #target color
        if 'red' in cond_text: chosen_color = 'red'
        elif 'yellow' in cond_text: chosen_color = 'yellow'    
        elif 'purple' in cond_text: chosen_color = 'purple'
        elif 'green' in cond_text: chosen_color = 'green'
        else: chosen_color = None
        #target shape
        if 'bowl' in cond_text: chosen_shape = 'bowl'
        elif 'cube' in cond_text: chosen_shape = 'cube'    
        elif 'cone' in cond_text: chosen_shape = 'cone'    
        elif 'sphere' in cond_text: chosen_shape = 'sphere' 
        else: chosen_shape = None 
        #target num
        samp = np.array(samp)
        if target_1_color == chosen_color and target_1_shape == chosen_shape:
            optimal = (~((np.linalg.norm(samp[:,[0,2]]-target_2_pos[[0,2]], axis=1) < thres) * \
                         ~(np.linalg.norm(samp[:,[0,2]]-target_1_pos[[0,2]], axis=1) < thres))).all()
            n_succ.append(1) if (ag_t1_H<ag_t1_0 and ag_t1_H < thres and optimal) else n_succ.append(0)
        elif target_2_color == chosen_color and target_2_shape == chosen_shape:
            optimal = (~((np.linalg.norm(samp[:,[0,2]]-target_1_pos[[0,2]], axis=1) < thres) * \
                         ~(np.linalg.norm(samp[:,[0,2]]-target_2_pos[[0,2]], axis=1) < thres))).all()            
            n_succ.append(1) if (ag_t2_H<ag_t2_0 and ag_t2_H < thres and optimal) else n_succ.append(0)
    print(f'test init acc: {np.sum(n_succ)/len(samps)}')
    return np.sum(n_succ)/len(samps)

def plot_traj(traj, init_s, save_fig_path, sample_state=None, cond_text=''):
    target_shapes = ['s', '^', 'o', MarkerStyle('o', fillstyle='bottom')] #'cube', 'cone', 'sphere', 'bowl'
    plt.figure()
    plt.xlim(-2,2)
    plt.ylim(-5,-3)
    plt.plot(traj[:,0],traj[:,2], marker='^')
    plt.plot(traj[0,0], traj[0,2], color='white', marker='^') #indicate agent (cone) direction (init state)
    plt.plot(traj[-1,0], traj[-1,2], color='black', marker='^') #indicate agent (cone) direction (final state)
    plt.plot(init_s[24], init_s[26], color=init_s[30:34], marker=target_shapes[np.argmax(init_s[34:38])]) #target 1
    plt.plot(init_s[38], init_s[40], color=init_s[44:48], marker=target_shapes[np.argmax(init_s[48:52])]) #target 2
    if sample_state is not None:
        plt.plot(sample_state[0], sample_state[2], color='red', marker='^') #indicate s_t_1
    plt.title(cond_text)
    plt.savefig(save_fig_path)
    print('fig saved at ', save_fig_path)


class AGENTSequenceDataset(Dataset):

    def __init__(self, horizon=150, max_path_length=1000, use_padding=True, dataset_path=None, *args, **kwargs):
        self.horizon = horizon #32
        self.max_path_length = max_path_length #150
        self.use_padding = use_padding

        with open(dataset_path, "rb") as input_file:
            self.observations, self.conditions, self.dummy_cond = pickle.load(input_file)
        self.observation_dim = 13
        self.action_dim = 0 # if use later on, force is x,y,z agent force
        self.cond_dim = self.conditions.shape[1] # 768 T5
        self.obs_cond_dim = self.observations.shape[2] # 52

        self.n_episodes = self.observations.shape[0]
        reshaped_obs = self.observations.reshape(self.n_episodes*self.max_path_length, -1)
        
        self.mins = reshaped_obs.min(axis=0)
        self.maxs = reshaped_obs.max(axis=0)        
        self.path_lengths = [obs.shape[0] for obs in self.observations]
        self.indices = self.make_indices(self.path_lengths, horizon)
        self.normalize()


    def normalize(self, keys=['observations']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        self.normed_observations = self.observations.reshape(self.observations.shape[0]*self.max_path_length, -1) #max_path_length 150
        ## [ 0, 1 ]
        self.normed_observations = (self.normed_observations - self.mins) / (self.maxs - self.mins + 1e-5)
        ## [ -1, 1 ]
        self.normed_observations = (self.normed_observations * 2) - 1
        self.normed_observations = self.normed_observations.reshape(self.observations.shape[0], self.max_path_length, -1)
    

    def normalize_init(self, init_states):
        normed_init_states = (init_states - self.mins) / (self.maxs - self.mins + 1e-5) # [ 0, 1 ]        
        normed_init_states = (normed_init_states * 2) - 1 # [ -1, 1 ]
        return normed_init_states


    # def unnormalize(self, x, eps=1e-4):
    def unnormalize(self, x, eps=1e-2):
        '''
            x : [ 0, 1 ]
            x [ horizon x obs_dim ]
        '''
        assert x.max() <= 1.0 + eps and x.min() >= -1.0 - eps, f'x range: ({x.min():.4f}, {x.max():.4f})' 

        mins, maxs = self.mins[:x.shape[-1]], self.maxs[:x.shape[-1]]
        ret = x + 1 #[-1,1]-->[0,2]
        ret /= 2 #[0,2]-->[0,1]
        return ret * (maxs - mins + 1e-5) + mins #[min,max]     


    def unnormalize_out_of_range(self, x, eps=1e-2):
        '''
            x : [ 0, 1 ]
            x [ horizon x obs_dim ]
        '''
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
        # return {0: observations[0]}
        return observations[0]


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        observations = self.normed_observations[path_ind, start:end, :self.observation_dim] #only agent position, rotation, velocity, angular_velocity. everything else is const.
        obs_conditions = self.normed_observations[path_ind, start, :] #full init state
        batch = Batch(observations, self.conditions[path_ind], self.dummy_cond, obs_conditions)
        return batch


    def get_item_render(self, idx=None):
        if idx is None:
            idx = np.random.choice(range(len(self.indices)))
        path_ind, start, end = self.indices[idx]
        gt_observations = self.observations[path_ind, start:end, :] #sub traj gt
        obs_conditions = self.normed_observations[path_ind, start, :] #init state normalized
        cond_feats = self.conditions[path_ind].reshape(1,-1)
        conds_map_keys = np.array([-0.0582, -0.0518, -0.0756, -0.1059])
        conds_map_vals = ['go to yellow object', 'go to red object', 'go to bowl', 'go to cube']
        cond_text = conds_map_vals[np.abs(conds_map_keys-float(cond_feats[0][0])).argmin()]
        return gt_observations, cond_feats, \
                self.dummy_cond.reshape(1,-1), obs_conditions.reshape(1,-1), cond_text
