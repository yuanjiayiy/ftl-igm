import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pdb
import os
import random
from argparse import Namespace
from diffuser.datasets.overcookedv3 import OvercookedSequenceDatasetV3
from termcolor import colored
import numpy as np

torch.manual_seed(0)

# ----- configs ------ # 

num_epochs = 4000
batch_size = 1024
lr = 6e-4 
save_interval = 50
dataset_name =  'overcooked'
horizon = 8
# ---------------------#

dataset_configs = {'overcooked' : {'loader': 'datasets.OvercookedSequenceDatasetV3',
                    'normalizer': 'GaussianNormalizer',
                    'horizon': 8,
                    'episode_length': 400,
                    'chunk_length': 64,
                    'preprocess_fns': [],
                    'clip_denoised': True,
                    'use_padding': False,
                    'max_path_length': 41,
                    'dataset_path': "data/overcooked_dataset/counter_circuit_o_1order_test/sp10_dataset.hdf5",
                    }                           
                }

class BC(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(BC, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=out_dim))
        self.embedding = nn.Embedding(10, 8) # 2 agents, 8 dim embedding
    
    def forward(self, init_state, cond):
        cond_and_state = torch.cat((init_state, self.embedding(cond)), dim=1)
        return self.mlp(cond_and_state)

class TrajectoryDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, horizon_max):
        self.base_dataset = base_dataset
        self.horizon_max = horizon_max

    def __len__(self):
        return self.base_dataset.__len__()
    
    def __getitem__(self, idx):

        path_idx, start, end = self.base_dataset.indices[idx]
        obs = self.base_dataset.observations[path_idx]
        action = self.base_dataset.actions[path_idx, start:start+horizon, 0, :]
        policy_id = self.base_dataset.policy_id[path_idx]
        one_hot_action = np.full((horizon, self.base_dataset.n_actions), -1)
        one_hot_action[np.arange(len(action)), action.flatten()] = 1

        # Condition on Past Trajectory or Previous Start State
        vectorized_func = np.vectorize(lambda x: x.flatten(), signature="(h,w,c)->(d)")
        conditions_obs = obs[start-1, 0]
        conditions_obs = vectorized_func(conditions_obs)
        conditions_obs = self.base_dataset.normalize_obs_cond(conditions_obs)

        # Condition on Partner (Agent ID = 1)
        conditions = policy_id[1]

        # if one_hot_action.sum() > -4:
        #     import pdb; pdb.set_trace()

        return (conditions_obs, conditions, one_hot_action.flatten())


dset_cfgs = dataset_configs['overcooked']
dataset = OvercookedSequenceDatasetV3(args=Namespace(**dset_cfgs))
dataset = TrajectoryDatasetWrapper(dataset, horizon_max=dset_cfgs['horizon'])
input_dim = 8 + 1040 #cond + init state (H x W x C)
hidden_dim = 512
out_dim = dataset.base_dataset.n_actions * horizon #ego state

device = torch.device('cpu')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = BC(input_dim, hidden_dim, out_dim).to(device)
model.train()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
ckpts_path = '/Users/carrie/logs/baselines/bc/overcooked/ckpts'
if not os.path.isdir(ckpts_path): os.mkdir(ckpts_path)

if __name__ == "__main__":

    for epoch in range(num_epochs):
        epoch_loss = 0.0 
        for batch in dataloader:
            init_state, cond, next_state = torch.Tensor(batch[0]).float().to(device), torch.Tensor(batch[1]).to(device), torch.Tensor(batch[2]).float().to(device)
            outputs = next_state

            optimizer.zero_grad()
            pred_out = model(init_state, cond)
            print("pred_out", pred_out, "\noutputs", outputs)
            loss = criterion(pred_out, outputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % save_interval == 0: # save checkpoints
        # if epoch == num_epochs-1:   
            torch.save(model.state_dict(), os.path.join(ckpts_path, f'_BC_{epoch + 1}_horizon_{horizon}.pth'))
        
        average_epoch_loss = epoch_loss / len(dataloader)
        print(colored(f'Epoch [{epoch + 1}/{num_epochs}]', 'magenta') + f', Average Loss: {colored(f"{average_epoch_loss:.4f}", "cyan")}')