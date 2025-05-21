import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pdb
import os
import random
from argparse import Namespace
from diffuser.datasets.overcookedv3 import OvercookedSequenceDatasetV3
from termcolor import colored
import numpy as np

torch.manual_seed(0)
# ----- configs ------ # 
num_epochs = 500
batch_size = 1024
lr = 6e-4 
save_interval = 50
eval_interval = 10  # Evaluate every N epochs
dataset_name = 'overcooked'
horizon = 1
train_ratio = 0.8  # 80% training, 20% validation
# ---------------------#

dataset_configs = {'overcooked' : {'loader': 'datasets.OvercookedSequenceDatasetV3',
                    'normalizer': 'GaussianNormalizer',
                    'horizon': 8,
                    'episode_length': 400,
                    'chunk_length': 64,
                    'preprocess_fns': [],
                    'clip_denoised': True,
                    'use_padding': False,
                    'max_path_length': 400,
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
        self.embedding = nn.Embedding(10, 8) # 8 dim embedding
        self.softmax = nn.Softmax(dim=-1)  # Add softmax layer
    
    def forward(self, init_state, cond):
        cond_and_state = torch.cat((init_state, self.embedding(cond)), dim=1)
        logits = self.mlp(cond_and_state)
        return self.softmax(logits)  # Return probabilities

class TrajectoryDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, horizon_max):
        self.base_dataset = base_dataset
        self.horizon_max = horizon_max

    def __len__(self):
        return self.base_dataset.__len__()
    
    def __getitem__(self, idx):

        path_idx, start, end = self.base_dataset.indices[idx]
        # print("path_idx", path_idx, "start", start, "end", end)
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

def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            init_state, cond, next_state = (
                torch.Tensor(batch[0]).float().to(device),
                torch.Tensor(batch[1]).to(device),
                torch.Tensor(batch[2]).float().to(device)
            )
            
            # Get probabilities from model
            action_probs = model(init_state, cond)
            
            # CrossEntropyLoss expects class indices (not one-hot)
            true_classes = torch.argmax(next_state, dim=-1)  # Convert one-hot to class indices
            
            # Calculate loss
            loss = criterion(action_probs, true_classes.long())
            total_loss += loss.item()
            
            # Get predicted class (from probabilities)
            pred_classes = torch.argmax(action_probs, dim=-1)
            
            # Calculate accuracy
            total_correct += (pred_classes == true_classes).sum().item()
            total_samples += true_classes.numel()
    
    avg_loss = total_loss / len(eval_loader)
    accuracy = total_correct / total_samples
    model.train()
    return avg_loss, accuracy

if __name__ == "__main__":
    # Load and split dataset
    dset_cfgs = dataset_configs['overcooked']
    full_dataset = OvercookedSequenceDatasetV3(args=Namespace(**dset_cfgs))
    full_dataset = TrajectoryDatasetWrapper(full_dataset, horizon_max=dset_cfgs['horizon'])
    
    # Split into train and eval set
    train_size = int(train_ratio * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # Model setup
    input_dim = 8 + 1040  # cond + init state (H x W x C)
    hidden_dim = 512
    out_dim = full_dataset.base_dataset.n_actions * horizon
    device = torch.device('cpu')
    model = BC(input_dim, hidden_dim, out_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Checkpoint directory
    ckpts_path = '/Users/carrie/logs/baselines/bc/overcooked/ckpts'
    if not os.path.isdir(ckpts_path):
        os.mkdir(ckpts_path)
    
    # Training loop
    best_eval_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training phase
        for batch in train_loader:
            init_state, cond, next_state = torch.Tensor(batch[0]).float().to(device), torch.Tensor(batch[1]).to(device), torch.Tensor(batch[2]).float().to(device)
            
            optimizer.zero_grad()
            action_probs = model(init_state, cond)
            true_classes = torch.argmax(next_state, dim=-1)  # Convert one-hot to indices
            loss = criterion(action_probs, true_classes.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        average_epoch_loss = epoch_loss / len(train_loader)
        print(colored(f'Epoch [{epoch + 1}/{num_epochs}]', 'magenta') + 
              f', Train Loss: {colored(f"{average_epoch_loss:.4f}", "cyan")}')
        
        # Evaluation phase
        if (epoch + 1) % eval_interval == 0:
            eval_loss, eval_accuracy = evaluate(model, eval_loader, criterion, device)
            print(colored(f'Epoch [{epoch + 1}/{num_epochs}]', 'magenta') + 
                  f', Eval Loss: {colored(f"{eval_loss:.4f} | {eval_accuracy:.4f}", "green")}')
            
            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), os.path.join(ckpts_path, f'best_BC_horizon_{horizon}.pth'))
        
        # Save checkpoints periodically
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(ckpts_path, f'BC_{epoch + 1}_horizon_{horizon}.pth'))