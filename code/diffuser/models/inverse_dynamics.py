import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)

class InverseDynamicsModel(nn.Module):
    def __init__(self, input_channels=52, hidden_channels=64, mlp_hidden=128, num_actions=1):
        super().__init__()
        # Initial 3x3 conv
        self.conv_in = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)

        # 3 residual 3x3 conv layers
        self.res_blocks = nn.Sequential(
            ResidualConvBlock(hidden_channels),
            ResidualConvBlock(hidden_channels),
            ResidualConvBlock(hidden_channels),
        )

        # MLP after global mean pooling
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, num_actions)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, obs, next_obs):
        # Input: obs and next_obs with shape (B, H, W, C)
        x = torch.cat([obs, next_obs], dim=-1)  # (B, H, W, 2C)
        x = x.permute(0, 3, 1, 2)  # (B, 2C, H, W)

        x = self.conv_in(x)
        x = self.res_blocks(x)

        # Global mean pooling across spatial dimensions
        x = x.mean(dim=[2, 3])  # (B, hidden_channels)

        return self.mlp(x)  # (B, num_actions)
    
    def loss(self, obs, next_obs, action):  
        target = action[:, 0]
        logits = self(obs, next_obs)
        # Use binary cross-entropy loss for action prediction
        loss = self.criterion(logits, target)  # Assuming action is of shape (B, 1)
        
        # Accuracy computation
        preds = logits.argmax(dim=1)          # (B,)
        correct = (preds == target).sum()
        accuracy = correct.item() / target.size(0)

        return loss, {"accuracy": accuracy}
    