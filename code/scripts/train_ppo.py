# from scripts_utils import Parser
import argparse
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo
from gymnasium.spaces import Box
import numpy as np
from highway_env import register_highway_envs
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import obs_as_tensor
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
register_highway_envs()

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None, help="Path to save/load the model")
parser.add_argument("--save-freq", type=int, default=10000, help="Frequency to save the model")
parser.add_argument("--eval", action='store_true', help="Eval or not")
parser.add_argument("--mlp", action='store_true', help="MLP policy")
parser.add_argument("--disable-video", action='store_true', help="Disable video recording")
parser.add_argument("--attn", action='store_true', help="Use attention layer in the policy")
parser.add_argument("--eval-vehicles-count", type=int, default=5, help="Number of vehicles in the evaluation environment")
parser.add_argument("--env", type=str, default='highway', help="Name of environment to train on")
parser.add_argument("--expert-data-path", type=str, default=None, help="Path to expert demonstrations")
parser.add_argument("--tensorboard", type=str, default="ppo_highway", help="Path to save tensorboard logs")


args = parser.parse_args()

def make_env(env_name, vehicles_count=5):
    env = gym.make(env_name, render_mode="rgb_array", vehicles_count=vehicles_count)
    assert env.observation_space.shape == (vehicles_count, 7)
    if not args.disable_video:
        video_folder = f"video/{env_name}/{wandb.run.id}"
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)
        env.unwrapped.set_record_video_wrapper(env)
    env = Monitor(env)  # Wrap for logging
    return env

# Load expert demonstrations
def load_expert_data(filepath):
    data = np.load(filepath, allow_pickle=True)
    obs, im, rew, done, trunc, info, idxs, scenario = data
    obs = np.array(obs)[:, :-1, :, :]  # Remove last observation
    obs = obs.reshape(-1, 5, 7)
    action = np.array([[inf['action'] for inf in info_item] for info_item in info]).flatten()
    print(obs.shape, obs[0], action.shape)
    return obs, action

# Behavior Cloning Loss Function
def bc_loss(policy, obs, expert_actions):
    obs_tensor = obs_as_tensor(obs, policy.device)
    dist = policy.get_distribution(obs_tensor)
    logits = dist.distribution.logits
    loss = nn.CrossEntropyLoss()(logits, expert_actions.long())
    return loss

class AttentionLayer(nn.Module):
    """ Self-attention layer for flexible feature extraction """
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        numerator = Q @ K.transpose(-2, -1)
        denominator = x.shape[-1] ** 0.5
        attn_weights = self.softmax(numerator / denominator)
        return attn_weights @ V

class FeatureExtractor(BaseFeaturesExtractor):
    """ Feature extractor with MLP and MaxPooling """
    def __init__(self, observation_space: Box, features_dim: int = 128, attention_layer: bool = False):
        super(FeatureExtractor, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[1]  # Assuming shape is (num_vehicles, feature_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
        if attention_layer:
            self.attn = AttentionLayer(features_dim)
        self.pool = nn.AdaptiveMaxPool1d(1)  # Max pooling over the vehicle dimension
    
    def forward(self, x):
        x = self.mlp(x)  # Shape: (batch_size, num_vehicles, features_dim)
        x = x.permute(0, 2, 1)  # Change to (batch_size, features_dim, num_vehicles)
        if hasattr(self, 'attn'):
            x = self.attn(x)
        else:
            x = self.pool(x).squeeze(-1)  # Pool over the vehicle dimension, resulting in (batch_size, features_dim)
        return x

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0, save_freq=50000, model_path="ppo_highway.zip"):
        super(WandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0
        self.save_freq = save_freq
        self.model_path = model_path
    
    def _on_step(self) -> bool:
        self.episode_rewards.append(self.locals["rewards"])
        # print('step', self.locals["n_steps"], 'rewards', self.locals["rewards"], 'done', self.locals.get('done', None))
        # print("Logger contents:", self.model.logger.name_to_value)

        
        if self.locals.get("dones")[0]:  # Episode ended
            total_reward = sum(self.episode_rewards)
            wandb.log({"total_episode_reward": total_reward, "episode": self.episode_count})
            self.episode_rewards = []  # Reset for next episode
            self.episode_count += 1
        
        logs = {}
        for key, value in self.model.logger.name_to_value.items():
            logs[key] = value
        
        wandb.log(logs)

        # Save model every save_freq timesteps
        if self.num_timesteps % self.save_freq == 0:
            model_save_path = f"logs/ppo_highway_{self.num_timesteps}.zip"
            self.model.save(model_save_path)
            print(f"Model saved at {self.num_timesteps} timesteps")
            artifact = wandb.Artifact(f"ppo_highway_model_{self.num_timesteps}", type="model")
            artifact.add_file(model_save_path)
            wandb.log_artifact(artifact)
            print(f"Model uploaded to Wandb at {self.num_timesteps} timesteps")
            test_env_name = "highway-v1"
            test_env = DummyVecEnv([lambda: make_env(test_env_name, vehicles_count=args.eval_vehicles_count)])
            evaluate_model(self.model, test_env)
            test_env.close()


        return True

def train_ppo(scenarios):

    for scenario_text,version in scenarios:
        SLOWER = 4 if scenario_text != 'intersection' else 0          
        env_name = f"{scenario_text.replace('_','-')}-v{version}"
        wandb.init(project=f"ppo_{env_name}")
        
        if args.eval:
            env = DummyVecEnv([lambda: make_env(env_name)])
        else:
            env = DummyVecEnv([lambda: make_env(env_name) for _ in range(8)])  # Vectorize

        # Create PPO model or load from path
        # Load model if exists, else create a new one
        model_path = args.model_path
        if model_path is not None:
            model = PPO.load(model_path, env=env)
            print(f"Model loaded successfully from {model_path}!")
        else:
            policy_kwargs = dict(
                features_extractor_class=FeatureExtractor,
                features_extractor_kwargs=dict(features_dim=128)
            )
            
            if args.mlp:
                model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./{args.tensorboard}", policy_kwargs=policy_kwargs)
            else:
                model = PPO(ActorCriticPolicy, env, verbose=1, tensorboard_log=f"./{args.tensorboard}", policy_kwargs=policy_kwargs)
            print("No saved model found, training a new one.")

        if args.eval:
            test_env_name = "highway-v1"
            test_env = DummyVecEnv([lambda: make_env(test_env_name, vehicles_count=args.eval_vehicles_count)])
            evaluate_model(model, test_env)

        else:
            if args.expert_data_path is not None:
                # Load expert demonstrations
                expert_obs, expert_actions = load_expert_data(args.expert_data_path)
                expert_actions = torch.tensor(expert_actions, dtype=torch.float32)
                # Pretrain with Behavior Cloning
                optimizer = optim.Adam(model.policy.parameters(), lr=3e-4)
                for epoch in range(5000):  # Number of pretraining steps
                    loss = bc_loss(model.policy, expert_obs, expert_actions)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if epoch % 100 == 0:
                        print(f"BC Loss: {loss.item()}")
                    wandb.log({"BC Loss": {loss.item()}})
                print("Evaluating model after BC pretraining")
                eval_env = DummyVecEnv([lambda: make_env(env_name)])
                evaluate_model(model, eval_env, num_episodes=64)
            # Train the model
            model.learn(total_timesteps=1000000, callback=WandbCallback(save_freq=args.save_freq, model_path=args.model_path))

            # Save the model
            model.save("ppo_highway")

        env.close()

def evaluate_model(model, env, num_episodes=64):
    total_rewards = []
    model.policy.observation_space = env.observation_space # Hack to get around vehicle count mismatch
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = []
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward.append(reward)
        total_rewards.append(sum(episode_reward))
        print(f"Episode {episode + 1}: Total Reward = {sum(episode_reward)}, Episode length = {len(episode_reward)}")
    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

# Parallel environments
# Create environment
if __name__ == "__main__":
    # args = Parser().parse_args('plan')
    
    train_ppo([(args.env,1)])

