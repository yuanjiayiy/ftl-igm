# from scripts_utils import Parser
import argparse
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import numpy as np
from highway_env import register_highway_envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import wandb
register_highway_envs()

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None, help="Path to save/load the model")
parser.add_argument("--save-freq", type=int, default=10000, help="Frequency to save the model")
parser.add_argument("--eval", action='store_true', help="Eval or not")
args = parser.parse_args()

class PadObservation(ObservationWrapper):
    def __init__(self, env, target_shape):
        super().__init__(env)
        orig_shape = env.observation_space.shape
        assert len(orig_shape) == len(target_shape), "Target shape must match original shape dimensions"
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=target_shape, dtype=np.float32
        )
        self.orig_shape = orig_shape
        self.target_shape = target_shape

    def observation(self, obs):
        padded_obs = np.zeros(self.target_shape, dtype=np.float32)
        slices = tuple(slice(0, s) for s in self.orig_shape)
        padded_obs[slices] = obs
        return padded_obs

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0, save_freq=10000, model_path="ppo_highway.zip"):
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
        
        wandb.log(logs, step=self.num_timesteps)


        # Save model every save_freq timesteps
        if self.num_timesteps % self.save_freq == 0:
            model_save_path = f"logs/ppo_highway_{self.num_timesteps}.zip"
            self.model.save(model_save_path)
            print(f"Model saved at {self.num_timesteps} timesteps")
            artifact = wandb.Artifact(f"ppo_highway_model_{self.num_timesteps}", type="model")
            artifact.add_file(model_save_path)
            wandb.log_artifact(artifact)
            print(f"Model uploaded to Wandb at {self.num_timesteps} timesteps")

        return True

def train_ppo(scenarios):

    for scenario_text,version in scenarios:
        SLOWER = 4 if scenario_text != 'intersection' else 0          
        env_name = f"{scenario_text.replace('_','-')}-v{version}"
        wandb.init(project=f"ppo_{env_name}")
        
        def make_env():
            env = gym.make(env_name, render_mode="rgb_array")
            if "padded" in env_name:
                # Get the original shape and define a new padded shape
                orig_shape = env.observation_space.shape
                padded_shape = (orig_shape[0] + 5, orig_shape[1])  # Example padding
                env = PadObservation(env, padded_shape)
            env = Monitor(env)  # Wrap for logging
            return env
        
        if args.eval:
            env = DummyVecEnv([lambda: make_env()])
        else:
            env = DummyVecEnv([lambda: make_env() for _ in range(8)])  # Vectorize

        # Create PPO model or load from path
        # Load model if exists, else create a new one
        model_path = args.model_path
        if model_path is not None:
            model = PPO.load(model_path, env=env)
            print(f"Model loaded successfully from {model_path}!")
        else:
            model = PPO("MlpPolicy", env, verbose=1, n_steps=128, tensorboard_log="./ppo_highway_tensorboard/")
            print("No saved model found, training a new one.")

        if args.eval:
            evaluate_model(model, env)

        else:
            # Train the model
            model.learn(total_timesteps=100000, callback=WandbCallback(save_freq=args.save_freq, model_path=args.model_path))

            # Save the model
            model.save("ppo_highway")

        env.close()

def evaluate_model(model, env, num_episodes=64):
    total_rewards = []
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
    
    train_ppo([("highway-padded",1)])

