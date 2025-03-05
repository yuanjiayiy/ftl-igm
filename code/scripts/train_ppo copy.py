# from scripts_utils import Parser
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

from highway_env import register_highway_envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import wandb
register_highway_envs()

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
        
        def make_env():
            env = gym.make(env_name, render_mode="rgb_array")
            env.env.env.default_config().update({"observation": {"vehicles_count": 9}})
            env = Monitor(env)  # Wrap for logging
            return env
        
        env = DummyVecEnv([lambda: make_env() for _ in range(8)])  # Vectorize

        # Create PPO model
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_highway_tensorboard/", n_steps=128)

        # Train the model
        model.learn(total_timesteps=100000, callback=WandbCallback())

        # Save the model
        model.save("ppo_highway")

        env.close()

# Parallel environments
# Create environment
if __name__ == "__main__":
    # args = Parser().parse_args('plan')
    

    wandb.init(project="ppo_highway_env")
    train_ppo([("highway",1)])

