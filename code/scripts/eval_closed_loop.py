import sys
from scripts.train_bc import BC, TrajectoryDatasetWrapper, dataset_configs

from overcooked_env.mdp.actions import Action, Direction

project_root = '/Users/carrie/ftl-igm/code'
if project_root not in sys.path:
    sys.path.append(project_root)
mapbt_path = '/Users/carrie/GAMMA-human-ai-collaboration/mapbt'
if mapbt_path not in sys.path:
    sys.path.append(mapbt_path)
overcooked_ai_py_src_path = '/Users/carrie/GAMMA-human-ai-collaboration/mapbt/envs/overcooked/overcooked_berkeley/src/overcooked_ai_py'
if overcooked_ai_py_src_path not in sys.path:
    sys.path.append(overcooked_ai_py_src_path)

import numpy as np
import torch as th
import os
import os.path as osp
import pickle
import warnings
import argparse
import torch.nn.functional as F
warnings.filterwarnings("ignore")
import diffuser.utils as utils
from diffuser.models.inverse_dynamics import InverseDynamicsModel
from diffuser.utils.arrays import to_np, to_torch
from mapbt.envs.overcooked.Overcooked_Env import Overcooked
from mapbt.envs.env_wrappers import *
from mapbt.algorithms.population.policy_pool import PolicyPool as Policy
from mapbt.config import get_config
from scripts_utils import Parser
from collections import deque
from overcooked_sample_renderer import OvercookedSampleRenderer
from einops.einops import rearrange
from diffuser.datasets.overcookedv3 import *

dset_cfgs = dataset_configs['overcooked']
dataset = OvercookedSequenceDatasetV3(args=argparse.Namespace(**dset_cfgs))
dataset = TrajectoryDatasetWrapper(dataset, horizon_max=dset_cfgs['horizon'])
input_dim = 8 + 1040 #cond + init state (H x W x C)
hidden_dim = 512
horizon = 1
out_dim = dataset.base_dataset.n_actions * horizon #ego state

def parse_args(args, parser):
    parser.add_argument("--old_dynamics", default=False, action='store_true', help="old_dynamics in mdp")
    parser.add_argument("--layout_name", type=str, default='counter_circuit_o_1order', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_horizon", type=int, default=2.5e6, help="Shaping factor of potential dense reward.")
    parser.add_argument("--use_phi", default=False, action='store_true', help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")  
    parser.add_argument("--use_hsp", default=False, action='store_true')   
    parser.add_argument("--random_index", default=False, action='store_true')
    parser.add_argument("--use_agent_policy_id", default=False, action='store_true', help="Add policy id into share obs, default False")
    parser.add_argument("--overcooked_version", default="old", type=str, choices=["new", "old"])
    parser.add_argument("--use_detailed_rew_shaping", default=False, action='store_true')
    parser.add_argument("--random_start_prob", default=0., type=float)
    parser.add_argument("--store_traj", default=False, action='store_true')
    # population
    parser.add_argument("--population_yaml_path", type=str, help="Path to yaml file that stores the population info.")
    
    # overcooked evaluation
    parser.add_argument("--agent0_policy_name", type=str, help="policy name of agent 0")
    parser.add_argument("--agent1_policy_name", type=str, help="policy name of agent 1")

    parser.add_argument("--base_model_loadpath", type=str, required=True, 
                      help="Path to the base_model model directory")
    parser.add_argument("--loadbase", type=str, default="logs",
                      help="Base directory for loading models")
    parser.add_argument("--dataset", type=str, default="overcooked",
                      help="Dataset name")
    parser.add_argument("--n_envs", type=int, default=3,
                      help="Number of parallel environments")
    parser.add_argument("--agent_id", type=int, default=5,
                      help="Agent ID for conditioning")
    parser.add_argument("--max_steps", type=int, default=400,
                      help="Maximum steps per episode")
    parser.add_argument("--run_dir", type=str, default="eval_run",
                      help="Directory for evaluation run")
    # parser.add_argument("--idm_loadpath", type=str, required=True, 
    #                   help="Path to the diffusion model directory")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def get_action(current_obs, next_obs):
    current_orientation, current_held, current_loc = np.argmax(current_obs[:4]), np.argmax(current_obs[4:9]), current_obs[9:]
    next_orientation, next_held, next_loc = np.argmax(next_obs[:4]), np.argmax(next_obs[4:9]), next_obs[9:]

    # Check if the next location is the same as the current location
    actions = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST, Action.STAY]
    best_move = Action.STAY
    min_dist = float('inf')

    # Try moving in each direction and see which one gets closest to next_loc
    for direction in actions:
        dx, dy = direction
        new_loc = (current_loc[0] + dx, current_loc[1] + dy)
        dist = (new_loc[0] - next_loc[0]) ** 2 + (new_loc[1] - next_loc[1]) ** 2
        if dist < min_dist:
            min_dist = dist
            best_move = direction

    # Now handle the case where we stayed in place
    if best_move == Action.STAY:
        if current_held != next_held:
            best_move = Action.INTERACT
        elif current_held == next_held and current_orientation != next_orientation:
            # Get the direction that corresponds to the new orientation
            direction = Direction.ALL_DIRECTIONS[next_orientation]
            best_move = direction
    
    print(current_orientation, current_held, current_loc)
    print(next_orientation, next_held, next_loc)
    print(best_move)
    return Action.ACTION_TO_INDEX[best_move]

def make_eval_env(all_args, run_dir, nenvs=3):
    def get_env_fn(rank):
        def init_env():
            env = Overcooked(all_args, run_dir, rank=rank)
            env.seed(all_args.seed * 50000 + rank * random.randint(0, 10000))
            return env
        return init_env
    return ShareDummyVecEnv([get_env_fn(i) for i in range(nenvs)])

def get_agent(population_yaml_path, policy_name, device):
    policy = Policy(None, None, None, None, device=device)
    featurize_type = policy.load_population(population_yaml_path, evaluation=True)
    policy = policy.policy_pool[policy_name]
    feat_type = featurize_type.get(policy_name, 'ppo')
    return policy, feat_type


def arg_max(obs):
    # Assume Obs -> [Batch, Horizon, H, W, C]
    if obs.dim() != 5:
        raise ValueError(f"Expected 5D input (B, T, H, W, C), got {obs.shape}")
    
    B, T, H, W, C = obs.shape

    # Flatten spatial dimensions [B, T, C, H*W]
    flat = rearrange(obs, "b t h w c -> b t c (h w)")

    # Get max indices along spatial dimension
    _, max_idxs = flat.max(dim=-1)  # [B, T, C]

    # Directly scatter 1.0 at max positions
    flat_mask = th.ones_like(flat)*-1
    flat_mask.scatter_(-1, max_idxs.unsqueeze(-1), 1.0)  
    # Reshape back to original dimensions
    peaks = rearrange(flat_mask, "b t c (h w) -> b t h w c", h=H, w=W)

    return peaks



def full_horizon_eval(args, base_model, dataset, policy, device, show_samples=False, eval_episodes=3, basedir="./eval_folder"):
    print(f"Starting Overcooked Evaluation; BaseDir {basedir}")
    video_dir = osp.join(basedir, "videos")
    frames_dir = osp.join(basedir, "frames")
    metrics_dir = osp.join(basedir, "metrics")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    renderer = OvercookedSampleRenderer()

    n_envs = args.n_envs if hasattr(args, 'n_envs') else 3
    envs = make_eval_env(args, run_dir=args.run_dir, nenvs=n_envs)
    
    all_metrics = []
    episode_rewards = []
    # agent_id = args.agent_id if hasattr(args, 'agent_id') else 5
    agent_id = 0 # test for sp10_final
    H, W, C = 8, 5, 26
    sample = dataset.__getitem__(0)
    for episode in range(eval_episodes):
        print(f"Starting episode {episode+1}/{eval_episodes}")

        #Reset Policy
        policy.reset(num_envs=n_envs, num_agents=2)
        for e in range(n_envs):
            policy.register_control_agent(e=e, a=1)

        # Setup diffusion conditioning
        cond = np.full((n_envs,), agent_id, dtype=np.int64)
        cond = th.tensor(cond, device=device)

        # Reset environment
        obs, _, _ = envs.reset()

        steps = 0
        done = False
        episode_reward = np.zeros((n_envs, 2))
        max_steps = args.max_steps if hasattr(args, 'max_steps') else 400
        frames = [[obs[i][0]] for i in range(n_envs)]
        samples_frames = [[] for _ in range(n_envs)]

        # Store the previous observation for conditioning
        grid = renderer.extract_grid_from_obs(obs[0][0])
        while not done and steps <= max_steps:
            print(f"Steps: {steps} / {max_steps}")

            # Setup Condition Obs Based on Obs
            obs_stack = np.stack([dataset.base_dataset.normalize_obs_cond(obs[e][0].flatten()) for e in range(n_envs)], axis=0)
            condition_obs = th.tensor(obs_stack, device=device, dtype=th.float32) # Shape: [n_envs, H x W x C]
            
            assert condition_obs.shape[-1] == 1040 # Double Check

            with th.no_grad():
                samples = base_model(condition_obs, cond)

        
            # We begin with the first ego obs (first obs of the environment)
            ego_obs_stack = np.stack(obs)
            obs_t = [extract_flat_features(ego_obs_stack[i][0]) for i in range(n_envs)]
            obs_t = to_torch(obs_t) # 3,D

            plan_horizon = 1
            for t in range(plan_horizon):
            
                step_actions = np.zeros((n_envs, 2, 1), dtype=np.int64)

                for env_i in range(n_envs):
                    one_hot_action = samples[env_i].reshape(-1, dataset.base_dataset.n_actions)[t]
                    ego_action = np.argmax(one_hot_action)
                    step_actions[env_i, 0] = to_np(ego_action)

                        
                partner_obs_lst = [obs[e][1] for e in range(n_envs)]
                partner_obs = np.stack(partner_obs_lst, axis=0)

                partner_action = policy.step(
                    partner_obs,
                    [(e, 1) for e in range(n_envs)],
                    deterministic=True,
                )

                step_actions[:, 1] = partner_action  # Fill partner action for step t

                print(step_actions)

                obs, shared_obs, reward, done, info, aval_actions = envs.step(step_actions)
                episode_reward += to_np(reward).squeeze(axis=2)
                    
                for e in range(n_envs):
                    frames[e].append(obs[e][0])

                # Check for early termination
                done = np.all(done)
                steps += 1
                if done or steps >= max_steps:
                    print(f"done = {done}, steps = {steps}")
                    break
        mean_episode_reward = episode_reward.mean(axis=0)
        print(f"Episode {episode+1} complete: steps={steps}, reward={mean_episode_reward}")
        metrics = {
            'episode': episode,
            'steps': steps,
            'rewards': episode_reward.tolist(),
            'mean_reward': mean_episode_reward.tolist(),
            'total_reward': episode_reward.sum().tolist()
        }
        all_metrics.append(metrics)
        episode_rewards.append(mean_episode_reward)
        with open(osp.join(metrics_dir, f"episode_{episode+1}_metrics.pkl"), 'wb') as f:
            pickle.dump(metrics, f)
        
        for e in range(n_envs):
            frames[e] = rearrange(frames[e], 't w h c -> t h w c')
            grid = renderer.extract_grid_from_obs(frames[e][0])
            env_dir = osp.join(video_dir, f"episode_{episode+1}_env_{e+1}")
            os.makedirs(env_dir, exist_ok=True)
            saved_video = renderer.render_trajectory_video(
                frames[e], 
                grid, 
                output_dir=env_dir,
                video_path=osp.join(env_dir, f"actual_trajectory.mp4"),
                fps=1)
            print(f"Video saved to {saved_video}")
    
    if episode_rewards:
        mean_reward = np.mean([r[0] for r in episode_rewards])  # Agent 0 rewards
        std_reward = np.std([r[0] for r in episode_rewards])
        coop_mean_reward = np.mean([r[1] for r in episode_rewards])  # Agent 1 rewards
        coop_std_reward = np.std([r[1] for r in episode_rewards])
        total_mean = np.mean([np.sum(r) for r in episode_rewards])  # Total team rewards
    else:
        mean_reward = std_reward = coop_mean_reward = coop_std_reward = total_mean = 0.0

    print(f"Evaluation complete!")
    print(f"Agent 0 (Diffusion+IDM) mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Agent 1 (Partner) mean reward: {coop_mean_reward:.2f} ± {coop_std_reward:.2f}")
    print(f"Team total mean reward: {total_mean:.2f}")
    
    # Save all metrics to file
    summary = {
        'args': args,
        'metrics': all_metrics,
        'episode_rewards': episode_rewards,
        'agent0_mean_reward': mean_reward,
        'agent0_std_reward': std_reward,
        'agent1_mean_reward': coop_mean_reward,
        'agent1_std_reward': coop_std_reward,
        'team_mean_reward': total_mean,
        'environment': args.layout_name,
        'agent_id': agent_id,
        'episodes': eval_episodes,
        'steps_per_episode': steps / max(1, eval_episodes)
    }
    
    metrics_path = osp.join(basedir, "eval_summary.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(summary, f)
    print(f"Summary metrics saved to {metrics_path}")
    envs.close()
    return summary
                

if __name__ == "__main__":
    parser = get_config()
    args = sys.argv[1:]
    args = parse_args(args, parser)
    device = th.device('cpu' if not th.cuda.is_available() else 'cuda')

    # load diffusion model function from disk
    base_model = BC(input_dim, hidden_dim, out_dim).to(device)
    base_model.load_state_dict(torch.load(args.base_model_loadpath, map_location=device))
    # diffusion = diffusion_experiment.ema
    # basedir = osp.join(args.loadbase, args.dataset, args.diffusion_loadpath)

    # MAPT Setup
    os.environ["layout"] = args.layout_name
    args.env_name = "Overcooked"
    population_yaml_path = args.population_yaml_path
    policy, featurize_type = get_agent(population_yaml_path, "sp10_final", "cpu")
    print("featurize_type: ", featurize_type)

    results = full_horizon_eval(
        args=args,
        base_model=base_model,
        dataset=dataset,
        policy=policy,
        device=device,
        show_samples=True,
        eval_episodes=1)