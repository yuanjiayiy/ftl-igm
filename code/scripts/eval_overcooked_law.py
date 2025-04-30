import sys

# project_root = '/home/law/Workspace/repos/ftl-igm/code'
# if project_root not in sys.path:
#     sys.path.append(project_root)
# mapbt_path = '/home/law/Workspace/repos/ftl-igm/mapbt_package/mapbt'
# if mapbt_path not in sys.path:
#     sys.path.append(mapbt_path)
# overcooked_ai_py_src_path = '/home/law/Workspace/repos/ftl-igm/mapbt_package/mapbt/envs/overcooked/overcooked_berkeley/src/overcooked_ai_py'
# if overcooked_ai_py_src_path not in sys.path:
#     sys.path.append(overcooked_ai_py_src_path)

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
from mapbt.envs.env_wrappers import ChooseSubprocVecEnv
from mapbt.algorithms.population.policy_pool import PolicyPool as Policy
from mapbt.config import get_config
from scripts_utils import Parser
from collections import deque
from overcooked_sample_renderer import OvercookedSampleRenderer

def parse_args(args, parser):
    parser.add_argument("--old_dynamics", default=False, action='store_true', help="old_dynamics in mdp")
    parser.add_argument("--layout_name", type=str, default='counter_circuit_o_1order', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int,
                        default=1, help="number of players")
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

    parser.add_argument("--diffusion_loadpath", type=str, required=True, 
                      help="Path to the diffusion model directory")
    parser.add_argument("--loadbase", type=str, default="/home/law/Workspace/repos/ftl-igm/code/logs",
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
    parser.add_argument("--idm_loadpath", type=str, required=True, 
                      help="Path to the diffusion model directory")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def get_idm_action(current_obs, next_obs, idm_model):
    with th.no_grad():
        logits = idm_model(current_obs, next_obs)
        probs = F.softmax(logits, dim=1)
        action = th.argmax(probs)
    return action

def make_eval_env(all_args, run_dir, nenvs=3):
    def get_env_fn(rank):
        def init_env():
            env = Overcooked(all_args, run_dir, rank=rank)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    return ChooseSubprocVecEnv([get_env_fn(i) for i in range(nenvs)])

def get_agent(population_yaml_path, policy_name, device):
    policy = Policy(None, None, None, None, device=device)
    featurize_type = policy.load_population(population_yaml_path, evaluation=True)
    policy = policy.policy_pool[policy_name]
    feat_type = featurize_type.get(policy_name, 'ppo')
    return policy, feat_type

def full_horizon_eval(args, basedir, diffusion, dataset, idm, policy, device, show_samples=False, eval_episodes=3):
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
    agent_id = 23
    H, W, C = dataset.observation_dim
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

        # Get dummy condition from dataset sample
        dummy_cond = th.tensor(np.stack([sample.dummy_cond] * n_envs, axis=0), device=device)

        # Reset environment
        obs, _, _ = envs.reset([True] * n_envs)

        steps = 0
        done = False
        episode_reward = np.zeros((n_envs, 2))
        max_steps = args.max_steps if hasattr(args, 'max_steps') else 400
        frames = [[obs[i][0]] for i in range(n_envs)]

        # Store the previous observation for conditioning
        prev_ego_obs_norm = np.stack([dataset.normalize_init(obs[e][0]) for e in range(n_envs)], axis=0)
        while not done and steps <= max_steps:
            # Condition Obs is the previous ego obs
            condition_obs = th.tensor(prev_ego_obs_norm, device=device, dtype=th.float32) # Shape: [n_envs, H, W, C]
            with th.no_grad():
                samples = diffusion.p_sample_loop(
                    shape=(n_envs, dataset.horizon, H, W, C),
                    cond=cond,
                    dummy_cond=dummy_cond,
                    cond_obs=condition_obs,
                )

            eval_actions = np.zeros((n_envs, dataset.horizon, 2, 1), dtype=np.int64)  # Assuming shape (envs, horizon, agents, action_dim)

            current_ego_obs_unnorm = np.stack([obs[e][0] for e in range(n_envs)], axis=0)
            for env_i in range(n_envs):
                pred_obs_seq_norm = samples[env_i].cpu().numpy()

                if show_samples:
                    grid = renderer.extract_grid_from_obs(current_ego_obs_unnorm[env_i])
                    pred_dir = osp.join(frames_dir, f"episode_{episode+1}_env_{env_i+1}_step_{steps}_predictions")
                    os.makedirs(pred_dir, exist_ok=True)
                    pred_video_path = osp.join(pred_dir, f"sample_diffusion_trajectory.mp4")
                    for i in range(3):
                        _ = renderer.render_trajectory_video(
                            pred_obs_seq_norm[i],
                            grid, output_dir=pred_dir, video_path=pred_video_path, fps=1,
                        )
                # (Current Ego Obs + Predicted Obs Sequence for Ego)
                full_obs = th.cat([prev_ego_obs_norm[env_i], pred_obs_seq_norm], dim=0)
                for t in range(dataset.horizon+1): # We added prev_ego_obs
                    obs_t = full_obs[t].unsqueeze(0)
                    obs_tp1 = full_obs[t+1].unsqueeze(0)
                    ego_action = get_idm_action(obs_t, obs_tp1, idm, device)
                    eval_actions[env_i, t, 0] = to_np(ego_action)

                
            # Now step through the environment using the 32-step plan
            plan_horizon = min(dataset.horizon, max_steps - steps)
            for t in range(plan_horizon):
                # Get current obs for partner and generate its action
                partner_obs_lst = [obs[e][1] for e in range(n_envs)]
                partner_obs = np.stack(partner_obs_lst, axis=0)

                partner_action = policy.step(
                    partner_obs,
                    [(e, 1) for e in range(n_envs)],
                    deterministic=True,
                )
                eval_actions[:, t, 1] = partner_action  # Fill partner action for step t
                print(f"partner action: ", partner_action)

                # Step env with actions at time t
                step_actions = eval_actions[:, t]
                obs, shared_obs, reward, done, info, aval_actions = envs.step(step_actions)
                
                for e in range(n_envs):
                    frames[e].append(obs[e][0])

                episode_reward += to_np(reward).squeeze(axis=2)
                steps += 1

                # Check for early termination
                done = np.all(done)
                if done or steps >= max_steps:
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
            grid = renderer.extract_grid_from_obs(frames[e][0])
            env_dir = osp.join(video_dir, f"episode_{episode+1}_env_{e+1}")
            os.makedirs(env_dir, exist_ok=True)
            saved_video = renderer.render_trajectory_video(
                frames[e], 
                grid, 
                output_dir=env_dir,
                video_path=osp.join(env_dir, f"trajectory.mp4"),
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
    diffusion_experiment = utils.load_diffusion(
        args.diffusion_loadpath,
        epoch="latest",
        seed=None,
        device=device,
        load_dataset=True,
    )
    diffusion = diffusion_experiment.diffusion
    diffusion.model.eval()
    dataset = diffusion_experiment.dataset
    basedir = osp.join(args.loadbase, args.dataset, args.diffusion_loadpath)

    # MAPT Setup
    os.environ["layout"] = args.layout_name
    args.env_name = "Overcooked"
    population_yaml_path = args.population_yaml_path
    policy, featurize_type = get_agent(population_yaml_path, "sp10_final", "cpu")
    print("featurize_type: ", featurize_type)

    idm_path = args.idm_loadpath
    if os.path.exists(idm_path):
        print(f"Loading IDM model from {idm_path}")
        idm = th.load(idm_path)
        idm_model = InverseDynamicsModel(num_actions=6)
        idm_model.load_state_dict(idm['model'])
        idm_model = idm_model.to(device)
        idm_model.eval()
    else:
        print(f"IDM model not found at {idm_path}, please provide the correct path")
        sys.exit(1)

    results = full_horizon_eval(
        args=args,
        basedir=basedir,
        diffusion=diffusion,
        dataset=dataset,
        idm=idm_model,
        policy=policy,
        device=device,
        show_samples=True,
        eval_episodes=3
    )