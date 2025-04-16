import sys
import numpy as np
import torch as th
import os
import os.path as osp
import pickle
import warnings
warnings.filterwarnings("ignore")
from scripts_utils import Parser
import diffuser.utils as utils
from diffuser.utils.arrays import to_np
from diffuser.datasets import object_rearrangement
from diffuser.datasets import AGENT
from AGENT_env import AGENT_env
from diffuser.datasets import highway
import gymnasium as gym
from mapbt.config import get_config
from gymnasium.wrappers import RecordVideo
from highway_env import register_highway_envs
from mapbt.algorithms.population.policy_pool import PolicyPool as Policy
register_highway_envs()

from mapbt.envs.overcooked.Overcooked_Env import Overcooked
from mapbt.envs.env_wrappers import ChooseSubprocVecEnv

def parse_args(args, parser):
    parser.add_argument("--old_dynamics", default=False, action='store_true', help="old_dynamics in mdp")
    parser.add_argument("--layout_name", type=str, default='cramped_room', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
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

    all_args = parser.parse_known_args(args)[0]

    return all_args

def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                env = Overcooked(all_args, run_dir, rank=rank)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def eval_overcooked(
    basedir,
    diffusion,
    dataset,
    renderer,
    proxy_policy,
    condition_guidance_w,
    device,
    
):
    """
    Evaluate the overcooked model.
    """
    parser = get_config()
    args = sys.argv[1:]
    all_args = parse_args(args, parser)
    # import pdb; pdb.set_trace()

    # assert all_args.algorithm_name == "population"
    run_dir = '/mmfs1/gscratch/cse/jiayiy9/GAMMA-human-ai-collaboration/mapbt/scripts/results/Overcooked/counter_circuit_o_1order/population/eval-comedi_oracle-proxy/run14'
    print(all_args)
    
    eval_envs = make_eval_env(all_args, run_dir=run_dir)
    obs, _, _ = eval_envs.reset()
    import pdb; pdb.set_trace()
    
    # TODO: implement this function
    # Pretrained proxy_policy is used to generate the condition features
    action = proxy_policy.predict(obs)
    


    all_samples = []
    for idx, (cond_features, init_s) in enumerate(zip(all_cond_features, all_cond_init)):  
        samples = diffusion(
                cond=th.tensor(cond_features.reshape(1,-1)).to(device),
                dummy_cond=th.tensor(dummy_cond).to(device),
                cond_obs=th.tensor(init_s.reshape(1,-1)).to(device),
            )
        action = samples.trajectories[1]
        step(action)
        all_samples.append(dataset.unnormalize(to_np(samples.trajectories)).squeeze())
        if idx >= n_samples_plot: break
    # save, render
    eval_dir = osp.join(basedir, f'eval_train_w_{condition_guidance_w}')
    if not osp.isdir(eval_dir): os.makedirs(eval_dir)    
    eval_train_gen_path = osp.join(eval_dir, 'eval_train.pkl')
    with open(eval_train_gen_path, 'wb') as f: pickle.dump([all_samples, all_cond_text], f)
    savenames = [f'gen-{i}' for i in range(n_samples_plot)]
    renderer.composite(eval_dir, savenames, np.array(all_samples)[:n_samples_plot], np.array(all_cond_text)[:n_samples_plot])


    



if __name__ == "__main__":
    args = Parser().parse_args('plan')
    device = th.device('cpu' if not th.cuda.is_available() else 'cuda')

    # load bc proxy
    print("population_yaml_path: ", args.population_yaml_path)
    policy = Policy(None, None, None, None, device=device)
    featurize_type = policy.load_population(args.population_yaml_path, evaluation=True)
    # policy.policy_pool['proxy'] is EvalPolicy object
    proxy_policy = policy.policy_pool['proxy']
    # proxy.step is a function that takes in a batch of observations and returns a batch of action

    print("featurize_type: ", featurize_type)

    # load diffusion model function from disk
    diffusion_experiment = utils.load_diffusion(
        args.loadbase, args.dataset, args.diffusion_loadpath,
        epoch=args.diffusion_epoch, seed=args.seed,
    )
    diffusion = diffusion_experiment.diffusion
    diffusion.model.eval()
    dataset = diffusion_experiment.dataset
    renderer = diffusion_experiment.renderer    

    # results path
    basedir = osp.join(args.loadbase, args.dataset, args.diffusion_loadpath)

    # sample from the base model, save, render and get accuracy
    diffusion.condition_guidance_w = args.condition_guidance_w
    if args.dataset == 'object_rearrangement':
        with open(f'data/{args.dataset}/eval_train.pkl', 'rb') as input_file: all_gt, all_cond_features, all_cond_text, dummy_cond = pickle.load(input_file)
        one_step_object_rearrangement(basedir, diffusion, dataset, renderer, dummy_cond, all_cond_features, all_cond_text, args.condition_guidance_w, device)
    elif args.dataset == 'AGENT':
        with open(f'data/{args.dataset}/eval_train.pkl', 'rb') as input_file: all_gt, all_cond_features, all_cond_init, all_cond_text, dummy_cond = pickle.load(input_file)
        open_loop_AGENT(basedir, diffusion, dataset, renderer, dummy_cond, all_cond_features, all_cond_init, all_cond_text, args.condition_guidance_w, device)
        closed_loop_AGENT(basedir, diffusion, dataset, renderer, dummy_cond, all_cond_features, all_cond_init, all_cond_text, args.condition_guidance_w, device)
    elif args.dataset == 'mocap':
        diffusion = diffusion_experiment.ema
        with open(f'data/{args.dataset}/train_gt.pkl', "rb") as input_file: _, all_cond_features, all_cond_init, all_cond_text, dummy_cond = pickle.load(input_file)        
        open_loop_mocap(basedir, diffusion, dataset, renderer, dummy_cond, all_cond_features, all_cond_init, all_cond_text, args.condition_guidance_w, device)
    elif args.dataset == 'highway':
        closed_loop_highway(osp.join(basedir, f'eval_train_w_{args.condition_guidance_w}'), diffusion, dataset, renderer, [("exit",1), ("highway",1), ("intersection",2), ("merge",1)], device, args.n_concepts)
    elif args.dataset == 'robot':
        open_loop_robot(basedir, diffusion, dataset, renderer, args.condition_guidance_w, device)
    elif args.dataset == 'overcooked':
        eval_overcooked(basedir, diffusion, dataset, renderer, proxy_policy, args.condition_guidance_w, device)
