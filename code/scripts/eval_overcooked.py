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
from gymnasium.wrappers import RecordVideo
from highway_env import register_highway_envs
from mapbt.algorithms.population.policy_pool import PolicyPool as Policy
register_highway_envs()


def eval_overcooked(
    basedir, diffusion, dataset, renderer, dummy_cond, all_cond_features, all_cond_text,
    condition_guidance_w, device
):
    """
    Evaluate the overcooked model.
    """
    



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

    import pdb; pdb.set_trace()

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
