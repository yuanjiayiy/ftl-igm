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
register_highway_envs()


def one_step_object_rearrangement(basedir, diffusion, dataset, renderer, dummy_cond, all_cond_features, all_cond_text, condition_guidance_w, device, mode='train', n_samples_plot=8):
    all_samples = []
    for cond_features in all_cond_features:
        samples = diffusion.p_sample_loop(
            shape=(1, dataset.observation_dim),
            cond=th.tensor(cond_features.reshape(1,-1)).to(device),
            dummy_cond=dummy_cond.to(device),
            compose=False
        )
        all_samples.append(dataset.unnormalize(to_np(samples.trajectories)).squeeze())
    # save, render, eval
    eval_dir = os.path.join(basedir, f'eval_{mode}')
    eval_gen_path = os.path.join(eval_dir, f'eval_{mode}_w_{condition_guidance_w}.pkl')
    if not os.path.isdir(os.path.join(eval_dir)): os.makedirs(eval_dir)
    render_path = os.path.join(eval_dir, f'eval_{mode}_w_{condition_guidance_w}.png')        
    with open(eval_gen_path, 'wb') as f: pickle.dump([all_samples, all_cond_text], f)
    renderer.composite(render_path, np.array(all_samples)[:n_samples_plot], np.array(all_cond_text)[:n_samples_plot])
    object_rearrangement.get_train_acc(all_samples, all_cond_text)

def open_loop_AGENT(basedir, diffusion, dataset, renderer, dummy_cond, all_cond_features, all_cond_init, all_cond_text, condition_guidance_w, device, n_samples_plot=8):  
    all_samples, all_inits = [], []
    for cond_features, cond_init in zip(all_cond_features, all_cond_init): 
        samples = diffusion.p_sample_loop(
            shape=(1, dataset.horizon, dataset.observation_dim),
            cond=th.tensor(cond_features).to(device),
            dummy_cond=dummy_cond.to(device),
            cond_obs=th.tensor(cond_init).to(device),
            compose=False
        )
        all_samples.append(dataset.unnormalize(to_np(samples.trajectories)).squeeze())
        all_inits.append(dataset.unnormalize(cond_init).squeeze())
    # save, render, eval
    eval_dir = osp.join(basedir, 'eval_train_open_loop')
    eval_train_gen_path = osp.join(eval_dir, f'eval_train_w_{condition_guidance_w}.pkl')
    if not osp.isdir(osp.join(eval_dir)): os.makedirs(eval_dir)
    render_path = osp.join(eval_dir, f'eval_train_w_{condition_guidance_w}.png')
    with open(eval_train_gen_path, 'wb') as f: pickle.dump([all_samples, all_inits, all_cond_text], f)
    renderer.composite(render_path, np.array(all_samples)[:n_samples_plot], np.array(all_cond_text)[:n_samples_plot], np.array(all_inits)[:n_samples_plot])
    AGENT.get_train_acc(all_samples, all_inits, all_cond_text)

def closed_loop_AGENT(basedir, diffusion, dataset, renderer, dummy_cond, all_cond_features, all_cond_init, all_cond_text, condition_guidance_w, device, n_samples_plot=2):
    eval_dir = osp.join(basedir, 'eval_train_closed_loop')
    env = AGENT_env.PyAgentEnv()
    if not osp.isdir(eval_dir): os.makedirs(eval_dir)
    all_samples, all_inits = [], []
    for init_s_idx, (cond_features, cond_init, cond_text) in enumerate(zip(all_cond_features, all_cond_init, all_cond_text)): 
        plot_dir = osp.join(eval_dir, f'acts_{init_s_idx}')
        if not osp.isdir(plot_dir): os.mkdir(plot_dir)    
        init_s = dataset.unnormalize(cond_init).squeeze()
        all_inits.append(init_s)
        env.load_scene(init_s)
        final_traj = [env._get_obs().astype(np.float32)]
        env.mode = 'train'
        env.cond_text = cond_text
        for t in range(1,dataset.horizon):
            init_s = th.tensor(dataset.normalize_init(final_traj[-1]).reshape(1,-1)).to(device) #current obs
            with th.no_grad():
                samples = diffusion.p_sample_loop(
                    shape=(1, dataset.horizon, dataset.observation_dim),
                    cond=th.tensor(cond_features).to(device),
                    dummy_cond=dummy_cond.to(device),
                    cond_obs=init_s,
                    compose=False,
                )
            s_t_unnorm = final_traj[-1][:dataset.observation_dim]
            s_t_1_unnorm = dataset.unnormalize(to_np(samples.trajectories)[0][min(t+5,dataset.horizon-1)]).squeeze()
            pred_force = (s_t_1_unnorm - s_t_unnorm)[[0,2]]
            obs, done = env.step(pred_force)
            final_traj.append(obs.astype(np.float32))
            AGENT.plot_traj(np.array(final_traj), final_traj[0], osp.join(plot_dir,f'inv_model_obs.png'), cond_text=cond_text) #updating obs
            AGENT.plot_traj(dataset.unnormalize(to_np(samples.trajectories)).squeeze(), final_traj[0], osp.join(plot_dir,f'inv_model_diffusion_{t}.png'), s_t_1_unnorm, cond_text=cond_text) #diffusion guidance
            if done: break
        all_samples.append(final_traj)
        if init_s_idx >= n_samples_plot: break
    # save, render, eval
    eval_train_gen_path = osp.join(eval_dir, f'eval_train_w_{condition_guidance_w}_acts.pkl')
    with open(eval_train_gen_path, 'wb') as f: pickle.dump([all_samples, all_inits, all_cond_text], f)
    render_path = osp.join(eval_dir, f'eval_train_w_{condition_guidance_w}_acts.png')
    renderer.composite(render_path, [np.array(samp) for samp in all_samples][:n_samples_plot], np.array(all_cond_text)[:n_samples_plot], np.array(all_inits)[:n_samples_plot])
    AGENT.get_train_acc(all_samples, all_inits, all_cond_text)

def open_loop_mocap(basedir, diffusion, dataset, renderer, dummy_cond, all_cond_features, all_cond_init, all_cond_text, condition_guidance_w, device, n_samples_plot=5):
    all_samples = []
    for idx, (cond_features, init_s) in enumerate(zip(all_cond_features, all_cond_init)):  
        samples = diffusion(
                cond=th.tensor(cond_features.reshape(1,-1)).to(device),
                dummy_cond=th.tensor(dummy_cond).to(device),
                cond_obs=th.tensor(init_s.reshape(1,-1)).to(device),
            )        
        all_samples.append(dataset.unnormalize(to_np(samples.trajectories)).squeeze())
        if idx >= n_samples_plot: break
    # save, render
    eval_dir = osp.join(basedir, f'eval_train_w_{condition_guidance_w}')
    if not osp.isdir(eval_dir): os.makedirs(eval_dir)    
    eval_train_gen_path = osp.join(eval_dir, 'eval_train.pkl')
    with open(eval_train_gen_path, 'wb') as f: pickle.dump([all_samples, all_cond_text], f)
    savenames = [f'gen-{i}' for i in range(n_samples_plot)]
    renderer.composite(eval_dir, savenames, np.array(all_samples)[:n_samples_plot], np.array(all_cond_text)[:n_samples_plot])


def closed_loop_highway(eval_dir, diffusion, dataset, renderer, scenarios, device, n_concepts, mode='train', cond=None, n_samples_plot=8):
    if not osp.isdir(eval_dir): os.makedirs(eval_dir)
    n_demos_eval = 1 #per scenario
    for scenario_text,version in scenarios:
        SLOWER = 4 if scenario_text != 'intersection' else 0          
        env_name = f"{scenario_text.replace('_','-')}-v{version}"
        env_save_dir = f"{eval_dir}/{scenario_text}_{version}"
        video_folder = f"{env_save_dir}/videos"
        demos_pkl = f"{env_save_dir}/eval_{mode}.pkl"
        if mode=='train': 
            cond = dataset.generate_representation(scenario_text)
            cond=th.tensor(cond.reshape(1,-1)).to(device)
        else:
            cond = th.tensor(cond).to(device)
        # Make env
        env = gym.make(env_name, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True) #default w/o episode_trigger: stop recording on terminated or truncated, causes issue with recorder.
        env.unwrapped.set_record_video_wrapper(env)
        # Run episodes
        trajs_obs, trajs_im, trajs_rew, trajs_done, trajs_trunc, trajs_info = [], [], [], [], [], []
        traj_num = 0
        while traj_num < n_demos_eval:
            (obs, info), done, truncated = env.reset(), False, False
            traj_obs, traj_im, traj_rew, traj_done, traj_trunc, traj_info = [obs], [], [], [], [], []
            all_inits = [obs]
            t = 0 # timestep
            while not (done or truncated):
                ######################
                # diffusion next state estimation
                init_s = th.tensor(dataset.normalize_init(traj_obs[-1]).flatten().reshape(1,-1)).to(device) #current obs
                with th.no_grad():
                    samples = diffusion.p_sample_loop(
                        shape=(1, dataset.horizon, dataset.observation_dim),
                        cond=cond,
                        dummy_cond=th.tensor(dataset.dummy_cond.reshape(1,-1)).to(device),
                        cond_obs=init_s,
                        compose=True if mode!='train' and n_concepts > 1 else False,
                        )
                s_t_1_unnorm = dataset.unnormalize(to_np(samples.trajectories)[0][min(t+1,dataset.horizon-1)]).squeeze() #future step
                # plot guidance
                guidance_dir = osp.join(env_save_dir, f'guidance_{traj_num}')
                if not osp.isdir(guidance_dir): os.mkdir(guidance_dir)
                highway.plot_traj(dataset.unnormalize(to_np(samples.trajectories)).squeeze(), traj_obs[-1], osp.join(guidance_dir, f'inv_model_diffusion_{t}.png'), dataset.n_vehicles, dataset.feat_dim, s_t_1_unnorm, cond_text=scenario_text) #diffusion guidance
                # inverse planning
                env_obs = env.observation_type.observe()
                inv_planning_obs = [] #n acts x n vehicles x features
                inv_planning_crashed = []
                for act in list(range(env.action_space.n)):
                    env_tmp = highway.safe_deepcopy_env(env.unwrapped)
                    env_tmp_obs = env_tmp.observation_type.observe()
                    assert (np.array(env_obs) == np.array(env_tmp_obs)).all()
                    obs, reward, done, truncated, info = env_tmp.step(act)
                    inv_planning_obs.append(obs)
                    inv_planning_crashed.append(info['crashed'])
                    assert (np.array(env_obs) == np.array(env.observation_type.observe())).all()
                    del env_tmp
                inv_planning_crashed = np.array(inv_planning_crashed)
                sim_vs_pred_states = np.linalg.norm(np.array(inv_planning_obs)[:,0,:]-s_t_1_unnorm, axis=1)
                if inv_planning_crashed.all():
                    best_act = SLOWER
                else:
                    best_act = np.random.choice(np.flatnonzero(sim_vs_pred_states == sim_vs_pred_states[~inv_planning_crashed].min())) #tie breaker best act not crashed (select act that gets us closest to diffusion pred w/o crashing)
                ######################
                obs, reward, done, truncated, info = env.step(best_act)
                traj_obs.append(obs)
                traj_rew.append(reward)
                traj_done.append(done)
                traj_trunc.append(truncated)
                traj_info.append(info) #speed, crashed, act
                img = env.render()
                traj_im.append(img)
                t += 1
            trajs_obs.append(traj_obs) # save as list, not numpy, not all same horizon (done or truncated)
            trajs_rew.append(traj_rew)
            trajs_done.append(traj_done)
            trajs_trunc.append(traj_trunc)
            trajs_info.append(traj_info) #has action
            trajs_im.append(traj_im)            
            traj_num += 1 
        # save, render, eval
        with open(demos_pkl, 'wb') as f: pickle.dump([trajs_obs, trajs_im, trajs_rew, trajs_done, trajs_trunc, trajs_info, scenario_text.replace('_',' ')], f)
        renderer.composite(osp.join(env_save_dir, f'closed_loop.png'), [np.array(samp) for samp in trajs_obs][:n_samples_plot], np.repeat(scenario_text,n_samples_plot), np.array(all_inits)[:n_samples_plot])
        highway.get_acc(scenario_text, trajs_rew, trajs_info, trajs_done, trajs_obs)
        env.close()

def open_loop_robot(basedir, diffusion, dataset, renderer, condition_guidance_w, device, n_demos_eval=10):
    all_samples, all_cond_text, all_inits, all_init_ims, all_gt = [], [], [], [], []
    for _ in range(n_demos_eval):
        sample = dataset.__getitem__(proprioception_dropout=False)
        with th.no_grad():
            samples = diffusion.p_sample_loop(
                shape=(1, dataset.horizon, dataset.observation_dim),
                cond=th.unsqueeze(utils.to_torch(sample.conditions).to(device),0),
                dummy_cond=th.unsqueeze(utils.to_torch(sample.dummy_cond).to(device),0),
                cond_obs=th.unsqueeze(utils.to_torch(sample.conditions_obs).to(device),0),
                cond_im=th.unsqueeze(utils.to_torch(sample.conditions_obs_im).to(device),0),
                compose=False,
            )
        all_inits.append(dataset.unnormalize(sample.conditions_obs))
        all_init_ims.append(dataset.unnormalize_im(np.expand_dims(sample.conditions_obs_im,axis=0)).squeeze())
        all_samples.append(dataset.unnormalize(to_np(samples.trajectories)).squeeze())
        all_cond_text.append('') #dummy to plot
        all_gt.append(dataset.unnormalize(to_np(sample.trajectories)).squeeze())
    # save, render
    eval_dir = osp.join(basedir, f'eval_train_w_{condition_guidance_w}')
    if not osp.isdir(eval_dir): os.makedirs(eval_dir)  
    with open(osp.join(eval_dir, f'samples.pkl'), 'wb') as f: pickle.dump([all_samples, all_cond_text, all_inits, all_init_ims, all_gt], f)
    renderer.composite(osp.join(eval_dir, f'render_samples.png'), np.array(all_samples), np.array(all_cond_text), np.array(all_inits), np.array(all_init_ims))
    renderer.composite(osp.join(eval_dir, f'render_gt.png'), np.array(all_gt), np.array(all_cond_text), np.array(all_inits), np.array(all_init_ims))



if __name__ == "__main__":
    args = Parser().parse_args('plan')
    device = th.device('cpu' if not th.cuda.is_available() else 'cuda')

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
