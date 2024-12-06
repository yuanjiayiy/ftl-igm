import os
import os.path as osp
import pickle
import numpy as np
import copy
import torch as th
import pylab as plt
import random
import warnings
warnings.filterwarnings("ignore")
from scripts_utils import Parser
import diffuser.utils as utils
from diffuser.utils.arrays import to_np, to_device
from diffuser.utils.serialization import get_latest_epoch
from diffuser.datasets import object_rearrangement
from diffuser.datasets import AGENT
from AGENT_env import AGENT_env
from eval_train import closed_loop_highway
from diffuser.datasets import robot
from diffuser.utils.training import TrainerROBOT


def learn_concept(args, diffusion, trainer, dataset, demos, device, dummy_cond=None):
    if not osp.isdir(trainer.logdir): os.makedirs(trainer.logdir)
    # disable model gradients
    diffusion.model.requires_grad_(False)
    trainer.model.requires_grad_(False)
    trainer.model.model.requires_grad_(False)
    trainer.ema_model.requires_grad_(False)
    trainer.ema_model.model.requires_grad_(False)
    #resnet18 batchnorm (robot)
    diffusion.eval()
    diffusion.model.eval()
    trainer.model.eval()
    trainer.model.model.eval()
    trainer.ema_model.eval()
    trainer.ema_model.model.eval()
    original_weights = []
    for w in trainer.model.model.state_dict().items(): original_weights.append(w[1].detach().clone())
    # init concepts: Gaussian and allow input update 
    init_cond_features = np.random.normal(0, 1, args.n_concepts*dataset.cond_dim).astype(np.float32)
    cond_features = th.nn.Parameter(th.tensor(copy.deepcopy(init_cond_features.reshape(args.n_concepts,-1)), device=device, requires_grad=True))
    if args.learn_weights:
        init_weights = np.random.uniform(0, 1, args.n_concepts).astype(np.float32) #weights: uniform  
        cond_weights = th.nn.Parameter(th.tensor(copy.deepcopy(init_weights), device=device, requires_grad=True))
    else:
        cond_weights = args.condition_guidance_w
    diffusion.condition_guidance_w = cond_weights
    # update dataset
    if args.dataset != 'robot':
        trainer.dataset.observations = demos
        if hasattr(trainer.dataset,'horizon'):
            trainer.dataset.path_lengths = [len(obs) for obs in trainer.dataset.observations]
        trainer.dataset.normalize()
        if args.dataset in ['object_rearrangement','AGENT']:
            trainer.dataset.observations = th.from_numpy(trainer.dataset.observations).to(th.float32)
        if hasattr(trainer.dataset,'horizon'):
            if args.dataset in ['mocap','highway']:
                trainer.dataset.normed_observations = [to_device(th.from_numpy(x).to(th.float32), device) for x in trainer.dataset.normed_observations]
            else:
                trainer.dataset.normed_observations = to_device(th.from_numpy(trainer.dataset.normed_observations).to(th.float32), device)
            trainer.dataset.indices = trainer.dataset.make_indices(trainer.dataset.path_lengths, trainer.dataset.horizon)
        if dummy_cond is not None:
            trainer.dataset.dummy_cond = dummy_cond.to(th.float32)
        else:
            trainer.dataset.dummy_cond = th.tensor(trainer.dataset.dummy_cond.reshape(1,-1)).to(device)
    trainer.dataset.conditions = cond_features
    # optimization
    trainer.model.condition_guidance_w = cond_weights
    if args.learn_weights:
        trainer.optimizer = th.optim.Adam([trainer.dataset.conditions, trainer.model.condition_guidance_w], lr=args.learning_rate)
    else:
        trainer.optimizer = th.optim.Adam([trainer.dataset.conditions], lr=args.learning_rate)
    # training concept representation (same as diffusion training but derive loss by inputs)
    for i in range(args.n_epochs):
        print(f'Epoch {i} / {args.n_epochs} | {trainer.logdir}')
        losses = trainer.train(n_train_steps=args.n_steps_per_epoch, invert_model=True)
    plt.figure()
    plt.plot(list(range(len(losses))),losses)
    plt.savefig(osp.join(trainer.logdir, f'learn_concept_training_loss.png'))
    # confirm only inputs changed not model weights
    curr_weights = trainer.model.model.state_dict()
    for w_orig, w_curr in zip(original_weights, curr_weights.items()): assert (w_orig == w_curr[1]).all()
    if args.learn_weights:
        assert th.equal(diffusion.condition_guidance_w, trainer.model.condition_guidance_w) and th.equal(trainer.model.condition_guidance_w, cond_weights)
    else:
        assert diffusion.condition_guidance_w == trainer.model.condition_guidance_w == cond_weights
    # save learned concept representations
    with open(osp.join(trainer.logdir, f'learned_concept.pkl'), 'wb') as f: pickle.dump([trainer.dataset.conditions, trainer.model.condition_guidance_w], f) 
    return trainer.dataset.conditions, trainer.model.condition_guidance_w 

def one_step_object_rearrangement(basedir, diffusion, dataset, renderer, dummy_cond, conditions, cond_text, weight_str, dataset_name, n_samples=50, n_samples_plot=8):
    all_samples = []
    for _ in range(n_samples):
        samples = diffusion.p_sample_loop(
            shape=(1, dataset.observation_dim),
            cond=conditions,
            dummy_cond=dummy_cond,
            compose=True 
        )
        all_samples.append(dataset.unnormalize(to_np(samples.trajectories)).squeeze())
    # save, render, eval
    savepath = osp.join(basedir, f'gen_concept_{weight_str}.pkl')
    with open(savepath, 'wb') as f: pickle.dump([all_samples, cond_text], f)    
    renderpath = osp.join(basedir, f'gen_concept_{weight_str}.png')
    renderer.composite(renderpath, np.array(all_samples)[:n_samples_plot], np.repeat(cond_text, n_samples_plot))
    object_rearrangement.get_learned_concept_acc(all_samples, cond_text, dataset_name)

def one_step_object_rearrangement_compos(basedir, compose_dataset, diffusion, dataset, renderer, dummy_cond, conditions, weights, cond_text, learn_weights, weight_str, device, n_samples=50, n_samples_plot=2):
    #load train concepts cond
    with open(compose_dataset, 'rb') as input_file: compos_cond_features, compos_cond_text = pickle.load(input_file)
    # generate composition with train concepts 
    for i, (seen_cond_feat, seen_cond_text) in enumerate(zip(compos_cond_features, compos_cond_text)):  
        all_samples = []
        if learn_weights:
            diffusion.condition_guidance_w = th.nn.Parameter(th.tensor([float(w) for w in weights]+[1.0]).to(device),requires_grad=False)
        for _ in range(n_samples):
            samples = diffusion.p_sample_loop(
                shape=(1, dataset.observation_dim),
                cond=th.cat([conditions, th.tensor(seen_cond_feat).to(device)], axis=0),                
                dummy_cond=dummy_cond,
                compose=True
            )
            all_samples.append(dataset.unnormalize(to_np(samples.trajectories)).squeeze())
        # save, render, eval
        savepath = osp.join(basedir, f'eval_concept_compos{i}_{weight_str}.pkl')
        with open(savepath, 'wb') as f: pickle.dump([all_samples, cond_text, seen_cond_text], f)        
        renderpath = osp.join(basedir, f'eval_concept_compos{i}_{weight_str}.png')
        renderer.composite(renderpath, np.array(all_samples)[:n_samples_plot], np.repeat(seen_cond_text,n_samples_plot))
        object_rearrangement.get_compos_learned_acc(all_samples, cond_text, seen_cond_text)

def closed_loop_AGENT(basedir, new_init_dataset, diffusion, dataset, renderer, demos, dummy_cond, conditions, cond_text, n_concepts, device, n_samples_plot=2, new_init=False):
    env = AGENT_env.PyAgentEnv(mode='test', cond_text=cond_text)
    new_init_str = '_new_init' if new_init else ''
    res_path = osp.join(basedir, f'pybullet_acts{new_init_str}')
    if not osp.isdir(res_path): os.makedirs(res_path)
    if new_init: #load test concepts cond
        with open(osp.join(new_init_dataset), 'rb') as input_file: 
            all_cond_init, _, _ = pickle.load(input_file) #init normalized  
    else: #get from demos
        all_cond_init = demos[:,0,:]
    all_samples, all_inits_unnorm, all_pred_forces = [], [], []
    for init_s_idx,init_s in enumerate(all_cond_init[:n_samples_plot]):
        plot_dir = osp.join(res_path, f'acts_{init_s_idx}')
        if not osp.isdir(plot_dir): os.mkdir(plot_dir)
        if new_init: 
            init_s_unnorm = dataset.unnormalize(init_s).squeeze()
        else: 
            init_s_unnorm = init_s
        all_inits_unnorm.append(init_s_unnorm)
        env.load_scene(init_s_unnorm)
        final_traj = [env._get_obs().astype(np.float32)]
        curr_pred_forces = []
        for t in range(1,dataset.horizon):
            init_s = th.tensor(dataset.normalize_init(final_traj[-1]).reshape(1,-1)).to(device) #current obs
            with th.no_grad():
                samples = diffusion.p_sample_loop(
                    shape=(1, dataset.horizon, dataset.observation_dim),
                    cond=th.tensor(conditions).to(device),
                    dummy_cond=dummy_cond.to(device),
                    cond_obs=init_s,
                    compose=True if n_concepts > 1 else False,
                )
            s_t_unnorm = final_traj[-1][:dataset.observation_dim]
            s_t_1_unnorm = dataset.unnormalize(to_np(samples.trajectories)[0][min(t+5,dataset.horizon-1)]).squeeze()
            pred_force = (s_t_1_unnorm - s_t_unnorm)[[0,2]]
            curr_pred_forces.append(pred_force)
            obs, done = env.step(pred_force) 
            final_traj.append(obs.astype(np.float32))
            AGENT.plot_traj(np.array(final_traj), final_traj[0], osp.join(plot_dir,f'inv_model_obs.png'), cond_text=cond_text) #updating obs
            AGENT.plot_traj(dataset.unnormalize(to_np(samples.trajectories)).squeeze(), final_traj[0], osp.join(plot_dir,f'inv_model_diffusion_{t}.png'), s_t_1_unnorm, cond_text=cond_text) #diffusion guidance
            if done: break
        all_samples.append(final_traj)
        all_pred_forces.append(curr_pred_forces)
    # save, render, eval
    savepath = osp.join(res_path, f'gen_concept_acts{new_init_str}.pkl')
    with open(savepath, 'wb') as f: pickle.dump([all_samples, all_inits_unnorm, cond_text, all_pred_forces], f)
    render_path = osp.join(res_path, f'gen_concept_acts{new_init_str}.png')
    renderer.composite(render_path, [np.array(samp) for samp in all_samples][:n_samples_plot], np.repeat(cond_text,n_samples_plot), np.array(all_inits_unnorm)[:n_samples_plot])
    AGENT.get_test_acc(all_samples, all_inits_unnorm, cond_text)

def open_loop_mocap(trainer, diffusion, dataset, renderer, dummy_cond, conditions, cond_text, n_concepts, device, n_samples_plot=8, new_init=False):
    savepath = osp.join(trainer.logdir, f'{"new" if new_init else "demos"}_inits.pkl')
    if not osp.isfile(savepath):
        all_samples = []
        if new_init:
            all_inits = [x[-1].reshape(1,-1) for x in trainer.dataset.normed_observations] # last obs in demos, sliding window won't reach, no padding
        else:
            all_inits = [x[0].reshape(1,-1) for x in trainer.dataset.normed_observations] # demo inits
        for init_s in all_inits:
            samples = diffusion.p_sample_loop(
                shape=(1, trainer.dataset.horizon, trainer.dataset.observation_dim),
                cond=conditions, #learned
                dummy_cond=dummy_cond,
                cond_obs=th.tensor(init_s).to(device),
                compose=True if n_concepts > 1 else False
            )
            all_samples.append(dataset.unnormalize(to_np(samples.trajectories)).squeeze())
        # save, render
        with open(savepath, 'wb') as f: pickle.dump([all_samples, cond_text], f)        
        render_savepath = osp.join(trainer.logdir, f'{"new" if new_init else "demos"}_inits')
        if not osp.isdir(render_savepath): os.mkdir(render_savepath)
        renderer.composite(render_savepath, [f'sample-{i}' for i in range(n_samples_plot)], np.array(all_samples)[:n_samples_plot], np.repeat(cond_text,n_samples_plot))

def open_loop_mocap_compos(trainer, diffusion, dataset, renderer, dummy_cond, conditions, weights, cond_text, learn_weights, device, n_samples_plot=8):
    with open('data/mocap/train_gt.pkl', 'rb') as input_file: _, compos_cond_features, compos_cond_init, compos_cond_text, _ = pickle.load(input_file) # load training concepts cond
    all_samples = []
    if learn_weights:
        diffusion.condition_guidance_w = th.nn.Parameter(th.tensor([float(w) for w in weights]+[1.0]).to(device),requires_grad=False) # compos weight 1
    for seen_cond_feat, init_s in zip(compos_cond_features, compos_cond_init): # training concept init
        samples = diffusion.p_sample_loop(
            shape=(1, trainer.dataset.horizon, trainer.dataset.observation_dim),
            cond=th.cat([conditions, th.tensor(seen_cond_feat.reshape(1,-1)).to(device)], axis=0),                
            dummy_cond=dummy_cond,
            cond_obs=th.tensor(init_s.reshape(1,-1)).to(device),
            compose=True
        )
        all_samples.append(dataset.unnormalize(to_np(samples.trajectories)).squeeze())
    # save, render
    savepath = osp.join(trainer.logdir, f'compos.pkl')
    with open(savepath, 'wb') as f: pickle.dump([all_samples, cond_text, compos_cond_text], f)
    render_savepath = osp.join(trainer.logdir, f'compos')
    if not osp.isdir(render_savepath): os.mkdir(render_savepath)
    renderer.composite(render_savepath, [f'sample-{i}' for i in range(n_samples_plot)], np.array(all_samples)[:n_samples_plot], np.array(compos_cond_text)[:n_samples_plot])

def open_loop_robot(trainer,  diffusion, dataset, renderer, dummy_cond, conditions, cond_text, n_concepts, weight_str, device, n_demos_eval=10):  
    all_samples, all_inits, all_init_ims, all_gt = [], [], [], []
    for _ in range(n_demos_eval):
        batch_idx = random.randint(0,int(trainer.dataset.indices.shape[0])-1)
        path_ind, start, end = trainer.dataset.indices[batch_idx]   
        cond_obs = th.from_numpy(trainer.dataset.normed_observations[path_ind][start]).to(device).reshape(1,-1)
        cond_im = th.from_numpy(np.transpose(trainer.dataset.cond_obs_imL[path_ind][start][:,:,:3],(2,0,1)).astype(np.float32)).to(device).unsqueeze(0)
        with th.no_grad():
            samples = diffusion.p_sample_loop(
                shape=(1, dataset.horizon, dataset.observation_dim),
                cond=conditions,
                dummy_cond=th.tensor(dummy_cond.reshape(1,-1)).to(device),
                cond_obs=cond_obs,
                cond_im=cond_im,
                compose=True if n_concepts > 1 else False,
            )
        all_inits.append(dataset.unnormalize(to_np(cond_obs).squeeze()))
        all_init_ims.append(dataset.unnormalize_im(cond_im).squeeze())
        all_samples.append(dataset.unnormalize(to_np(samples.trajectories).squeeze()))
        all_gt.append(dataset.unnormalize(dataset.normed_observations[path_ind][start:end]))
    # save, render
    with open(osp.join(trainer.logdir, f'samples.pkl'), 'wb') as f: pickle.dump([all_samples, all_inits, all_init_ims, all_gt], f)
    renderer.composite(osp.join(trainer.logdir, f'render_samples_{weight_str}.png'), np.array(all_samples), cond_text*n_demos_eval, np.array(all_inits), np.array(all_init_ims)) 
    renderer.composite(osp.join(trainer.logdir, f'render_gt_{weight_str}.png'), np.array(all_gt), cond_text*n_demos_eval, np.array(all_inits), np.array(all_init_ims)) #gt
    robot.plot(osp.join(trainer.logdir, f'render_samples_gt_2D_{weight_str}.png'), np.array(all_samples), np.array(all_gt), cond_text*n_demos_eval, np.array(all_inits), np.array(all_init_ims), plot_type='2D') #both


if __name__ == "__main__":
    args = Parser().parse_args('plan')
    device = th.device('cpu' if not th.cuda.is_available() else 'cuda')  

    # load diffusion model function from disk
    load_dataset = False if args.dataset=='robot' else True
    diffusion_experiment = utils.load_diffusion(
        args.loadbase, args.dataset, args.diffusion_loadpath,
        epoch=args.diffusion_epoch, seed=args.seed, load_dataset=load_dataset,
    )
    diffusion = diffusion_experiment.diffusion
    renderer = diffusion_experiment.renderer
    if args.dataset != 'robot':
        dataset = diffusion_experiment.dataset
        trainer = diffusion_experiment.trainer
    basedir = osp.join(args.loadbase, args.dataset, args.diffusion_loadpath)
    
    # load new task demos
    if args.dataset == 'object_rearrangement':
        with open(args.dataset_name, 'rb') as input_file: shapes, cond_text, dummy_cond = pickle.load(input_file)
        demos = np.array(shapes).reshape(len(shapes),-1)
        new_concept_name = args.dataset_name.split("/")[-1][:-4]
    elif args.dataset == 'AGENT':
        with open(args.dataset_name, 'rb') as input_file: demos, gt_cond_features, dummy_cond, cond_text = pickle.load(input_file)     
        new_concept_name = args.dataset_name.split("/")[-1][:-4]
    elif args.dataset == 'mocap':
        with open(args.dataset_name, 'rb') as input_file: demos, gt_cond_features, dummy_cond, cond_text = pickle.load(input_file) 
        demos = [obs[::args.sample_rate,:] for obs in demos] 
        new_concept_name = args.dataset_name.split("/")[-1].split('mocap_dataset_test_')[-1].split('.pkl')[0]         
    elif args.dataset == 'highway':
        with open(args.dataset_name, 'rb') as input_file: demos, _, _, _, _, _, _, cond_text = pickle.load(input_file)
        new_concept_name = 'eval_new'
        dummy_cond = None
    elif args.dataset == 'robot':
        with open(args.dataset_name, 'rb') as input_file: demos, cond_obs, cond_imL, _, dummy_cond, cond_text = pickle.load(input_file)
        dataset = robot.RobotSequenceDataset(horizon=args.horizon,
                                        use_padding=args.use_padding,
                                        max_path_length=args.max_path_length,
                                        dataset_stats_path=args.dataset_stats_path,
                                        mode=args.mode
                                        )
        dataset.reshaped_observations = np.vstack(copy.deepcopy(demos))
        dataset.n_episodes = len(demos)
        dataset.reshaped_cond_obs_robot = np.vstack(copy.deepcopy(cond_obs))
        dataset.cond_obs_imL = np.vstack(cond_imL)
        dataset.path_lengths = [len(obs) for obs in demos]
        dataset.normalize()
        dataset.indices = dataset.make_indices(dataset.path_lengths, dataset.horizon)
        dataset.dummy_cond = th.tensor(dataset.dummy_cond.reshape(1,-1)).to(device)
        #
        trainer = TrainerROBOT(diffusion, dataset, renderer, results_folder=args.savepath, 
                            train_batch_size=args.batch_size, train_lr=args.learning_rate,
                            )
        data = th.load(osp.join(basedir, f'state_{get_latest_epoch((args.loadbase, args.dataset, args.diffusion_loadpath))}.pt'))
        trainer.step = data['step']
        trainer.model.load_state_dict(data['model'])
        trainer.ema_model.load_state_dict(data['ema'])
        new_concept_name = 'eval_new'

    # set paths
    weight_str = '' if args.learn_weights else f'w_{args.condition_guidance_w}'    
    trainer.logdir = osp.join(basedir, f'{new_concept_name}', f'n_cond_{args.n_concepts}', f'{weight_str}')

    # learn new concept
    conditions, weights = learn_concept(args, diffusion, trainer, dataset, demos, device, dummy_cond)
    if args.learn_weights:
        weight_str = 'w'
        for cond_idx in range(args.n_concepts): weight_str += f'_{str(round(float(diffusion.condition_guidance_w[cond_idx]),2))}'   

    # evaluate learned concept
    if args.dataset == 'object_rearrangement':
        one_step_object_rearrangement(trainer.logdir, diffusion, dataset, renderer, dummy_cond, conditions, cond_text, weight_str, args.dataset_name)
        if args.compose_dataset is not None: # compos new inferred with known concpet
            one_step_object_rearrangement_compos(trainer.logdir, args.compose_dataset, diffusion, dataset, renderer, dummy_cond, conditions, weights, cond_text, args.learn_weights, weight_str, device)
    elif args.dataset == 'AGENT':
        closed_loop_AGENT(trainer.logdir, args.new_init_dataset, diffusion, dataset, renderer, demos, dummy_cond, conditions, cond_text, args.n_concepts, device)
        closed_loop_AGENT(trainer.logdir, args.new_init_dataset, diffusion, dataset, renderer, demos, dummy_cond, conditions, cond_text, args.n_concepts, device, new_init=True)
    elif args.dataset == 'mocap':
        open_loop_mocap(trainer, diffusion, dataset, renderer, dummy_cond, conditions, cond_text, args.n_concepts, device)
        open_loop_mocap(trainer, diffusion, dataset, renderer, dummy_cond, conditions, cond_text, args.n_concepts, device, new_init=True)
        open_loop_mocap_compos(trainer, diffusion, dataset, renderer, dummy_cond, conditions, weights, cond_text, args.learn_weights, device)
    elif args.dataset == 'highway':
        closed_loop_highway(osp.join(basedir, f'{new_concept_name}', f'n_cond_{args.n_concepts}', f'{weight_str}'), diffusion, dataset, renderer, [("roundabout", 1)], device, args.n_concepts, mode='test', cond=conditions)
    elif args.dataset == 'robot':
        open_loop_robot(trainer,  diffusion, dataset, renderer, dummy_cond, conditions, cond_text, args.n_concepts, weight_str, device)

