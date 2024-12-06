from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.MLPnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 1,
        'n_diffusion_steps': 100,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': True, 
        'dim_mults': (1, 2, 4, 8),
        'returns_condition': True,
        'condition_guidance_w': 1., 
        'attention': False,
        'renderer': 'utils.ObjectRearrangementRenderer', 

        ## dataset
        'dataset': 'object_rearrangement',
        'loader': 'datasets.ObjectRearrangement',
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 1000,
        'dataset_path': "data/object_rearrangement/training_dataset.pkl",

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'trainer': 'utils.TrainerObjectRearrangement',
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 200e3, #number of training steps
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 20000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'plan': {
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 64,
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 1,
        'n_diffusion_steps': 100,
        'learning_rate': 2e-4,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}/YYYYMMDD-HHMMSS', #TODO update path
        'diffusion_epoch': 'latest',

        'verbose': True,
        'suffix': '0',

        'n_concepts': None,
        'dataset_name': None,
        'compose_dataset': None,
        'learn_weights': True,
        'condition_guidance_w': 1.,
        'dataset': 'object_rearrangement',
        'n_epochs': 5,
        'n_steps_per_epoch': 10000,
    },
}
