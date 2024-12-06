from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 100,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': True, 
        'dim_mults': (1, 2, 4, 8),
        'returns_condition': True,
        'condition_guidance_w': 1.,
        'attention': False,
        'renderer': 'utils.RobotRenderer', 

        ## dataset
        'dataset': 'robot',
        'loader': 'datasets.RobotSequenceDataset',
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 1000,
        'dataset_path': 'data/robot/training_dataset',
        'dataset_stats_path': 'data/robot/training_dataset_stats.pkl',
        'mode': 'train',

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'trainer': 'utils.TrainerROBOT',
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6, #number of training steps
        'batch_size': 32, #8,
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
        'guide': 'sampling.ValueGuide',
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
        #load dataset stats
        'use_padding': False, 
        'max_path_length': 1000,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 100,
        'learning_rate': 2e-4,

        ## value function
        'discount': 0.997,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}/YYYYMMDD-HHMMSS', #TODO update path
        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',

        'n_concepts': 2,
        'dataset_name': "data/robot/test_dataset.pkl",
        'new_init_dataset': None,     
        'learn_weights': True,
        'condition_guidance_w': 1.,
        'dataset_stats_path': 'data/robot/training_dataset_stats.pkl',
        'mode': 'plan',
        'dataset': 'robot',
        'n_epochs': 5,
        'n_steps_per_epoch': 10000,        
    },
}
