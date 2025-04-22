from diffuser.models.inverse_dynamics import InverseDynamicsModel
from scripts.eval_train import closed_loop_overcooked
from scripts_utils import Parser
import diffuser.utils as utils
import torch
import wandb

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#


if __name__ == "__main__":
    
    args = Parser().parse_args('diffusion')
    wandb.init(
        project="overcooked_idm",
        entity="social-rl",
        name="overcooked_idm",
        config=args)

    # dataset
    train_dataset_config = utils.Config(
        args.loader,
        args=args,
        savepath=(args.savepath, 'dataset_config.pkl'),
        split="train",
        # horizon=args.horizon,
        # use_padding=args.use_padding,
        # max_path_length=args.max_path_length,
        # dataset_path=args.dataset_path,
        # dataset_stats_path=None if not hasattr(args,'dataset_stats_path') else args.dataset_stats_path,
    )

    eval_dataset_config = utils.Config(
        args.loader,
        args=args,
        savepath=(args.savepath, 'dataset_config.pkl'),
        split="test"
    )
    
    render_config = utils.Config(
        args.renderer,
        savepath=(args.savepath, 'render_config.pkl'),
    )
    dataset = train_dataset_config()
    renderer = render_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    trainer_config = utils.Config(
        args.trainer,
        savepath=(args.savepath, 'trainer_config.pkl'),
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        sample_freq=args.sample_freq,
        save_freq=args.save_freq,
        label_freq=int(args.n_train_steps // args.n_saves),
        save_parallel=args.save_parallel,
        results_folder=args.savepath,
        bucket=args.bucket,
        n_reference=args.n_reference,
    )

    # instantiate
    # Example usage:
    model = InverseDynamicsModel(num_actions=6).to(args.device)
    
    obs = torch.randn(32, 8, 5, 26).to(args.device)       # batch of 32
    next_obs = torch.randn(32, 8, 5, 26).to(args.device)
    logits = model(obs, next_obs)        # output shape: (32, 1) if 1 action
    
    idm = model
    trainer = trainer_config(idm, dataset, renderer)

    # test forward & backward pass
    utils.report_parameters(model)
    print('Testing forward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0])
    loss, _ = idm.loss(*batch) 
    
    loss.backward()
    print('✓')

    eval_dataset = eval_dataset_config()
    eval_trainer = trainer_config(idm, eval_dataset, renderer)
    eval_trainer.eval(n_eval_steps=args.n_steps_per_epoch)

    # main loop
    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} | {args.savepath}')
        trainer.train(n_train_steps=args.n_steps_per_epoch)

        # eval with eval dataset
        eval_trainer.eval(n_eval_steps=args.n_steps_per_epoch)
        
        # if i % 2 == 0:
        #     closed_loop_overcooked()
