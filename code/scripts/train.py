from scripts_utils import Parser
import diffuser.utils as utils
import wandb


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#


if __name__ == "__main__":
    args = Parser().parse_args('diffusion')

    # dataset
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        horizon=args.horizon,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        dataset_path=args.dataset_path,
        dataset_stats_path=None if not hasattr(args,'dataset_stats_path') else args.dataset_stats_path,
    )
    render_config = utils.Config(
        args.renderer,
        savepath=(args.savepath, 'render_config.pkl'),
    )
    dataset = dataset_config()
    renderer = render_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # model & trainer
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=observation_dim + action_dim, #output
        cond_dim=dataset.cond_dim,
        obs_cond_dim=dataset.obs_cond_dim, #input
        dim_mults=args.dim_mults,
        attention=args.attention,
        device=args.device,
    )
    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        ## loss weighting
        action_weight=args.action_weight,
        loss_weights=args.loss_weights,
        loss_discount=args.loss_discount,
        device=args.device,
    )
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
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)
    uncond_model = None
    if not args.train_uncond:
        uncond_model = utils.load_diffusion(
            args.loadbase, args.dataset, args.diffusion_loadpath,
            epoch=args.diffusion_epoch, seed=args.seed,
        ).diffusion

    # test forward & backward pass
    utils.report_parameters(model)
    print('Testing forward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0])
    loss, _ = diffusion.loss(*batch, train_uncond = args.train_uncond, uncond_model=uncond_model)
    loss.backward()
    print('✓')

    # main loop
    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
    wandb.init(entity="social-rl", project="diffusion-training", config=vars(args))
    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} | {args.savepath}')
        trainer.train(n_train_steps=args.n_steps_per_epoch, invert_model = False, train_uncond = args.train_uncond, uncond_model=uncond_model)
    
    wandb.finish()
