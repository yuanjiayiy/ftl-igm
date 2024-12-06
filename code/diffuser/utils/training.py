import os
import copy
import numpy as np
import torch
import random
from random import sample
from transformers import T5Tokenizer, T5EncoderModel

from .arrays import batch_to_device, to_np, to_device, to_torch
from .timer import Timer
from diffuser.datasets.object_rearrangement import * 
from diffuser.datasets.AGENT import *
from diffuser.datasets.mocap import *
from diffuser.datasets.highway import *


def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.reset_parameters()
        self.step = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.t5_model = T5EncoderModel.from_pretrained("google/flan-t5-base").to(self.device)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self, n_train_steps, invert_model=False):
        losses = []
        timer = Timer()
        for _ in range(n_train_steps):
            for _ in range(self.gradient_accumulate_every):
                if not invert_model:
                    batch = next(self.dataloader)
                    batch = batch_to_device(batch)
                    loss, infos = self.model.loss(*batch)
                else:
                    loss, infos = self.invert_model()
                loss = loss / self.gradient_accumulate_every
                losses.append(round(loss.item(), 4))
                if not invert_model:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.step % self.update_ema_every == 0 and not invert_model:
                self.step_ema()
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)
            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)
            if self.step == 0 and self.sample_freq and not invert_model:
                self.render_reference(self.n_reference)
            if self.sample_freq and self.step % self.sample_freq == 0 and not invert_model:
                self.render_samples()
            self.step += 1
        return losses

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


    def invert_model(self,):
        '''
            concept inference
        '''
        pass


    def render_reference(self, batch_size=10):
        '''
            render training points
        '''
        pass


    def render_samples(self, batch_size=2, n_samples=8):
        '''
            generate and render samples
        '''
        pass


#############################################
class TrainerObjectRearrangement(Trainer):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
    ):
        super().__init__(
            diffusion_model,
            dataset,
            renderer,
            ema_decay=ema_decay,
            train_batch_size=train_batch_size,
            train_lr=train_lr,
            gradient_accumulate_every=gradient_accumulate_every,
            step_start_ema=step_start_ema,
            update_ema_every=update_ema_every,
            log_freq=log_freq,
            sample_freq=sample_freq,
            save_freq=save_freq,
            label_freq=label_freq,
            save_parallel=save_parallel,
            results_folder=results_folder,
            n_reference=n_reference,
            bucket=bucket,
        )

    def invert_model(self, invert_model=True):
        batch_idx = random.randint(0,int(self.dataset.observations.shape[0])-1)
        x, cond, dummy_cond = to_device(self.dataset.observations[batch_idx].reshape(1,-1), 'cuda:0'), \
                                self.dataset.conditions, \
                                to_device(self.dataset.dummy_cond, 'cuda:0')
        loss, infos = self.model.loss(x, cond, dummy_cond, invert_model=invert_model)
        return loss, infos

    def render_reference(self, batch_size=10):
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()
        trajectories = to_np(batch.trajectories)
        conditions = [get_task(cond) for cond in to_np(batch.conditions)]
        normed_observations = trajectories
        observations = self.dataset.unnormalize(normed_observations)
        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations, conditions)

    def render_samples(self, batch_size=2, n_samples=8):
        for i in range(batch_size):
            batch = self.dataloader_vis.__next__()
            shapes = [0,1,2]
            shapes_text = ['circle', 'triangle', 'square']
            cond_shapes = sample(shapes, 2)
            relations = ['right of', 'above']
            cond_relation = sample(relations, 1)[0]
            conditions_text = [shapes_text[cond_shapes[0]] + ' ' + cond_relation + ' ' + shapes_text[cond_shapes[1]]]
            conditions_embedding = self.t5_tokenizer(conditions_text, return_tensors="pt").input_ids.to(self.device)
            conditions_embedding = self.t5_model(conditions_embedding).last_hidden_state.detach().mean(axis=1)
            conditions = to_device(conditions_embedding, 'cuda:0')
            dummy_cond = to_device(batch.dummy_cond, 'cuda:0')
            conditions = conditions.repeat(n_samples,1)
            dummy_cond = dummy_cond.repeat(n_samples,1)
            samples = self.ema_model(conditions, dummy_cond)
            trajectories = to_np(samples.trajectories)
            normed_observations = trajectories
            observations = self.dataset.unnormalize(normed_observations)
            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations, np.array(conditions_text).repeat(n_samples).reshape(n_samples,-1))



#############################################
class TrainerAGENT(Trainer):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
    ):
        super().__init__(
            diffusion_model,
            dataset,
            renderer,
            ema_decay=ema_decay,
            train_batch_size=train_batch_size,
            train_lr=train_lr,
            gradient_accumulate_every=gradient_accumulate_every,
            step_start_ema=step_start_ema,
            update_ema_every=update_ema_every,
            log_freq=log_freq,
            sample_freq=sample_freq,
            save_freq=save_freq,
            label_freq=label_freq,
            save_parallel=save_parallel,
            results_folder=results_folder,
            n_reference=n_reference,
            bucket=bucket,
        )

    def invert_model(self, invert_model=True):
        batch_idx = random.randint(0,int(self.dataset.indices.shape[0])-1)
        path_ind, start, end = self.dataset.indices[batch_idx]                   
        x = self.dataset.normed_observations[path_ind, start:end, :self.dataset.observation_dim].reshape(1,self.dataset.horizon,-1)
        cond = self.dataset.conditions  # learned
        dummy_cond = self.dataset.dummy_cond
        cond_obs = self.dataset.normed_observations[path_ind, start, :].reshape(1,-1)
        loss, infos = self.model.loss(x, cond, dummy_cond, cond_obs, invert_model=invert_model)
        return loss, infos     

    def render_reference(self, batch_size=10):
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()
        trajectories = to_np(batch.trajectories)
        conditions = [get_task(cond) for cond in to_np(batch.conditions)]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.unnormalize(normed_observations)
        init_states = self.dataset.unnormalize(to_np(batch.conditions_obs))
        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations, conditions, init_states)

    def render_samples(self, batch_size=2, n_samples=8):
        all_samples = []
        all_cond_text = []
        all_inits = []
        for i in range(n_samples):
            init_s = copy.deepcopy(self.dataset.observations[1][0])
            cond_features, cond_text, init_s = generate_cond(init_s)
            all_inits.append(init_s.squeeze())
            init_s = torch.tensor(self.dataset.normalize_init(init_s))
            samples = self.ema_model(
                cond=cond_features.to(self.device),
                dummy_cond=cond_dummy,
                cond_obs=init_s.to(self.device),
                history_cond=init_s[:,:self.dataset.observation_dim].to(self.device),
            )
            all_samples.append(self.dataset.unnormalize(to_np(samples.trajectories)).squeeze())
            all_cond_text.append(cond_text)
        savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
        print(savepath)
        self.renderer.composite(savepath, np.array(all_samples), np.array(all_cond_text), np.array(all_inits))


#############################################
class TrainerMocap(Trainer):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
    ):
        super().__init__(
            diffusion_model,
            dataset,
            renderer,
            ema_decay=ema_decay,
            train_batch_size=train_batch_size,
            train_lr=train_lr,
            gradient_accumulate_every=gradient_accumulate_every,
            step_start_ema=step_start_ema,
            update_ema_every=update_ema_every,
            log_freq=log_freq,
            sample_freq=sample_freq,
            save_freq=save_freq,
            label_freq=label_freq,
            save_parallel=save_parallel,
            results_folder=results_folder,
            n_reference=n_reference,
            bucket=bucket,
        )

    def invert_model(self, invert_model=True):
        batch_idx = random.randint(0,int(self.dataset.indices.shape[0])-1)
        path_ind, start, end = self.dataset.indices[batch_idx]                   
        x = self.dataset.normed_observations[path_ind][start:end, :self.dataset.observation_dim].reshape(1,self.dataset.horizon,-1)
        cond = self.dataset.conditions  # learned
        dummy_cond = self.dataset.dummy_cond
        cond_obs = self.dataset.normed_observations[path_ind][start, :].reshape(1,-1)
        loss, infos = self.model.loss(x, cond, dummy_cond, cond_obs, invert_model=invert_model)
        return loss, infos

    def render_reference(self, batch_size=10):
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()
        trajectories = to_np(batch.trajectories)
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.unnormalize(normed_observations)
        conditions = [''] * observations.shape[0] #dummy val for rendering
        savenames = [f'_sample-reference-{i}' for i in range(observations.shape[0])]
        self.renderer.composite(self.logdir, savenames, observations, conditions)

    def render_samples(self, batch_size=2, n_samples=4):
        all_samples = []
        all_cond_text = []
        for _ in range(n_samples):
            _, cond_features, cond_dummy, init_s, cond_text = self.dataset.get_item_render()
            samples = self.ema_model(
                cond=torch.tensor(cond_features).to(self.device),
                dummy_cond=torch.tensor(cond_dummy).to(self.device),
                cond_obs=torch.tensor(init_s).to(self.device),
            )
            all_samples.append(self.dataset.unnormalize(to_np(samples.trajectories)).squeeze())
            all_cond_text.append(cond_text.replace(' ','_').replace('/','_').replace('(','').replace(')',''))
        savenames = [f'sample-{self.step}-{i}' for i in range(n_samples)]
        self.renderer.composite(self.logdir, savenames, np.array(all_samples), np.array(all_cond_text))

 

#############################################
class TrainerHighway(Trainer):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
    ):
        super().__init__(
            diffusion_model,
            dataset,
            renderer,
            ema_decay=ema_decay,
            train_batch_size=train_batch_size,
            train_lr=train_lr,
            gradient_accumulate_every=gradient_accumulate_every,
            step_start_ema=step_start_ema,
            update_ema_every=update_ema_every,
            log_freq=log_freq,
            sample_freq=sample_freq,
            save_freq=save_freq,
            label_freq=label_freq,
            save_parallel=save_parallel,
            results_folder=results_folder,
            n_reference=n_reference,
            bucket=bucket,
        )

    def invert_model(self, invert_model=True): 
        batch_idx = random.randint(0,int(self.dataset.indices.shape[0])-1)
        path_ind, start, end = self.dataset.indices[batch_idx]
        x = torch.vstack([x[0] for x in self.dataset.normed_observations[path_ind][start:end]]).reshape(1,self.dataset.horizon,-1)
        cond = self.dataset.conditions  # learned
        dummy_cond = self.dataset.dummy_cond
        cond_obs = self.dataset.normed_observations[path_ind][start].flatten().reshape(1,-1)
        loss, infos = self.model.loss(x, cond, dummy_cond, cond_obs, invert_model=invert_model)
        return loss, infos

    def render_reference(self, batch_size=10):
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()
        trajectories = to_np(batch.trajectories)        
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.unnormalize(normed_observations)
        conditions = [''] * observations.shape[0] #dummy val for rendering
        init_states = self.dataset.unnormalize(to_np(batch.conditions_obs))
        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations, conditions, init_states)

    def render_samples(self, batch_size=2, n_samples=4):
        all_samples = []
        all_cond_text = []
        all_inits = []
        for i in range(n_samples):
            _, cond_features, cond_dummy, init_s, cond_text = self.dataset.get_item_render()
            samples = self.ema_model(
                cond=torch.tensor(cond_features).to(self.device),
                dummy_cond=torch.tensor(cond_dummy).to(self.device),
                cond_obs=torch.tensor(init_s).to(self.device),
            )
            all_samples.append(self.dataset.unnormalize(to_np(samples.trajectories)).squeeze())
            all_inits.append(self.dataset.unnormalize(to_np(init_s)))
            all_cond_text.append(cond_text.replace(' ','_').replace('/','_'))
        savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
        self.renderer.composite(savepath, np.array(all_samples), np.array(all_cond_text), np.array(all_inits))



#############################################
class TrainerROBOT(Trainer):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
    ):
        super().__init__(
            diffusion_model,
            dataset,
            renderer,
            ema_decay=ema_decay,
            train_batch_size=train_batch_size,
            train_lr=train_lr,
            gradient_accumulate_every=gradient_accumulate_every,
            step_start_ema=step_start_ema,
            update_ema_every=update_ema_every,
            log_freq=log_freq,
            sample_freq=sample_freq,
            save_freq=save_freq,
            label_freq=label_freq,
            save_parallel=save_parallel,
            results_folder=results_folder,
            n_reference=n_reference,
            bucket=bucket,
        )

    def invert_model(self, invert_model=True):
        batch_idx = random.randint(0,int(self.dataset.indices.shape[0])-1)
        path_ind, start, end = self.dataset.indices[batch_idx]
        x = torch.from_numpy(self.dataset.normed_observations[path_ind][start:end,:]).to(self.device).unsqueeze(0)
        cond = self.dataset.conditions  # learned
        dummy_cond = self.dataset.dummy_cond
        cond_obs = torch.from_numpy(self.dataset.normed_observations[path_ind][start]).to(self.device).reshape(1,-1)
        cond_im = torch.from_numpy(np.transpose(self.dataset.cond_obs_imL[path_ind][start][:,:,:3],(2,0,1)).astype(np.float32)).to(self.device).unsqueeze(0)
        loss, infos = self.model.loss(x, cond, dummy_cond, cond_obs, cond_im, invert_model)        
        return loss, infos

    def render_reference(self, batch_size=10):
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()
        conditions = to_np(batch.conditions[0])[:,None] #dummy val for rendering
        observations = self.dataset.unnormalize(to_np(batch.trajectories))
        init_states = self.dataset.unnormalize(to_np(batch.conditions_obs))
        init_ims = self.dataset.unnormalize_im(to_np(batch.conditions_obs_im))
        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations, conditions, init_states, init_ims)        

    def render_samples(self, batch_size=2, n_samples=2):
        all_samples = []
        all_gt_samples = []
        all_cond_text = []
        all_inits = []
        all_init_ims = []
        for i in range(n_samples):
            sample = self.dataset.__getitem__(proprioception_dropout=False)
            samples = self.ema_model(
                cond=torch.unsqueeze(to_torch(sample.conditions).to(self.device),0),
                dummy_cond=torch.unsqueeze(to_torch(sample.dummy_cond).to(self.device),0),
                cond_obs=torch.unsqueeze(to_torch(sample.conditions_obs).to(self.device),0),
                cond_im=torch.unsqueeze(to_torch(sample.conditions_obs_im).to(self.device),0)
            )
            all_inits.append(self.dataset.unnormalize(sample.conditions_obs))
            all_init_ims.append(self.dataset.unnormalize_im(np.expand_dims(sample.conditions_obs_im,axis=0)).squeeze())
            all_samples.append(self.dataset.unnormalize(to_np(samples.trajectories)).squeeze())
            all_gt_samples.append(self.dataset.unnormalize(to_np(sample.trajectories)).squeeze())
            all_cond_text.append('') #dummy to plot
        self.renderer.composite(os.path.join(self.logdir, f'sample-{self.step}-pred.png'), np.array(all_samples), np.array(all_cond_text), np.array(all_inits), np.array(all_init_ims))
        self.renderer.composite(os.path.join(self.logdir, f'sample-{self.step}-gt.png'), np.array(all_gt_samples), np.array(all_cond_text), np.array(all_inits), np.array(all_init_ims))
