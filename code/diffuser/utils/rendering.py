import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pylab as plt

from mocap_env.sequence import SequenceVisualizer

#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#

def env_map(env_name):
    '''
        map D4RL dataset names to custom fully-observed
        variants for rendering
    '''
    if 'halfcheetah' in env_name:
        return 'HalfCheetahFullObs-v2'
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
    else:
        return env_name

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

class ObjectRearrangementRenderer:

    def __init__(self):
        self.n_shapes = 3
        self.obs_dim = 7 * self.n_shapes
        self.shape_type_idx = 3

    def composite(self, savepath, observations, conditions, dim=(1024, 256), **kwargs):
        """observations: batch x obs_dim"""
        ims_per_row = 2
        fig, axs = plt.subplots((observations.shape[0]//ims_per_row), ims_per_row, figsize=(15, 15))
        for ax, obs, cond in zip(axs.flat, observations, conditions):
            ax.set_xlim(-2,7)
            ax.set_ylim(-2,7)
            ax.set_title(cond)
            for shape_st_idx in range(0,len(obs),len(obs)//self.n_shapes): #each sample has n_shapes shapes
                shape = obs[shape_st_idx:shape_st_idx+len(obs)//self.n_shapes]
                shape_type = np.argmax(shape[-self.shape_type_idx:]) #argmax last 3 entries
                #plot based on shape type (cases)
                if shape_type == 0: #circle
                    pp = mpatches.Circle((shape[0], shape[1]), shape[2], color='#60D937')
                    ax.add_patch(pp)
                elif shape_type == 1: #triangle
                    pp = mpatches.RegularPolygon((shape[0], shape[1]), 3, radius=shape[2], orientation=shape[3], color='#FF42A1') #orientation radians
                    ax.add_patch(pp)
                elif shape_type == 2: #sqaure
                    pp = mpatches.Rectangle((shape[0], shape[1]), shape[2], shape[2], angle=shape[3]*(180/np.pi), color='#00A1FF')#[0], #angle degrees
                    ax.add_patch(pp)
        if savepath is not None:
            fig.tight_layout()
            plt.savefig(savepath)
            print(f'Saved {len(obs)} samples to: {savepath}')
        return fig


from matplotlib.markers import MarkerStyle
class AGENTRenderer:
    def __init__(self):
        self.obs_dim = 52
        self.horizon = 32 #150
        self.cond_dim = 768
        self.target_shapes = ['s', '^', 'o', MarkerStyle('o', fillstyle='bottom')] #'cube', 'cone', 'sphere', 'bowl'

    def composite(self, savepath, observations, conditions, init_states, dim=(1024, 256), **kwargs):
        """observations: batch x horizon x obs_dim"""
        ims_per_row = 2
        fig, axs = plt.subplots(((len(observations)+1)//ims_per_row), ims_per_row, figsize=(15, 15))
        for ax, obs, cond, init_s in zip(axs.flat, observations, conditions, init_states):
            ax.set_xlim(-2,2)
            ax.set_ylim(-5,-3)
            ax.set_title(cond)
            #access relevant parts and plot by pos and color and shape:
            ax.plot(obs[:,0], obs[:,2], marker='^') #agent trajectory (x,z)
            ax.plot(obs[0,0], obs[0,2], color='white', marker='^') #indicate agent (cone) direction (init state)
            ax.plot(obs[-1,0], obs[-1,2], color='black', marker='^') #indicate agent (cone) direction (final state)
            ax.plot(init_s[24], init_s[26], color=init_s[30:34], marker=self.target_shapes[np.argmax(init_s[34:38])]) #target 1
            ax.plot(init_s[38], init_s[40], color=init_s[44:48], marker=self.target_shapes[np.argmax(init_s[48:52])]) #target 2

        if savepath is not None:
            fig.tight_layout()
            plt.savefig(savepath)
            print(f'Saved {len(obs)} samples to: {savepath}')
        return fig


class mocapRenderer:
    def __init__(self):
        self.obs_dim = 14 * 3 #13 joints, 3D pos
        self.horizon = 256
        self.cond_dim = 768

    def composite(self, savedir, savenames, observations, conditions, dim=(1024, 256), video_fps=25, **kwargs):
        """observations: batch x horizon x obs_dim"""
        if savedir is not None:
            for obs, cond, sname in zip(observations, conditions, savenames):
                vis = SequenceVisualizer(savedir, sname,  # mandatory parameters
                                        plot_fn=None, 
                                        vmin=-1, vmax=1,  # min and max values of the 3D plot scene
                                        to_file=True,  # if True writes files to the given directory
                                        subsampling=1,  # subsampling of sequences
                                        with_pauses=False,  # if True pauses after each frame
                                        fps=20,  # fps for visualization
                                        mark_origin=False)  # if True draw cross at origin
                vis.plot(obs.reshape(obs.shape[0],-1,3),
                        seq2=None,
                        parallel=False,
                        plot_fn1=None, plot_fn2=None,  # defines how seq/seq2 are drawn
                        views=[(45, 45)],  # [(elevation, azimuth)]  # defines the view(s)
                        lcolor='#099487', rcolor='#F51836',
                        lcolor2='#E1C200', rcolor2='#5FBF43',
                        noaxis=True,  # if True draw person against white background
                        noclear=False, # if True do not clear the scene for next frame
                        toggle_color=False,  # if True toggle color after each frame
                        plot_cbc=None,  # alternatve plot function: fn(ax{matplotlib}, seq{n_frames x dim}, frame:{int})
                        last_frame=None, # {int} define the last frame < len(seq)
                        definite_cbc=None, # fn(ax{matplotlib}, iii{int}|enueration, frame{int})
                        name=cond.replace(' ','_').replace('/','_').replace('(','').replace(')',''),
                        plot_jid=False,
                        create_video=True,
                        video_fps=video_fps,
                        if_video_keep_pngs=True)


class HighwayRenderer:
    def __init__(self):
        self.n_vehicles = 5
        self.feat_dim = 7
        self.obs_dim = self.n_vehicles * self.feat_dim #5 vehicles, 7 features
        self.horizon = 8
        self.cond_dim = 768

    def composite(self, savepath, observations, conditions, init_states, dim=(1024, 256), **kwargs):
        """observations: batch x horizon x obs_dim"""
        # observations are "unnormalized" but esentially aren't
        ims_per_row = 2
        fig, axs = plt.subplots(((len(observations)+1)//ims_per_row), ims_per_row, figsize=(15, 15))
        for ax, obs, cond, init_s in zip(axs.flat, observations, conditions, init_states):
            obs = obs.reshape(-1, self.feat_dim) #horizon x features (ego vehicle)
            ax.set_xlim(-120,600)
            ax.set_ylim(-120,100)
            ax.set_title(cond)
            #access relevant parts and plot by pos and color and shape:
            ax.plot(obs[:,1], obs[:,2], color='blue', marker='^') #ego trajectory (x,y)
            ax.plot(obs[0,1], obs[0,2], color='white', marker='^') #indicate agent (cone) direction (init state)
            ax.plot(obs[-1,1], obs[-1,2], color='black', marker='^') #indicate agent (cone) direction (final state)
            #other vehicles init pos
            init_s = init_s.reshape(self.n_vehicles, self.feat_dim)
            for i in range(1,self.n_vehicles):
                ax.scatter(init_s[i,1], init_s[i,2], color='red', marker='^') #(x,y)

        if savepath is not None:
            fig.tight_layout()
            plt.savefig(savepath)
            print(f'Saved {len(obs)} samples to: {savepath}')
        return fig


class RobotRenderer:
    def __init__(self):
        pass
    
    def composite(self, savepath, observations, conds_text, init_states, init_ims, dim=(1024, 256), **kwargs):
        """observations: batch x horizon x obs_dim
        plot (obs pred + init gt + final pred) + (init im)"""
        ims_per_row = 2
        n_cols = len(observations)
        fig, axs = plt.subplots(n_cols, ims_per_row, figsize=(15, 15))
        # for ax, obs, cond, init_im, init_s in zip(axs.flat, observations, conds_text, init_states_ims, init_states_robot):
        for col in range(n_cols):
            obs = observations[col]
            init_s = init_states[col]
            init_im = init_ims[col]
            colors = sns.color_palette('hls', len(obs))
            axs[col,0].remove()
            axs[col,0] = fig.add_subplot(n_cols,ims_per_row,(col*2)+1, projection='3d')                   
            axs[col,0].scatter3D(obs[:,0], obs[:,1], obs[:,2], alpha=0.5, color=colors) #obs
            axs[col,0].scatter3D(init_s[0], init_s[1], init_s[2], color='black', marker='o') #init gt
            axs[col,0].scatter3D(obs[-1,0], obs[-1,1], obs[-1,2], color='black', marker='x') #final obs               
            axs[col,0].set_title(conds_text[col])  
            axs[col,1].imshow(init_im.astype(np.uint8))

        if savepath is not None:
            fig.tight_layout()
            plt.savefig(savepath)
            print(f'Saved {len(obs)} samples to: {savepath}')
        return fig

