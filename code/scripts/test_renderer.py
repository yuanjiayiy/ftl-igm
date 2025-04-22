import os
import sys
import numpy as np
import pygame
from PIL import Image, ImageChops

project_root = '/home/law/Workspace/repos/ftl-igm/code'
if project_root not in sys.path:
    sys.path.append(project_root)
mapbt_path = '/home/law/Workspace/repos/ftl-igm/mapbt_package/mapbt'
if mapbt_path not in sys.path:
    sys.path.append(mapbt_path)
overcooked_ai_py_src_path = '/home/law/Workspace/repos/ftl-igm/mapbt_package/mapbt/envs/overcooked/overcooked_berkeley/src/overcooked_ai_py'
if overcooked_ai_py_src_path not in sys.path:
    sys.path.append(overcooked_ai_py_src_path)

# Import using direct overcooked_ai_py classes
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_env.visualization.pygame_utils import scale_surface_by_factor
from overcooked_sample_renderer import OvercookedSampleRenderer

pygame.init()

output_dir = os.path.join(os.getcwd(), "renderer_comparison")
os.makedirs(output_dir, exist_ok=True)

layout = "counter_circuit_o_1order"
ae = AgentEvaluator.from_layout_name(mdp_params={"layout_name": layout, "old_dynamics": True}, 
                                     env_params={"horizon": 400})

grid_layout = ae.env.mdp.terrain_mtx

sample_renderer = OvercookedSampleRenderer()
state_visualizer = StateVisualizer()



def run_comparison(num_frames=10):
    ap = AgentPair(RandomAgent(), RandomAgent())
    
    # trajs = ae.evaluate_agent_pair(ap, num_frames)
    trajs = ae.evaluate_human_model_pair(1)
    
    for i, state in enumerate(trajs["ep_states"][0][:num_frames]):  # Using first episode, up to num_frames
        feature_tensor = ae.env.mdp.lossless_state_encoding(state)

        obs_0, obs_1 = feature_tensor
        obs_0 = np.transpose(obs_0, (1,0,2))
        
        # Sample Renderer
        sample_path = os.path.join(output_dir, f"sample_frame_{i:03d}.png")
        sample_renderer.save_obs_image(obs_0, grid_layout, sample_path) # grid , [8,5,26]
        
        # State Visualizer
        state_path = os.path.join(output_dir, f"state_frame_{i:03d}.png")
        state_visualizer.display_rendered_state(state, grid=grid_layout, img_path=state_path)
    
        
        
    
print("Starting renderer comparison...")
results = run_comparison(num_frames=400)
print(f"Images saved to {output_dir}")
