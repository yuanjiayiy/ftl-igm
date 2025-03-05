import pickle
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from highway_env import register_highway_envs
from rl_agents.agents.common.factory import agent_factory
register_highway_envs()

save_dir = 'data'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
n_demos = 100
for scenario_text,version in [("highway-20car",1)]:
    print(scenario_text)
    env_name = f"{scenario_text.replace('_','-')}-v{version}"
    video_folder = f"{save_dir}/{scenario_text}_{version}/videos"
    demos_pkl = f"{save_dir}/{scenario_text}_{version}/demos.pkl"

    # Make environment
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True) #default w/o episode_trigger: stop recording on terminated or truncated, causes issue with recorder.
    env.unwrapped.set_record_video_wrapper(env)

    # Make agent
    agent_config = {
        "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
        "env_preprocessors": [{"method":"simplify"}],
        "budget": 50,
        "gamma": 0.7,
    }
    agent = agent_factory(env, agent_config)

    # Run episodes
    trajs_obs, trajs_im, trajs_rew, trajs_done, trajs_trunc, trajs_info = [], [], [], [], [], []
    save_idxs = [] # all videos recorded, not all data saved
    traj_num = 0
    while len(save_idxs) < n_demos:
        print(traj_num)
        (obs, info), done, truncated = env.reset(), False, False
        import pdb; pdb.set_trace()
        traj_obs, traj_im, traj_rew, traj_done, traj_trunc, traj_info = [obs], [], [], [], [], []
        while not (done or truncated):
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            traj_obs.append(obs)
            traj_rew.append(reward)
            traj_done.append(done)
            traj_trunc.append(truncated)
            traj_info.append(info) #speed, crashed, act
            img = env.render()
            traj_im.append(img)
        traj_num += 1 #(should have been updated at the end of for but was created like this)
        if info['crashed']: continue # don't save episodes where ego vehicle crashes      
        if scenario_text == 'exit' and not info['is_success']: continue # vehicle didn't take exit in exit scenario. don't save
        if scenario_text == 'intersection' and truncated and not done: continue # vehicle didn't reach target in intersection scenario. don't save
        if scenario_text == 'intersection' and len(traj_obs) < 8: print(f'not saving {len(traj_obs)}'); continue # short horizon
        trajs_obs.append(traj_obs) # save as list, not numpy, not all same horizon (done or truncated)
        trajs_rew.append(traj_rew)
        trajs_done.append(traj_done)
        trajs_trunc.append(traj_trunc)
        trajs_info.append(traj_info)
        trajs_im.append(traj_im)
        print(f'saving {traj_num} len {len(traj_obs)}, reward {sum(traj_rew)}')
        save_idxs.append(traj_num)

    # Save obs, act, text description
    if scenario_text == 'highway_fast': scenario_text = 'highway less vehicles'
    if scenario_text == 'two_way' and version == 1: scenario_text = 'two way left hand side'
    if scenario_text == 'u_turn': scenario_text = 'u-turn'
    with open(demos_pkl, 'wb') as f:
        pickle.dump([trajs_obs, trajs_im, trajs_rew, trajs_done, trajs_trunc, trajs_info, save_idxs, scenario_text.replace('_',' ')], f)
    env.close()
    