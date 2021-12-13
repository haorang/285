import gym
import argparse
import yaml
import os

from collections import OrderedDict
from stable_baselines3 import PPO_REWARD
from stable_baselines3 import PPO_AIRL

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_dmcontrol_env, make_vec_metaworld_env
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from stable_baselines3.common.vec_env import VecNormalize
from reward_model import RewardModel
from replay_buffer import ReplayBuffer



import numpy as np
import torch

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="walker_walk", help="environment ID")
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="logs/PrefPPO/", type=str)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=123)
    parser.add_argument("--n-envs", help="# of parallel environments", type=int, default=16)
    parser.add_argument("--n-steps", help="# of steps to run for each environment per update", type=int, default=500)
    parser.add_argument("--lr", help="learning rate", type=float, default=3e-4)
    parser.add_argument("--total-timesteps", help="total timesteps", type=int, default=2000000)
    parser.add_argument("-b", "--batch-size", help="batch size", type=int, default=64)
    parser.add_argument("--ent-coef", help="coeff for entropy", type=float, default=0.0)
    parser.add_argument("--hidden-dim", help="dim of hidden features", type=int, default=1024)
    parser.add_argument("--num-layer", help="# of layers", type=int, default=2)
    parser.add_argument("--use-sde", help="Whether to use generalized State Dependent Exploration", type=int, default=1)
    parser.add_argument("--sde-freq", help="Sample a new noise matrix every n steps", type=int, default=4)
    parser.add_argument("--gae-lambda", help="Factor for trade-off of bias vs variance", type=float, default=0.92)
    parser.add_argument("--clip-init", help="Initial value of clipping", type=float, default=0.4)
    parser.add_argument("--n-epochs", help="Number of epoch when optimizing the surrogate loss", type=int, default=20)
    parser.add_argument("--normalize", help="Normalization", type=int, default=1)
    parser.add_argument("--unsuper-step", help="# of steps for unsupervised learning", type=int, default=32000)
    parser.add_argument("--unsuper-n-epochs", help="# of steps for unsupervised learning", type=int, default=50)
    
    parser.add_argument("--expert-demos", help="expert demos", type=str, default="")
    parser.add_argument("--expert-demos2", help="expert demo2s", type=str, default="")
    parser.add_argument("--airl-timesteps", help="expert demo2s", type=int, default=1000000)
    parser.add_argument("--airl-trajs", help="expert demo2s", type=str, default=5)

    # reward learning
    parser.add_argument("--re-lr", help="Learning rate of reward fn", type=float, default=0.0003)
    parser.add_argument("--re-segment", help="Size of segment", type=int, default=50)
    parser.add_argument("--re-act", help="Last activation for reward fn", type=str, default='tanh')
    parser.add_argument("--re-num-interaction", help="# of interactions", type=int, default=16000)
    parser.add_argument("--re-num-feed", help="# of feedbacks", type=int, default=1)
    parser.add_argument("--re-batch", help="Batch size", type=int, default=128)
    parser.add_argument("--re-update", help="Gradient update of reward fn", type=int, default=100)
    parser.add_argument("--re-feed-type", help="type of feedback", type=int, default=0)
    parser.add_argument("--re-large-batch", help="size of buffer for ensemble uncertainty", type=int, default=10)
    parser.add_argument("--re-max-feed", help="# of total feedback", type=int, default=1400)
    parser.add_argument("--teacher-beta", type=float, default=-1)
    parser.add_argument("--teacher-gamma", type=float, default=1.0)
    parser.add_argument("--teacher-eps-mistake", type=float, default=0.0)
    parser.add_argument("--teacher-eps-skip", type=float, default=0.0)
    parser.add_argument("--teacher-eps-equal", type=float, default=0.0)
    args = parser.parse_args()
    
    metaworld_flag = False
    max_ep_len = 1000
    if 'metaworld' in args.env:
        metaworld_flag = True
        max_ep_len = 500
    
    env_name = args.env
        
    if args.normalize == 1:
        args.tensorboard_log += 'normalized_' + env_name
    else:
        args.tensorboard_log += env_name
    
    args.tensorboard_log += '/teacher_' + str(args.teacher_beta)
    args.tensorboard_log += '_' + str(args.teacher_gamma)
    args.tensorboard_log += '_' + str(args.teacher_eps_mistake)
    args.tensorboard_log += '_' + str(args.teacher_eps_skip)
    args.tensorboard_log += '_' + str(args.teacher_eps_equal)
    
    args.tensorboard_log += '/lr_'+str(args.lr)
    args.tensorboard_log += '_reward_lr' + str(args.re_lr)
    args.tensorboard_log += '_seg' + str(args.re_segment)
    args.tensorboard_log += '_act' + str(args.re_act)
    args.tensorboard_log += '_inter' + str(args.re_num_interaction)
    args.tensorboard_log += '_type' + str(args.re_feed_type)
    args.tensorboard_log += '_large' + str(args.re_large_batch)
    args.tensorboard_log += '_rebatch' + str(args.re_batch)
    args.tensorboard_log += '_reupdate' + str(args.re_update)
    args.tensorboard_log += '_batch_' + str(args.batch_size)
    args.tensorboard_log += '_nenvs_' + str(args.n_envs)
    args.tensorboard_log += '_nsteps_' + str(args.n_steps)
    args.tensorboard_log += '_ent_' + str(args.ent_coef)
    args.tensorboard_log += '_hidden_' + str(args.hidden_dim)
    args.tensorboard_log += '_sde_' + str(args.use_sde)
    args.tensorboard_log += '_sdefreq_' + str(args.sde_freq)
    args.tensorboard_log += '_gae_' + str(args.gae_lambda)
    args.tensorboard_log += '_clip_' + str(args.clip_init)
    args.tensorboard_log += '_nepochs_' + str(args.n_epochs)
    args.tensorboard_log += '_maxfeed_' + str(args.re_max_feed)
    args.tensorboard_log += '_unsuper_' + str(args.unsuper_step)
    args.tensorboard_log += '_update_' + str(args.unsuper_n_epochs)
    args.tensorboard_log += '_seed_' + str(args.seed) 
    
    # extra params
    if args.use_sde == 0:
        use_sde = False
    else:
        use_sde = True
    
    clip_range = linear_schedule(args.clip_init)
    
    # Parallel environments
    if metaworld_flag:
        env = make_vec_metaworld_env(
            args.env, 
            n_envs=args.n_envs, 
            monitor_dir=args.tensorboard_log,
            seed=args.seed)
        eval_env = make_vec_metaworld_env(
            args.env, 
            n_envs=1, 
            monitor_dir=args.tensorboard_log,
            seed=args.seed)
    else:
        env = make_vec_dmcontrol_env(
            args.env,     
            n_envs=args.n_envs, 
            monitor_dir=args.tensorboard_log,
            seed=args.seed)
        eval_env = make_vec_dmcontrol_env(
            args.env,     
            n_envs=1, 
            monitor_dir=args.tensorboard_log,
            seed=args.seed)
    
    # instantiating the reward model
    reward_model = RewardModel(
        env.envs[0].observation_space.shape[0],
        env.envs[0].action_space.shape[0],
        size_segment=args.re_segment,
        activation=args.re_act, 
        lr=args.re_lr,
        mb_size=args.re_batch, 
        teacher_beta=args.teacher_beta, 
        teacher_gamma=args.teacher_gamma, 
        teacher_eps_mistake=args.teacher_eps_mistake,
        teacher_eps_skip=args.teacher_eps_skip, 
        teacher_eps_equal=args.teacher_eps_equal,
        large_batch=args.re_large_batch)
    
    if args.normalize == 1:
        env = VecNormalize(env, norm_reward=False)
        eval_env = VecNormalize(eval_env, norm_reward=False)
    # AIRL
    # Load expert demos and format into buffer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    demos1 = np.load(args.expert_demos)
    demos2 = np.load(args.expert_demos2)
    print(demos1.files)
    # (500, 32, 24) (500, 32, 6) (500, 32) (500, 32)
    obs_steps, obs_envs, obs_shape = demos1['obs'].shape
    _, _, acs_shape = demos1['acs'].shape

    exp_obs = demos1['obs'].transpose(1,0,2)
    exp_acs = demos1['acs'].transpose(1,0,2)
    exp_rews = demos1['rews'].transpose(1, 0)
    exp_dones = demos1['dones'].transpose(1, 0)
    
    exp_obs2 = demos2['obs'].transpose(1,0,2)
    exp_acs2 = demos2['acs'].transpose(1,0,2)
    exp_rews2 = demos2['rews'].transpose(1, 0)
    exp_dones2 = demos2['dones'].transpose(1, 0)
    
    expert_buffer = ReplayBuffer((obs_shape,), (acs_shape,), obs_steps * obs_envs * 2, device, window=500)
    
    for i in range(obs_envs):
        for j in range(obs_steps):
            expert_buffer.add(exp_obs[i][j], exp_acs[i][j], exp_rews[i][j], np.zeros(obs_shape), exp_dones[i][j], 0)
            expert_buffer.add(exp_obs2[i][j], exp_acs2[i][j], exp_rews2[i][j], np.zeros(obs_shape), exp_dones2[i][j], 0)
    
    print("asdf", expert_buffer.sample(2)[0].shape)
    
    policy_buffer = ReplayBuffer((obs_shape,), (acs_shape,), obs_steps * 32, device, window=32)
    
    net_arch = [dict(pi=[args.hidden_dim]*args.num_layer, 
                     vf=[args.hidden_dim]*args.num_layer)]
    policy_kwargs = dict(net_arch=net_arch)
    
    airl_ppo = PPO_AIRL(
        MlpPolicy, env,
        tensorboard_log=args.tensorboard_log, 
        seed=args.seed, 
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        use_sde=use_sde,
        sde_sample_freq=args.sde_freq,
        gae_lambda=args.gae_lambda,
        clip_range=clip_range,
        n_epochs=args.n_epochs,
        metaworld_flag=metaworld_flag,
        verbose=1,
        device=device,
        reward_model = reward_model)
    
    airl_steps = 0
    while airl_steps < args.airl_timesteps:
        # Sample trajectories using policy
        print("airl step", airl_steps)
        last_obs = env.reset()
        n_steps = 0
        #expert_buffer.update_log_probs(airl_ppo.policy, device)
        while n_steps < obs_steps:
            print(n_steps)
            obs_tensor = torch.as_tensor(last_obs).to(device)
            # do we want this part of the grad graph?
            with torch.no_grad():
                actions, values, log_probs = airl_ppo.policy.forward(obs_tensor)
                actions = actions.cpu().numpy()
                clipped_actions = actions
                # Clip the actions to avoid out of bound error
                if isinstance(env.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions, env.action_space.low, env.action_space.high)
                new_obs, rewards, dones, infos = env.step(clipped_actions)
                #print(policy_buffer.actions.shape, policy_buffer.obses.shape)
                print(actions.shape, log_probs.shape, new_obs.shape, rewards[:,np.newaxis].shape)
                policy_buffer.add_batch(last_obs, actions, log_probs.cpu().numpy()[:, np.newaxis], new_obs, dones[:,np.newaxis], dones[:,np.newaxis])
                n_steps += 1
        
        reward_model.train_reward_airl(expert_buffer, policy_buffer)
        
        # Update policy PPO with the reward from reward model
        do_setup = airl_steps == 0
        if do_setup:
            total_timesteps, callback = airl_ppo._setup_learn(
                64000, eval_env, None, 64000, 1, None, True, "OnPolicyAlgorithm"
            )
        print(" --------------- START AIRL PPO LEARN AIRL --------------") 
        airl_ppo.learn_airl(64000, callback=callback, do_setup=False)
        airl_steps += obs_steps * obs_envs
    
    
    print(" --------------- START PREF PPO --------------") 
    
    
    # train model
    model = PPO_REWARD(
        reward_model,
        MlpPolicy, env,
        tensorboard_log=args.tensorboard_log, 
        seed=args.seed, 
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        use_sde=use_sde,
        sde_sample_freq=args.sde_freq,
        gae_lambda=args.gae_lambda,
        clip_range=clip_range,
        n_epochs=args.n_epochs,
        num_interaction=args.re_num_interaction,
        num_feed=args.re_num_feed,
        feed_type=args.re_feed_type,
        re_update=args.re_update,
        metaworld_flag=metaworld_flag,
        max_feed=args.re_max_feed, 
        unsuper_step=args.unsuper_step,
        unsuper_n_epochs=args.unsuper_n_epochs,
        size_segment=args.re_segment,
        max_ep_len=max_ep_len,
        verbose=1)

    # save args
    with open(os.path.join(args.tensorboard_log, "args.yml"), "w") as f:
        ordered_args = OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
        yaml.dump(ordered_args, f)
        
    model.learn(total_timesteps=args.total_timesteps, unsuper_flag=1)
    model.reward_model.save(args.tensorboard_log, args.total_timesteps)
