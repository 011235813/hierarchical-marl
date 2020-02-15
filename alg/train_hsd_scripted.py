"""Entry point for training HSD-scripted.

Trains high-level role assignment policy with environment reward
Trains low-level action policies with role-specific rewards given by environment
"""

import json
import os
import random
import sys
import time

sys.path.append('../env/')

import numpy as np
import tensorflow as tf

import alg_hsd_scripted
import alg_qmix
import env_wrapper
import evaluate
import replay_buffer


def train_function(config):

    config_env = config['env']
    config_main = config['main']
    config_alg = config['alg']
    config_h = config['h_params']
    
    seed = config_main['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    
    alg_name = config_main['alg_name']
    dir_name = config_main['dir_name']
    model_name = config_main['model_name']
    summarize = config_main['summarize']
    save_period = config_main['save_period']
    
    os.makedirs('../results/%s'%dir_name, exist_ok=True)
    
    N_train = config_alg['N_train']
    N_eval = config_alg['N_eval']
    period = config_alg['period']
    buffer_size = config_alg['buffer_size']
    batch_size = config_alg['batch_size']
    pretrain_episodes = config_alg['pretrain_episodes']
    steps_per_train = config_alg['steps_per_train']
    
    epsilon_start = config_alg['epsilon_start']
    epsilon_end = config_alg['epsilon_end']
    epsilon_div = config_alg['epsilon_div']
    epsilon_step = (epsilon_start - epsilon_end)/float(epsilon_div)
    epsilon = epsilon_start
    
    N_roles = config_h['N_roles']
    steps_per_assign = config_h['steps_per_assign']
    # Each <steps_per_assign> is one "step" for the high-level policy
    # This means we train the high-level policy once for every
    # <steps_per_train> high-level steps
    steps_per_train_h = steps_per_assign * steps_per_train
    
    env = env_wrapper.Env(config_env, config_main)
    
    l_state = env.state_dim
    l_action = env.action_dim
    l_obs = env.obs_dim
    N_home = config_env['num_home_players']
    
    if config_main['alg_name'] == 'qmix':
        alg = alg_qmix.Alg(config_alg, N_home, l_state, l_obs, l_action, config['nn_qmix'])
    elif alg_name == 'hsd-scripted' or alg_name == 'mara-c':
        alg = alg_hsd_scripted.Alg(alg_name, config_alg, N_home, l_state, l_obs, l_action, N_roles, config['nn_hsd_scripted'])
    
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    
    sess.run(alg.list_initialize_target_ops)
    
    if summarize:
        writer = tf.summary.FileWriter('../results/%s' % dir_name, sess.graph)
    saver = tf.train.Saver(max_to_keep=config_main['max_to_keep'])
    
    # Buffer for high level role assignment policy
    buf_high = replay_buffer.Replay_Buffer(size=buffer_size)
    # Buffer for low level agent policy
    buf_low = replay_buffer.Replay_Buffer(size=buffer_size)
    
    # Logging
    header = "Episode,Step,Step_train,R_avg,R_eval,Steps_per_eps,Opp_win_rate,Win_rate,T_env,T_alg\n"
    with open("../results/%s/log.csv" % dir_name, 'w') as f:
        f.write(header)
    
    t_start = time.time()
    t_env = 0
    t_alg = 0
    
    reward_period = 0
    
    step = 0
    step_train = 0
    step_h = 0
    for idx_episode in range(1, N_train+1):
    
        state_home, state_away, list_obs_home, list_obs_away, done = env.reset()
    
        # Variables with suffix _h are high-level quantities for training the role assignment policy
        # These are the high-level equivalent of the s_t in a usual transition tuple (s_t, a_t, s_{t+1})
        state_home_h, state_away_h, list_obs_home_h, list_obs_away_h = state_home, state_away, list_obs_home, list_obs_away
        # Cumulative discounted reward for high-level policy
        reward_h = 0
        # Action taken by high-level role assignment policy
        roles_int = np.random.randint(0, N_roles, N_home)
    
        reward_episode = 0
        summarized = 0
        summarized_h = 0
        step_episode = 0 # steps within an episode
        while not done:
            
            if step_episode % steps_per_assign == 0:
                if step_episode != 0:
                    # The environment state at this point, e.g. <state_home>,
                    # acts like the "next state" for the high-level policy
                    # All of the intervening environment steps act as a single step for the high-level policy
                    r_discounted = reward_h * (config_alg['gamma']**steps_per_assign)
                    if alg_name == 'hsd-scripted':
                        buf_high.add( np.array([ state_home_h, np.array(list_obs_home_h), roles_int, r_discounted, state_home, np.array(list_obs_home), done ]) )
                    elif alg_name == 'mara-c':
                        buf_high.add( np.array([ state_home_h, idx_action_centralized, r_discounted, state_home, done ]) )
                step_h += 1
                    
                # Get new role assignment, i.e. take high-level action
                if idx_episode < pretrain_episodes:
                    roles_int = np.random.randint(0, N_roles, N_home)
                    if alg_name == 'mara-c':
                        idx_action_centralized = np.random.randint(0, alg.dim_role_space)
                else:
                    t_alg_start = time.time()
                    if alg_name == 'hsd-scripted':
                        roles_int = alg.assign_roles(list_obs_home, epsilon, sess)
                    elif alg_name == 'mara-c':
                        roles_int, idx_action_centralized = alg.assign_roles_centralized(state_home, epsilon, sess)
                    t_alg += time.time() - t_alg_start
                roles = np.zeros([N_home, N_roles])
                roles[np.arange(N_home), roles_int] = 1
    
                if (idx_episode >= pretrain_episodes) and (step_h % steps_per_train == 0):
                    # Conduct training of high-level policy
                    batch = buf_high.sample_batch(batch_size)
                    t_alg_start = time.time()
                    if summarize and idx_episode % period == 0 and not summarized_h:
                        alg.train_step(sess, batch, step_train, summarize=True, writer=writer)
                        summarized_h = True
                    else:
                        alg.train_step(sess, batch, step_train, summarize=False, writer=None)
                    step_train += 1
                    t_alg += time.time() - t_alg_start
    
                # Update high-level state
                state_home_h, state_away_h, list_obs_home_h, list_obs_away_h = state_home, state_away, list_obs_home, list_obs_away
    
                reward_h = 0
    
            # Take low-level actions, conditioned on roles
            if idx_episode < pretrain_episodes:
                actions_int = env.random_actions()
            else:
                t_alg_start = time.time()
                actions_int = alg.run_actor(list_obs_home, roles, epsilon, sess)
                t_alg += time.time() - t_alg_start
                
            t_env_start = time.time()
            state_home_next, state_away_next, list_obs_home_next, list_obs_away_next, reward, local_rewards, done, info = env.step(actions_int, roles_int)
            t_env += time.time() - t_env_start
    
            step += 1
            step_episode += 1
    
            l_temp = [np.array(list_obs_home), actions_int, local_rewards, np.array(list_obs_home_next), roles]
            a_temp = np.empty(len(l_temp), dtype=object)
            a_temp[:] = l_temp
            buf_low.add( a_temp )
    
            if (idx_episode >= pretrain_episodes) and (step % steps_per_train == 0):
                # Train low-level policies
                batch = buf_low.sample_batch(batch_size)
                t_alg_start = time.time()
                if summarize and idx_episode % period == 0 and not summarized:
                    alg.train_step_low(sess, batch, step_train, summarize=True, writer=writer)
                    summarized = True
                else:
                    alg.train_step_low(sess, batch, step_train, summarize=False, writer=None)
                step_train += 1
                t_alg += time.time() - t_alg_start
    
            state_home = state_home_next
            list_obs_home = list_obs_home_next
            reward_episode += reward
            reward_h += reward
    
            if done:
                # Since the episode is done, we also terminate the current role assignment period,
                # even if not all <steps_per_assign> have been completed
                r_discounted = reward_h * config_alg['gamma']**(step_episode % steps_per_assign)
                if alg_name == 'hsd-scripted':
                    buf_high.add( np.array([ state_home_h, np.array(list_obs_home_h), roles_int, r_discounted, state_home, np.array(list_obs_home), done]) )
                elif alg_name == 'mara-c':
                    buf_high.add( np.array([ state_home_h, idx_action_centralized, r_discounted, state_home, done ]) )
    
        if idx_episode >= pretrain_episodes and epsilon > epsilon_end:
            epsilon -= epsilon_step
    
        reward_period += reward_episode
    
        if idx_episode == 1 or idx_episode % (5*period) == 0:
            print('{:>10s}{:>10s}{:>12s}{:>8s}{:>8s}{:>15s}{:>15s}{:>10s}{:>12s}{:>12s}'.format(*(header.strip().split(','))))
    
        if idx_episode % period == 0:
            # Evaluation episodes
            r_avg_eval, steps_per_episode, win_rate, win_rate_opponent = evaluate.test_hierarchy(alg_name, N_eval, env, sess, alg, steps_per_assign)
            if win_rate >= config_main['save_threshold']:
                saver.save(sess, '../results/%s/%s-%d' % (dir_name, "model_good.ckpt", idx_episode))
    
            s = '%d,%d,%d,%.2f,%.2f,%d,%.2f,%.2f,%.5e,%.5e\n' % (idx_episode, step, step_train, reward_period/float(period), r_avg_eval, steps_per_episode, win_rate_opponent, win_rate, t_env, t_alg)
            with open('../results/%s/log.csv' % dir_name, 'a') as f:
                f.write(s)
            print('{:10d}{:10d}{:12d}{:8.2f}{:8.2f}{:15d}{:15.2f}{:10.2f}{:12.5e}{:12.5e}\n'.format(idx_episode, step, step_train, reward_period/float(period), r_avg_eval, int(steps_per_episode), win_rate_opponent, win_rate, t_env, t_alg))
            reward_period = 0
    
        if idx_episode % save_period == 0:
            saver.save(sess, '../results/%s/%s-%d' % (dir_name, "model.ckpt", idx_episode))
            
    saver.save(sess, '../results/%s/%s' % (dir_name, model_name))
    
    with open('../results/%s/time.txt' % dir_name, 'a') as f:
        f.write('t_env_total,t_env_per_step,t_alg_total,t_alg_per_step\n')
        f.write('%.5e,%.5e,%.5e,%.5e' % (t_env, t_env/step, t_alg, t_alg/step))


if __name__ == '__main__':

    with open('config.json', 'r') as f:
        config = json.load(f)
    train_function(config)        
