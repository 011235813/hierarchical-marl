"""Trains off-policy algorithms, such as QMIX and IQL."""

import json
import os
import random
import sys
import time

sys.path.append('../env/')

import numpy as np
import tensorflow as tf

import alg_iql
import alg_qmix
import env_wrapper
import evaluate
import replay_buffer


def train_function(config):

    config_env = config['env']
    config_main = config['main']
    config_alg = config['alg']
    
    seed = config_main['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    
    dir_name = config_main['dir_name']
    model_name = config_main['model_name']
    summarize = config_main['summarize']
    save_period = config_main['save_period']
    
    os.makedirs('../results/%s'%dir_name, exist_ok=True)
    with open('../results/%s/%s'
              % (dir_name, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
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
    
    env = env_wrapper.Env(config_env, config_main)
    config_env_mod = config_env.copy()
    config_env_mod['self_play'] = False # test against stock AI during evaluation episodes
    config_env_mod['num_away_ai_players'] = config_env_mod['num_away_players'] # set number of stock AI
    env_eval = env_wrapper.Env(config_env_mod, config_main)
    self_play = config_env['self_play']
    if self_play:
        assert(config_env['num_away_ai_players'] == 0)
    
    l_state = env.state_dim
    l_action = env.action_dim
    l_obs = env.obs_dim
    N_home = config_env['num_home_players']
    
    if config_main['alg_name'] == 'qmix':
        alg = alg_qmix.Alg(config_alg, N_home, l_state, l_obs, l_action, config['nn_qmix'])
    elif config_main['alg_name'] == 'iql':
        alg = alg_iql.Alg(config_alg, N_home, l_state, l_obs, l_action, config['nn_iql'])
    
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    
    sess.run(alg.list_initialize_target_ops)
    
    if summarize:
        writer = tf.summary.FileWriter('../results/%s' % dir_name, sess.graph)
    saver = tf.train.Saver(max_to_keep=config_main['max_to_keep'])
    
    buf = replay_buffer.Replay_Buffer(size=buffer_size)
    
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
    for idx_episode in range(1, N_train+1):
    
        state_home, state_away, list_obs_home, list_obs_away, done = env.reset()
    
        reward_episode = 0
        summarized = 0
        while not done:
            
            if idx_episode < pretrain_episodes:
                if self_play:
                    actions_int_h, actions_int_a = env.random_actions()
                    actions_int = (actions_int_h, actions_int_a)
                else:
                    actions_int = env.random_actions()
            else:
                t_alg_start = time.time()
                if self_play:
                    actions_int_h = alg.run_actor(list_obs_home, epsilon, sess)
                    actions_int_a = alg.run_actor(list_obs_away, epsilon, sess)
                    actions_int = (actions_int_h, actions_int_a)
                else:
                    actions_int = alg.run_actor(list_obs_home, epsilon, sess)
                t_alg += time.time() - t_alg_start
                
            t_env_start = time.time()
            state_home_next, state_away_next, list_obs_home_next, list_obs_away_next, reward, local_rewards, done, info = env.step(actions_int)
            t_env += time.time() - t_env_start
        
            step += 1
    
            if self_play:
                buf.add( np.array([ state_home, np.array(list_obs_home), actions_int_h, reward[0], state_home_next, np.array(list_obs_home_next), done] ) )
                buf.add( np.array([ state_away, np.array(list_obs_away), actions_int_a, reward[1], state_away_next, np.array(list_obs_away_next), done] ) )
            else:
                buf.add( np.array([ state_home, np.array(list_obs_home), actions_int, reward, state_home_next, np.array(list_obs_home_next), done] ) )
    
            if (idx_episode >= pretrain_episodes) and (step % steps_per_train == 0):
                batch = buf.sample_batch(batch_size)
                t_alg_start = time.time()
                if summarize and idx_episode % period == 0 and not summarized:
                    alg.train_step(sess, batch, step_train, summarize=True, writer=writer)
                    summarized = True
                else:
                    alg.train_step(sess, batch, step_train, summarize=False, writer=None)
                step_train += 1
                t_alg += time.time() - t_alg_start
    
            state_home = state_home_next
            list_obs_home = list_obs_home_next
            state_away = state_away_next
            list_obs_away = list_obs_away_next
            if self_play:
                reward_episode += reward[0]
            else:
                reward_episode += reward
    
        if idx_episode >= pretrain_episodes and epsilon > epsilon_end:
            epsilon -= epsilon_step
    
        reward_period += reward_episode
    
        if idx_episode == 1 or idx_episode % (5*period) == 0:
            print('{:>10s}{:>10s}{:>12s}{:>8s}{:>8s}{:>15s}{:>15s}{:>10s}{:>12s}{:>12s}'.format(*(header.strip().split(','))))
    
        if idx_episode % period == 0:
            # Evaluation episodes
            r_avg_eval, steps_per_episode, win_rate, win_rate_opponent = evaluate.test(N_eval, env_eval, sess, alg)
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


if __name__ == "__main__":
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    train_function(config)        
