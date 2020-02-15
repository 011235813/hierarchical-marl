"""Tests a saved policy."""
import sys
import random
import json
import time
import os

sys.path.append('../env/')

import numpy as np
import tensorflow as tf

import env_wrapper
import replay_buffer
import evaluate
import alg_iql
import alg_qmix
import alg_hsd_scripted
import alg_hsd


def test_function(config):

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
    summarize = False
    
    N_test = config_main['N_test']
    measure = config_main['measure']
    test_filename = config_main['test_filename']
    
    N_roles = config_h['N_roles']
    steps_per_assign = config_h['steps_per_assign']
    
    env = env_wrapper.Env(config_env, config_main, test=True, N_roles=N_roles)
    
    l_state = env.state_dim
    l_action = env.action_dim
    l_obs = env.obs_dim
    N_home = config_env['num_home_players']
    
    if alg_name == 'qmix':
        alg = alg_qmix.Alg(config_alg, N_home, l_state, l_obs, l_action, config['nn_qmix'])
    elif alg_name == 'hsd-scripted' or alg_name == 'mara-c':
        alg = alg_hsd_scripted.Alg(alg_name, config_alg, N_home, l_state, l_obs, l_action, N_roles, config['nn_hsd_scripted'])
    elif alg_name == 'iql':
        alg = alg_iql.Alg(config_alg, N_home, l_state, l_obs, l_action, config['nn_iql'])
    elif alg_name == 'hsd':
        alg = alg_hsd.Alg(config_alg, config_h, N_home, l_state, l_obs, l_action, N_roles, config['nn_hsd'])
    
    saver = tf.train.Saver()
    
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    
    print("Restoring variables from %s" % dir_name)
    saver.restore(sess, '../results/%s/%s' % (dir_name, model_name))
    
    if alg_name == 'qmix' or alg_name == 'iql':
        result = evaluate.test(N_test, env, sess, alg,
                               dir_name=dir_name, log=True,
                               test_filename=test_filename)
        r_avg_eval, steps_per_episode, win_rate_home, win_rate_away = result
    elif alg_name == 'hsd-scripted' or alg_name == 'mara-c' or alg_name == 'hsd':
        result = evaluate.test_hierarchy(alg_name, N_test, env, sess,
                                         alg, steps_per_assign,
                                         dir_name=dir_name, log=True,
                                         measure=measure,
                                         test_filename=test_filename,
                                         fixed_idx_skill=config_h['fixed_idx_skill'])
        r_avg_eval, steps_per_episode, win_rate, win_rate_opponent = result


if __name__ == '__main__':

    with open('config.json', 'r') as f:
        config = json.load(f)
    test_function(config)
