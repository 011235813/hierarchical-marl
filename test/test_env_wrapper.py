import sys
import random
import json

sys.path.append('../env/')

import numpy as np
import tensorflow as tf

import env_wrapper

with open('../alg/config.json', 'r') as f:
    config = json.load(f)

config_env = config['env']
config_main = config['main']
config_alg = config['alg']

config_main['render'] = True

seed = config_main['seed']
np.random.seed(seed)
random.seed(seed)
tf.set_random_seed(seed)

dir_name = config_main['dir_name']
model_name = config_main['model_name']
N_train = config_alg['N_train']
period = config_alg['period']

buffer_size = config_alg['buffer_size']
batch_size = config_alg['batch_size']
pretrain_episodes = config_alg['pretrain_episodes']
steps_per_train = config_alg['steps_per_train']

N_home = config_env['num_home_players']
N_away = config_env['num_away_players']
N_roles = config['h_params']['N_roles']

activate_role_reward = config_env['activate_role_reward']

env = env_wrapper.Env(config_env, config_main)

state_dim = env.state_dim
action_dim = env.action_dim

random_actions = 0
random_roles = 0

print_json = 0
print_state = 0
print_obs = 0
print_action = 0
print_reward = 0
print_local_rewards = 0

step = 0

for idx_episode in range(1, N_train+1):

    print("Episode", idx_episode)

    state_home, state_away, list_obs_home, list_obs_away, done = env.reset()

    while not done:
        
        if random_actions:
            actions_int = env.random_actions()
            count = 1
            if print_action:
                print("Actions_int")
                print(actions_int)
                print('')
        else:
            action = input("Enter actions for all agents separated by comma, or -1 for all random:")
            if action == '-1':
                actions_int = np.random.randint(0, action_dim, N_home)
            else:
                actions_int = list(map(int, action.split(',')))
            count = int(input("Enter number of times to repeat action: "))
                
        if activate_role_reward:
            if random_roles:
                roles_int = np.random.randint(0, N_roles, N_home)
            else:
                roles = input("Enter roles for all agents separated by comma, or -1 for all random:")
                if roles == '-1':
                    roles_int = np.random.randint(0, N_roles, N_home)
                else:
                    roles_int = list(map(int, roles.split(',')))

        idx_count = 0
        while idx_count < count:
            if activate_role_reward:
                state_home, state_away, list_obs_home, list_obs_away, reward, local_rewards, done, info = env.step(actions_int, roles=roles_int)
            else:
                state_home, state_away, list_obs_home, list_obs_away, reward, local_rewards, done, info = env.step(actions_int)
            step += 1
            idx_count += 1

        if print_json:
            for key, val in env.state_json.items():
                print(key, val)

        if print_reward:
            print("Reward", reward)

        if print_local_rewards:
            print("Local rewards", local_rewards)
    
        if print_state:
            print("State_home")
            # print(state_home)
            idx_start = 0
            print("Puck (x,z,v_x,v_z)", state_home[idx_start : idx_start + 4])
            idx_start += 4
            print("Team puck carrier", state_home[idx_start : idx_start + N_home])
            idx_start += N_home
            # print("Team score", state_home[idx_start])
            # idx_start += 1
            for idx in range(N_home):
                print("Team agent", idx, "(x,z,v_x,v_z)", state_home[idx_start + idx*4 : idx_start + (idx+1)*4])
            print('')
            idx_start += 4 * N_home
            print("Opponent puck carrier", state_home[idx_start : idx_start+N_away])
            idx_start += N_away
            # print("Opponent score", state_home[idx_start])
            # idx_start += 1
            for idx in range(N_away):
                print("Opponent agent", idx, "(x,z,v_x,v_z)", state_home[idx_start + idx*4 : idx_start + (idx+1)*4])
            print('')

        if print_obs:
            print("Observations")
            for idx1, obs in enumerate(list_obs_home):
                print("Agent", idx1, "observations")
                idx = 0
                print("Puck relative (x,z,v_x,v_z)", obs[idx : idx + 4])
                idx += 4
                print("Self has puck", obs[idx])
                idx += 1
                print("Team has puck", obs[idx])
                idx += 1
                print("Self (x,z,v_x,v_z)", obs[idx : idx + 4])
                idx += 4
                counter = 0
                for idx2 in range(N_home):
                    if idx2 == idx1:
                        continue
                    else:
                        print("Team agent", idx2, "(x,z,v_x,v_z)", obs[idx + counter*4 : idx + (counter+1)*4])
                    counter += 1
                idx += 4 * (N_home-1)
                print("Opponent has puck", obs[idx])
                idx += 1
                counter = 0
                for idx2 in range(N_away):
                    print("Opponent agent", idx2, "(x,z,v_x,v_z)", obs[idx + counter*4 : idx + (counter+1)*4])
                    counter += 1
                print('')
    
        # input("Enter: ")
        if done:
            print("Episode %d | Total step %d | Episode step %d" % (idx_episode, step, env.env_step))
            # print("info", info)
            input("DONE: ")

