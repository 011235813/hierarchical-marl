from multiprocessing import Process
import json

import train_hsd_scripted
import train_hsd
import train_offpolicy

processes = []

with open('config.json', 'r') as f:
    config = json.load(f)

alg_name = config['main']['alg_name']

N_seeds = config['main']['N_seeds']
seed_base = config['main']['seed']
dir_name_base = config['main']['dir_name']
dir_idx_start = config['main']['dir_idx_start']

if alg_name == 'qmix' or alg_name == 'iql':
    train_function = train_offpolicy.train_function
elif alg_name == 'hsd-scripted' or alg_name == 'mara-c':
    train_function = train_hsd_scripted.train_function
elif alg_name == 'hsd' and config['h_params']['low_level_alg'] == 'iql':
    train_function = train_hsd.train_function

for idx_run in range(N_seeds):

    # Give each run a unique seed and log folder
    config_copy = config.copy()
    config_copy['main']['seed'] = seed_base + idx_run
    config_copy['main']['dir_name'] = dir_name_base + '_{:1d}'.format(dir_idx_start + idx_run)
    
    p = Process(target=train_function, args=(config_copy,))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
