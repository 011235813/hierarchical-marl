from multiprocessing import Process
import json

import test

processes = []

with open('config.json', 'r') as f:
    config = json.load(f)

N_seeds = config['main']['N_seeds']
seed_base = config['main']['seed']
dir_name_base = config['main']['dir_name']
dir_idx_start = config['main']['dir_idx_start']

test_function = test.test_function

for idx_run in range(N_seeds):

    # Give each run a unique seed and log folder
    config_copy = config.copy()
    config_copy['main']['seed'] = seed_base + idx_run
    config_copy['main']['dir_name'] = dir_name_base + '_{:1d}'.format(dir_idx_start + idx_run)
    
    p = Process(target=test_function, args=(config_copy,))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
