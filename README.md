# Hierarchial Cooperative Multi-Agent Reinforcement Learning with Skill Discovery (HSD)

This is the code for experiments in the paper [Hierarchial Cooperative Multi-Agent Reinforcement Learning with Skill Discovery](https://arxiv.org/abs/1912.03558), published in AAMAS 2020. Ablations and baselines are included.


## Prerequisites

- Python version >= 3.5.2
- TensorFlow 1.13.1
- PyGame 1.9.4
- [STS2 1.0.0](https://github.com/electronicarts/SimpleTeamSportsSimulator). In case of future API changes, our algorithm is compatible with at least this [submit](https://github.com/electronicarts/SimpleTeamSportsSimulator/tree/80b4c688a4b8fb3113f466fae5de060e29c79fbe).

## Project structure

- `alg` : implementation of algorithm, neural networks, `config.json` containing all hyperparameters.
- `env` : implementation of multi-agent wrapper around STS2 simulator.
- `results` : each experiment will create a subfolder that contains log files recorded during training and eval.
- `test` : test scripts

## Training

Each algorithm named `alg_*.py` is run through a script with name `train_*.py`.
The pairings are as follows:
- `train_hsd.py` runs `alg_hsd.py` (HSD)
- `train_offpolicy.py` runs `alg_qmix.py` (QMIX) and `alg_iql.py` (IQL)
- `train_hsd_scripted.py` runs `alg_hsd_scripted.py`

To do multi-seed runs that sweep over the initial random seed, set appropriate choices in config.json and use `train_multiprocess.py`. See example below.

For all algorithms, 
- Activate your TensorFlow (if using `virtualenv`) and allocate GPU using `export CUDA_VISIBLE_DEVICES=<n>` where `n` is some GPU number.
- `cd` into the `alg` folder
- Execute training script, e.g. `python train_hsd.py`
- Periodic training progress is logged in `log.csv`, along with saved models, under `results/<dir_name>`.

### Example 1: training HSD

- Select correct settings in `alg/config.json`. Refer to `config_hsd.json` for an example. The key parameters to set are
  - `"alg_name" : "hsd"`
  - everything under `"h_params"`
  - neural network parameters under `"nn_hsd"`

### Example 2: training QMIX

- Select correct settings in `alg/config.json`. Refer to `config_qmix.json` for an example. The key parameters to set are
  - `"alg_name" : "qmix"`
  - neural network parameters under `"nn_qmix"`

### Example 3 for multi-seed runs

For example, to conduct 5 parallel runs with seeds 12341,12342,...,12345 and save into directory names hsd_1, hsd_2,...,hsd_3 (all under `results/`), set the following parameters in config.json:
- `"N_seeds" : 5`
- `"seed" : 12341`
- `"dir_name" : "hsd"`
- `"dir_idx_start" : 1`


## Testing

### Example 1 for testing HSD

- Choose appropriate settings in `alg/config.json`.

  - `"dir_name" : "hsd_1"`
  - `"model_name" : "model_good.ckpt-<some number>"`
  - `"render" : true` (to see PyGame)
  - `"N_test" : 100` (for 100 test episodes)
  - `"measure" : true` (to enable generation of additional .csv files for analysis of behavior)

- `cd` into the `alg` folder. Execute test script `python test.py`

- Results will be stored in `test.csv` under `results/<dir_name>/`. If `"measure" : true`, then files `matrix_role_counts.pkl`, `count_skills.pkl` and `count_low_actions.pkl` will also be generated.


## Citation

<pre>
@inproceedings{yang2020hierarchical,
  title={Hierarchical Cooperative Multi-Agent Reinforcement Learning with Skill Discovery},
  author={Yang, Jiachen and Borovikov, Igor and Zha, Hongyuan},
  booktitle={Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems},
  pages={1566--1574},
  year={2020}
}
</pre>

## License

HSD is distributed under the terms of the BSD-3 license. All new contributions must be made under this license.

See [LICENSE](LICENSE) for details.

SPDX-License-Identifier: BSD-3

