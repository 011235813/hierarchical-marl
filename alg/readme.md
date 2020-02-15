# Description of parameters in config.json

## main

- `N_seeds` : number of random seeds to use by `train_multiprocess.py` for parallel training runs
- `seed` : random seed, either for single training run, or the starting seed for multi-seed runs
- `dir_name` : name of subfolder under `results/` that will be automatically created to store log and models
- `dir_idx_start` : used only by `train_multiprocess.py`. See the example in the main README.
- `model_name` : when training, this is the name for the final model; when testing, this specifies the name of the model to be tested
- `render` : if true, displays the PyGame interface
- `summarize` : if true, Tensorboard information will be logged during training
- `alg_name` : options are "iql", "qmix", "hsd-scripted", "hsd"
- `save_period` : number of episodes to elapse before each model save
- `max_to_keep` : maximum number of models to save. Earlier models will be deleted
- `save_threshold` : if win rate (evaluated periodically during training) exceeds this threshold, a `model_good.ckpt-<episode>` will be saved
- `N_test` : number of test episodes to run when using `test.py`
- `measure` : if true, additional files "matrix\_role\_counts.pkl", "count\_skills.pkl", and "count\_low\_actions.pkl" will be created

## alg

- `N_train` : total number of game episodes during training
- `N_eval` : number of episodes used for each evaluation step
- `period` : number of training episodes between each evaluation step
- `epsilon_start` : exploration parameter for methods such as Q-learning
- `epsilon_end` : lowerbound on epsilon
- `epsilon_div` : number of episodes to get from `epsilon_start` to `epsilon_end`
- `buffer_size` : size of replay buffer
- `lr_Q`, `lr_actor`, `lr_V`, `lr_decoder` : learning rates for optimizers
- `gamma` : discount factor
- `tau` : scalar coefficient for target network updates
- `batch_size` : size of minibatch sampled from replay buffer for each gradient step
- `pretrain_episodes` : number of episodes to run using random actions before training starts
- `steps_per_train` : number of environment steps between each gradient step for off-policy methods

## h_params
- `N_roles` : max number of skills to learn
- `steps_per_assign` : number of low-level environment steps between each high-level skill selection step
- `N_roles_start` : initial number of skills (currently this must equal `N_roles`, as curriculum is not supported yet)
- `curriculum_threshold` : number of skills increases when decoder performance exceeds this threshold (currently not supported)
- `alpha_start` : initial value of the scalar that trades off intrinsic versus extrinsic reward
- `alpha_end` : minimum value of alpha
- `alpha_step` : alpha decreases by this amount when evaluation win rate exceeds `alpha_threshold`
- `alpha_threshold` : alpha decreases by `alpha_step` when evaluation win rate exceeds `alpha_threshold`
- `N_batch_hsd` : size of dataset used for each decoder training step
- `traj_skip` : downsampling rate for each trajectory used in decoder
- `obs_truncate_length` : each observation vector in the trajectories given to the decoder is truncated to this length
- `use_state_difference` : if true, use the difference between observations, instead of the raw observation itself, for the trajectory input to the decoder
- `low_level_alg` : options are `iql` and `reinforce`. Our method uses `iql`
- `fixed_idx_skill` : either null, or a list with format `[[agent_id_1, skill_idx_1], [agent_id_2, skill_idx_2]]`. Only used for testing HSD policy with two teammates using fixed skills.

## nn_*
- width of each hidden layer in neural nets

## env
- `self_play` : trains with self-play if true, currently only supported for QMIX
- `max_steps` : maximum number of environment steps per episode (at which point it ends in a draw)
- `max_tick` : ignore by setting to absurdly high value
- `episode_max_tick` : ignore by setting to absurdly high value
- `num_home_players` : total number of players on home team
- `num_away_players` : total number of players on away team
- `num_home_ai_players` : number of built-in agents used for the home team
- `num_away_ai_players` : number of built-in agents used for the away team
- `render` and `record_game_state` : used by internal STS2, don't need to change these
- `record_game_state` : keep at False
- `dense_reward` : if true, give reward for having ball possession
- `draw_penalty` : if true, give penalty for episode ending in a draw
- `anneal_init_exp` : if true, player's initial random distance to their own goal will be annealed from small to large
- `init_exp_start`, `init_exp_end`, `init_exp_div` : used if `anneal_init_exp` is true
- `activate_role_reward` : if true, computes individual hand-crafted subtask rewards. Must be true for `hsd-scripted`.
- `role_reward_param_file` : JSON file specifying coefficients for each reward term for each skill. Possible if `activate_role_reward` is true.

