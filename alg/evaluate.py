"""Defines functions for running test episodes."""
import numpy as np
import pickle


def test(n_eval, env, sess, alg, dir_name='', log=False, test_filename='test.csv'):
    """
    Args:
        n_eval: number of eval episodes
        env: Environment object
        sess: TF session
        alg: Alg object
        dir_name: folder under '/results/' to save
        log: if True, produces a CSV file
        test_filename: name of log file to write

    Returns:
        1. average episode reward
        2. average total steps in episode
        3. win percentage
        4. opponent win percentage
    """
    epsilon = 0
    reward_total = 0
    score_home = 0
    score_away = 0
    total_steps = 0

    if log:
        header = "Episode,Steps,Reward,Draw,Away_win,Home_win\n"
        with open("../results/%s/%s" % (dir_name, test_filename), 'w') as f:
            f.write(header)

    for idx_eval in range(1, n_eval+1):

        state_home, state_away, list_obs_home, list_obs_away, done = env.reset()
        reward_episode = 0

        while not done:
            actions_int = alg.run_actor(list_obs_home, epsilon, sess)

            state_home_next, state_away_next, list_obs_home_next, list_obs_away_next, reward, local_rewards, done, info = env.step(actions_int)

            state_home = state_home_next
            list_obs_home = list_obs_home_next
            reward_episode += reward

        reward_total += reward_episode
        total_steps += env.env_step
        if info['winning_team'] == 0:
            draw, away_win, home_win = (0, 0, 1)
            score_home += 1
        elif info['winning_team'] == 1:
            draw, away_win, home_win = (0, 1, 0)
            score_away += 1
        else:
            draw, away_win, home_win = (1, 0, 0)

        if log:
            s = "%d,%d,%.2f,%d,%d,%d\n" % (idx_eval, env.env_step, reward_episode, draw, away_win, home_win)
            # print(s)
            with open("../results/%s/%s" % (dir_name, test_filename), 'a') as f:
                f.write(s)

    r_avg_eval, steps_per_episode, win_rate_home, win_rate_away = np.array([reward_total, total_steps, score_home, score_away]) / float(n_eval)

    if log:
        with open("../results/%s/%s" % (dir_name, test_filename), 'a') as f:
            f.write('\n')
            f.write('R_avg,Steps_per_eps,Win_rate_home,Win_rate_away\n')
            s = '%.2f,%d,%.2f,%.2f' % (r_avg_eval, int(steps_per_episode), win_rate_home, win_rate_away)
            f.write(s)
            print(s)

    return r_avg_eval, steps_per_episode, win_rate_home, win_rate_away


def test_hierarchy(alg_name, n_eval, env, sess, alg, steps_per_assign,
                   dir_name='', log=False, measure=False,
                   test_filename='test.csv', fixed_idx_skill=None):
    """
    Args:
        alg_name: string, options are ['hsd', 'hsd-scripted', 'mara-c']
        n_eval: number of eval episodes
        env: Environment object
        sess: TF session
        alg: Alg object
        steps_per_assign: hyperparam of HSD and HSD-scripted
        dir_name: folder under '/results/' to save
        log: if True, produces a CSV file
        measure: if True, measure more stats
        test_filename: name of log file to write
        fixed_idx_skill: if not None, must be a length-2 list of [agent idx, skill idx]
                         Used for testing HSD policy with two teammates with fixed skills

    Returns 
        1. average episode reward
        2. average total steps in episode
        3. win percentage
        4. opponent win percentage
    """
    epsilon = 0
    reward_total = 0
    score_home = 0
    score_away = 0
    total_steps = 0

    if alg_name == 'hsd':
        N_roles = alg.l_z
    else:
        N_roles = alg.N_roles

    if log:
        header = "Episode,Steps,Reward,Draw,Away_win,Home_win\n"
        with open("../results/%s/%s" % (dir_name, test_filename), 'w') as f:
            f.write(header)
            
    if measure:
        # number of times each of 8 event occurs under each skill
        matrix_role_counts = np.zeros([N_roles, n_eval, 8]) # 8 hardcoded events
        # number of times each skill was selected, recorded separately for cases with and without possession
        count_skills = np.zeros([2, n_eval, N_roles]) 
        # number of primitive actions under each skill
        count_low_actions = np.zeros([N_roles, n_eval, env.action_dim])
        # one sequence of skill integer for each agent
        time_series_skills = [ [ [] for idx_agent in range(env.N_home) ] for idx_eval in range(n_eval) ]
        # count of skill usage over (x,y) position, one 2D matrix for each skill
        count_skills_position = np.zeros([N_roles, int(env.z_tot)+1, int(env.x_tot)+1])

    for idx_eval in range(1, n_eval+1):

        state_home, state_away, list_obs_home, list_obs_away, done = env.reset()
        reward_episode = 0

        step_episode = 0
        while not done:

            if step_episode % steps_per_assign == 0:
                if alg_name == 'hsd-scripted':
                    roles_int = alg.assign_roles(list_obs_home, epsilon, sess)
                elif alg_name == 'mara-c':
                    roles_int, idx_action_centralized = alg.assign_roles_centralized(state_home, epsilon, sess)
                elif alg_name == 'hsd':
                    roles_int = alg.assign_roles(list_obs_home, epsilon, sess, N_roles)
                    if fixed_idx_skill:
                        # if not None, then it's a length-2 list of [agent idx, skill idx]
                        roles_int[fixed_idx_skill[0]] = fixed_idx_skill[1]
                roles = np.zeros([env.N_home, N_roles])
                roles[np.arange(env.N_home), roles_int] = 1

                if measure:
                    for idx_agent in range(env.N_home):
                        if env.control_team == 0 and env.control_index == idx_agent:
                            count_skills[0, idx_eval-1, roles_int[idx_agent]] += 1
                        else:
                            count_skills[1, idx_eval-1, roles_int[idx_agent]] += 1
                        time_series_skills[idx_eval-1][idx_agent].append(roles_int[idx_agent])
                        agent_x = int(round(env.state_json['home%d'%idx_agent + '_pos_x'] - env.arena_min_x))
                        agent_z = int(round(env.state_json['home%d'%idx_agent + '_pos_z'] - env.arena_min_z))
                        count_skills_position[roles_int[idx_agent], agent_z, agent_x] += 1
            
            actions_int = alg.run_actor(list_obs_home, roles, epsilon, sess)

            state_home_next, state_away_next, list_obs_home_next, list_obs_away_next, reward, local_rewards, done, info = env.step(actions_int, roles_int, measure)
            if measure:
                matrix_role_counts[:, idx_eval-1, :] += info['matrix_role_counts']
                for idx_agent in range(env.N_home):
                    count_low_actions[roles_int[idx_agent], idx_eval-1, actions_int[idx_agent]] += 1

            step_episode += 1

            state_home = state_home_next
            list_obs_home = list_obs_home_next
            reward_episode += reward

        reward_total += reward_episode
        total_steps += env.env_step
        if info['winning_team'] == 0:
            draw, away_win, home_win = (0, 0, 1)
            score_home += 1
        elif info['winning_team'] == 1:
            draw, away_win, home_win = (0, 1, 0)
            score_away += 1
        else:
            draw, away_win, home_win = (1, 0, 0)

        if log:
            s = "%d,%d,%.2f,%d,%d,%d\n" % (idx_eval, env.env_step, reward_episode, draw, away_win, home_win)
            # print(s)
            with open("../results/%s/%s" % (dir_name, test_filename), 'a') as f:
                f.write(s)

    r_avg_eval, steps_per_episode, win_rate_home, win_rate_away = np.array([reward_total, total_steps, score_home, score_away]) / float(n_eval)

    if log:
        with open("../results/%s/%s" % (dir_name, test_filename), 'a') as f:
            f.write('\n')
            f.write('R_avg,Steps_per_eps,Win_rate_home,Win_rate_away\n')
            s = '%.2f,%d,%.2f,%.2f' % (r_avg_eval, int(steps_per_episode), win_rate_home, win_rate_away)
            f.write(s)
            print(s)
        if measure:
            with open('../results/%s/matrix_role_counts.pkl' % dir_name, 'wb') as f:
                pickle.dump(matrix_role_counts, f)
            with open('../results/%s/count_skills.pkl' % dir_name, 'wb') as f:
                pickle.dump(count_skills, f)
            with open('../results/%s/count_low_actions.pkl' % dir_name, 'wb') as f:
                pickle.dump(count_low_actions, f)
            with open('../results/%s/time_series_skills.pkl' % dir_name, 'wb') as f:
                pickle.dump(time_series_skills, f)
            with open('../results/%s/count_skills_position.pkl' % dir_name, 'wb') as f:
                pickle.dump(count_skills_position, f)

    return r_avg_eval, steps_per_episode, win_rate_home, win_rate_away
