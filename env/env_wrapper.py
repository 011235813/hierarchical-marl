"""Wrapper around STS2 simulator.

This implementation supports self-play training of a single team
by transforming the global state to be state_home and state_away,
where each is from the perspective of the home or away team, respectively.

A "role" here is synonymous with a "skill" in the HSD paper.

Assumptions:
1. Equal number of players on each team
2. Either 3 or 5 players on each team
"""

import sys, os
sys.path.append('../../sts2_mf_mcts/')

import json
import random

import numpy as np

from sts2.environment import STS2Environment

puck_actions_3 = ['NONE', 'SHOOT', 'PASS_1', 'PASS_2', 'PASS_3']
puck_actions_5 = ['NONE', 'SHOOT', 'PASS_1', 'PASS_2', 'PASS_3', 'PASS_4', 'PASS_5']
directions_home = [[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]]
directions_away = [[0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]]

actions_home_3 = list(puck_actions_3) + list(directions_home)
actions_home_5 = list(puck_actions_5) + list(directions_home)

actions_away_3 = list(puck_actions_3) + list(directions_away)
actions_away_5 = list(puck_actions_5) + list(directions_away)

map_agent_selfpassaction_3 = {0 : 2, 1 : 3, 2 : 4}
map_agent_selfpassaction_5 = {0 : 2, 1 : 3, 2 : 4, 3 : 5, 4 : 6}


class Env(object):

    def __init__(self, config_env, config_main, test=False, N_roles=None):
        """Set up variables

        Args:
            config_env: dictionary of environment parameters
            config_main: dictionary of main experiment parameters
            test: if True, turns on measurements (e.g. for testing a policy)
            N_roles: only used for extra measurements during test
        """
        self.N_home = config_env['num_home_players']
        self.N_away = config_env['num_away_players']
        self.N_home_ai = config_env['num_home_ai_players']  # Number of home-team scripted agents
        self.N_away_ai = config_env['num_away_ai_players']  # Number of away-team scripted agents

        assert self.N_home == self.N_away, "Must have equal numbers of home and away players"
        assert self.N_home == 3 or self.N_home == 5, "Only 3v3 or 5v5 supported"

        self.env = STS2Environment(
            num_home_players=self.N_home, # Number of home-team players
            num_away_players=self.N_away, # Number of away-team players
            num_home_agents=self.N_home - self.N_home_ai,
            num_away_agents=self.N_away - self.N_away_ai,
            with_pygame=config_main['render'], # Turn on pygame feature, so that you can `render()`
            timeout_ticks=config_env['max_tick'])  # Max-length of one round of game
        self.render = config_main['render']

        if self.N_home == 3:
            self.actions_home = actions_home_3
            self.actions_away = actions_away_3
            self.map_agent_selfpassaction = map_agent_selfpassaction_3
            self.list_idx_shoot_pass = np.arange(1,5)
            self.list_idx_puck_actions = np.arange(5)
        elif self.N_home == 5:
            self.actions_home = actions_home_5
            self.actions_away = actions_away_5
            self.map_agent_selfpassaction = map_agent_selfpassaction_5
            self.list_idx_shoot_pass = np.arange(1,7)
            self.list_idx_puck_actions = np.arange(7)

        self.test = test
        self.N_roles = N_roles

        self.max_steps = config_env['max_steps']
        self.anneal_init_exp = config_env['anneal_init_exp']
        if self.anneal_init_exp and not self.test:
            init_exp_start = config_env['init_exp_start']
            self.init_exp_end = config_env['init_exp_end']
            self.init_exp_step = (self.init_exp_end - init_exp_start) / config_env['init_exp_div']
            self.init_exp = init_exp_start

        self.activate_role_reward = config_env['activate_role_reward']
        self.role_reward_param_file = config_env['role_reward_param_file']
        if self.role_reward_param_file:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.role_reward_param_file), 'r') as f:
                self.map_role_reward_params = json.load(f)

        self.config_env = config_env
        self.config_main = config_main

        self.self_play = config_env['self_play']

        # Do this once at start of training to get state dimension
        self.state_json, status = self.env.reset()
        self.control_team = self.state_json['control_team']
        self.control_index = self.state_json['control_index']
        self.update_puck_state()
        self.populate_constants(self.state_json)
        state_home = self.get_state_home(self.state_json)
        list_obs_home = self.get_obs_home(self.state_json)
        self.state_dim = len(state_home)
        self.action_dim = len(self.actions_home)
        self.obs_dim = len(list_obs_home[0])

    def calc_reward(self, state_prev, state):
        """Computes global team reward.

        Args:
            state_prev: json
            state: json

        Returns:
            either scalar or (scalar, scalar), depending on self_play
        """
        r = 0
        # goal reward
        if state['home_score'] - state_prev['home_score'] > 0:
            r += 1.0
        elif state['away_score'] - state_prev['away_score'] > 0:
            r -= 1.0
                
        # get or lose possession reward
        if state['control_team'] == 0 and state_prev['control_team'] == 1:
            r += 0.1
        elif state['control_team'] == 1 and state_prev['control_team'] == 0:
            r -= 0.1

        # Dense reward for possession. Do not allow sum of dense reward to exceed goal scoring reward
        if self.config_env['dense_reward']:
            if state['control_team'] == 0:
                r += 1.0 / (2 * self.max_steps)
            else:
                r -= 1.0 / (2 * self.max_steps)

        r_home = r
        r_away = -r
        if self.config_env['draw_penalty'] and self.env_step == self.max_steps:
            r_home -= 0.5
            r_away -= 0.5

        if self.self_play:
            return r_home, r_away
        else:
            return r_home

    def calc_local_rewards(self, state_prev, state, roles):
        """Compute individual reward for each agent given its role.
        
        Only used by HSD-scripted
        The following manually designed role-reward pairs are available:
        role 0 - agent is rewarded for having possession at previous state and scoring a goal at current state
        role 1 - agent does not have possession at previous state and gained possession at current state
        role 2 - agent stays near opponent net and there is no opponent in the line between the agent and the teammate with ball
        role 3 - agent stays near its own net and obstructs the line between opponent with the ball and the net

        Args:
            state_prev: json
            state: json
            roles: np.array of integers

        Returns:
            np.array of reward values
        """
        local_rewards = np.zeros(self.N_home)
        for idx_agent in range(self.N_home):
            role = roles[idx_agent]
            if role == 0:
                # Check for possession at previous state
                if state_prev['control_team'] == 0 and state_prev['control_index'] == idx_agent:
                    # Check for scoring at current state
                    if state['home_score'] - state_prev['home_score'] > 0:
                        # give_reward = True
                        local_rewards[idx_agent] = 1
            elif role == 1:
                # Check for lack of possession at previous state and possession at current state
                if state_prev['control_team'] == 1 and state['control_team'] == 0 and state['control_index'] == idx_agent:
                    # give_reward = True
                    local_rewards[idx_agent] = 1
            elif role == 2:
                # This role receives reward only when this agent's team, but not this agent, has possession
                if state['control_team'] == 0 and state['control_index'] != idx_agent:
                    # Part 1: Compute distance to goal
                    x = state['home%d'%idx_agent + '_pos_x']
                    z = state['home%d'%idx_agent + '_pos_z']
                    dist_to_goal = np.linalg.norm( [x - self.away_net_x, z - self.away_net_z] )
                    # Higher reward for being closer to goal. Range [0,1]
                    r_dist = 1 - dist_to_goal / self.max_dist_from_goal
                    
                    # Part 2: compute smallest orthogonal distance from any opponent to the line between this agent
                    # and the teammate with the ball
                    v_ball_agent = np.array([self.p_x - x, self.p_z - z]) # vector from teammate with ball to agent
                    v_ball_agent_normed = v_ball_agent / np.linalg.norm(v_ball_agent)
                    min_orthogonal_distance = np.inf
                    for idx_opponent in range(self.N_away):
                        x_opp = state['away%d'%idx_opponent + '_pos_x']
                        z_opp = state['away%d'%idx_opponent + '_pos_z']
                        v_ball_opponent = np.array([self.p_x - x_opp, self.p_z - z_opp])
                        if v_ball_opponent.dot(v_ball_agent) <= 0 or np.linalg.norm(v_ball_opponent) > np.linalg.norm(v_ball_agent):
                            # opponent is behind the line between teammate with ball and the agent
                            # or opponent is behind the agent, so it cannot intercept
                            continue
                        v_orthogonal = v_ball_opponent - v_ball_agent_normed.dot(v_ball_opponent) * v_ball_agent_normed
                        distance = np.linalg.norm(v_orthogonal)
                        if distance < min_orthogonal_distance:
                            min_orthogonal_distance = distance
                    # Higher reward for longer orthogonal distance
                    if min_orthogonal_distance == np.inf:
                        # no opponent can intercept
                        r_opponent = 1
                    else:
                        r_opponent = min_orthogonal_distance / self.z_tot
                    # scale to [0,1]
                    local_rewards[idx_agent] = r_opponent
            elif role == 3:
                # This role receives reward only when the other team has possession
                if state['control_team'] == 1:
                    # Part 1: Compute distance to its own goal
                    x = state['home%d'%idx_agent + '_pos_x']
                    z = state['home%d'%idx_agent + '_pos_z']
                    dist_to_goal = np.linalg.norm( [x - self.home_net_x, z - self.home_net_z] )
                    # Higher reward for being closer to its own goal. Range [0,1]
                    r_dist = 1 - dist_to_goal / self.max_dist_from_goal

                    # Part 2: compute the orthogonal distance from this agent to the line
                    # between oppponent with the ball and the agent's own goal (where the opponent should attack)
                    v_goal_ball = np.array([self.home_net_x - self.p_x, self.home_net_z - self.p_z])
                    v_goal_ball_normed = v_goal_ball / np.linalg.norm(v_goal_ball)
                    v_goal_agent = np.array([self.home_net_x - x, self.home_net_z - z])
                    if np.linalg.norm(v_goal_agent) > np.linalg.norm(v_goal_ball):
                        # agent is behind the opponent, so it cannot block
                        distance = np.inf
                    else:
                        v_orthogonal = v_goal_agent - v_goal_ball_normed.dot(v_goal_agent) * v_goal_ball_normed
                        distance = np.linalg.norm(v_orthogonal)
                    # Higher reward for shorter orthogonal distance
                    if distance == np.inf:
                        r_block = 0
                    else:
                        r_block = 1 - distance / self.z_tot
                    local_rewards[idx_agent] = r_block
            
        return local_rewards

    def calc_local_rewards_parameterized(self, state_prev, state, roles):
        """Currently only supports home team.
        
        Deprecated.
        """
        # Find out whether opponent team made a shoot or pass attempt
        self.away_action_has_shoot = False
        self.away_action_has_pass = False
        for idx_away in range(self.N_away):
            a = state_prev['away%d_action'%idx_away]
            if a == 'SHOOT':
                self.away_action_has_shoot = True
                break
            elif 'PASS' in a:
                self.away_action_has_pass = True
                break

        self.home_action_has_shoot = False
        for idx_home in range(self.N_home):
            a = state_prev['home%d_action'%idx_home]
            if a == 'SHOOT':
                self.home_action_has_shoot = True
                break

        local_rewards = np.zeros(self.N_home)
        for idx_agent in range(self.N_home):
            role = roles[idx_agent]
            z = state['home%d'%idx_agent + '_pos_z']
            # Vector of values for each reward condition
            reward_params = self.map_role_reward_params[str(role)]
            if reward_params[0] != 0:
                # Role: Score a goal
                # Check for possession at previous state
                if state_prev['control_team'] == 0 and state_prev['control_index'] == idx_agent:
                    # Check for scoring at current state
                    if state['home_score'] - state_prev['home_score'] > 0:
                        local_rewards[idx_agent] += reward_params[0]
            if reward_params[1] != 0:
                # Role: offensive rebound
                if self.home_action_has_shoot and state_prev['home_score'] == state['home_score'] and state['control_team'] == 0 and state['control_index'] == idx_agent:
                    local_rewards[idx_agent] += reward_params[1]
            if reward_params[2] != 0:
                # Role: staying on offensive side of field
                if np.abs( z - self.away_net_z ) < self.z_tot / 3:
                    local_rewards[idx_agent] += reward_params[2]
            if reward_params[3] != 0:
                # Role: staying on defensive side of field
                if np.abs( z - self.home_net_z ) < self.z_tot / 3:
                    local_rewards[idx_agent] += reward_params[3]
            if reward_params[4] != 0:
                # Role: staying in midfield
                if self.away_net_z + self.z_tot/3 <= z and z <= self.home_net_z - self.z_tot/3:
                    local_rewards[idx_agent] += reward_params[4]
            if reward_params[5] != 0:
                # Role: getting ball from opponent by direct physical contact
                if state_prev['control_team'] == 1 and not self.away_action_has_shoot and not self.away_action_has_pass and state['control_team'] == 0 and state['control_index'] == idx_agent:
                    local_rewards[idx_agent] += reward_params[5]
            if reward_params[6] != 0:
                # Role: Defensive rebound
                if state_prev['control_team'] == 1 and self.away_action_has_shoot and state['control_team'] == 0 and state['control_index'] == idx_agent:
                    local_rewards[idx_agent] += reward_params[6]
            if reward_params[7] != 0:
                # Role: Blocked pass
                if state_prev['control_team'] == 1 and self.away_action_has_pass and state['control_team'] == 0 and state['control_index'] == idx_agent:
                    local_rewards[idx_agent] += reward_params[7]
            # if reward_params[1] != 0:
            #     # Check for lack of possession at previous state and possession at current state
            #     if state_prev['control_team'] == 1 and state['control_team'] == 0 and state['control_index'] == idx_agent:
            #         # give_reward = True
            #         local_rewards[idx_agent] += reward_params[1]

        return local_rewards


    def measure_role_counts(self, state_prev, state, roles):
        """Measure number of occurrences of skills.

        Args:
            state_prev: json
            state: json
            roles: np.array of integers

        Returns:
            matrix of counts, where [r,c] entry is the count for role r and quantity c
        """
        K = 8 # hardcoded number of events to count, see the conditionals below
        matrix_role_counts = np.zeros([ self.N_roles, K ], dtype=np.int)

        # Find out whether opponent team made a shoot or pass attempt
        self.away_action_has_shoot = False
        self.away_action_has_pass = False
        for idx_away in range(self.N_away):
            a = state_prev['away%d_action'%idx_away]
            if a == 'SHOOT':
                self.away_action_has_shoot = True
                break
            elif 'PASS' in a:
                self.away_action_has_pass = True
                break

        self.home_shot_attempt_array = np.zeros(self.N_home, dtype=np.int)
        self.home_prev_action_has_shoot = False
        self.home_prev_action_has_pass = False
        for idx_home in range(self.N_home):
            a_prev = state_prev['home%d_action'%idx_home]
            a = state['home%d_action'%idx_home]
            if a_prev == 'SHOOT':
                self.home_prev_action_has_shoot = True
            elif 'PASS' in a_prev:
                self.home_prev_action_has_pass = True
            if a == 'SHOOT':
                self.home_shot_attempt_array[idx_home] = 1

        for idx_role in range(self.N_roles):
            # Go through each agent who was assigned to that role
            for idx_agent in range(self.N_home):
                if roles[idx_agent] == idx_role:
                    # A whole bunch of conditionals to detect if count should be incremented
                    # Goals
                    if state_prev['control_team'] == 0 and state_prev['control_index'] == idx_agent and state['home_score'] - state_prev['home_score'] > 0:
                        matrix_role_counts[idx_role, 0] += 1
                    # Offensive rebound
                    if self.home_prev_action_has_shoot > 0 and state_prev['home_score'] == state['home_score'] and state['control_team'] == 0 and state['control_index'] == idx_agent:
                        matrix_role_counts[idx_role, 1] += 1
                    # Shot attempts
                    if self.home_shot_attempt_array[idx_agent] == 1:
                        matrix_role_counts[idx_role, 2] += 1
                    # Made pass
                    if self.home_prev_action_has_pass and state_prev['control_team'] == 0 and state_prev['control_index'] == idx_agent and state['control_team'] == 0 and state['control_index'] != idx_agent:
                        matrix_role_counts[idx_role, 3] += 1
                    # Received pass
                    if self.home_prev_action_has_pass and state_prev['control_team'] == 0 and state_prev['control_index'] != idx_agent and state['control_team'] == 0 and state['control_index'] == idx_agent:
                        matrix_role_counts[idx_role, 4] += 1
                    # Steals by direct physical contact
                    if state_prev['control_team'] == 1 and not self.away_action_has_shoot and not self.away_action_has_pass and state['control_team'] == 0 and state['control_index'] == idx_agent:
                        matrix_role_counts[idx_role, 5] += 1
                    # Defensive rebound
                    if state_prev['control_team'] == 1 and self.away_action_has_shoot and state['control_team'] == 0 and state['control_index'] == idx_agent:
                        matrix_role_counts[idx_role, 6] += 1
                    # Blocked pass
                    if state_prev['control_team'] == 1 and self.away_action_has_pass and state['control_team'] == 0 and state['control_index'] == idx_agent:
                        matrix_role_counts[idx_role, 7] += 1

        return matrix_role_counts

    def normalize(self, x, z):
        """Normalize by arena dimension.
        
        Args:
            x: int
            z: int

        Returns:
            normalized (int, int)
        """
        # return (x - self.arena_min_x)/self.x_tot, (z - self.arena_min_z) / self.z_tot
        return x / self.x_tot, z / self.z_tot

    def rel_norm_pos(self, x, z, team):
        """Gets normalized relative distance from own net.

        Computes relative position from goal and normalizes by arena dimension
        This function is invariant under 180 degree rotation and switch of team perspective

        Args:
            x: int
            z: int
            team: whether the transformation is for producing the state vector
            from the 'home' or 'away' perspective

        Returns:
            (int, int)
        """
        if team == 'home':
            x_rel_norm = 2 * (x - self.home_net_x) / self.x_tot
            z_rel_norm = 2 * z / self.z_tot
        elif team == 'away':
            x_rel_norm = 2 * (self.away_net_x - x) / self.x_tot
            z_rel_norm = -2 * z / self.z_tot

        return x_rel_norm, z_rel_norm

    def get_state_home(self, state_json):
        """Gets global state of the home team.

        Transforms the state vector to be from the perspective of the home team
        [position and velocity of puck carrier,
        one-hot vector of team indicating puck carrier,
        position (relative to home goal) and velocity of all team players,
        one-hot vector of opponent team indicating puck carrier,
        position (relative to home goal) and velocity of all opponents]

        Args:
            state_json: json object from STS2 simulator

        Returns:
            np.array
        """
        state = []

        p_x_rel_norm, p_z_rel_norm = self.rel_norm_pos(self.p_x, self.p_z, 'home')
        state += [p_x_rel_norm, p_z_rel_norm, self.p_v_x, self.p_v_z]

        # Home quantities
        l_control = [0] * self.N_home
        if state_json['control_team'] == 0:
            l_control[ int(state_json['control_index']) ] = 1
        state += list(l_control)
        
        for idx in range(self.N_home):
            x = state_json['home%d'%idx + '_pos_x']
            z = state_json['home%d'%idx + '_pos_z']
            x_rel_norm, z_rel_norm = self.rel_norm_pos(x, z, 'home')

            v_x = state_json['home%d'%idx + '_vel_x']
            v_z = state_json['home%d'%idx + '_vel_z']
            
            state += [x_rel_norm, z_rel_norm, v_x, v_z]

        # Away quantities
        l_control = [0] * self.N_away
        if state_json['control_team'] == 1:
            l_control[ int(state_json['control_index']) ] = 1
        state += list(l_control)
        
        for idx in range(self.N_away):
            x = state_json['away%d'%idx + '_pos_x']
            z = state_json['away%d'%idx + '_pos_z']
            x_rel_norm, z_rel_norm = self.rel_norm_pos(x, z, 'home') # from home perspective

            v_x = state_json['away%d'%idx + '_vel_x']
            v_z = state_json['away%d'%idx + '_vel_z']
            
            state += [x_rel_norm, z_rel_norm, v_x, v_z]

        return np.array(state)

    def get_state_away(self, state_json):
        """Gets global state of the away team.

        Transforms the state vector to be from the perspective of the away team
        Same quantities as get_state_home, except that self and opponent are flipped
        e.g. positions are relative to away goal

        Args:
            state_json: json object from STS2 simulator

        Returns:
            np.array
        """
        state = []

        # Puck position and velocity
        p_x_rel_norm, p_z_rel_norm = self.rel_norm_pos(self.p_x, self.p_z, 'away')
        state += [p_x_rel_norm, p_z_rel_norm, -self.p_v_x, -self.p_v_z]

        # Away quantities
        l_control = [0] * self.N_away
        if state_json['control_team'] == 1:
            l_control[ int(state_json['control_index']) ] = 1
        state += list(l_control)
        
        for idx in range(self.N_away):
            x = state_json['away%d'%idx + '_pos_x']
            z = state_json['away%d'%idx + '_pos_z']
            x_rel_norm, z_rel_norm = self.rel_norm_pos(x, z, 'away')

            v_x = state_json['away%d'%idx + '_vel_x']
            v_z = state_json['away%d'%idx + '_vel_z']
            
            state += [x_rel_norm, z_rel_norm, -v_x, -v_z]

        # Home quantities
        l_control = [0] * self.N_home
        if state_json['control_team'] == 0:
            l_control[ int(state_json['control_index']) ] = 1
        state += list(l_control)
        
        for idx in range(self.N_home):
            x = state_json['home%d'%idx + '_pos_x']
            z = state_json['home%d'%idx + '_pos_z']
            x_rel_norm, z_rel_norm = self.rel_norm_pos(x, z, 'away')

            v_x = state_json['home%d'%idx + '_vel_x']
            v_z = state_json['home%d'%idx + '_vel_z']
            
            state += [x_rel_norm, z_rel_norm, -v_x, -v_z]

        return np.array(state)

    def get_obs_home(self, state_json):
        """Gets local observations for each home agent.

        Returns list of np.array, each array is the egocentric observation
        of one agent on the home team
        [relative pos and vel of puck,
        self has puck, team has puck,
        self position relative to goal, self velocity, 
        all teammates' position and velocity relative to self,
        opponent team has puck,
        all opponents' position and velocity relative to self]

        Args:
            state_json: json object from STS2 simulator

        Returns:
            list of np.arrays
        """
        list_obs = []

        if self.control_team == 0:
            team_has_puck = 1
        else:
            team_has_puck = 0
        
        # Create egocentric observation vector for each agent
        for idx in range(self.N_home):
            obs = []

            # Agent's absolute pos and vel
            x = state_json['home%d'%idx + '_pos_x']
            z = state_json['home%d'%idx + '_pos_z']
            v_x = state_json['home%d'%idx + '_vel_x']
            v_z = state_json['home%d'%idx + '_vel_z']

            # rel pos and vel of puck
            x_rel, z_rel = self.normalize( self.p_x - x, self.p_z - z )
            obs += [ x_rel, z_rel, self.p_v_x - v_x, self.p_v_z - v_z ]

            # self has puck, team has puck
            if team_has_puck and idx == self.control_index:
                obs.append( 1 )
            else:
                obs.append( 0 )
            obs.append( team_has_puck )

            # self position relative to goal, self velocity, 
            x_rel_norm, z_rel_norm = self.rel_norm_pos(x, z, 'home')
            v_x = state_json['home%d'%idx + '_vel_x']
            v_z = state_json['home%d'%idx + '_vel_z']
            obs += [x_rel_norm, z_rel_norm, v_x, v_z]

            # all teammates' position and velocity relative to self
            for idx2 in range(self.N_home):
                if idx2 == idx:
                    continue
                else:
                    x2 = state_json['home%d'%idx2 + '_pos_x']
                    z2 = state_json['home%d'%idx2 + '_pos_z']
                    v_x2 = state_json['home%d'%idx2 + '_vel_x']
                    v_z2 = state_json['home%d'%idx2 + '_vel_z']
                    x_rel, z_rel = self.normalize( x2 - x, z2 - z )
                    obs += [x_rel, z_rel, v_x2 - v_x, v_z2 - v_z]

            # opponent team has puck
            obs.append( 1 - team_has_puck )

            # all opponents' position and velocity relative to self
            for idx2 in range(self.N_away):
                x2 = state_json['away%d'%idx2 + '_pos_x']
                z2 = state_json['away%d'%idx2 + '_pos_z']
                v_x2 = state_json['away%d'%idx2 + '_vel_x']
                v_z2 = state_json['away%d'%idx2 + '_vel_z']
                x_rel, z_rel = self.normalize( x2 - x, z2 - z )
                obs += [x_rel, z_rel, v_x2 - v_x, v_z2 - v_z]
            
            list_obs.append( np.array(obs) )

        return list_obs

    def get_obs_away(self, state_json):
        """Gets local observations for each home agent.

        Returns list of np.array, each array is the egocentric observation
        of one agent on the away team.

        Args:
            state_json: json object from STS2 simulator

        Returns:
            list of np.arrays
        """
        list_obs = []

        if self.control_team == 1:
            team_has_puck = 1
        else:
            team_has_puck = 0
        
        # Create egocentric observation vector for each agent
        for idx in range(self.N_away):
            obs = []

            # Agent's absolute pos and vel
            x = state_json['away%d'%idx + '_pos_x']
            z = state_json['away%d'%idx + '_pos_z']
            v_x = state_json['away%d'%idx + '_vel_x']
            v_z = state_json['away%d'%idx + '_vel_z']

            # rel pos and vel of puck
            x_rel, z_rel = self.normalize( -(self.p_x - x), -(self.p_z - z) )
            obs += [ x_rel, z_rel, -(self.p_v_x - v_x), -(self.p_v_z - v_z) ]

            # self has puck, team has puck
            if team_has_puck and idx == self.control_index:
                obs.append( 1 )
            else:
                obs.append( 0 )
            obs.append( team_has_puck )

            # self position relative to goal, self velocity, 
            x_rel_norm, z_rel_norm = self.rel_norm_pos(x, z, 'away')
            v_x = state_json['away%d'%idx + '_vel_x']
            v_z = state_json['away%d'%idx + '_vel_z']
            obs += [x_rel_norm, z_rel_norm, -v_x, -v_z]

            # all teammates' position and velocity relative to self
            for idx2 in range(self.N_away):
                if idx2 == idx:
                    continue
                else:
                    x2 = state_json['away%d'%idx2 + '_pos_x']
                    z2 = state_json['away%d'%idx2 + '_pos_z']
                    v_x2 = state_json['away%d'%idx2 + '_vel_x']
                    v_z2 = state_json['away%d'%idx2 + '_vel_z']
                    x_rel, z_rel = self.normalize( -(x2 - x), -(z2 - z) )
                    obs += [x_rel, z_rel, -(v_x2 - v_x), -(v_z2 - v_z)]

            # opponent team has puck
            obs.append( 1 - team_has_puck )

            # all opponents' position and velocity relative to self
            for idx2 in range(self.N_home):
                x2 = state_json['home%d'%idx2 + '_pos_x']
                z2 = state_json['home%d'%idx2 + '_pos_z']
                v_x2 = state_json['home%d'%idx2 + '_vel_x']
                v_z2 = state_json['home%d'%idx2 + '_vel_z']
                x_rel, z_rel = self.normalize( -(x2 - x), -(z2 - z) )
                obs += [x_rel, z_rel, -(v_x2 - v_x), -(v_z2 - v_z)]
            
            list_obs.append( np.array(obs) )

        return list_obs

    def process_actions(self, actions_int, roles=None, team='home'):
        """Converts integer actions into suitable format for input to STS2.

        Args:
            actions_int: np.array of integers for Home team
            roles: list of ints (only used by HSD-scripted)

        Returns:
            dictionary in format
            {'h_ai_1' : {'action' : <puck action>, 'input' : <direction action>},
             'h_ai_2' : { ... }
            }
        """
        if team == 'home':
            team_count = self.N_home
            team_idx = 0
            prefix = 'h'
            actions_array = self.actions_home
        else:
            team_count = self.N_away
            team_idx = 1
            prefix = 'a'
            actions_array = self.actions_away
        
        actions = {}
        for idx_agent in range(team_count):
            action_int = actions_int[idx_agent]
            if action_int in self.list_idx_shoot_pass and (self.control_team != team_idx or self.control_index != idx_agent):
                # action is to shoot or pass and it does not have puck
                # make it do nothing
                action = {'action' : actions_array[0],
                          'input' : [0.0, 0.0]}
            elif action_int == self.map_agent_selfpassaction[idx_agent] and self.control_team == team_idx and self.control_index == idx_agent:
                # action is to pass to self and it has puck
                # make it do nothing
                action = {'action' : actions_array[0],
                          'input' : [0.0, 0.0]}
            else:
                # Start with do-nothing; only puck action XOR direction action
                action = {'action' : actions_array[0],
                          'input' : [0.0, 0.0]}
                if action_int in self.list_idx_puck_actions:
                    # Puck actions
                    action['action'] = actions_array[action_int]
                else:
                    # Direction actions
                    action['input'] = actions_array[action_int]

            if team == 'home' and roles is not None:  # We may want show roles just for one team?
                action['role'] = roles[idx_agent]

            actions[prefix+'_ai_%d'%(idx_agent+1)] = action

        return actions

    def is_done(self, state_prev, state):
        """Check whether episode is done."""
        info = {}
        if state['home_score'] - state_prev['home_score'] > 0:
            score_changed = 1
            info['winning_team'] = 0
        elif state['away_score'] - state_prev['away_score'] > 0:
            score_changed = 1
            info['winning_team'] = 1
        else:
            score_changed = 0
            info['winning_team'] = -1
            
        done = True if (score_changed or self.env_step == self.max_steps) else False

        return done, info

    def update_puck_state(self):
        """Updates state of the puck."""
        # Puck position and velocity
        if self.control_team == 0:
            prefix = 'home%d' % self.control_index
        else:
            prefix = 'away%d' % self.control_index
        self.p_x, self.p_z, self.p_v_x, self.p_v_z = self.state_json[prefix+'_pos_x'], self.state_json[prefix+'_pos_z'], self.state_json[prefix+'_vel_x'], self.state_json[prefix+'_vel_z']

    def step(self, actions_int, roles=None, measure=False):
        """Main environment step.

        Args:
            actions_int: either np.array of integers or 2-tuple of such arrays (if self_play)
            roles: np.array of integers (only used by HSD-scripted)

        Returns:
            home global state, away global state, 
            list of home players' observations, list of away players' observations,
            global reward, list of local rewards (only for HSD-scripted),
            done indicator, info dict
        """
        state_json_prev = self.state_json

        if self.self_play:
            actions_int_home, actions_int_away = actions_int
            actions_h = self.process_actions(actions_int_home, team='home')
            actions_a = self.process_actions(actions_int_away, team='away')
            actions = {**actions_h, **actions_a}
        else:
            actions = self.process_actions(actions_int, roles=roles, team='home')

        self.state_json, r, d, i = self.env.step(actions)
        self.env.render() if self.render else None
        self.env_step += 1
        self.control_team = self.state_json['control_team']
        self.control_index = self.state_json['control_index']
        self.update_puck_state()

        state_home = self.get_state_home(self.state_json)
        list_obs_home = self.get_obs_home(self.state_json)

        if self.self_play:
            state_away = self.get_state_away(self.state_json)
            list_obs_away = self.get_obs_away(self.state_json)
        else:
            state_away = None
            list_obs_away = []

        reward = self.calc_reward(state_json_prev, self.state_json)

        if self.activate_role_reward:
            if roles is None:
                raise ValueError("env_wrapper.py : activate_role_reward is True but roles is None in step()")
            if self.role_reward_param_file:
                local_rewards = self.calc_local_rewards_parameterized(state_json_prev, self.state_json, roles)
            else:
                local_rewards = self.calc_local_rewards(state_json_prev, self.state_json, roles)
        else:
            local_rewards = np.zeros(self.N_home)

        done, info = self.is_done(state_json_prev, self.state_json)
        if self.test and measure:
            # numpy matrix; entry at (r,c) is count for quantity c for role r
            matrix_role_counts = self.measure_role_counts(state_json_prev, self.state_json, roles)
            info['matrix_role_counts'] = matrix_role_counts

        return state_home, state_away, list_obs_home, list_obs_away, reward, local_rewards, done, info

    def populate_constants(self, state_json):
        """Set up constants."""
        self.arena_min_x = state_json['arena_min_x']
        self.arena_max_x = state_json['arena_max_x']
        self.arena_min_z = state_json['arena_min_z']
        self.arena_max_z = state_json['arena_max_z']

        self.x_tot = self.arena_max_x - self.arena_min_x
        self.z_tot = self.arena_max_z - self.arena_min_z

        self.away_net_x = state_json['away_net_x']
        self.away_net_z = state_json['away_net_z']
        self.home_net_x = state_json['home_net_x']
        self.home_net_z = state_json['home_net_z']

        self.max_dist_from_goal = np.sqrt( self.z_tot**2 + (0.5*self.x_tot)**2 )

        self.home_attack_z = state_json['home_attack_z']
        self.away_attack_z = state_json['away_attack_z']

    def do_nothing_action(self):
        """For dealing with STS2 delays."""
        actions = {}
        for idx_agent in range(self.N_home):
            actions['h_ai_%d'%(idx_agent+1)] = {'action' : self.actions_home[0],
                                                'input' : [0.0, 0.0]}
        return actions

    def step_until_game_on(self):
        """For dealing with STS2 delays."""
        counter = 0
        while self.state_json['current_phase'] != 'GAME_ON':
            # print("Step until game on, current_phase = %s" % self.state_json['current_phase'])
            actions = self.do_nothing_action()
            self.state_json, r, d, i = self.env.step(actions)
            self.env.render() if self.render else None
            counter += 1
            if counter > 20:
                print("env_wrapper.py : stuck in step_until_game_on() due to current_phase=%s. Manually breaking" % self.state_json['current_phase'])
                break

    def reset(self):
        """Episode reset.

        Returns:
        global state from home team perspective, 
        global state from away team perspective, 
        list of home players' observations, 
        list of away players' observations,
        done indicator
        """
        if self.anneal_init_exp and not self.test:
            self.env.game.init_exp = self.init_exp
            if self.init_exp < self.init_exp_end:
                self.init_exp += self.init_exp_step
        self.env.game.SetGamePhase('START_PLAY', False)
        self.env.game.PhaseUpdate(False)

        self.step_until_game_on()

        self.state_json, status = self.env.reset()
        # Game is paused for some simulator steps after reset is called after a goal.
        # Take at least one extra step to check for this paused phase
        tick1 = self.state_json['tick']
        tick2 = tick1
        while tick2 == tick1:
            # print("extra steps")
            self.state_json, r, d, i = self.env.step(self.do_nothing_action())
            self.env.render() if self.render else None
            tick2 = self.state_json['tick']
        
        self.env_step = 0
        
        self.control_team = self.state_json['control_team']
        self.control_index = self.state_json['control_index']
        self.update_puck_state()

        done = False

        state_home = self.get_state_home(self.state_json)
        list_obs_home = self.get_obs_home(self.state_json)

        if self.self_play:
            state_away = self.get_state_away(self.state_json)
            list_obs_away = self.get_obs_away(self.state_json)
        else:
            state_away = None
            list_obs_away = []

        return state_home, state_away, list_obs_home, list_obs_away, done

    def random_actions(self):
        """Returns either np.array of integers, or tuple of such if self_play."""
        if self.self_play:
            actions_int_h = np.random.randint(0, self.action_dim, self.N_home)
            actions_int_a = np.random.randint(0, self.action_dim, self.N_away)
            return actions_int_h, actions_int_a
        else:
            actions_int_h = np.random.randint(0, self.action_dim, self.N_home)
            return actions_int_h
