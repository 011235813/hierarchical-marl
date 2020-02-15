"""Implementation of HSD-scripted.

High-level Q-functions Q(s,\zbf) are trained either with QMIX (with decentralized execution)
or Q-learning (centralized execution) using the global reward
Low-level Q-functions Q(o^n,z^n,a^n) are trained with independent Q-learning using local rewards
"""

import tensorflow as tf
import numpy as np
import sys
import networks


class Alg(object):

    def __init__(self, alg_name, config_alg, n_agents, l_state, l_obs, l_action, N_roles, nn):
        """
        Args:
            alg_name: currently supports 'hsd-scripted' for QMIX at high level or 'mara-c' for Q-learning at high level. Low level is always independent Q-learning
            config_alg: dictionary of general RL params
            n_agents: number of agents on the team controlled by this alg
            l_state, l_obs, l_action, N_roles: int
            nn: dictionary with neural net sizes
        """
        self.alg_name = alg_name

        self.l_state = l_state
        self.l_obs = l_obs
        self.l_action = l_action
        self.N_roles = N_roles
        self.nn = nn

        self.n_agents = n_agents
        self.tau = config_alg['tau']
        self.lr_Q = config_alg['lr_Q']
        self.gamma = config_alg['gamma']

        self.agent_labels = np.eye(self.n_agents)

        if self.alg_name == 'mara-c':
            # Combinatorial action space, but we don't allow duplicate roles
            assert(self.N_roles >= self.n_agents)
            self.dim_role_space = int(np.math.factorial(self.N_roles) / np.math.factorial(self.N_roles - self.n_agents))
            self.list_list_indices = []
            self.populate_list_list_roles(0, [])

        # Initialize computational graph
        self.create_networks()
        self.list_initialize_target_ops, self.list_update_target_ops, self.list_update_target_ops_low = self.get_assign_target_ops()
        self.create_train_op()
        self.create_train_op_IQL()

        # TF summaries
        self.create_summary()

    def populate_list_list_roles(self, agent_idx, list_indices):
        
        if len(list_indices) == self.n_agents:
            self.list_list_indices.append(list_indices)
        else:
            for idx_role in range(self.N_roles):
                if idx_role in list_indices:
                    # Skip over duplicate roles
                    continue
                else:
                    l = list(list_indices)
                    l.append(idx_role)
                    self.populate_list_list_roles(agent_idx+1, l)

    def create_networks(self):

        # Placeholders
        self.state = tf.placeholder(tf.float32, [None, self.l_state], 'state')
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'obs')
        self.role = tf.placeholder(tf.float32, [None, self.N_roles], 'role')
        
        # Low-level Q-functions
        with tf.variable_scope("Qlow_main"):
            self.Q_low = networks.Q_low(self.obs, self.role, self.nn['n_h1_low'], self.nn['n_h2_low'], self.l_action)
        with tf.variable_scope("Qlow_target"):
            self.Q_low_target = networks.Q_low(self.obs, self.role, self.nn['n_h1_low'], self.nn['n_h2_low'], self.l_action)

        self.argmax_Q_low = tf.argmax(self.Q_low, axis=1)
        self.argmax_Q_low_target = tf.argmax(self.Q_low_target, axis=1)

        # Low level action
        self.actions_low_1hot = tf.placeholder(tf.float32, [None, self.l_action], 'actions_low_1hot')

        # High-level Q-functions
        if self.alg_name == 'hsd-scripted':
            # Individual agent networks
            # output dimension is [time * n_agents, q-values]
            with tf.variable_scope("Agent_main"):
                self.agent_qs = networks.Qmix_single(self.obs, self.nn['n_h1'], self.nn['n_h2'], self.N_roles)
            with tf.variable_scope("Agent_target"):
                self.agent_qs_target = networks.Qmix_single(self.obs, self.nn['n_h1'], self.nn['n_h2'], self.N_roles)
            
            self.argmax_Q = tf.argmax(self.agent_qs, axis=1)
            self.argmax_Q_target = tf.argmax(self.agent_qs_target, axis=1)
            
            # To extract Q-value from agent_qs and agent_qs_target
            # [batch*n_agents, N_roles]
            self.actions_1hot = tf.placeholder(tf.float32, [None, self.N_roles], 'actions_1hot')
            # [batch*n_agents, 1]
            self.q_selected = tf.reduce_sum(tf.multiply(self.agent_qs, self.actions_1hot), axis=1)
            # [batch, n_agents]
            self.mixer_q_input = tf.reshape( self.q_selected, [-1, self.n_agents] )
            
            self.q_target_selected = tf.reduce_sum(tf.multiply(self.agent_qs_target, self.actions_1hot), axis=1)
            self.mixer_target_q_input = tf.reshape( self.q_target_selected, [-1, self.n_agents] )
            
            # Mixing network
            with tf.variable_scope("Mixer_main"):
                self.mixer = networks.Qmix_mixer(self.mixer_q_input, self.state, self.l_state, self.n_agents, self.nn['n_h_mixer'])
            with tf.variable_scope("Mixer_target"):
                self.mixer_target = networks.Qmix_mixer(self.mixer_target_q_input, self.state, self.l_state, self.n_agents, self.nn['n_h_mixer'])
        elif self.alg_name == 'mara-c':
            # Standard Q-learning for role assignment
            with tf.variable_scope("Qhigh_main"):
                self.Q_high = networks.Q_high(self.state, self.nn['n_h1'], self.nn['n_h2'], self.dim_role_space)
            with tf.variable_scope("Qhigh_target"):
                self.Q_high_target = networks.Q_high(self.state, self.nn['n_h1'], self.nn['n_h2'], self.dim_role_space)
            self.argmax_Q_high = tf.argmax(self.Q_high, axis=1)
            self.argmax_Q_high_target = tf.argmax(self.Q_high_target, axis=1)
            self.actions_high_1hot = tf.placeholder(tf.float32, [None, self.dim_role_space], 'actions_high_1hot')
                
    def get_assign_target_ops(self):

        # ops for equating main and target
        list_initial_ops = []
        # ops for slow update of target toward main
        list_update_ops = []
        # ops for slow update of low-level target toward low-level main
        list_update_ops_low = []

        if self.alg_name == 'hsd-scripted':
            list_Agent_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Agent_main')
            map_name_Agent_main = {v.name.split('main')[1] : v for v in list_Agent_main}
            list_Agent_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Agent_target')
            map_name_Agent_target = {v.name.split('target')[1] : v for v in list_Agent_target}
            
            if len(list_Agent_main) != len(list_Agent_target):
                raise ValueError("get_initialize_target_ops : lengths of Agent_main and Agent_target do not match")
            
            for name, var in map_name_Agent_main.items():
                # create op that assigns value of main variable to
                # target variable of the same name
                list_initial_ops.append( map_name_Agent_target[name].assign(var) )
            
            for name, var in map_name_Agent_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_Agent_target[name].assign( self.tau*var + (1-self.tau)*map_name_Agent_target[name] ) )
            
            list_Mixer_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_main')
            map_name_Mixer_main = {v.name.split('main')[1] : v for v in list_Mixer_main}
            list_Mixer_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_target')
            map_name_Mixer_target = {v.name.split('target')[1] : v for v in list_Mixer_target}
            
            if len(list_Mixer_main) != len(list_Mixer_target):
                raise ValueError("get_initialize_target_ops : lengths of Mixer_main and Mixer_target do not match")
            
            # ops for equating main and target
            for name, var in map_name_Mixer_main.items():
                # create op that assigns value of main variable to
                # target variable of the same name
                list_initial_ops.append( map_name_Mixer_target[name].assign(var) )
            
            # ops for slow update of target toward main
            for name, var in map_name_Mixer_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_Mixer_target[name].assign( self.tau*var + (1-self.tau)*map_name_Mixer_target[name] ) )
        elif self.alg_name == 'mara-c':
            list_Qhigh_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Qhigh_main')
            map_name_Qhigh_main = {v.name.split('main')[1] : v for v in list_Qhigh_main}
            list_Qhigh_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Qhigh_target')
            map_name_Qhigh_target = {v.name.split('target')[1] : v for v in list_Qhigh_target}
            
            if len(list_Qhigh_main) != len(list_Qhigh_target):
                raise ValueError("get_initialize_target_ops : lengths of Qhigh_main and Qhigh_target do not match")
            
            for name, var in map_name_Qhigh_main.items():
                # create op that assigns value of main variable to
                # target variable of the same name
                list_initial_ops.append( map_name_Qhigh_target[name].assign(var) )
            
            for name, var in map_name_Qhigh_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_Qhigh_target[name].assign( self.tau*var + (1-self.tau)*map_name_Qhigh_target[name] ) )
        
        list_Qlow_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Qlow_main')
        map_name_Qlow_main = {v.name.split('main')[1] : v for v in list_Qlow_main}
        list_Qlow_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Qlow_target')
        map_name_Qlow_target = {v.name.split('target')[1] : v for v in list_Qlow_target}
        
        if len(list_Qlow_main) != len(list_Qlow_target):
            raise ValueError("get_initialize_target_ops : lengths of Qlow_main and Qlow_target do not match")
        
        for name, var in map_name_Qlow_main.items():
            # create op that assigns value of main variable to
            # target variable of the same name
            list_initial_ops.append( map_name_Qlow_target[name].assign(var) )
        
        for name, var in map_name_Qlow_main.items():
            # incremental update of target towards main
            list_update_ops_low.append( map_name_Qlow_target[name].assign( self.tau*var + (1-self.tau)*map_name_Qlow_target[name] ) )

        return list_initial_ops, list_update_ops, list_update_ops_low

    def run_actor(self, list_obs, roles, epsilon, sess):
        """Get low-level actions for all agents as a batch.

        Args:
            list_obs: list of vectors, one per agent
            roles: np.array where each row is a 1-hot vector
            epsilon: exploration parameter
            sess: TF session

        Returns: np.array of action integers
        """
        # convert to batch
        obs = np.array(list_obs)
        feed = {self.obs : obs, self.role : roles}
        actions_argmax = sess.run(self.argmax_Q_low, feed_dict=feed)

        actions = np.zeros(self.n_agents, dtype=int)
        for idx in range(self.n_agents):
            if np.random.rand() < epsilon:
                actions[idx] = np.random.randint(0, self.l_action)
            else:
                actions[idx] = actions_argmax[idx]

        return actions

    def assign_roles(self, list_obs, epsilon, sess):
        """Get high-level role assignment actions for all agents.
        
        Args:
            list_obs: list of vectors, one per agent
            epsilon: exploration parameter
            sess: TF session

        Returns: np.array of role indices
        """
        obs = np.array(list_obs)
        feed = {self.obs : obs}
        roles_argmax = sess.run(self.argmax_Q, feed_dict=feed)

        roles = np.zeros(self.n_agents, dtype=int)
        for idx in range(self.n_agents):
            if np.random.rand() < epsilon:
                roles[idx] = np.random.randint(0, self.N_roles)
            else:
                roles[idx] = roles_argmax[idx]

        return roles

    def assign_roles_centralized(self, state, epsilon, sess):
        """Centralized skill selection for all agents.
        
        Directly samples one single action index from high-level Q function
        Maps action index to roles for all agents

        Returns np.array of role indices
        """
        if np.random.rand() < epsilon:
            idx = np.random.randint(0, self.dim_role_space)
        else:
            feed = {self.state : np.array([state])}
            idx = sess.run(self.argmax_Q_high, feed_dict=feed)
        roles = np.array( self.list_list_indices[int(idx)] )

        return roles, np.squeeze(idx)

    def create_train_op(self):

        self.td_target = tf.placeholder(tf.float32, [None], 'td_target')
        if self.alg_name == 'hsd-scripted':
            # TD target calculated in train_step() using Mixer_target
            self.loss_Q_high = tf.reduce_mean(tf.square(self.td_target - tf.squeeze(self.mixer)))
        elif self.alg_name == 'mara-c':
            # Treat the role assignment as single-agent Q-learning
            self.td_error_Q = self.td_target - tf.reduce_sum(tf.multiply(self.Q_high, self.actions_high_1hot), axis=1)
            self.loss_Q_high = tf.reduce_mean(tf.square(self.td_error_Q))
            
        self.Q_opt = tf.train.AdamOptimizer(self.lr_Q)
        self.Q_op = self.Q_opt.minimize(self.loss_Q_high)

    def create_train_op_IQL(self):
        self.td_target_IQL = tf.placeholder(tf.float32, [None], 'td_target_IQL')
        self.td_error = self.td_target_IQL - tf.reduce_sum(tf.multiply(self.Q_low, self.actions_low_1hot), axis=1)
        self.loss_IQL = tf.reduce_mean(tf.square(self.td_error))

        self.IQL_opt = tf.train.AdamOptimizer(self.lr_Q)
        self.IQL_op = self.IQL_opt.minimize(self.loss_IQL)

    def create_summary(self):
        
        if self.alg_name == 'hsd-scripted':
            summaries = [tf.summary.scalar('loss_Q_high', self.loss_Q_high)]
            mixer_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_main')
            for v in mixer_main_variables:
                summaries.append(tf.summary.histogram(v.op.name, v))
            grads = self.Q_opt.compute_gradients(self.loss_Q_high, mixer_main_variables)
            for grad, var in grads:
                if grad is not None:
                    summaries.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
            
            agent_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Agent_main')
            for v in agent_main_variables:
                summaries.append(tf.summary.histogram(v.op.name, v))
            grads = self.Q_opt.compute_gradients(self.loss_Q_high, agent_main_variables)
            for grad, var in grads:
                if grad is not None:
                    summaries.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
        elif self.alg_name == 'mara-c':
            summaries = [tf.summary.scalar('loss_Q_high', self.loss_Q_high)]
            Q_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Qhigh_main')
            for v in Q_main_variables:
                summaries.append(tf.summary.histogram(v.op.name, v))
            grads = self.Q_opt.compute_gradients(self.loss_Q_high, Q_main_variables)
            for grad, var in grads:
                if grad is not None:
                    summaries.append( tf.summary.histogram(var.op.name+'/gradient', grad) )

        self.summary_op = tf.summary.merge(summaries)

        summaries_low = [tf.summary.scalar('loss_IQL', self.loss_IQL)]
        Qlow_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Qlow_main')
        for v in Qlow_main_variables:
            summaries_low.append(tf.summary.histogram(v.op.name, v))
        grads = self.IQL_opt.compute_gradients(self.loss_IQL, Qlow_main_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_low.append( tf.summary.histogram(var.op.name+'/gradient', grad) )

        self.summary_op_low = tf.summary.merge(summaries_low)

    def process_actions(self, n_steps, actions, n_actions):
        """
        Args:
            n_steps: number of steps in trajectory
            actions: must have shape [time, n_agents], and values are action indices
            n_actions: dimension of action space

        Returns: 1-hot representation of actions
        """
        # Each row of actions is one time step,
        # row contains action indices for all agents
        # Convert to [time, agents, N_roles]
        # so each agent gets its own 1-hot row vector
        actions_1hot = np.zeros([n_steps, self.n_agents, n_actions], dtype=int)
        grid = np.indices((n_steps, self.n_agents))
        actions_1hot[grid[0], grid[1], actions] = 1

        # In-place reshape of actions to [time*n_agents, N_roles]
        actions_1hot.shape = (n_steps*self.n_agents, n_actions)

        return actions_1hot

    def process_batch(self, batch):
        """Used for high-level buffer
        
        Extract quantities of the same type from batch.
        Format batch so that each agent at each time step is one batch entry.
        """
        # shapes are [time, ...original dims...]
        if self.alg_name == 'hsd-scripted':
            state = np.stack(batch[:,0]) # [time, l_state]
            obs = np.stack(batch[:,1]) # [time, agents, l_obs]
            actions = np.stack(batch[:,2]) # [time, agents]
            reward = np.stack(batch[:,3]) # [time]
            state_next = np.stack(batch[:,4]) # [time, l_state]
            obs_next = np.stack(batch[:,5]) # [time, agents, l_obs]
            done = np.stack(batch[:,6]) # [time]
        elif self.alg_name == 'mara-c':
            state = np.stack(batch[:,0]) # [time, l_state]
            actions = np.stack(batch[:,1]) # [time]
            reward = np.stack(batch[:,2]) # [time]
            state_next = np.stack(batch[:,3]) # [time, l_state]
            done = np.stack(batch[:,4]) # [time]

        # Try to free memory
        batch = None
    
        n_steps = state.shape[0]

        if self.alg_name == 'hsd-scripted':
            # In-place reshape for obs, so that one time step
            # for one agent is considered one batch entry
            obs.shape = (n_steps * self.n_agents, self.l_obs)
            obs_next.shape = (n_steps * self.n_agents, self.l_obs)
            actions_1hot = self.process_actions(n_steps, actions, self.N_roles)
            return n_steps, state, obs, actions_1hot, reward, state_next, obs_next, done
        elif self.alg_name == 'mara-c':
            actions_1hot = np.zeros([n_steps, self.dim_role_space])
            actions_1hot[np.arange(n_steps), actions] = 1
            return n_steps, state, actions_1hot, reward, state_next, done


    def process_batch_low(self, batch):
        """
        Extract quantities of the same type from batch.
        Format batch so that each agent at each time step is one batch entry.
        """
        # shapes are [time, ...original dims...]
        # state = np.stack(batch[:,0]) # [time, l_state]
        obs = np.stack(batch[:,0]) # [time, agents, l_obs]
        actions = np.stack(batch[:,1]) # [time, agents]
        rewards = np.stack(batch[:,2]) # [time, agents]
        # state_next = np.stack(batch[:,4]) # [time, l_state]
        obs_next = np.stack(batch[:,3]) # [time, agents, l_obs]
        # done = np.stack(batch[:,6]) # [time]
        roles = np.stack(batch[:,4]) # [time, agents, N_roles]

        # Try to free memory
        batch = None
    
        n_steps = obs.shape[0]

        # In-place reshape for obs, so that one time step
        # for one agent is considered one batch entry
        obs.shape = (n_steps * self.n_agents, self.l_obs)
        obs_next.shape = (n_steps * self.n_agents, self.l_obs)
        rewards.shape = (n_steps * self.n_agents)
        roles.shape = (n_steps * self.n_agents, self.N_roles)

        actions_1hot = self.process_actions(n_steps, actions, self.l_action)
            
        return n_steps, obs, actions_1hot, rewards, obs_next, roles

    def train_step(self, sess, batch, step_train=0, summarize=False, writer=None):
        """Training step for role assignment policy via QMIX or Q-learning."""
        if self.alg_name == 'hsd-scripted':
            # Each agent for each time step is now a batch entry
            n_steps, state, obs, actions_1hot, reward, state_next, obs_next, done = self.process_batch(batch)

            # Get argmax actions from target networks
            feed = {self.obs : obs_next}
            argmax_actions = sess.run(self.argmax_Q_target, feed_dict=feed) # [batch*n_agents]
            # Convert to 1-hot
            actions_target_1hot = np.zeros([n_steps * self.n_agents, self.N_roles], dtype=int)
            actions_target_1hot[np.arange(n_steps*self.n_agents), argmax_actions] = 1
            
            # Get Q_tot target value
            feed = {self.state : state_next,
                    self.actions_1hot : actions_target_1hot,
                    self.obs : obs_next}
            Q_tot_target = sess.run(self.mixer_target, feed_dict=feed)

            done_multiplier = -(done - 1)
            target = reward + self.gamma * np.squeeze(Q_tot_target) * done_multiplier
        elif self.alg_name == 'mara-c':
            n_steps, state, actions_1hot, reward, state_next, done = self.process_batch(batch)

            feed = {self.state : state_next}
            Q_target = sess.run(self.Q_high_target, feed_dict=feed)
            target = reward + self.gamma * np.max(Q_target, axis=1)

        feed = {self.state : state, self.td_target : target}
        if self.alg_name == 'hsd-scripted':
            feed[self.obs] = obs
            feed[self.actions_1hot] = actions_1hot
        elif self.alg_name == 'mara-c':
            feed[self.actions_high_1hot] = actions_1hot

        if summarize:
            summary, _ = sess.run([self.summary_op, self.Q_op], feed_dict=feed)
            writer.add_summary(summary, step_train)
        else:
            _ = sess.run(self.Q_op, feed_dict=feed)

        sess.run(self.list_update_target_ops)

    def train_step_low(self, sess, batch, step_train=0, summarize=False, writer=None):
        """Training step for low-level action policy.
        
        Runs independent Q-learning on each agent's experiences using local rewards
        """
        n_steps, obs, actions_1hot, rewards, obs_next, roles = self.process_batch_low(batch)

        # Get target values
        feed = {self.obs : obs_next, self.role : roles}
        Q_target = sess.run(self.Q_low_target, feed_dict=feed)
        target = rewards + self.gamma * np.max(Q_target, axis=1)

        feed = {self.obs : obs,
                self.actions_low_1hot : actions_1hot,
                self.role : roles,
                self.td_target_IQL : target}
        if summarize:
            summary, _ = sess.run([self.summary_op_low, self.IQL_op], feed_dict=feed)
            writer.add_summary(summary, step_train)
        else:
            _ = sess.run(self.IQL_op, feed_dict=feed)

        sess.run(self.list_update_target_ops_low)
