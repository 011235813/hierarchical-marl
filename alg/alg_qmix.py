"""Implementation of QMIX."""

import numpy as np
import tensorflow as tf

import networks
import sys


class Alg(object):

    def __init__(self, config_alg, n_agents, l_state, l_obs, l_action, nn):
        """
        Args:
            config_alg: dictionary of general RL params
            n_agents: number of agents on the team controlled by this alg
            l_state, l_obs, l_action: int
            nn: dictionary with neural net sizes
        """
        self.l_state = l_state
        self.l_obs = l_obs
        self.l_action = l_action
        self.nn = nn

        self.n_agents = n_agents
        self.tau = config_alg['tau']
        self.lr_Q = config_alg['lr_Q']
        self.gamma = config_alg['gamma']

        self.agent_labels = np.eye(self.n_agents)

        # Initialize computational graph
        self.create_networks()
        self.list_initialize_target_ops, self.list_update_target_ops = self.get_assign_target_ops(tf.trainable_variables())
        self.create_train_op()

        # TF summaries
        self.create_summary()

    def create_networks(self):

        # Placeholders
        self.state = tf.placeholder(tf.float32, [None, self.l_state], 'state')
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'obs')

        # Individual agent networks
        # output dimension is [time * n_agents, q-values]
        with tf.variable_scope("Agent_main"):
            self.agent_qs = networks.Qmix_single(self.obs, self.nn['n_h1'], self.nn['n_h2'], self.l_action)
        with tf.variable_scope("Agent_target"):
            self.agent_qs_target = networks.Qmix_single(self.obs, self.nn['n_h1'], self.nn['n_h2'], self.l_action)

        self.argmax_Q = tf.argmax(self.agent_qs, axis=1)
        self.argmax_Q_target = tf.argmax(self.agent_qs_target, axis=1)

        # To extract Q-value from agent_qs and agent_qs_target
        # [batch*n_agents, l_action]
        self.actions_1hot = tf.placeholder(tf.float32, [None, self.l_action], 'actions_1hot')
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
                
    def get_assign_target_ops(self, list_vars):

        # ops for equating main and target
        list_initial_ops = []
        # ops for slow update of target toward main
        list_update_ops = []

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
        
        return list_initial_ops, list_update_ops

    def run_actor(self, list_obs, epsilon, sess):
        """Get actions for all agents as a batch.
        
        Args:
            list_obs: list of vectors, one per agent
            epsilon: exploration parameter
            sess: TF session

        Returns: np.array of action integers
        """
        # convert to batch
        obs = np.array(list_obs)
        feed = {self.obs : obs}
        actions_argmax = sess.run(self.argmax_Q, feed_dict=feed)

        actions = np.zeros(self.n_agents, dtype=int)
        for idx in range(self.n_agents):
            if np.random.rand() < epsilon:
                actions[idx] = np.random.randint(0, self.l_action)
            else:
                actions[idx] = actions_argmax[idx]

        return actions

    def create_train_op(self):
        # TD target calculated in train_step() using Mixer_target
        self.td_target = tf.placeholder(tf.float32, [None], 'td_target')
        self.loss_mixer = tf.reduce_mean(tf.square(self.td_target - tf.squeeze(self.mixer)))

        self.mixer_opt = tf.train.AdamOptimizer(self.lr_Q)
        self.mixer_op = self.mixer_opt.minimize(self.loss_mixer)

    def create_summary(self):

        summaries = [tf.summary.scalar('loss_mixer', self.loss_mixer)]
        mixer_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_main')
        # mixer_main_variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_main')]
        for v in mixer_main_variables:
            summaries.append(tf.summary.histogram(v.op.name, v))
        grads = self.mixer_opt.compute_gradients(self.loss_mixer, mixer_main_variables)
        for grad, var in grads:
            if grad is not None:
                summaries.append( tf.summary.histogram(var.op.name+'/gradient', grad) )

        agent_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Agent_main')
        for v in agent_main_variables:
            summaries.append(tf.summary.histogram(v.op.name, v))
        grads = self.mixer_opt.compute_gradients(self.loss_mixer, agent_main_variables)
        for grad, var in grads:
            if grad is not None:
                summaries.append( tf.summary.histogram(var.op.name+'/gradient', grad) )

        self.summary_op = tf.summary.merge(summaries)

    def process_actions(self, n_steps, actions):
        """
        actions must have shape [time, n_agents],
        and values are action indices
        """
        # Each row of actions is one time step,
        # row contains action indices for all agents
        # Convert to [time, agents, l_action]
        # so each agent gets its own 1-hot row vector
        actions_1hot = np.zeros([n_steps, self.n_agents, self.l_action], dtype=int)
        grid = np.indices((n_steps, self.n_agents))
        actions_1hot[grid[0], grid[1], actions] = 1

        # In-place reshape of actions to [time*n_agents, l_action]
        actions_1hot.shape = (n_steps*self.n_agents, self.l_action)

        return actions_1hot

    def process_batch(self, batch):
        """
        Extract quantities of the same type from batch.
        Format batch so that each agent at each time step is one
        batch entry.
        Duplicate global quantities <n_agents> times to be
        compatible with this scheme.
        """
        # shapes are [time, ...original dims...]
        state = np.stack(batch[:,0]) # [time, l_state]
        obs = np.stack(batch[:,1]) # [time, agents, l_obs]
        actions = np.stack(batch[:,2]) # [time, agents]
        reward = np.stack(batch[:,3]) # [time]
        state_next = np.stack(batch[:,4]) # [time, l_state]
        obs_next = np.stack(batch[:,5]) # [time, agents, l_obs]
        done = np.stack(batch[:,6]) # [time]

        # Try to free memory
        batch = None
    
        n_steps = state.shape[0]
    
        # In-place reshape for obs, so that one time step
        # for one agent is considered one batch entry
        obs.shape = (n_steps * self.n_agents, self.l_obs)
        obs_next.shape = (n_steps * self.n_agents, self.l_obs)

        actions_1hot = self.process_actions(n_steps, actions)
            
        return n_steps, state, obs, actions_1hot, reward, state_next, obs_next, done

    def train_step(self, sess, batch, step_train=0, summarize=False, writer=None):

        # Each agent for each time step is now a batch entry
        n_steps, state, obs, actions_1hot, reward, state_next, obs_next, done = self.process_batch(batch)

        # Get argmax actions from target networks
        feed = {self.obs : obs_next}
        argmax_actions = sess.run(self.argmax_Q_target, feed_dict=feed) # [batch*n_agents]
        # Convert to 1-hot
        actions_target_1hot = np.zeros([n_steps * self.n_agents, self.l_action], dtype=int)
        actions_target_1hot[np.arange(n_steps*self.n_agents), argmax_actions] = 1

        # Get Q_tot target value
        feed = {self.state : state_next,
                self.actions_1hot : actions_target_1hot,
                self.obs : obs_next}
        Q_tot_target = sess.run(self.mixer_target, feed_dict=feed)

        done_multiplier = -(done - 1)
        target = reward + self.gamma * np.squeeze(Q_tot_target) * done_multiplier

        feed = {self.state : state,
                self.actions_1hot : actions_1hot,
                self.obs : obs,
                self.td_target : target}
        if summarize:
            summary, _ = sess.run([self.summary_op, self.mixer_op], feed_dict=feed)
            writer.add_summary(summary, step_train)
        else:
            _ = sess.run(self.mixer_op, feed_dict=feed)

        sess.run(self.list_update_target_ops)
