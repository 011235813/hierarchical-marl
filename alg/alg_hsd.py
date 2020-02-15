"""Implementation of hierarchical cooperative multi-agent RL with skill discovery.

High-level Q-function Q(s,\zbf) is trained with QMIX (with decentralized execution)
using global environment reward

Low-level policies are either 
1. parameterized as policy networks pi(a^n|o^n,z^n) and trained with policy gradient
using the intrinsic reward log P(z|tau) + entropy
or 
2. induced from Q-functions Q(o^n,z^n,a^n) and trained with independent Q-learning
using only log P(z|tau) as delayed reward
"""

import tensorflow as tf
import numpy as np
import sys
import networks


class Alg(object):

    def __init__(self, config_alg, config_h, n_agents, l_state, l_obs, l_action, l_z, nn):
        """
        Args:
            config_alg: dictionary of general RL params
            config_h: dictionary of HSD params
            n_agents: number of agents on the team controlled by this alg
            l_state, l_obs, l_action, l_z: int
            nn: dictionary with neural net sizes
        """
        self.l_state = l_state
        self.l_obs = l_obs
        self.l_action = l_action
        self.l_z = l_z
        self.nn = nn

        self.n_agents = n_agents
        self.tau = config_alg['tau']
        self.lr_Q = config_alg['lr_Q']
        self.lr_actor = config_alg['lr_actor']
        self.lr_decoder = config_alg['lr_decoder']
        self.gamma = config_alg['gamma']

        self.traj_length = config_h['steps_per_assign']
        self.traj_skip = config_h['traj_skip']
        self.traj_length_downsampled = int(np.ceil( self.traj_length / self.traj_skip ))
        self.use_state_difference = config_h['use_state_difference']
        if self.use_state_difference:
            self.traj_length_downsampled -= 1

        # Domain-specific removal of information from agent observation
        # Either none (deactivate) or scalar index where obs should be truncated for use by decoder
        self.obs_truncate_length = config_h['obs_truncate_length']
        assert( (self.obs_truncate_length is None) or (self.obs_truncate_length <= self.l_obs) )

        self.low_level_alg = config_h['low_level_alg']
        assert(self.low_level_alg == 'reinforce' or self.low_level_alg == 'iac' or self.low_level_alg == 'iql')
        if self.low_level_alg == 'iac':
            self.lr_V = config_alg['lr_V']

        # Initialize computational graph
        self.create_networks()
        self.list_initialize_target_ops, self.list_update_target_ops, self.list_update_target_ops_low = self.get_assign_target_ops()
        self.create_train_op_high()
        self.create_train_op_low()
        self.create_train_op_decoder()

        # TF summaries
        self.create_summary()

    def create_networks(self):

        # Placeholders
        self.state = tf.placeholder(tf.float32, [None, self.l_state], 'state')
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'obs')
        self.z = tf.placeholder(tf.float32, [None, self.l_z], 'z')

        # Decoder p(z|tau)
        if self.obs_truncate_length:
            self.traj = tf.placeholder(dtype=tf.float32, shape=[None, self.traj_length_downsampled, self.obs_truncate_length])
        else:
            self.traj = tf.placeholder(dtype=tf.float32, shape=[None, self.traj_length_downsampled, self.l_obs])
        with tf.variable_scope("Decoder"):
            self.decoder_out, self.decoder_probs = networks.decoder(self.traj, self.traj_length_downsampled, self.nn['n_h_decoder'], self.l_z)

        # Low-level policy
        if self.low_level_alg == 'reinforce' or self.low_level_alg == 'iac':
            self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')
            with tf.variable_scope("Policy_main"):
                probs = networks.actor(self.obs, self.z, self.nn['n_h1_low'], self.nn['n_h2_low'], self.l_action)
            self.probs = (1-self.epsilon) * probs + self.epsilon/float(self.l_action)
            self.action_samples = tf.multinomial(tf.log(self.probs), 1)

        if self.low_level_alg == 'iac':
            with tf.variable_scope("V_main"):
                self.V = networks.critic(self.obs, self.z, self.nn['n_h1_low'], self.nn['n_h2_low'])
            with tf.variable_scope("V_target"):
                self.V_target = networks.critic(self.obs, self.z, self.nn['n_h1_low'], self.nn['n_h2_low'])

        # Low-level Q-functions
        if self.low_level_alg == 'iql':
            with tf.variable_scope("Qlow_main"):
                self.Q_low = networks.Q_low(self.obs, self.z, self.nn['n_h1_low'], self.nn['n_h2_low'], self.l_action)
            with tf.variable_scope("Qlow_target"):
                self.Q_low_target = networks.Q_low(self.obs, self.z, self.nn['n_h1_low'], self.nn['n_h2_low'], self.l_action)
            self.argmax_Q_low = tf.argmax(self.Q_low, axis=1)
            self.actions_low_1hot = tf.placeholder(tf.float32, [None, self.l_action], 'actions_low_1hot')

        # High-level QMIX
        # Individual agent networks
        # output dimension is [time * n_agents, q-values]
        with tf.variable_scope("Agent_main"):
            self.agent_qs = networks.Qmix_single(self.obs, self.nn['n_h1'], self.nn['n_h2'], self.l_z)
        with tf.variable_scope("Agent_target"):
            self.agent_qs_target = networks.Qmix_single(self.obs, self.nn['n_h1'], self.nn['n_h2'], self.l_z)
        
        self.argmax_Q = tf.argmax(self.agent_qs, axis=1)
        self.argmax_Q_target = tf.argmax(self.agent_qs_target, axis=1)
        
        # To extract Q-value from agent_qs and agent_qs_target
        # [batch*n_agents, N_roles]
        self.actions_1hot = tf.placeholder(tf.float32, [None, self.l_z], 'actions_1hot')
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

    def get_assign_target_ops(self):

        # ops for equating main and target
        list_initial_ops = []
        # ops for slow update of target toward main
        list_update_ops = []
        # ops for slow update of low-level target toward low-level main
        list_update_ops_low = []

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

        if self.low_level_alg == 'iac':
            list_V_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_main')
            map_name_V_main = {v.name.split('main')[1] : v for v in list_V_main}
            list_V_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_target')
            map_name_V_target = {v.name.split('target')[1] : v for v in list_V_target}
            if len(list_V_main) != len(list_V_target):
                raise ValueError("get_initialize_target_ops : lengths of V_main and V_target do not match")
            for name, var in map_name_V_main.items():
                list_initial_ops.append( map_name_V_target[name].assign(var) )
            for name, var in map_name_V_main.items():
                list_update_ops_low.append( map_name_V_target[name].assign( self.tau*var + (1-self.tau)*map_name_V_target[name] ) )
        elif self.low_level_alg == 'iql':
            list_Qlow_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Qlow_main')
            map_name_Qlow_main = {v.name.split('main')[1] : v for v in list_Qlow_main}
            list_Qlow_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Qlow_target')
            map_name_Qlow_target = {v.name.split('target')[1] : v for v in list_Qlow_target}
            if len(list_Qlow_main) != len(list_Qlow_target):
                raise ValueError("get_initialize_target_ops : lengths of Qlow_main and Qlow_target do not match")
            for name, var in map_name_Qlow_main.items():
                list_initial_ops.append( map_name_Qlow_target[name].assign(var) )
            for name, var in map_name_Qlow_main.items():
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

        if self.low_level_alg == 'reinforce' or self.low_level_alg == 'iac':
            feed = {self.obs : obs, self.z : roles, self.epsilon : epsilon}
            actions = sess.run(self.action_samples, feed_dict=feed)
        elif self.low_level_alg == 'iql':
            feed = {self.obs : obs, self.z : roles}
            actions_argmax = sess.run(self.argmax_Q_low, feed_dict=feed)
            actions = np.zeros(self.n_agents, dtype=int)
            for idx in range(self.n_agents):
                if np.random.rand() < epsilon:
                    actions[idx] = np.random.randint(0, self.l_action)
                else:
                    actions[idx] = actions_argmax[idx]

        return actions.flatten()

    def assign_roles(self, list_obs, epsilon, sess, N_roles_current):
        """Get high-level role assignment actions for all agents.
        
        Args:
            list_obs: list of vectors, one per agent
            epsilon: exploration parameter
            sess: TF session
            N_roles_current: number of activated role dimensions

        Returns: np.array of role indices
        """
        obs = np.array(list_obs)
        feed = {self.obs : obs}
        Q_values = sess.run(self.agent_qs, feed_dict=feed)
        # Limit the number of activated options based on curriculum
        roles_argmax = np.argmax(Q_values[:, 0:N_roles_current], axis=1)

        roles = np.zeros(self.n_agents, dtype=int)
        for idx in range(self.n_agents):
            if np.random.rand() < epsilon:
                roles[idx] = np.random.randint(0, N_roles_current)
            else:
                roles[idx] = roles_argmax[idx]

        return roles

    def create_train_op_high(self):
        """Ops for training high-level policy."""
        self.td_target = tf.placeholder(tf.float32, [None], 'td_target')
        self.loss_Q_high = tf.reduce_mean(tf.square(self.td_target - tf.squeeze(self.mixer)))
        self.Q_opt = tf.train.AdamOptimizer(self.lr_Q)
        self.Q_op = self.Q_opt.minimize(self.loss_Q_high)

    def create_train_op_low(self):
        """Ops for training low-level policy."""
        if self.low_level_alg == 'reinforce' or self.low_level_alg == 'iac':
            self.actions_taken = tf.placeholder(tf.float32, [None, self.l_action], 'action_taken')
            # self.probs shape is [batch size * traj length, l_action]
            # now log_probs shape is [batch size * traj length]
            log_probs = tf.log(tf.reduce_sum(tf.multiply(self.probs, self.actions_taken), axis=1)+1e-15) 

            if self.low_level_alg == 'reinforce':
                # Rehape to [batch size, traj length]
                log_probs_reshaped = tf.reshape( log_probs, [-1, self.traj_length])
                self.traj_reward = tf.placeholder(tf.float32, [None], 'traj_reward')
                # E [ \sum_t \log \pi(a_t|o_t,z) * R ]
                self.policy_loss = - tf.reduce_mean( tf.reduce_sum(log_probs_reshaped, axis=1) * self.traj_reward )
            elif self.low_level_alg == 'iac':
                # Critic train op
                self.V_td_target = tf.placeholder(tf.float32, [None], 'V_td_target')
                self.loss_V = tf.reduce_mean(tf.square(self.V_td_target - tf.squeeze(self.V)))
                self.V_opt = tf.train.AdamOptimizer(self.lr_V)
                self.V_op = self.V_opt.minimize(self.loss_V)
                # Policy train op
                self.V_evaluated = tf.placeholder(tf.float32, [None], 'V_evaluated')
                self.V_td_error = self.V_td_target - self.V_evaluated
                self.policy_loss = -tf.reduce_mean( tf.multiply( log_probs, self.V_td_error ) )

            self.policy_opt = tf.train.AdamOptimizer(self.lr_actor)
            self.policy_op = self.policy_opt.minimize(self.policy_loss)

        elif self.low_level_alg == 'iql':
            self.td_target_IQL = tf.placeholder(tf.float32, [None], 'td_target_IQL')
            self.td_error = self.td_target_IQL - tf.reduce_sum(tf.multiply(self.Q_low, self.actions_low_1hot), axis=1)
            self.loss_IQL = tf.reduce_mean(tf.square(self.td_error))
            self.IQL_opt = tf.train.AdamOptimizer(self.lr_Q)
            self.IQL_op = self.IQL_opt.minimize(self.loss_IQL)

    def create_train_op_decoder(self):
        """Ops for training skill decoder."""
        self.onehot_z = tf.placeholder(tf.float32, [None, self.l_z], 'onehot_z')
        self.decoder_loss = tf.losses.softmax_cross_entropy(self.onehot_z, self.decoder_out)
        self.decoder_opt = tf.train.AdamOptimizer(self.lr_decoder)
        self.decoder_op = self.decoder_opt.minimize(self.decoder_loss)

    def create_summary(self):

        summaries_Q = [tf.summary.scalar('loss_Q_high', self.loss_Q_high)]
        mixer_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_main')
        for v in mixer_main_variables:
            summaries_Q.append(tf.summary.histogram(v.op.name, v))
        grads = self.Q_opt.compute_gradients(self.loss_Q_high, mixer_main_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_Q.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
        
        agent_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Agent_main')
        for v in agent_main_variables:
            summaries_Q.append(tf.summary.histogram(v.op.name, v))
        grads = self.Q_opt.compute_gradients(self.loss_Q_high, agent_main_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_Q.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
        self.summary_op_Q = tf.summary.merge(summaries_Q)

        if self.low_level_alg == 'reinforce' or self.low_level_alg == 'iac':
            summaries_policy = [tf.summary.scalar('policy_loss', self.policy_loss)]
            policy_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy_main')
            for v in policy_variables:
                summaries_policy.append(tf.summary.histogram(v.op.name, v))
            grads = self.policy_opt.compute_gradients(self.policy_loss, policy_variables)
            for grad, var in grads:
                if grad is not None:
                    summaries_policy.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
            self.summary_op_policy = tf.summary.merge(summaries_policy)

        summaries_decoder = [tf.summary.scalar('decoder_loss', self.decoder_loss)]
        decoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Decoder')
        for v in decoder_variables:
            summaries_decoder.append(tf.summary.histogram(v.op.name, v))
        grads = self.decoder_opt.compute_gradients(self.decoder_loss, decoder_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_decoder.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
        self.summary_op_decoder = tf.summary.merge(summaries_decoder)

        if self.low_level_alg == 'iac':
            summaries_V = [tf.summary.scalar('V_loss', self.loss_V)]
            V_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_main')
            for v in V_variables:
                summaries_V.append(tf.summary.histogram(v.op.name, v))
            grads = self.V_opt.compute_gradients(self.loss_V, V_variables)
            for grad, var in grads:
                if grad is not None:
                    summaries_V.append( tf.summary.histogram(var.op.name+'/gradient', grad) )                
            self.summary_op_V = tf.summary.merge(summaries_V)
        
        if self.low_level_alg == 'iql':
            summaries_Qlow = [tf.summary.scalar('loss_IQL', self.loss_IQL)]
            Qlow_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Qlow_main')
            for v in Qlow_main_variables:
                summaries_Qlow.append(tf.summary.histogram(v.op.name, v))
            grads = self.IQL_opt.compute_gradients(self.loss_IQL, Qlow_main_variables)
            for grad, var in grads:
                if grad is not None:
                    summaries_Qlow.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
            self.summary_op_Qlow = tf.summary.merge(summaries_Qlow)


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
        """Used for high-level buffer.
        
        Extract quantities of the same type from batch.
        Format batch so that each agent at each time step is one batch entry.
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
        actions_1hot = self.process_actions(n_steps, actions, self.l_z)

        return n_steps, state, obs, actions_1hot, reward, state_next, obs_next, done

    def train_policy_high(self, sess, batch, step_train, summarize=False, writer=None):
        """Training step for high-level policy."""
        # Each agent for each time step is now a batch entry
        n_steps, state, obs, actions_1hot, reward, state_next, obs_next, done = self.process_batch(batch)

        # Get argmax actions from target networks
        feed = {self.obs : obs_next}
        argmax_actions = sess.run(self.argmax_Q_target, feed_dict=feed) # [batch*n_agents]
        # Convert to 1-hot
        actions_target_1hot = np.zeros([n_steps * self.n_agents, self.l_z], dtype=int)
        actions_target_1hot[np.arange(n_steps*self.n_agents), argmax_actions] = 1
        
        # Get Q_tot target value
        feed = {self.state : state_next,
                self.actions_1hot : actions_target_1hot,
                self.obs : obs_next}
        Q_tot_target = sess.run(self.mixer_target, feed_dict=feed)

        done_multiplier = -(done - 1)
        target = reward + self.gamma * np.squeeze(Q_tot_target) * done_multiplier

        feed = {self.state : state, self.td_target : target}
        feed[self.obs] = obs
        feed[self.actions_1hot] = actions_1hot

        if summarize:
            summary, _ = sess.run([self.summary_op_Q, self.Q_op], feed_dict=feed)
            writer.add_summary(summary, step_train)
        else:
            _ = sess.run(self.Q_op, feed_dict=feed)

        sess.run(self.list_update_target_ops)

    def process_batch_low(self, batch):
        """
        Extract quantities of the same type from batch.
        Format batch so that each agent at each time step is one batch entry.
        """
        # shapes are [time, ...original dims...]
        obs = np.stack(batch[:,0]) # [time, agents, l_obs]
        actions = np.stack(batch[:,1]) # [time, agents]
        rewards = np.stack(batch[:,2]) # [time, agents]
        obs_next = np.stack(batch[:,3]) # [time, agents, l_obs]
        roles = np.stack(batch[:,4]) # [time, agents, N_roles]
        done = np.stack(batch[:,5]) # [time]

        batch = None # Try to free memory
        n_steps = obs.shape[0]

        # In-place reshape for obs, so that one time step
        # for one agent is considered one batch entry
        obs.shape = (n_steps * self.n_agents, self.l_obs)
        obs_next.shape = (n_steps * self.n_agents, self.l_obs)
        rewards.shape = (n_steps * self.n_agents)
        roles.shape = (n_steps * self.n_agents, self.l_z)
        done = np.repeat(done, self.n_agents, axis=0)

        actions_1hot = self.process_actions(n_steps, actions, self.l_action)
            
        return n_steps, obs, actions_1hot, rewards, obs_next, roles, done

    def train_policy_low(self, sess, batch, step_train=0, summarize=False, writer=None):
        """Training step for low-level action policy.
        
        Runs independent Q-learning on each agent's experiences using local rewards
        """
        n_steps, obs, actions_1hot, rewards, obs_next, roles, done = self.process_batch_low(batch)

        # Get target values
        feed = {self.obs : obs_next, self.z : roles}
        Q_target = sess.run(self.Q_low_target, feed_dict=feed)
        done_multiplier = - (done - 1)
        target = rewards + self.gamma * np.max(Q_target, axis=1) * done_multiplier

        feed = {self.obs : obs,
                self.actions_low_1hot : actions_1hot,
                self.z : roles,
                self.td_target_IQL : target}
        if summarize:
            summary, _ = sess.run([self.summary_op_Qlow, self.IQL_op], feed_dict=feed)
            writer.add_summary(summary, step_train)
        else:
            _ = sess.run(self.IQL_op, feed_dict=feed)

        sess.run(self.list_update_target_ops_low)

    def train_decoder(self, sess, dataset, step_train, summarize=False, writer=None):
        """Training step for skill decoder.

        Args:
            sess: TF session
            dataset: list of np.array objects
            step_train: int
        """
        dataset = np.array(dataset)
        obs = np.stack(dataset[:,0])
        z = np.stack(dataset[:,1])

        # Downsample obs along the traj dimension
        # obs has shape [batch, traj, l_obs]
        obs_downsampled = obs[ :, ::self.traj_skip, : ]
        if self.obs_truncate_length:
            obs_downsampled = obs_downsampled[ : , : , :self.obs_truncate_length]
        if self.use_state_difference:
            # use the difference between consecutive states in a trajectory, rather than the state
            obs_downsampled = obs_downsampled[ : , 1: , : ] - obs_downsampled[ : , :-1 , : ]
        assert( obs_downsampled.shape[1] == self.traj_length_downsampled )

        # Train decoder
        feed = {self.onehot_z : z, self.traj : obs_downsampled}
        if summarize:
            summary, _, decoder_probs = sess.run([self.summary_op_decoder, self.decoder_op, self.decoder_probs], feed_dict=feed)
            writer.add_summary(summary, step_train)
        else:
            _, decoder_probs = sess.run([self.decoder_op, self.decoder_probs], feed_dict=feed)
            
        # decoder_probs has shape [batch, N_roles]
        prob = np.sum(np.multiply( decoder_probs, z ), axis=1)
        expected_prob = np.mean( prob )

        return expected_prob

    def process_dataset(self, dataset):
        """
        Extract batches of obs, action, reward and z for training low-level policy and decoder
        Each batch entry corresponds to a trajectory segment

        dataset: np.array
        """
        # shapes are [batch, ...original dims...]
        obs = np.stack(dataset[:,0]) # [batch, traj, l_obs]
        action = np.stack(dataset[:,1]) # [batch, traj, l_action]
        reward = np.stack(dataset[:,2]) # [batch, traj]
        obs_next = np.stack(dataset[:,3]) # [batch, traj, l_obs]
        done = np.stack(dataset[:,4]) # [batch, traj]
        z = np.stack(dataset[:,5]) # [batch, N_roles]
        
        return obs, action, reward, obs_next, done, z

    def compute_reward(self, sess, agents_traj_obs, z):
        """Computes P(z|traj) as the reward for low-level policy.
        
        Args:
            sess: TF session
            agents_traj_obs: np.array shape [n_agents, traj length, l_obs]
            z: np.array shape [n_agents, N_roles]
        """
        # Downsample along traj dimension
        obs_downsampled = agents_traj_obs[ : , ::self.traj_skip , : ]
        if self.obs_truncate_length:
            obs_downsampled = obs_downsampled[ : , : , :self.obs_truncate_length]

        if self.use_state_difference:
            # use the difference between consecutive states in a trajectory, rather than the state
            obs_downsampled = obs_downsampled[ : , 1: , : ] - obs_downsampled[ : , :-1 , : ]
        assert( obs_downsampled.shape[1] == self.traj_length_downsampled )
        
        decoder_probs = sess.run(self.decoder_probs, feed_dict={self.traj : obs_downsampled})
        prob = np.sum(np.multiply( decoder_probs, z ), axis=1)

        return prob

    def train_policy_and_decoder(self, sess, dataset, alpha, epsilon, step_train, summarize=False, writer=None):
        """DEPRECATED.

        dataset: list of np.array objects
        alpha:  scalar coefficient for computing extrinsic versus intrinsic reward
        epsilon: scalar exploration parameter

        Return E[ log p(z|traj) ] of the batch
        """
        obs, action, reward_e, obs_next, done, z = self.process_dataset(np.array(dataset))

        # Downsample obs along the traj dimension
        # obs has shape [batch, traj, l_obs]
        obs_downsampled = obs[ :, ::self.traj_skip, : ]
        
        if self.obs_truncate_length:
            obs_downsampled = obs_downsampled[ : , : , :self.obs_truncate_length]

        if self.use_state_difference:
            # use the difference between consecutive states in a trajectory, rather than the state
            obs_downsampled = obs_downsampled[ : , 1: , : ] - obs_downsampled[ : , :-1 , : ]
        assert( obs_downsampled.shape[1] == self.traj_length_downsampled )

        # Train decoder
        feed = {self.onehot_z : z, self.traj : obs_downsampled}
        if summarize:
            summary, _, decoder_probs = sess.run([self.summary_op_decoder, self.decoder_op, self.decoder_probs], feed_dict=feed)
            writer.add_summary(summary, step_train)
        else:
            _, decoder_probs = sess.run([self.decoder_op, self.decoder_probs], feed_dict=feed)
            
        # Compute mean and std of batch of P(z|traj)
        prob = np.sum(np.multiply( decoder_probs, z ), axis=1)
        log_probs = np.log( prob + 1e-15 )
        log_probs_mean = np.mean( log_probs )
        log_probs_std = np.std( log_probs )

        # decoder_probs has shape [batch, N_roles]
        expected_prob = np.mean( prob )
        
        # Reshape quantities for low-level policy training
        N_batch = obs.shape[0]
        N_traj = obs.shape[1]
        obs_reshaped = np.reshape(obs, [N_batch * N_traj, self.l_obs])
        action_reshaped = np.reshape(action, [N_batch * N_traj, self.l_action])
        reward_reshaped = np.reshape(reward_e, [N_batch * N_traj])
        obs_next_reshaped = np.reshape(obs_next, [N_batch * N_traj, self.l_obs])
        done_reshaped = np.reshape(done, [N_batch * N_traj])
        # duplicate s.t. each time step in trajectory gets the same z
        z_repeated = np.repeat(z, N_traj, axis=0) 

        if self.low_level_alg == 'reinforce':
            # Compute intrinsic reward for each batch entry
            reward_i = (log_probs - log_probs_mean) / log_probs_std
            # Environment reward, sum along trajectory
            reward_e = np.sum(reward_e, axis=1)
            reward = alpha * reward_e + (1 - alpha) * reward_i
            
            # Train low-level policy
            feed = {self.epsilon : epsilon,
                    self.obs : obs_reshaped,
                    self.z : z_repeated,
                    self.actions_taken : action_reshaped,
                    self.traj_reward : reward}
            if summarize:
                summary, _ = sess.run([self.summary_op_policy, self.policy_op], feed_dict=feed)
                writer.add_summary(summary, step_train)
            else:
                _ = sess.run(self.policy_op, feed_dict=feed)
        elif self.low_level_alg == 'iac':
            # NOTE: intrinsic reward not implemented yet
            # Train critic
            feed = {self.obs : obs_next_reshaped, self.z : z_repeated}
            V_target_res, V_next_res = sess.run([self.V_target, self.V], feed_dict=feed)
            V_target_res = np.squeeze(V_target_res)
            V_next_res = np.squeeze(V_next_res)
            # if true, then 0, else 1
            done_multiplier = -(done_reshaped - 1)
            V_td_target = reward_reshaped + self.gamma * V_target_res * done_multiplier
            feed = {self.V_td_target : V_td_target, self.obs : obs_reshaped, self.z : z_repeated}
            if summarize:
                summary, _, V_res = sess.run([self.summary_op_V, self.V_op, self.V], feed_dict=feed)
                writer.add_summary(summary, step_train)
            else:
                _, V_res = sess.run([self.V_op, self.V], feed_dict=feed)
            
            # Train policy
            V_res = np.squeeze(V_res)
            V_td_target = reward_reshaped + self.gamma * V_next_res * done_multiplier
            feed = {self.epsilon : epsilon, self.obs : obs_reshaped, self.z : z_repeated,
                    self.actions_taken : action_reshaped, self.V_td_target : V_td_target,
                    self.V_evaluated : V_res}
            if summarize:
                summary, _ = sess.run([self.summary_op_policy, self.policy_op], feed_dict=feed)
                writer.add_summary(summary, step_train)
            else:
                _ = sess.run(self.policy_op, feed_dict=feed)

            sess.run(self.list_update_target_ops_low)
        else:
            raise ValueError("alg_hsd.py : self.low_level_alg invalid")

        return expected_prob
