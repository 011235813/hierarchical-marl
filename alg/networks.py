import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def get_variable(name, shape):

    return tf.get_variable(name, shape, tf.float32,
                           tf.initializers.truncated_normal(0,0.01))


def actor(obs, role, n_h1, n_h2, n_actions):
    """
    Args:
        obs: TF placeholder
        role: TF placeholder
        n_h1: int
        n_h2: int
        n_actions: int
    """
    concated = tf.concat( [obs, role], axis=1 )
    h1 = tf.layers.dense(inputs=concated, units=n_h1, activation=tf.nn.relu, use_bias=True, name='actor_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu, use_bias=True, name='actor_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='actor_out')
    probs = tf.nn.softmax(out, name='actor_softmax')

    return probs


def critic(obs, role, n_h1, n_h2):
    """
    Args:
        obs: np.array
        role: 1-hot np.array
        n_h1: int
        n_h2: int
    """
    concated = tf.concat( [obs, role], axis=1 )
    h1 = tf.layers.dense(inputs=concated, units=n_h1, activation=tf.nn.relu, use_bias=True, name='V_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu, use_bias=True, name='V_h2')
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=True, name='V_out')

    return out


def decoder(trajs, timesteps, n_h=128, n_logits=8):
    """Bidirectional LSTM with mean pool over time.
    
    Args:
        trajs: shape (batch size, timesteps, obs dim)
        timesteps: int
        n_h:  number of LSTM hidden units
        n_logits: number of output units
    """
    # Unstack produces a list of length timesteps, where
    # each list entry has shape [batch, obs dim], for use by rnn
    trajs = tf.unstack(trajs, timesteps, 1)

    lstm_forward_cell = rnn.LSTMCell(num_units=n_h, forget_bias=1.0)
    lstm_backward_cell = rnn.LSTMCell(num_units=n_h, forget_bias=1.0)
                               
    rnn_out, state_forward, state_backward = rnn.static_bidirectional_rnn(lstm_forward_cell, lstm_backward_cell, trajs, dtype=tf.float32)

    # Mean-pool over time
    rnn_mean = tf.reduce_mean(rnn_out, axis=0)

    # Output, interpreted as unnormalized probabilities
    out = tf.layers.dense(inputs=rnn_mean, units=n_logits, activation=None, use_bias=True)
    
    probs = tf.nn.softmax(out, name='decoder_probs')

    return out, probs


def Qmix_single(obs, n_h1, n_h2, n_actions):
    """
    Args:
        obs: tf placeholder
        n_h1, n_h2, n_actions: ints
    """
    h1 = tf.layers.dense(inputs=obs, units=n_h1, activation=tf.nn.relu,
                         use_bias=True, name='h1')

    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                         use_bias=True, name='h2')

    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None,
                          use_bias=True, name='out')

    return out


def Q_low(obs, role, n_h1, n_h2, n_actions):
    """
    Args:
        obs: tf placeholder
        role: tf placeholder
        n_h1, n_h2, n_actions: integers
    """
    concated = tf.concat( [obs, role], axis=1 )
    h1 = tf.layers.dense(inputs=concated, units=n_h1, activation=tf.nn.relu,
                         use_bias=True, name='h1')

    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                         use_bias=True, name='h2')

    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None,
                          use_bias=True, name='out')

    return out


def Q_high(state, n_h1, n_h2, n_actions):
    """
    Args:
        state: tf placeholder
        n_h1, n_h2, n_actions: ints
    """
    h1 = tf.layers.dense(inputs=state, units=n_h1, activation=tf.nn.relu,
                         use_bias=True, name='h1')

    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                         use_bias=True, name='h2')

    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None,
                          use_bias=True, name='out')

    return out


def Qmix_mixer(agent_qs, state, state_dim, n_agents, n_h_mixer):
    """
    Args:
        agent_qs: shape [batch, n_agents]
        state: shape [batch, state_dim]
        state_dim: integer
        n_agents: integer
        n_h_mixer: integer
    """
    agent_qs_reshaped = tf.reshape(agent_qs, [-1, 1, n_agents])

    # n_h_mixer * n_agents because result will be reshaped into matrix
    hyper_w_1 = get_variable('hyper_w_1', [state_dim, n_h_mixer*n_agents]) 
    hyper_w_final = get_variable('hyper_w_final', [state_dim, n_h_mixer])

    hyper_b_1 = tf.get_variable('hyper_b_1', [state_dim, n_h_mixer])

    hyper_b_final_l1 = tf.layers.dense(inputs=state, units=n_h_mixer, activation=tf.nn.relu,
                                       use_bias=False, name='hyper_b_final_l1')
    hyper_b_final = tf.layers.dense(inputs=hyper_b_final_l1, units=1, activation=None,
                                    use_bias=False, name='hyper_b_final')

    # First layer
    w1 = tf.abs(tf.matmul(state, hyper_w_1))
    b1 = tf.matmul(state, hyper_b_1)
    w1_reshaped = tf.reshape(w1, [-1, n_agents, n_h_mixer]) # reshape into batch of matrices
    b1_reshaped = tf.reshape(b1, [-1, 1, n_h_mixer])
    # [batch, 1, n_h_mixer]
    hidden = tf.nn.elu(tf.matmul(agent_qs_reshaped, w1_reshaped) + b1_reshaped)
    
    # Second layer
    w_final = tf.abs(tf.matmul(state, hyper_w_final))
    w_final_reshaped = tf.reshape(w_final, [-1, n_h_mixer, 1]) # reshape into batch of matrices
    b_final_reshaped = tf.reshape(hyper_b_final, [-1, 1, 1])

    # [batch, 1, 1]
    y = tf.matmul(hidden, w_final_reshaped) + b_final_reshaped

    q_tot = tf.reshape(y, [-1, 1])

    return q_tot
