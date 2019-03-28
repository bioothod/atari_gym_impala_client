import tensorflow as tf

import logging

weight_decay = 1e-5

def conv_network(config, input, scope='conv', reuse=False):
    input_map = input['map']

    with tf.variable_scope(scope, reuse=reuse):
        out = input_map

        out = tf.layers.dense(out, 2, activation=None, use_bias=True)
        out = tf.layers.dense(out, 3, activation=tf.nn.relu, use_bias=True)
        return out

def create_conv_network(config, input_dict, scope='input', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = conv_network(config, input_dict, 'conv', reuse)

        one_hot_action = input_dict['action']
        clipped_reward = input_dict['reward']
        params = input_dict['params']

        out = tf.concat([out, clipped_reward, one_hot_action, params], axis=1)
        logging.info('conv layer: conv state: {}'.format(out))

        return out

class rnn_head:
    def __init__(self, config, input_features, scope='lstm', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            logging.info('lstm layer: input: {}'.format(input_features))

            batch_size = config['batch_size']

            num_lstm_outputs = 256
            self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_lstm_outputs, state_is_tuple=True)

            self.c_state_ph = tf.placeholder(tf.float32, shape=[batch_size, self.lstm_cell.state_size.c], name='input/c_state')
            self.h_state_ph = tf.placeholder(tf.float32, shape=[batch_size, self.lstm_cell.state_size.h], name='input/h_state')

            _ = tf.constant((self.lstm_cell.state_size.c, self.lstm_cell.state_size.h), dtype=tf.int32, name='output/lstm_state_sizes')

            state_in = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_ph, self.h_state_ph)

            logging.info('c_states: {}, h_states: {}, state_in: {}'.format(self.c_state_ph, self.h_state_ph, state_in))

            #lstm_outputs, lstm_state = tf.nn.dynamic_rnn(self.lstm_cell, input_features, initial_state=state_in, dtype=tf.float32, time_major=False)
            self.lstm_state = tf.identity(state_in, 'output/lstm_state')

            #logging.info('lstm_outputs: {}, lstm_state: {}'.format(lstm_outputs, lstm_state))

            #num_timesteps = tf.shape(lstm_outputs)[1]
            #reshaped_output_to_batch = tf.reshape(lstm_outputs, [-1, num_lstm_outputs])

            batched_policy_logits = tf.layers.dense(inputs=input_features, units=config['num_actions'], activation=None, use_bias=True, name='batched_policy_logits_layer')
            batched_baseline = tf.layers.dense(inputs=input_features, units=1, activation=None, use_bias=True, name='batched_baseline_layer')

            num_timesteps = tf.shape(input_features)[1]
            self.policy_logits = tf.reshape(batched_policy_logits, [-1, num_timesteps, config['num_actions']])
            self.baseline = tf.reshape(batched_baseline, [-1, num_timesteps])

    def init_sizes(self):
        return (self.lstm_cell.state_size.c, self.lstm_cell.state_size.h)

