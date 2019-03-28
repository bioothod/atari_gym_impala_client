import tensorflow as tf

import logging

weight_decay = 1e-5

def conv_network(config, input, scope='conv', reuse=False):
    input_map = input['map']

    with tf.variable_scope(scope, reuse=reuse):
        conv_out = input_map

        for i, (num_ch, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
            # Downscale.
            conv_out = tf.layers.conv2d(
                    inputs=conv_out,
                    filters=num_ch,
                    kernel_size=3,
                    strides=1,
                    padding='SAME',
                    use_bias=True,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
            )
            conv_out = tf.nn.pool(
                    conv_out,
                    window_shape=[3, 3],
                    pooling_type='MAX',
                    padding='SAME',
                    strides=[2, 2],
            )

            # Residual block(s).
            for j in range(num_blocks):
                with tf.variable_scope('residual_%d_%d' % (i, j)):
                    block_input = conv_out
                    conv_out = tf.nn.relu(conv_out)
                    conv_out = tf.layers.conv2d(
                            inputs=conv_out,
                            filters=num_ch,
                            kernel_size=3,
                            strides=1,
                            padding='SAME',
                            use_bias=True,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    )
                    conv_out = tf.nn.relu(conv_out)
                    conv_out = tf.layers.conv2d(
                            inputs=conv_out,
                            filters=num_ch,
                            kernel_size=3,
                            strides=1,
                            padding='SAME',
                            use_bias=True,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    )
                    conv_out += block_input

        conv_out = tf.nn.relu(conv_out)
        conv_out = tf.layers.flatten(conv_out)

        conv_out = tf.layers.dense(conv_out, 256, activation=tf.nn.relu)
        return conv_out

def create_conv_network(config, input_dict, scope='input', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        conv_out = conv_network(config, input_dict, 'conv', reuse)

        one_hot_action = input_dict['action']
        clipped_reward = input_dict['reward']
        params = input_dict['params']

        conv_state = tf.concat([conv_out, clipped_reward, one_hot_action, params], axis=1)
        logging.info('conv layer: conv state: {}'.format(conv_state))

        return conv_state

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

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(self.lstm_cell, input_features, initial_state=state_in, dtype=tf.float32, time_major=False)
            self.lstm_state = tf.identity(lstm_state, 'output/lstm_state')

            logging.info('lstm_outputs: {}, lstm_state: {}'.format(lstm_outputs, lstm_state))

            num_timesteps = tf.shape(lstm_outputs)[1]
            reshaped_output_to_batch = tf.reshape(lstm_outputs, [-1, num_lstm_outputs])

            batched_policy_logits = tf.layers.dense(inputs=reshaped_output_to_batch, units=config['num_actions'], activation=None, use_bias=True, name='batched_policy_logits_layer')
            batched_baseline = tf.layers.dense(inputs=reshaped_output_to_batch, units=1, activation=None, use_bias=True, name='batched_baseline_layer')

            self.policy_logits = tf.reshape(batched_policy_logits, [-1, num_timesteps, config['num_actions']])
            self.baseline = tf.reshape(batched_baseline, [-1, num_timesteps])

    def init_sizes(self):
        return (self.lstm_cell.state_size.c, self.lstm_cell.state_size.h)

