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
    def __init__(self, config, input_features, dones_r, scope='lstm', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            logging.info('lstm layer: input: {}'.format(input_features))

            num_lstm_outputs = 256
            self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_lstm_outputs, state_is_tuple=True)
            c_zeros = tf.zeros([1, self.lstm_cell.state_size.c], dtype=tf.float32)
            h_zeros = tf.zeros([1, self.lstm_cell.state_size.h], dtype=tf.float32)
            initial_zero_state = tf.nn.rnn_cell.LSTMStateTuple(c_zeros, h_zeros)

            logging.info('lstm: {}. initial_zero_state: {}'.format(self.lstm_cell, initial_zero_state))

            self.c_state_ph = tf.placeholder(tf.float32, shape=[None, self.lstm_cell.state_size.c], name='input/c_state')
            self.h_state_ph = tf.placeholder(tf.float32, shape=[None, self.lstm_cell.state_size.h], name='input/h_state')


            _ = tf.constant((self.lstm_cell.state_size.c, self.lstm_cell.state_size.h), dtype=tf.int32, name='output/lstm_state_sizes')

            batch_index = tf.constant(0, dtype=tf.int32)
            batch_size = tf.shape(dones_r)[0]
            time_steps = tf.shape(dones_r)[1]

            x = input_features[0, 0, :]
            x = tf.expand_dims(x, 0)
            x = tf.expand_dims(x, 0)
            logging.info('input_feature_per_timestamp: {}'.format(x))

            rnn_outputs = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=0, dynamic_size=True)
            rnn_states = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=0, dynamic_size=True)

            def batch_cond(i, _rnn_outputs, _rnn_states):
                return tf.less(i, batch_size)

            def batch_body(i, rnn_outputs, rnn_states):
                lstm_state = tf.nn.rnn_cell.LSTMStateTuple(tf.expand_dims(self.c_state_ph[i], 0), tf.expand_dims(self.h_state_ph[i], 0))
                done = dones_r[i, :]

                time_index = tf.constant(0, dtype=tf.int32)
                def time_cond(j, _lstm_state, _rnn_outputs):
                    return tf.less(j, time_steps)

                def time_body(j, lstm_state, rnn_outputs):
                    state_in = tf.where(tf.logical_and(j > 0, done[j-1]), initial_zero_state, lstm_state)

                    x = input_features[i, j, :]
                    x = tf.expand_dims(x, 0)
                    x = tf.expand_dims(x, 0)

                    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(self.lstm_cell, x, initial_state=initial_zero_state, time_major=False)
                    lstm_outputs = tf.squeeze(lstm_outputs, 0)
                    rnn_outputs = rnn_outputs.write(i * time_steps + j, lstm_outputs)

                    j = tf.add(j, 1)
                    return j, lstm_state, rnn_outputs

                time_index, lstm_state, rnn_outputs = tf.while_loop(time_cond, time_body, [time_index, lstm_state, rnn_outputs])

                rnn_states = rnn_states.write(i, lstm_state)
                i = tf.add(i, 1)
                return i, rnn_outputs, rnn_states

            batch_index, rnn_outputs, rnn_states = tf.while_loop(batch_cond, batch_body, [batch_index, rnn_outputs, rnn_states])

            self.lstm_state = rnn_states.stack()
            self.lstm_state = tf.reshape(self.lstm_state, [batch_size, 2, num_lstm_outputs], name='output/lstm_state')

            reshaped_output_to_batch = rnn_outputs.stack()
            reshaped_output_to_batch = tf.reshape(reshaped_output_to_batch, [batch_size * time_steps, num_lstm_outputs])
            logging.info('reshaped_output_to_batch: {}'.format(reshaped_output_to_batch))

            batched_policy_logits = tf.layers.dense(inputs=reshaped_output_to_batch, units=config['num_actions'], activation=None, use_bias=True, name='batched_policy_logits_layer')
            batched_baseline = tf.layers.dense(inputs=reshaped_output_to_batch, units=1, activation=None, use_bias=True, name='batched_baseline_layer')

            self.policy_logits = tf.reshape(batched_policy_logits, [-1, time_steps, config['num_actions']])
            self.baseline = tf.reshape(batched_baseline, [-1, time_steps])

    def init_sizes(self):
        return (self.lstm_cell.state_size.c, self.lstm_cell.state_size.h)

