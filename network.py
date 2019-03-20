import tensorflow as tf

import logging

weight_decay = 1e-5

def conv_single_image(image, params, scope='img', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope('flat', reuse=reuse):
            flat_map = tf.layers.flatten(image)
            flat_params = tf.layers.flatten(params)
            concat = tf.concat([flat_map, flat_params], 1)

            #flat = flat_map
            flat = concat

            return flat

def conv_network(config, input, scope='conv', reuse=False):
    input_map = input['map']

    with tf.variable_scope(scope, reuse=reuse):
        conv_out = input_map

        for i, (num_ch, num_blocks) in enumerate([(16, 2), (32, 2)]):
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

        return conv_out

def create_conv_network(config, input_dict, scope='input', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        conv_out = conv_network(config, input_dict, 'conv', reuse)

        one_hot_action = input_dict['action']
        clipped_reward = input_dict['reward']
        params = input_dict['params']

        conv_state = tf.concat([conv_out, clipped_reward, one_hot_action, params], axis=1)
        logging.info('conv lyaer: conv state: {}'.format(conv_state))

        return conv_state

def create_rnn_network(config, input_features, scope='lstmp', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        logging.info('lstm lyaer: input: {}'.format(input_features))

        rnn_layers = [tf.nn.rnn_cell.LSTMCell(lstm_size) for lstm_size in [8, config.get('num_actions') + 1]]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        outputs, state = tf.nn.dynamic_rnn(multi_rnn_cell, input_features, dtype=tf.float32)

        # only get batch of one last timestamp predictions
        #output = outputs[:, -1, :]
        logging.info('lstm lyaer: input: {}, outputs: {}, state: {}'.format(input_features, outputs, state))

        policy_logits, baseline = tf.split(outputs, [config.get('num_actions'), 1], axis=-1)
        baseline = tf.squeeze(baseline, axis=-1)

    return policy_logits, baseline
