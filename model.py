import numpy as np
import tensorflow as tf

import logging
import os

import network
from env import *


class Model:
    def __init__(self, config, is_client=False):
        self.config = config

        self.graph = tf.Graph()

        checkpoint_dir = config.get('checkpoint_dir')

        input_map_shape = config.get('input_map_shape')
        input_params_shape = config.get('input_params_shape')
        num_actions = config.get('num_actions')

        logging.info('creating model: input_map_shape: {}, input_params_shape: {}, num_actions: {}'.format(input_map_shape, input_params_shape, num_actions))

        _ = tf.constant(input_map_shape, dtype=tf.int64, name="input_map_shape")
        _ = tf.constant(input_params_shape, dtype=tf.int64, name="input_params_shape")
        _ = tf.constant([num_actions], dtype=tf.int64, name="input_policy_logits_shape")

        self.map_ph = tf.placeholder(tf.float32, [None] + input_map_shape, name='input/map')
        self.params_ph = tf.placeholder(tf.float32, [None] + input_params_shape, name='input/params')
        self.bh_policy_logits_ph = tf.placeholder(tf.float32, [None, num_actions], name='input/policy_logits')
        self.bh_action_taken_ph = tf.placeholder(tf.int32, [None], name='input/action_taken')
        self.reward_ph = tf.placeholder(tf.float32, [None], name='input/reward')
        self.done_ph = tf.placeholder(tf.bool, [None], name='input/done')
        self.time_steps_ph = tf.placeholder(tf.int32, [], name='input/time_steps')

        one_hot_actions = tf.one_hot(self.bh_action_taken_ph, config['num_actions'])

        #clipped_reward = self.reward_ph
        clipped_reward = tf.clip_by_value(self.reward_ph, -1, 1)

        dones_r = tf.reshape(self.done_ph, [-1, self.time_steps_ph])
        clipped_rewards_r = tf.reshape(clipped_reward, [-1, self.time_steps_ph])

        network_dict = {
                'map': self.map_ph,
                'params': self.params_ph,
                'action': one_hot_actions,
                'reward': tf.expand_dims(clipped_reward, -1)
        }

        conv_output = network.create_conv_network(config, network_dict, scope='impala_cnn')
        reshaped_output = tf.reshape(conv_output, [-1, self.time_steps_ph, conv_output.shape[-1]])

        rnn_state = network.rnn_head(config, reshaped_output, dones_r, scope='impala_rnn')

        policy_logits = rnn_state.policy_logits
        baseline = rnn_state.baseline
        logging.info('policy_logits: {}, baseline: {}'.format(policy_logits, baseline))

        self.policy_logits = tf.identity(policy_logits, name='output/policy_logits')

        last_logits = self.policy_logits[:, -1, :]
        self.new_action_predict = tf.argmax(last_logits, axis=1, name='output/new_action_predict')
        self.new_action_train = tf.random.multinomial(last_logits, num_samples=1, output_dtype=tf.int32)
        self.new_action_train = tf.squeeze(self.new_action_train, 1, name='output/new_action_train')

        pshape = policy_logits.shape[-1]
        policy_flattened_to_batch = tf.reshape(policy_logits, [-1, pshape])

        actions_r = tf.reshape(self.bh_action_taken_ph, [-1, self.time_steps_ph])

        target_action_log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy_flattened_to_batch, labels=self.bh_action_taken_ph)
        bh_action_log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.bh_policy_logits_ph, labels=self.bh_action_taken_ph)
        log_rhos = target_action_log_probs - bh_action_log_probs
        log_rhos_r = tf.reshape(log_rhos, [-1, self.time_steps_ph])

        logging.info('network has been created, number of variables: {}'.format(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='impala_network'))))

        policy_gradient_loss = tf.constant(0., dtype=tf.float32)
        cross_entropy_loss = tf.constant(0., dtype=tf.float32)
        baseline_loss = tf.constant(0., dtype=tf.float32)

        logging.info('rhos: {}, values: {}'.format(log_rhos.shape, baseline.shape))
        i = tf.constant(0, dtype=tf.int32)
        bsize = tf.shape(self.map_ph)[0]
        iter_steps_float = tf.divide(bsize, self.time_steps_ph)
        max_iter_steps = tf.cast(tf.round(iter_steps_float), tf.int32)
        c = lambda i, _l1, _l2, _l3: tf.less(i, max_iter_steps)

        def body(i, policy_gradient_loss, cross_entropy_loss, baseline_loss):
            log_rhos_local = log_rhos_r[i, :]
            discounts = tf.cast(~dones_r[i, :], tf.float32) * config['discount_gamma']

            values = baseline[i, :]
            logging.info('values: {}'.format(values))
            bootstrap_value = values[0]

            local_rewards = clipped_rewards_r[i, :]

            vs, pg_advantages = self.build_vtrace(log_rhos=log_rhos_local, discounts=discounts, rewards=local_rewards, values=values, bootstrap_value=bootstrap_value)
            logging.info('vs: {}, pg_advantages: {}'.format(vs, pg_advantages))

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions_r[i, :], logits=policy_logits[i, :, :])
            advantages = tf.stop_gradient(pg_advantages)
            policy_gradient_loss_per_timestep = cross_entropy * advantages
            policy_gradient_loss = tf.add(policy_gradient_loss, tf.reduce_sum(policy_gradient_loss_per_timestep) * config['policy_gradient_loss_scale'])

            policy = tf.nn.softmax(policy_logits[i, :, :])
            log_policy = tf.nn.log_softmax(policy_logits[i, :, :])
            entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
            cross_entropy_loss = tf.add(cross_entropy_loss, -tf.reduce_sum(entropy_per_timestep) * config['cross_entropy_loss_scale'])

            advantage = vs - values
            baseline_loss = tf.add(baseline_loss, 0.5 * tf.reduce_sum(tf.square(advantage)) * config['baseline_loss_scale'])

            i = tf.add(i, 1)
            return i, policy_gradient_loss, cross_entropy_loss, baseline_loss

        i, policy_gradient_loss, cross_entropy_loss, baseline_loss = tf.while_loop(c, body, [i, policy_gradient_loss, cross_entropy_loss, baseline_loss])

        policy_gradient_loss = tf.divide(policy_gradient_loss, tf.cast(iter_steps_float, dtype=tf.float32))
        cross_entropy_loss = tf.divide(cross_entropy_loss, tf.cast(iter_steps_float, dtype=tf.float32))
        baseline_loss = tf.divide(baseline_loss, tf.cast(iter_steps_float, dtype=tf.float32))

        self.policy_gradient_loss = tf.identity(policy_gradient_loss, name='output/policy_gradient_loss')
        tf.summary.scalar('policy_gradient_loss', self.policy_gradient_loss)

        self.cross_entropy_loss = tf.identity(cross_entropy_loss, name='output/cross_entropy_loss')
        tf.summary.scalar('cross_entropy_loss', self.cross_entropy_loss)

        self.baseline_loss = tf.identity(baseline_loss, name='output/baseline_loss')
        tf.summary.scalar('baseline_loss', self.baseline_loss)


        self.total_loss = tf.add_n([self.policy_gradient_loss, self.baseline_loss, self.cross_entropy_loss])
        self.total_loss = tf.identity(self.total_loss, name='output/total_loss')
        tf.summary.scalar('total_loss', self.total_loss)

        if False:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if len(regularization_losses) > 0:
                rl = tf.add_n(regularization_losses)
                tf.summary.scalar('regularization_loss', rl)
                self.total_loss += rl

        self.global_step_op = tf.train.get_or_create_global_step()
        self.lr_ph = tf.placeholder(tf.float32, shape=[], name='learning_rate_ph')
        tf.summary.scalar('learning_rate', self.lr_ph)

        #opt = tf.train.AdamOptimizer(lr_ph, beta1=0.9, beta2=0.999, epsilon=0.01)
        opt = tf.train.RMSPropOptimizer(self.lr_ph, decay=0.99, name='optimizer')

        self.train_op = opt.minimize(self.total_loss, global_step=self.global_step_op, name='output/train_op')

        self.total_saver = tf.train.Saver()

        if is_client:
            return


        self.steps = 0
        hooks = []

        variables_to_restore = []
        variables_to_init = []
        for v in tf.global_variables():
            if False:
                names_to_init = ['transform_', 'first_layer']
                initialized = False
                for ni in names_to_init:
                    if ni in v.name:
                        variables_to_init.append(v)
                        initialized = True
                        break

                if initialized:
                    continue

            variables_to_restore.append(v)

        init_op = tf.initializers.variables(variables_to_init)

        saver = tf.train.Saver(var_list=variables_to_restore)

        def init_fn(scaffold, sess):
            sess.run([init_op])

            if checkpoint_dir:
                path = tf.train.latest_checkpoint(checkpoint_dir)

                logging.info('checkpoint_dir: {}, restore path: {}'.format(checkpoint_dir, path))
                if path:
                    saver.restore(sess, path)

            logging.info('init_fn has been completed')

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        self.summary_op = tf.summary.merge(summaries)
        if checkpoint_dir:
            self.summary_writer = tf.summary.FileWriter(checkpoint_dir)

        scaffold = tf.train.Scaffold(saver=None, init_fn=init_fn)

        logging.info('going to create a session')

        session_config = tf.ConfigProto()
        session_config.allow_soft_placement = True
        #session_config.gpu_options.visible_device_list = str(hvd.local_rank())

        self.mon_sess = tf.train.MonitoredTrainingSession(config=session_config, hooks=hooks, checkpoint_dir=None, scaffold=scaffold)
        self.sess = self.mon_sess._tf_sess()

        step = self.sess.run(self.global_step_op)
        logging.info('step: {}'.format(step))

        self.save_graph(step, want_const=False)

        path = os.path.join(checkpoint_dir, 'saver_def.pb')
        with tf.gfile.GFile(path, "wb+") as f:
            sd = self.total_saver.saver_def
            ser = sd.SerializeToString()

            f.write(ser)

    def serialize_graph(self, want_const=False):
        sd = self.total_saver.saver_def
        logging.info('filename_tensor_name: {}, save_tensor_name: {}, filename_tensor_name: {}, restore_op_name: {}'.format(
            sd.filename_tensor_name[:-2], sd.save_tensor_name[:-2], sd.filename_tensor_name[:-2], sd.restore_op_name))

        if not want_const:
            return self.sess.graph_def.SerializeToString()

        output_variables = ['output/new_action_train', 'output/new_action_predict', 'output/policy_logits']
        output_graph_def = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, output_variables)
        return output_graph_def.SerializeToString()

    def save_graph(self, global_step, want_const=False):
        checkpoint_dir = self.config.get('checkpoint_dir')
        if not checkpoint_dir:
            return

        if want_const:
            path = os.path.join(checkpoint_dir, 'graph.{}.frozen.pb'.format(global_step))
        else:
            path = os.path.join(checkpoint_dir, 'graph.{}.pb'.format(global_step))

        with tf.gfile.GFile(path, "wb+") as f:
            ser_graph = self.serialize_graph(want_const=want_const)
            f.write(ser_graph)
            logging.info('saved {} graph into {}, size: {}'.format('CONST' if want_const else 'VARIABLE', path, len(ser_graph)))

    def save_checkpoint(self):
        checkpoint_dir = self.config.get('checkpoint_dir')
        if not checkpoint_dir:
            return
        self.total_saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=self.global_step_op, write_meta_graph=True)

    def build_vtrace(self, log_rhos, discounts, rewards, values, bootstrap_value, clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0, scope='vtrace_from_importance_weights'):
        r"""V-trace from log importance weights.

        Calculates V-trace actor critic targets as described in

        "IMPALA: Scalable Distributed Deep-RL with
        Importance Weighted Actor-Learner Architectures"
        by Espeholt, Soyer, Munos et al.
      
        In the notation used throughout documentation and comments, T refers to the
        time dimension ranging from 0 to T-1. B refers to the batch size and
        NUM_ACTIONS refers to the number of actions. This code also supports the
        case where all tensors have the same number of additional dimensions, e.g.,
        `rewards` is [T, B, C], `values` is [T, B, C], `bootstrap_value` is [B, C].
      
        Args:
          log_rhos: A float32 tensor of shape [T, B, NUM_ACTIONS] representing the log
            importance sampling weights, i.e.
            log(target_policy(a) / behaviour_policy(a)). V-trace performs operations
            on rhos in log-space for numerical stability.
          discounts: A float32 tensor of shape [T, B] with discounts encountered when
            following the behaviour policy.
          rewards: A float32 tensor of shape [T, B] containing rewards generated by
            following the behaviour policy.
          values: A float32 tensor of shape [T, B] with the value function estimates
            wrt. the target policy.
          bootstrap_value: A float32 of shape [B] with the value function estimate at
            time T.
          clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
            importance weights (rho) when calculating the baseline targets (vs).
            rho^bar in the paper. If None, no clipping is applied.
          clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
            on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)). If
            None, no clipping is applied.
          name: The name scope that all V-trace operations will be created in.
      
        Returns:
          A VTraceReturns namedtuple (vs, pg_advantages) where:
            vs: A float32 tensor of shape [T, B]. Can be used as target to
              train a baseline (V(x_t) - vs_t)^2.
            pg_advantages: A float32 tensor of shape [T, B]. Can be used as the
              advantage in the calculation of policy gradients.
        """

        # Make sure tensor ranks are consistent.
        rho_rank = log_rhos.shape.ndims  # Usually 2.
        values.shape.assert_has_rank(rho_rank)
        bootstrap_value.shape.assert_has_rank(rho_rank - 1)
        discounts.shape.assert_has_rank(rho_rank)
        rewards.shape.assert_has_rank(rho_rank)
        if clip_rho_threshold is not None:
            clip_rho_threshold = tf.convert_to_tensor(clip_rho_threshold, dtype=tf.float32)
            clip_rho_threshold.shape.assert_has_rank(0)
        if clip_pg_rho_threshold is not None:
            clip_pg_rho_threshold = tf.convert_to_tensor(clip_pg_rho_threshold, dtype=tf.float32)
            clip_pg_rho_threshold.shape.assert_has_rank(0)
      
        with tf.name_scope(scope, values=[log_rhos, discounts, rewards, values, bootstrap_value]):
            rhos = tf.exp(log_rhos)
            if clip_rho_threshold is not None:
                clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name='clipped_rhos')
            else:
                clipped_rhos = rhos
        
            cs = tf.minimum(1.0, rhos, name='cs')
            # Append bootstrapped value to get [v1, ..., v_t+1]
            values_t_plus_1 = tf.concat([values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
            deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
        
            # Note that all sequences are reversed, computation starts from the back.
            sequences = (
                tf.reverse(discounts, axis=[0]),
                tf.reverse(cs, axis=[0]),
                tf.reverse(deltas, axis=[0]),
            )
            # V-trace vs are calculated through a scan from the back to the beginning
            # of the given trajectory.
            def scanfunc(acc, sequence_item):
                discount_t, c_t, delta_t = sequence_item
                return delta_t + discount_t * c_t * acc
        
            initial_values = tf.zeros_like(bootstrap_value)
            vs_minus_v_xs = tf.scan(
                    fn=scanfunc,
                    elems=sequences,
                    initializer=initial_values,
                    parallel_iterations=1,
                    back_prop=False,
                    name='scan')

            # Reverse the results back to original order.
            vs_minus_v_xs = tf.reverse(vs_minus_v_xs, [0], name='vs_minus_v_xs')
        
            # Add V(x_s) to get v_s.
            vs = tf.add(vs_minus_v_xs, values, name='vs')
        
            # Advantage for policy gradient.
            vs_t_plus_1 = tf.concat([vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
            if clip_pg_rho_threshold is not None:
                clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos, name='clipped_pg_rhos')
            else:
                clipped_pg_rhos = rhos

            pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)
        
            # Make sure no gradients backpropagated through the returned values.
            return tf.stop_gradient(vs), tf.stop_gradient(pg_advantages)

def create_model(config, is_client=False):
    #tf.logging.set_verbosity(tf.logging.ERROR)
    #logging.getLogger('tensorflow').setLevel(logging.ERROR)
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.get('visible_devices', '')

    config_internal = {
        'discount_gamma': 0.99,
        'policy_gradient_loss_scale': 1,
        'cross_entropy_loss_scale': 0.00025,
        'baseline_loss_scale': 0.5,
        'summary_steps': 1000,
        'checkpoint_steps': 10000,
        'initial_learning_rate': 2.5e-4,
        'minimal_learning_rate': 1e-5,
        'learning_rate_decay_factor': 0.7,
        'learning_rate_decay_steps': 100000,
    }

    if not is_client:
        checkpoint_dir = config.get('train_dir')
        os.makedirs(checkpoint_dir, exist_ok=True)
        config_internal['checkpoint_dir'] = checkpoint_dir


    config.update(config_internal)

    m = Model(config, is_client=is_client)
    return m
