import numpy as np
import tensorflow as tf

import argparse
import cv2
import grpc
import gym
import logging
import os
import random
import sys
import time

import env
import game
import halite_model_pb2
import halite_model_pb2_grpc
import model

from state import stacked_env

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class GameWrapper:
    def __init__(self, config):
        self.config = config
        self.owner_id = config['owner_id']
        self.env_id = config['env_id']

        self.trajectory = []

        self.input_map_shape = config['input_map_shape']
        self.input_params_shape = config['input_params_shape']
        self.state_stack_size = config['state_stack_size']

        self.env = env.Env()
        #self.env = gym.make(config['game'])
        #self.env = game.MaxAndSkipEnv(self.env)
        #self.env = game.FireResetEnv(self.env)

        monitor_dir = config.get('monitor_dir')
        if monitor_dir:
            self.env = gym.wrappers.Monitor(self.env, directory=monitor_dir, video_callable=False, write_upon_reset=True)

        self.config['num_actions'] = self.env.action_space.n

        self.step = 0
        self.train_step = -1
        self.trajectory_len = -1

        self.have_graph = False

        opts = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(config.get('remote_addr'), options=opts)
        self.stub = halite_model_pb2_grpc.HaliteProcessStub(self.channel)

        self.graph = tf.Graph()
        with self.graph.as_default():
            config['batch_size'] = 1
            self.model = model.create_model(config, is_client=True)
            self.have_graph = True

            session_config = tf.ConfigProto()
            session_config.allow_soft_placement = True
            self.sess = tf.Session(config=session_config)

            self.load_model()

        self.input_lstm_state_sizes = self.graph.get_tensor_by_name('impala_rnn/output/lstm_state_sizes:0').eval(session=self.sess)
        c_state_size = self.input_lstm_state_sizes[0]
        h_state_size = self.input_lstm_state_sizes[1]

        self.c_state = np.zeros([1, c_state_size], np.float32)
        self.h_state = np.zeros([1, h_state_size], np.float32)

        self.init_c_state = self.c_state.copy()
        self.init_h_state = self.h_state.copy()


        self.state = stacked_env(self.config)
        self.prev_action = -1
        self.prev_reward = 0
        self.prev_model_st = None
        self.prev_st = None

    def new_state(self, state):
        res = np.reshape(state, self.input_map_shape)

        params = np.zeros(shape=self.input_params_shape, dtype=np.float32)

        self.state.append(res, params)
        return self.state.current()

    def load_model(self):
        start_time = time.time()

        proto = self.stub.GetFrozenGraph(halite_model_pb2.Status())
        if not proto:
            logging.error('could not get frozen graph, method returned None')
            return

        self.train_step = proto.train_step
        self.trlen = proto.trajectory_len

        request_time = time.time() - start_time

        with self.graph.as_default():
            start_time = time.time()
            checkpoint_time = 0.0

            od_graph_def = tf.GraphDef()

            if len(proto.frozen_graph) != 0:
                od_graph_def.ParseFromString(proto.frozen_graph)
                tf.import_graph_def(od_graph_def, name='')
                graph_def_parsing_time = time.time() - start_time
            else:
                if not self.have_graph:
                    od_graph_def.ParseFromString(proto.graph_def)
                    tf.import_graph_def(od_graph_def, name='')
                    self.have_graph = True

                graph_def_parsing_time = time.time() - start_time

                start_time = time.time()
                prefix = '{}/{}.tmp.{}'.format(self.config['tmp_dir'], str(random.randint(0, 10000000)), os.path.basename(proto.prefix))
                index_file = '{}.index'.format(prefix)
                data_file = '{}.data-00000-of-00001'.format(prefix)
                with open(index_file, 'wb+') as f:
                    f.write(proto.checkpoint_index)
                with open(data_file, 'wb+') as f:
                    f.write(proto.checkpoint_data)

                self.model.total_saver.restore(self.sess, prefix)

                os.remove(index_file)
                os.remove(data_file)

                checkpoint_time = time.time() - start_time

            start_time = time.time()
            with tf.device('/cpu:0'):
                self.policy_logits_op = self.graph.get_tensor_by_name('output/policy_logits:0')
                self.action_op = self.graph.get_tensor_by_name('output/new_action_train:0')
                self.lstm_state_op = self.graph.get_tensor_by_name('impala_rnn/output/lstm_state:0')

                #logging.info('policy_logits: {}, action: {}'.format(self.policy_logits_op, self.action_op))

            tensor_lookup_time = time.time() - start_time

            logging.info('reloaded: train_step: {}, trajectory len: {}, network request: {:.1f} ms, graph_def parsing: {:.1f} ms, checkpoint recovery: {:.1f} ms, tensor lookup: {:.1f} ms'.format(
                self.train_step, self.trlen, request_time * 1000, graph_def_parsing_time * 1000, checkpoint_time * 1000, tensor_lookup_time * 1000))

    def get_action(self, input_maps, input_params, last_actions, last_rewards):
        fd = {
            'input/map:0': input_maps,
            'input/params:0': input_params,
            'input/action_taken:0': last_actions,
            'input/reward:0': last_rewards,
            'input/time_steps:0': self.state_stack_size,
            'impala_rnn/input/c_state:0': self.c_state,
            'impala_rnn/input/h_state:0': self.h_state,
        }

        logits, action, state = self.sess.run([self.policy_logits_op, self.action_op, self.lstm_state_op], feed_dict = fd)
        self.c_state, self.h_state = state

        return logits[0][0], action[0]

    def reset(self):
        self.state = stacked_env(self.config)
        obs = self.env.reset()
        return self.new_state(obs)

    def env_step(self, action):
        obs, reward, done, info = self.env.step(action)
        new_st = self.new_state(obs)

        return reward, new_st, done

    def loop_body(self):
        logits, action = self.get_action([self.prev_st.state], [self.prev_st.params], [self.prev_action], [self.prev_reward])

        reward, new_st, done = self.env_step(action)
        new_model_st = halite_model_pb2.State(state=new_st.state.tobytes(), params=new_st.params.tobytes())

        logging.info('loop: {}: {}.{}: state: {}, action: {}, new_state: {}, reward: {}, done: {}, logits: {}'.format(
            self.step, self.owner_id, self.env_id, self.prev_st.state, action, new_st.state, reward, done, softmax(logits)))

        model_he = halite_model_pb2.HistoryEntry(
                owner_id = self.owner_id,
                env_id = self.env_id,
                step = self.step,
                state = self.prev_model_st,
                action = action,
                reward = reward,
                new_state = new_model_st,
                done = done,
                logits = logits,
            )

        self.trajectory.append(model_he)
        if len(self.trajectory) == self.trlen:
            #logging.info('c_state: {}, init_c_state: {}'.format(self.c_state, self.init_c_state))
            tr = halite_model_pb2.Trajectory(
                owner_id = self.owner_id,
                env_id = self.env_id,
                entries = self.trajectory,
                train_step = self.train_step,
                c_state = self.init_c_state[0],
                h_state = self.init_h_state[0],

            )
            self.stub.HistoryAppend(tr)
            self.trajectory = []

            self.init_c_state = self.c_state.copy()
            self.init_h_state = self.h_state.copy()

            self.load_model()

        if done:
            self.c_state = np.zeros_like(self.c_state)
            self.h_state = np.zeros_like(self.h_state)

        self.prev_st = new_st
        self.prev_model_st = new_model_st
        self.prev_action = action
        self.prev_reward = reward

        return done

def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, help='Training directory')
    parser.add_argument('--tmp_dir', type=str, default='/tmp', help='Temporary directory to store model checkpoint for restoration process')
    parser.add_argument('--game', type=str, default='Breakout-v0', help='Game name')
    parser.add_argument('--dump_model', action='store_true', help='Dump model into checkpoint/graph and exit')
    parser.add_argument('--remote_addr', default='localhost:5001', type=str, help='Remote service address to connect to for inference')
    parser.add_argument('--logfile', type=str, help='Logfile')
    parser.add_argument('--player_id', default=0, type=int, help='Player ID used to index history entries')
    parser.add_argument('--num_clients', default=32, type=int, help='Maximum number of clients simultaneously connected to the training server')
    parser.add_argument('--num_episodes', default=1000, type=int, help='Number of episodes to run')

    FLAGS = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['NVIDIA_VISIBLE_DEVICES'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    tf.logging.set_verbosity(tf.logging.ERROR)

    logging.basicConfig(filename=FLAGS.logfile, filemode='a', level=logging.INFO, format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')

    config = {
            'game': FLAGS.game,
            'tmp_dir': FLAGS.tmp_dir,
            'state_stack_size': 1,
            'remote_addr': FLAGS.remote_addr,
            'train_dir': FLAGS.train_dir,
            'input_map_shape': [1],
            'input_params_shape': [1],
            'owner_id': FLAGS.player_id,
            'env_id': 0,
    }

    if FLAGS.dump_model:
        #e = gym.make(config['game'])
        e = env.Env()
        config['num_actions'] = e.action_space.n
        config['batch_size'] = FLAGS.num_clients

        import model

        m = model.create_model(config)
        m.save_checkpoint()
        exit(0)

    game = GameWrapper(config)

    episode = 0
    episode_rewards = []

    while FLAGS.num_episodes < 0 or episode < FLAGS.num_episodes:

        game.prev_st = game.reset()
        game.prev_model_st = halite_model_pb2.State(state=game.prev_st.state.tobytes(), params=game.prev_st.params.tobytes())
        done = False
        rewards = []

        while not done:
            done = game.loop_body()
            rewards.append(game.prev_reward)

        er = np.sum(rewards)
        episode_rewards.append(er)
        if len(episode_rewards) > 100:
            episode_rewards = episode_rewards[1:]

        logging.info('{}: last train_step: {}, steps: {}, episode reward: {}, mean episode reward: {:.1f}, std: {:.1f}'.format(
            episode, game.train_step, len(rewards), er, np.mean(episode_rewards), np.std(episode_rewards)))
        episode += 1

def main():
    try:
        run_main()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()

        logging.error("got error: {}".format(e))

        import traceback

        lines = traceback.format_exc().splitlines()
        for l in lines:
            logging.error(l)

        #traceback.print_exception(exc_type, exc_value, exc_traceback)
        exit(-1)

if __name__ == '__main__':
    main()
