import argparse
import logging
from multiprocessing import Process
import os
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--bot', required=True, type=str, help='Path to bot file')
parser.add_argument('--remote_addr', required=True, type=str, help='Service address to connect to, format: addr:port')
parser.add_argument('--python', required=True, type=str, help='Path to python3 interpreter')
parser.add_argument('--logfile', type=str, help='Logfile')
parser.add_argument('--num_games', type=int, default=40, help='Maximum number of games')
parser.add_argument('--players_prefix', type=int, default=100, help='Players ID starts with this number')
parser.add_argument('--tmp_dir', type=str, default='/tmp', help='Temporary directory to store model checkpoint for client\'s restoration process')
parser.add_argument('--state_prefix', required=True, type=str, help='Client\'s logfile prefix')

FLAGS = parser.parse_args()

logging.basicConfig(filename=FLAGS.logfile, filemode='a', level=logging.INFO, format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')

def target_func(players_prefix):
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    args = '{} {} --remote_addr {} --player_id {} --logfile {}.{}.log --tmp_dir {}'.format(
            os.path.abspath(FLAGS.python), os.path.abspath(FLAGS.bot),
            FLAGS.remote_addr,
            players_prefix,
            os.path.abspath(FLAGS.state_prefix), players_prefix,
            FLAGS.tmp_dir).split()

    logging.info('starting: {}'.format(' '.join(args)))

    while True:
        p = subprocess.run(args)
        logging.info('I am {} and I am exiting with {} exit code'.format(os.getpid(), p.returncode))
        #break
    return p.returncode

def main():
    processes = []
    start_time = time.time()
    players_prefix = FLAGS.players_prefix
    for i in range(FLAGS.num_games):

        p = Process(target=target_func, args=(players_prefix,))
        p.start()

        players_prefix += 1

        processes.append(p)

    success = 0
    for p in processes:
        ret = p.join()

        logging.info('process {} exited with {} exit code'.format(p.pid, p.exitcode))
        if p.exitcode == 0:
            success += 1

    duration = time.time() - start_time
    logging.info('successfully completed tasks: {}, time to recode input: {} seconds'.format(success, int(duration)))

if __name__ == '__main__':
    main()
