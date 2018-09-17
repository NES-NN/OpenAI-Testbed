import argparse
import logging
import pickle
import neat
import gym
import numpy as np
import os
import csv
from ppaquette_gym_super_mario.wrappers import *
from testbed.logging import visualize
from testbed.training import neat as neat_


# -----------------------------------------------------------------------------
#  GLOBALS
# -----------------------------------------------------------------------------

GYM_NAME = 'ppaquette/SavingSuperMarioBros-1-1-Tiles-v0'
STATE_DIR = None
SESSION_DIR = None
CHECKPOINTS_DIR = None
SAVE_INTERVAL = 5
ENV = None


# -----------------------------------------------------------------------------
#  HELPERS
# -----------------------------------------------------------------------------

def get_env():
    env = gym.make(GYM_NAME)
    save_wrapper = EnableStateSavingAndLoading(STATE_DIR)
    env = save_wrapper(env)
    return env


def mkdir_p(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# -----------------------------------------------------------------------------
#  NEAT TRAINING
# -----------------------------------------------------------------------------

def play_best(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    stuck = 0
    stuck_max = 600
    done = False
    observation = ENV.reset()

    while not done:
        outputs = neat_.clean_outputs(net.activate(observation.flatten()))

        observation, reward, done, info = ENV.step(outputs)

        # Check if Mario is progressing in level
        stuck += 1 if reward <= 0 else 0

        if stuck > stuck_max:
            break

    ENV.close()

    return info['distance']


def evaluate(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    train_length = 300

    for stuck_point in [530, 1320, 1600, 2204, 2800]:
        done = False
        stuck = 0
        stuck_max = 600
        ENV.loadSaveStateFile(stuck_point)
        observation = ENV.reset()
        info = {}

        while not done:
            # Get move from NN
            outputs = neat_.clean_outputs(net.activate(observation.flatten()))

            # Make move
            observation, reward, done, info = ENV.step(outputs)

            # Check if Mario is progressing in level
            stuck += 1 if reward <= 0 else 0

            if stuck > stuck_max or info['distance'] > stuck_point + train_length:
                break

        # print("STUCKPOINT : " + str(stuck_point) + " Fitness : " + str(info['distance'] - stuck_point))
        fitnesses.append(max(0, info['distance'] - stuck_point))

        ENV.close()

    return np.array(fitnesses).mean()


def load_checkpoint(config):
    try:
        checkpoint = max([x.split("-")[-1] for x in os.listdir(CHECKPOINTS_DIR) if x.startswith("neat-checkpoint-")])
        print("Found checkpoint at gen : " + str(checkpoint) + "... Loading...")
        return neat.Checkpointer.restore_checkpoint(CHECKPOINTS_DIR + "neat-checkpoint-" + checkpoint)
    except Exception as e:
        print("No saved session found, creating new population")
        return neat.Population(config)


def evolve(config, num_cores):
    pop = load_checkpoint(config)
    pop.add_reporter(neat.Checkpointer(1, 600, CHECKPOINTS_DIR + "neat-checkpoint-"))
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pe = neat.ParallelEvaluator(num_cores, evaluate)

    for gen in range(500):
        winner = pop.run(pe.evaluate, 1)

        winner_distance = play_best(winner, config)

        with open('stats.csv', mode='a') as stats_file:
            stats_writer = csv.writer(stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            stats_writer.writerow([gen, winner_distance])

        with open(SESSION_DIR + 'Best-{}.pkl'.format(len(stats.most_fit_genomes)), 'wb') as output:
            pickle.dump(winner, output, 1)


def main():
    parser = argparse.ArgumentParser(description='Mario NEAT Agent Trainer')
    parser.add_argument('--config-path', type=str, default="/opt/train/NEAT/config-feedforward",
                        help="The path to the NEAT parameter config file to use")
    parser.add_argument('--num-cores', type=int, default=1,
                        help="The number of cores on your computer for parallel execution")
    parser.add_argument('--state-path', type=str, default="/opt/train/NEAT/AllYouCanEat-SavePoints/",
                        help="The directory to pull and store states from")
    parser.add_argument('--session-path', type=str, default="/opt/train/NEAT/session/",
                        help="The directory to store output files within")

    args = parser.parse_args()

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Load Config
    config = neat_.load_config_with_defaults(args.config_path)

    # Setup globals
    global STATE_DIR
    STATE_DIR = args.state_path
    mkdir_p(STATE_DIR)

    global SESSION_DIR
    SESSION_DIR = args.session_path
    mkdir_p(SESSION_DIR)

    global CHECKPOINTS_DIR
    CHECKPOINTS_DIR = SESSION_DIR + "checkpoints/"
    mkdir_p(CHECKPOINTS_DIR)

    global ENV
    ENV = get_env()

    # Evolve!
    evolve(config=config, num_cores=args.num_cores)


if __name__ == '__main__':
    main()
