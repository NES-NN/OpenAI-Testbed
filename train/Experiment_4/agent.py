import argparse
import logging
import pickle
import neat
import gym
import csv
import numpy as np
import os
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
STUCK_POINT = 0
DRILL_LENGTH = 0
PASS_LENGTH = 0
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

def evaluate(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    done = False
    stuck = 0
    stuck_max = 600

    ENV.loadSaveStateFile(STUCK_POINT)
    observation = ENV.reset()

    while not done:
        # Get move from NN
        outputs = neat_.clean_outputs(net.activate(observation.flatten()))

        # Make move
        observation, reward, done, info = ENV.step(outputs)

        # Check if Mario is progressing in level
        stuck += 1 if reward <= 0 else 0

        # TODO: Needs improvement, need to disable at end of level and when in a pipe.
        # Also not sure what will happen with END_DISTANCE when in a pipe..
        if stuck > stuck_max or info['distance'] > STUCK_POINT+DRILL_LENGTH:
            break

    ENV.close()

    return neat_.calculate_fitness(info)


def log(stats):
    generation = len(stats.most_fit_genomes)
    best_fitness = [c.fitness for c in stats.most_fit_genomes]
    avg_fitness = np.array(stats.get_fitness_mean())
    stdev_fitness = np.array(stats.get_fitness_stdev())

    with open(str(STUCK_POINT) + '_stats.csv', mode='a') as stats_file:
        stats_writer = csv.writer(stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        stats_writer.writerow([generation, best_fitness[-1], avg_fitness[-1], stdev_fitness[-1]])


def evolve(config, num_cores):
    pop = neat.Population(config)

    for sp in [530, 1320, 1600, 2204, 2800]:
        global STUCK_POINT
        STUCK_POINT = sp

        pop.add_reporter(neat.Checkpointer(1, 600, CHECKPOINTS_DIR + "neat-checkpoint-" + str(STUCK_POINT) + "-"))
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

        pe = neat.ParallelEvaluator(num_cores, evaluate)

        while True:
            winner = pop.run(pe.evaluate, 1)

            visualize.plot_stats(stats, ylog=False, view=False,
                filename=SESSION_DIR + 'avg_fitness-' + str(STUCK_POINT) + '.svg')

            visualize.plot_species(stats, view=False,
                filename=SESSION_DIR + 'speciation-' + str(STUCK_POINT) + '.svg')

            log(stats)

            # Save the best Genome from the last 5 gens.
            with open(SESSION_DIR + 'Best-{}-{}.pkl'.format(str(STUCK_POINT),
                                                            len(stats.most_fit_genomes)), 'wb'
                      ) as output:
                pickle.dump(winner, output, 1)

            if stats.get_fitness_mean()[-1] >= STUCK_POINT+PASS_LENGTH:
                break


def main():
    parser = argparse.ArgumentParser(description='Mario NEAT Agent Trainer')
    parser.add_argument('--config-path', type=str, default="/opt/train/NEAT/config-feedforward",
                        help="The path to the NEAT parameter config file to use")
    parser.add_argument('--num-cores', type=int, default=1,
                        help="The number of cores on your computer for parallel execution")
    parser.add_argument('--state-path', type=str, default="/opt/train/Experiment_4/states/",
                        help="The directory to pull and store states from")
    parser.add_argument('--session-path', type=str, default="/opt/train/Experiment_4/session/",
                        help="The directory to store output files within")
    parser.add_argument('--input-distance', type=int, default=40,
                        help="The target distance Mario should start training from")
    parser.add_argument('--target-distance', type=int, default=1000,
                        help="The target distance Mario should achieve before closing")
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

    global DRILL_LENGTH
    DRILL_LENGTH = 350

    global PASS_LENGTH
    PASS_LENGTH = DRILL_LENGTH - 100

    global ENV
    ENV = get_env()

    # Evolve!
    evolve(config=config, num_cores=args.num_cores)


if __name__ == '__main__':
    main()
