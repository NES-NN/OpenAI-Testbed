import os
import sys
import csv
import neat
import pickle
import argparse
import logging
import numpy as np
from testbed.logging import visualize
from testbed.training import neat as neat_
from ppaquette_gym_super_mario.wrappers import *


# -----------------------------------------------------------------------------
#  HELPERS
# -----------------------------------------------------------------------------


def mkdir_p(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# -----------------------------------------------------------------------------
#  NEAT TRAINING
# -----------------------------------------------------------------------------


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    stuck_max = 600
    info = {}


    for i in range(0, 32):
        observation = ENV_ARR[i].reset()
        done = False
        stuck = 0

        while not done:
            # Get move from NN
            outputs = neat_.clean_outputs(net.activate(observation.flatten()))

            # Make move
            observation, reward, done, info = ENV_ARR[i].step(outputs)

            # Check if Mario is progressing in level
            stuck += 1 if reward <= 0 else 0

            # TODO: Needs improvement, need to disable at end of level and when in a pipe.
            if stuck > stuck_max:
                ENV_ARR[i].close()
                return neat_.calculate_fitness(info)

        # If mario dies, don't move to next level, exit.
        if info['life'] == 0:
            break

        ENV_ARR[i].close()
    return neat_.calculate_fitness(info)


# TODO: This works, but could be better... need to make a more 'elegant' version
def load_checkpoint(config):
    try:
        checkpoint = max([x.split("-")[-1] for x in os.listdir(CHECKPOINTS_DIR) if x.startswith("neat-checkpoint-")])
        print("Found checkpoint at gen : " + str(checkpoint) + "... Loading...")
        return neat.Checkpointer.restore_checkpoint(CHECKPOINTS_DIR + "neat-checkpoint-" + checkpoint)
    except Exception as e:
        print("No saved session found, creating new population")
        return neat.Population(config)


def log(stats):
    generation = len(stats.most_fit_genomes)
    best_fitness = [c.fitness for c in stats.most_fit_genomes]
    avg_fitness = np.array(stats.get_fitness_mean())
    stdev_fitness = np.array(stats.get_fitness_stdev())

    with open(SESSION_DIR + 'stats.csv', mode='a') as stats_file:
        stats_writer = csv.writer(stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        stats_writer.writerow([generation, best_fitness[-1], avg_fitness[-1], stdev_fitness[-1]])


def run(config, num_cores):
    pop = load_checkpoint(config)

    pop.add_reporter(neat.Checkpointer(1, 600, CHECKPOINTS_DIR + "neat-checkpoint-"))

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pe = neat.ParallelEvaluator(num_cores, eval_genome)

    for gen in range(500):
        winner = pop.run(pe.evaluate, 1)

        visualize.plot_stats(stats, ylog=False, view=False,
                             filename=SESSION_DIR + 'avg_fitness.svg')
        visualize.plot_species(stats, view=False,
                               filename=SESSION_DIR + 'speciation.svg')

        log(stats)

        # Save the best Genome from the last 5 gens.
        with open(SESSION_DIR + 'Best-{}.pkl'.format(len(stats.most_fit_genomes)), 'wb') as output:
            pickle.dump(winner, output, 1)


def main():
    parser = argparse.ArgumentParser(description='Mario NEAT Trainer')
    parser.add_argument('--config-path', type=str, default="/opt/train/NEAT/config-feedforward",
                        help="The path to the NEAT parameter config file to use")
    parser.add_argument('--num-cores', type=int, default=1,
                        help="The number of cores on your computer for parallel execution")
    parser.add_argument('--state-path', type=str, default="/opt/train/NEAT/states/",
                        help="The directory to pull and store states from")
    parser.add_argument('--session-path', type=str, default="/opt/train/NEAT/session/",
                        help="The directory to store output files within")
    parser.add_argument('--display', type=str, default=":1",
                        help="The display to bind to to allow FCEUX to launch")
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

    global ENV_ARR
    ENV_ARR = neat_.generate_env_arr(STATE_DIR)

    global SAVE_INTERVAL
    SAVE_INTERVAL = 5

    # Ensure the display variable is bound
    os.environ["DISPLAY"] = args.display

    run(config, args.num_cores)


if __name__ == '__main__':
    main()
