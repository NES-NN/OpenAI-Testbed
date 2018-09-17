import argparse
import logging
import pickle
import neat
import gym
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
START_DISTANCE = 0
END_DISTANCE = 0
MAX_DISTANCE = 0
SAVE_INTERVAL = 5
EPISODES = 5
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
    fitnesses = []

    for e in range(EPISODES):
        done = False
        stuck = 0
        stuck_max = 600

        ENV.loadSaveStateFile(START_DISTANCE)
        observation = ENV.reset()

        while not done:
            # Get move from NN
            outputs = neat_.clean_outputs(net.activate(observation.flatten()))

            # Make move
            observation, reward, done, info = ENV.step(outputs)

            # Check if Mario is progressing in level
            stuck += 1 if reward <= 0 else 0

            # Save out state progress
            global MAX_DISTANCE
            if info['distance'] > MAX_DISTANCE and info['distance'] % SAVE_INTERVAL == 0:
                MAX_DISTANCE = info['distance']
                ENV.saveToStateFile()

            # TODO: Needs improvement, need to disable at end of level and when in a pipe.
            # Also not sure what will happen with END_DISTANCE when in a pipe..
            if stuck > stuck_max or info['distance'] > END_DISTANCE:
                break

            fitnesses.append(info['distance'])

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

    while True:
        winner = pop.run(pe.evaluate, 1)

        visualize.plot_stats(stats, ylog=False, view=False,
            filename=SESSION_DIR + 'avg_fitness.svg')
        visualize.plot_species(stats, view=False,
            filename=SESSION_DIR + 'speciation.svg')

        # Save the best Genome from the last 5 gens.
        with open(SESSION_DIR + 'Best-{}.pkl'.format(len(stats.most_fit_genomes)), 'wb') as output:
            pickle.dump(winner, output, 1)

        if stats.get_fitness_mean()[-1] >= END_DISTANCE:
            break


def main():
    parser = argparse.ArgumentParser(description='Mario NEAT Agent Trainer')
    parser.add_argument('--config-path', type=str, default="/opt/train/NEAT/config-feedforward",
                        help="The path to the NEAT parameter config file to use")
    parser.add_argument('--num-cores', type=int, default=1,
                        help="The number of cores on your computer for parallel execution")
    parser.add_argument('--state-path', type=str, default="/opt/train/NEAT/states/",
                        help="The directory to pull and store states from")
    parser.add_argument('--session-path', type=str, default="/opt/train/NEAT/session/",
                        help="The directory to store output files within")
    parser.add_argument('--input-distance', type=int, default=40,
                        help="The target distance Mario should start training from")
    parser.add_argument('--target-distance', type=int, default=1000,
                        help="The target distance Mario should achieve before closing")
    parser.add_argument('--episodes', type=int, default=5,
                        help="The number of episodes to run for each genome")
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

    global START_DISTANCE
    START_DISTANCE = args.input_distance

    global END_DISTANCE
    END_DISTANCE = args.target_distance

    global MAX_DISTANCE
    MAX_DISTANCE = 0

    global SAVE_INTERVAL
    SAVE_INTERVAL = 5

    global EPISODES
    EPISODES = args.episodes

    global ENV
    ENV = get_env()

    # Evolve!
    evolve(config=config, num_cores=args.num_cores)


if __name__ == '__main__':
    main()
