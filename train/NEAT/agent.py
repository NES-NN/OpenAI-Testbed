import os
import gym
import neat
import pickle
import logging
import argparse
from random import randint
from neat.six_util import iteritems
from neat.genome import DefaultGenome
from testbed.training import neat as neat_
from ppaquette_gym_super_mario.wrappers import *


# -----------------------------------------------------------------------------
#  GLOBALS
# -----------------------------------------------------------------------------

GYM_NAME = 'ppaquette/SavingSuperMarioBros-1-1-Tiles-v0'
STATE_DIR = None
SESSION_DIR = None
CHECKPOINTS_DIR = None
SAVE_INTERVAL = 5
DRILL_LENGTH = 350
PASS_LENGTH = DRILL_LENGTH - 100
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


def crossover(genome1, genome2):
    """ Configure a new genome by crossover from two parent genomes. """
    assert isinstance(genome1.fitness, (int, float))
    assert isinstance(genome2.fitness, (int, float))
    if genome1.fitness > genome2.fitness:
        parent1, parent2 = genome1, genome2
    else:
        parent1, parent2 = genome2, genome1

    connections = {}
    nodes = {}

    # Inherit connection genes
    for key, cg1 in iteritems(parent1.connections):
        cg2 = parent2.connections.get(key)
        if cg2 is None:
            # Excess or disjoint gene: copy from the fittest parent.
            connections[key] = cg1.copy()
        else:
            # Homologous gene: combine genes from both parents.
            connections[key] = cg1.crossover(cg2)

    # Inherit node genes
    parent1_set = parent1.nodes
    parent2_set = parent2.nodes

    for key, ng1 in iteritems(parent1_set):
        ng2 = parent2_set.get(key)
        assert key not in nodes
        if ng2 is None:
            # Extra gene: copy from the fittest parent
            nodes[key] = ng1.copy()
        else:
            # Homologous gene: combine genes from both parents.
            nodes[key] = ng1.crossover(ng2)

    result_genome = DefaultGenome(randint(10000, 99999))
    result_genome.connections = connections
    result_genome.nodes = nodes

    # Set fitness to average of input genomes
    result_genome.fitness = ((genome1.fitness + genome2.fitness) / 2)

    return result_genome


def save_genome(fname, genome):
    with open(SESSION_DIR + fname, 'wb') as output:
        pickle.dump(genome, output, 1)


def eval_stuck_point(config, num_cores):
    # Create a new Network
    pop = neat.Population(config)
    pop.add_reporter(neat.Checkpointer(1, 600, CHECKPOINTS_DIR + "neat-checkpoint-" + str(STUCK_POINT) + "-"))
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pe = neat.ParallelEvaluator(num_cores, evaluate)

    while True:
        best = pop.run(pe.evaluate, 1)
        if stats.get_fitness_mean()[-1] >= STUCK_POINT+PASS_LENGTH:
            save_genome('Best_{:d}.pkl'.format(STUCK_POINT), best)
            return stats.best_genome()


def evolve(config, num_cores):
    master = None
    for sp in [530, 1320, 1600, 2204, 2800]:
        # I don't like using a global for this, would prefer to pass the value...
        global STUCK_POINT
        STUCK_POINT = sp

        # If its the first stuck point, set its best to master and move to the next point
        if master is None:
            master = eval_stuck_point(config, num_cores)
        else:
            sp_best = eval_stuck_point(config, num_cores)
            master = crossover(master, sp_best)

        save_genome('Master_{:d}.pkl'.format(sp), master)

    save_genome('SuperMario.pkl', master)


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

    global END_DISTANCE
    END_DISTANCE = args.target_distance

    global MAX_DISTANCE
    MAX_DISTANCE = 0

    global SAVE_INTERVAL
    SAVE_INTERVAL = 5

    global ENV
    ENV = get_env()

    # Evolve!
    evolve(config=config, num_cores=args.num_cores)


if __name__ == '__main__':
    main()
