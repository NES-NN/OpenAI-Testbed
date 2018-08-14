import argparse
import logging
import pickle
import neat
import gym
from ppaquette_gym_super_mario.wrappers import *
from testbed.logging import visualize
from testbed.training import neat as neat_


# -----------------------------------------------------------------------------
#  GLOBALS
# -----------------------------------------------------------------------------

GYM_NAME = 'ppaquette/SavingSuperMarioBros-1-1-Tiles-v0'
GYM_ENV = None


def evaluate(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    done = False
    stuck = 0
    stuck_max = 600
    info = {}

    GYM_ENV.loadSaveStateFile()
    observation = GYM_ENV.reset()

    while not done:
        outputs = neat_.clean_outputs(net.activate(observation.flatten()))
        observation, reward, done, info = GYM_ENV.step(outputs)
        stuck += 1 if reward <= 0 else 0

        # TODO: Needs improvement, need to disable at end of level and when in a pipe.
        if stuck > stuck_max:
            break
        if info['life'] == 0:
            break

    GYM_ENV.close()
    return neat_.calculate_fitness(info)


def evolve(config, num_cores):
    pop = neat.Population(config)

    pop.add_reporter(neat.Checkpointer(1, 600))
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pe = neat.ParallelEvaluator(num_cores, evaluate)

    while True:
        winner = pop.run(pe.evaluate, 1)

        visualize.plot_stats(stats, ylog=False, view=False)
        visualize.plot_species(stats, view=False)

        # Save the best Genome from the last 5 gens.
        with open('Best-{}.pkl'.format(len(stats.most_fit_genomes)), 'wb') as output:
            pickle.dump(winner, output, 1)


def main():
    parser = argparse.ArgumentParser(description='Mario NEAT Agent Trainer')
    parser.add_argument('--config-path', type=str, default="/Users/joshuabeemster/Documents/GitHub/OpenAI-Testbed/train/NEAT/config-feedforward",
                        help="The path to the NEAT parameter config file to use")
    parser.add_argument('--num-cores', type=int, default=1,
                        help="The number of cores on your computer for parallel execution")
    parser.add_argument('--state-path', type=str, default="/Users/joshuabeemster/Documents/GitHub/OpenAI-Testbed/train/NEAT/states/test.fcs",
                        help="The path to the state file to commence training from")
    parser.add_argument('--target-distance', type=int, default=1,
                        help="The target distance Mario should achieve before closing")
    args = parser.parse_args()

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Load Config
    config = neat_.load_config_with_defaults(args.config_path)

    # Create Gym Environment
    env = gym.make(GYM_NAME)
    save_wrapper = SetSaveStateFolder(args.state_path)
    # control_wrapper = SetPlayingMode("human")
    global GYM_ENV
    GYM_ENV = save_wrapper(env)

    # Evolve!
    evolve(config=config, num_cores=args.num_cores)


if __name__ == '__main__':
    main()
