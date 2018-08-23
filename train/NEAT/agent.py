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


# -----------------------------------------------------------------------------
#  HELPERS
# -----------------------------------------------------------------------------

def get_env():
    env = gym.make(GYM_NAME)
    save_wrapper = EnableStateSavingAndLoading(STATE_PATH)
    env = save_wrapper(env)
    return env


# -----------------------------------------------------------------------------
#  NEAT TRAINING
# -----------------------------------------------------------------------------

def evaluate(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    done = False
    stuck = 0
    stuck_max = 600

    ENV.loadSaveStateFile(START_DISTANCE)
    observation = ENV.reset()

    while not done:
        outputs = neat_.clean_outputs(net.activate(observation.flatten()))
        observation, reward, done, info = ENV.step(outputs)
        stuck += 1 if reward <= 0 else 0

        # TODO: Needs improvement, need to disable at end of level and when in a pipe.
        # Also not sure what will happen with END_DISTANCE when in a pipe..
        if stuck > stuck_max or info['distance'] > END_DISTANCE:
            break

    ENV.close()

    return neat_.calculate_fitness(info)


def evolve(config, num_cores, checkpoint):
    if checkpoint:
        pop = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
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

        if stats.get_fitness_mean()[-1:][0] >= END_DISTANCE:
            break


def main():
    parser = argparse.ArgumentParser(description='Mario NEAT Agent Trainer')
    parser.add_argument('--config-path', type=str, default="/opt/train/NEAT/config-feedforward",
                        help="The path to the NEAT parameter config file to use")
    parser.add_argument('--num-cores', type=int, default=1,
                        help="The number of cores on your computer for parallel execution")
    parser.add_argument('--state-path', type=str, default="/opt/train/stateSaving/saveStates/",
                        help="The path to the state file to commence training from")
    parser.add_argument('--input-distance', type=int, default=40,
                        help="The target distance Mario should start training from")
    parser.add_argument('--target-distance', type=int, default=1000,
                        help="The target distance Mario should achieve before closing")
    parser.add_argument('--checkpoint', type=str, default="",
                        help="Resume training from a saved checkpoint")
    args = parser.parse_args()

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Load Config
    config = neat_.load_config_with_defaults(args.config_path)

    # Setup globals
    global STATE_PATH
    STATE_PATH = args.state_path
    global START_DISTANCE
    START_DISTANCE = args.input_distance
    global END_DISTANCE
    END_DISTANCE = args.target_distance
    global ENV
    ENV = get_env()

    # Evolve!
    evolve(config=config, num_cores=args.num_cores, checkpoint=args.checkpoint)


if __name__ == '__main__':
    main()
