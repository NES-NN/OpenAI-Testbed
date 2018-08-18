import os
import numpy as np
import pickle
import argparse
import neat
import gym
from ppaquette_gym_super_mario.wrappers import *
from testbed.logging import visualize

# -----------------------------------------------------------------------------
#  GLOBALS
# -----------------------------------------------------------------------------


def generate_env_arr():
    env = []
    for i in range(0, 32):
        env.append(
            gym.make(
                'ppaquette/SuperMarioBros-{:d}-{:d}-Tiles-v0'.format(
                    int((i / 4) + 1),
                    int((i % 4) + 1)
                )
            )
        )
    return env


ENV_ARR = generate_env_arr()

# -----------------------------------------------------------------------------
#  NEAT TRAINING
# -----------------------------------------------------------------------------


def clean_outputs(outputs):
    """Creates valid button presses to pass to gym"""
    return [1 if sigmoid(b) > 0.5 else 0 for b in outputs]


def sigmoid(x):
    """Simple sigmoid function"""
    return 1 / (1 + np.exp(-x))


def calculate_fitness(info):
    # TODO: Need a better fitness function.
    return info['distance']


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
            outputs = clean_outputs(net.activate(observation.flatten()))

            # Make move
            observation, reward, done, info = ENV_ARR[i].step(outputs)

            # Check if Mario is progressing in level
            stuck += 1 if reward <= 0 else 0

            # TODO: Needs improvement, need to disable at end of level and when in a pipe.
            if stuck > stuck_max:
                ENV_ARR[i].close()
                return calculate_fitness(info)

        if info['life'] == 0:
            break

        ENV_ARR[i].close()
    return calculate_fitness(info)


def run(config, num_cores, checkpoint=None):
    if checkpoint:
        pop = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        pop = neat.Population(config)

    # Checkpoint every 5 generations or 10 min.
    pop.add_reporter(neat.Checkpointer(5, 600))

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pe = neat.ParallelEvaluator(num_cores, eval_genome)

    # We could use this to set a limit of generations... But i dont think we need too... 
    # for x in range(args.generations if args.generations > 0 else sys.maxsize):
    while True:
        winner = pop.run(pe.evaluate, 5)

        visualize.plot_stats(stats, ylog=False, view=False)
        visualize.plot_species(stats, view=False)

        # Save the best Genome from the last 5 gens.
        with open('Best-{}.pkl'.format(len(stats.most_fit_genomes)), 'wb') as output:
            pickle.dump(winner, output, 1)


def play_best(config, play_best_file):
    wrapper = SetPlayingMode('normal')
    e = wrapper(ENV_ARR[0])

    genome = pickle.load(open(play_best_file, 'rb'))

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    done = False
    observation = e.reset()
    while not done:
        action = clean_outputs(net.activate(observation.flatten()))
        print('Action taken by NN : [{} {} {} {} {} {}]'.format(
                ('UP' if action[0] else ''),
                ('Left' if action[1] else ''),
                ('Right' if action[2] else ''),
                ('Down' if action[3] else ''),
                ('A' if action[4] else ''),
                ('B' if action[5] else '')))
        e.step(action)


def load_config(config_file):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    return config


def main():
    parser = argparse.ArgumentParser(description='Mario NEAT Trainer')
    parser.add_argument('--config-file', type=str, default="config-feedforward",
                        help="The path to the NEAT parameter config file to use")
    parser.add_argument('--generations', type=int, default=-1,
                        help="The number of generations to evolve the network")
    parser.add_argument('--play-best', type=str, default="",
                        help="Play the best of a trained network")
    parser.add_argument('--checkpoint', type=str, default="",
                        help="Resume training from a saved checkpoint")
    parser.add_argument('--num-cores', type=int, default=4,
                        help="The number of cores on your computer for parallel execution")
    args = parser.parse_args()

    # Load the NEAT config file
    config = load_config(args.config_file)

    # Ensure the display variable is bound
    os.environ["DISPLAY"] = ":1"

    if args.play_best:
        play_best(config, args.play_best)
    else:
        run(config, args.num_cores, args.checkpoint)


if __name__ == '__main__':
    main()
