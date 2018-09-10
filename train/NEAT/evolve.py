import os
import sys
import neat
import pickle
import argparse
from testbed.logging import visualize
from testbed.training import neat as neat_
from ppaquette_gym_super_mario.wrappers import *

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
    checkpoint_directory = SESSION_DIR + "checkpoints/"
    try:
        # Check if a checkpoint exists
        checkpoint = max([x.split("-")[-1] for x in os.listdir(checkpoint_directory) if x.startswith("neat-checkpoint-")])
        print ("Found checkpoint at gen :" + str(checkpoint) + "... Loading...")
        return neat.Checkpointer.restore_checkpoint(checkpoint_directory + "neat-checkpoint-" + checkpoint)
    except Exception:
        print("No saved session found, creating new population")
        return neat.Population(config)


def run(config, num_cores):
    pop = load_checkpoint(config)

    # Checkpoint every 5 generations or 10 min.
    pop.add_reporter(neat.Checkpointer(1, 600, SESSION_DIR + "checkpoints/neat-checkpoint-"))

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

        # Save the best Genome from the last 5 gens.
        with open(SESSION_DIR + 'Best/{}.pkl'.format(len(stats.most_fit_genomes)), 'wb') as output:
            pickle.dump(winner, output, 1)


def main():
    parser = argparse.ArgumentParser(description='Mario NEAT Trainer')
    parser.add_argument('--config-file', type=str, default="config-feedforward",
                        help="The path to the NEAT parameter config file to use")
    parser.add_argument('--session', type=str, default="session",
                        help="Where to put states and checkpoints")
    parser.add_argument('--num-cores', type=int, default=1,
                        help="The number of cores on your computer for parallel execution")
    parser.add_argument('--display', type=str, default=":1",
                        help="The display to bind to to allow FCEUX to launch")
    args = parser.parse_args()

    global SESSION_DIR
    SESSION_DIR = os.path.abspath(os.path.dirname(sys.argv[0])) + "/" + args.session + "/"

    global ENV_ARR
    ENV_ARR = neat_.generate_env_arr(SESSION_DIR + "States/")

    global SAVE_INTERVAL
    SAVE_INTERVAL = 5

    # Load the NEAT config file
    config = neat_.load_config_with_defaults(SESSION_DIR + args.config_file)

    # Ensure the display variable is bound
    os.environ["DISPLAY"] = args.display

    run(config, args.num_cores)


if __name__ == '__main__':
    main()
