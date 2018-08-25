import neat
import pickle
import argparse
from ppaquette_gym_super_mario.wrappers import *
from testbed.training import neat as neat_


# -----------------------------------------------------------------------------
#  GLOBALS
# -----------------------------------------------------------------------------

ENV_ARR = neat_.generate_env_arr()


# -----------------------------------------------------------------------------
#  NEAT PLAYER
# -----------------------------------------------------------------------------
def play_best(config, play_best_file):
    wrapper = SetPlayingMode('normal')
    e = wrapper(ENV_ARR[0])

    genome = pickle.load(open(play_best_file, 'rb'))

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    done = False
    observation = e.reset()
    while not done:
        action = neat_.clean_outputs(net.activate(observation.flatten()))
        print('Action taken by NN : [{} {} {} {} {} {}]'.format(
                ('UP' if action[0] else ''),
                ('Left' if action[1] else ''),
                ('Right' if action[2] else ''),
                ('Down' if action[3] else ''),
                ('A' if action[4] else ''),
                ('B' if action[5] else '')))
        e.step(action)


def main():
    parser = argparse.ArgumentParser(description='Play the best player form a trained NEAT Network')
    parser.add_argument('--config-file', type=str, default="config-feedforward",
                        help="The path to the NEAT parameter config file to use")

    args = parser.parse_args()

    config = neat_.load_config_with_defaults(args.config_file)

    play_best(config, args.play_best)


if __name__ == '__main__':
    main()