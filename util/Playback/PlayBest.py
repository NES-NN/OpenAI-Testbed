import argparse
import logging
import pickle
import neat
import gym
import numpy as np
import os
from ppaquette_gym_super_mario.wrappers import *
from testbed.training import neat as neat_


# -----------------------------------------------------------------------------
#  GLOBALS
# -----------------------------------------------------------------------------

GYM_NAME = 'ppaquette/SavingSuperMarioBros-1-1-Tiles-v0'
ENV = None


# -----------------------------------------------------------------------------
#  HELPERS
# -----------------------------------------------------------------------------

def get_env():
    return gym.make(GYM_NAME)


# -----------------------------------------------------------------------------
#  NEAT PLAYER
# -----------------------------------------------------------------------------

def play_best(config, play_best_file):
    wrapper = SetPlayingMode('normal')
    e = wrapper(ENV)

    genome = pickle.load(open(play_best_file, 'rb'))

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    done = False
    e.reset()

    observation = e.step([0,0,0,0,0,0])

    while not done:
        action = neat_.clean_outputs(net.activate(observation[0].flatten()))

        print('Action taken by NN : [{} {} {} {} {} {}]'.format(
                ('UP' if action[0] else ''),
                ('Left' if action[1] else ''),
                ('Right' if action[2] else ''),
                ('Down' if action[3] else ''),
                ('A' if action[4] else ''),
                ('B' if action[5] else '')))

        observation = e.step(action)


def main():
    parser = argparse.ArgumentParser(description='Play the best player form a trained NEAT Network')
    parser.add_argument('--config-file', type=str, default='config-feedforward',
                        help="The path to the NEAT parameter config file to use")
    parser.add_argument('--play-best', type=str, default='Best.pkl',
                        help="The path to the pickle output from training to load the best network")
    args = parser.parse_args()

    # Load Config
    config = neat_.load_config_with_defaults(args.config_file)

    global ENV
    ENV = get_env()

    play_best(config, args.play_best)


if __name__ == '__main__':
    main()
