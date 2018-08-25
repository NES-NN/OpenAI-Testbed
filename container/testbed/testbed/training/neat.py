"""
    neat.py
"""
import gym
import neat
import numpy as np
from ppaquette_gym_super_mario.wrappers import *


# -----------------------------------------------------------------------------
#  CONFIGURATION
# -----------------------------------------------------------------------------


def load_config_with_defaults(config_path):
    """Creates a NEAT config with default settings and an absolute config path."""
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    return config


# -----------------------------------------------------------------------------
#  HELPERS
# -----------------------------------------------------------------------------


def sigmoid(x):
    """Simple sigmoid function"""
    return 1 / (1 + np.exp(-x))


def clean_outputs(outputs):
    """Creates valid button presses to pass to gym"""
    return [1 if sigmoid(b) > 0.5 else 0 for b in outputs]


# TODO: Need a better fitness function
def calculate_fitness(info):
    """Calculates a fitness score based on Gym information"""
    return info['distance']


def generate_env_arr(start = 0, end = 32):
    env = []
    for i in range(start, end):
        env.append(
            gym.make(
                'ppaquette/SuperMarioBros-{:d}-{:d}-Tiles-v0'.format(
                    int((i / 4) + 1),
                    int((i % 4) + 1)
                )
            )
        )
    return env