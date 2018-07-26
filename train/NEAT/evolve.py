import math
import os
import sys
import time
import numpy as np
import pickle
import argparse

import neat
import gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario.wrappers import *

import visualise

# -----------------------------------------------------------------------------
#  GLOBALS
# -----------------------------------------------------------------------------

args = None
env = []

for i in range(0, 32):
    env.append(gym.make('ppaquette/SuperMarioBros-{:d}-{:d}-Tiles-v0'.format(int((i/4)+1), int((i%4)+1))))

# -----------------------------------------------------------------------------
#  NEAT TRAINING
# -----------------------------------------------------------------------------

# Creates valid button presses to pass to gym
def clean_outputs(outputs):
    return [1 if sigmoid(b) > 0.5 else 0 for b in outputs]

# Simple sigmoid function. 
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

#TODO: Need a better fitness function.
def calculate_fitness(info):
    return info['distance']

def eval_genome(genome, config):

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    stuckMax = 600
    
    for i in range(0, 32):
        observation = env[i].reset()
        done = False
        stuck = 0
        while not done:
            # Get move from NN
            outputs = clean_outputs(net.activate(observation.flatten()))

            # Make move
            observation, reward, done, info = env[i].step(outputs)

            # Check if Mario is progressing in level
            stuck += 1 if reward <= 0 else 0

            #TODO: Needs improvment, need to disable at end of level and when in a pipe.
            if (stuck > stuckMax):
                env[i].close()
                return calculate_fitness(info)

        if info['life'] == 0:
            break

        env[i].close()
    return calculate_fitness(info)

def run(config):
    if args.checkpoint:
        pop = neat.Checkpointer.restore_checkpoint(args.checkpoint)
    else:
        pop = neat.Population(config)

    # Checkpoint every 5 generations or 10 min.
    pop.add_reporter(neat.Checkpointer(5, 600))

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pe = neat.ParallelEvaluator(args.numCores, eval_genome)
    
    # We could use this to set a limit of generations... But i dont think we need too... 
    #for x in range(args.generations if args.generations > 0 else sys.maxsize): 
    while True:
        winner = pop.run(pe.evaluate, 5)
        
        if args.vineLogging:
            #TODO: Fix Vine Logging, everything needed should be in stats. 
            visualise.save_statistics(stats)
        else:
            visualise.plot_stats(stats, ylog=False, view=False)
            visualise.plot_species(stats, view=False)

        # Save the best Genome from the last 5 gens.
        with open('Best-{}.pkl'.format(len(stats.most_fit_genomes)), 'wb') as output:
            pickle.dump(winner, output, 1)

def playBest(config, bestFile):
    #TODO: Set to normal...
    # Error - The mode "normal" is not supported. Supported options are "algo" or "human"
    wrapper = SetPlayingMode('human')
    e = wrapper(env[0])

    genome = pickle.load(open(bestFile, 'rb'))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mario NEAT Trainer')
    parser.add_argument('--config-file', dest="configFile", type=str, default="config-feedforward",
                        help="The path to the NEAT parameter config file to use")
    parser.add_argument('--generations', dest="generations", type=int, default=-1,
                        help="The number of generations to evolve the network")
    parser.add_argument('--play-best', dest="playBest", type=str, default="",
                        help="Play the best of a trained network")
    parser.add_argument('--checkpoint', dest="checkpoint", type=str, default="",
                        help="Resmume training from a saved checkpoint")
    parser.add_argument('--num-cores', dest="numCores", type=int, default=4,
                        help="The number of cores on your computer for parallel execution")
    parser.add_argument('--vine-logging', dest="vineLogging", action='store_true',
                        help="Use Vine Logging")
    args = parser.parse_args()

    config = load_config(args.configFile)

    if args.playBest:
        playBest(config, args.playBest)
    else:
        run(config)
