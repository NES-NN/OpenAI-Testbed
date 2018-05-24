import argparse
import gym
import ppaquette_gym_super_mario
import os
import numpy as np
from neat import nn, population, statistics, parallel

### User Params ###
# The name of the game to solve
game_name = 'ppaquette/SuperMarioBros-1-1-Tiles-v0'

### End User Params ###

parser = argparse.ArgumentParser(description='Mario NEAT Trainer')
parser.add_argument('--episodes', type=int, default=1,
                    help="The number of times to run a single genome. This takes the fitness score from the worst run")
parser.add_argument('--generations', type=int, default=10,
                    help="The number of generations to evolve the network")
parser.add_argument('--save-file', dest="saveFile", type=str, default="network.bin",
                    help="Uses a checkpoint to start the simulation")
parser.add_argument('--num-cores', dest="numCores", type=int, default=1,
                    help="The number cores on your computer for parallel execution")
args = parser.parse_args()

def simulate_species(net, env, episodes=1):
    fitnesses = []

    for runs in range(episodes):     
        observation = env.reset()
        done = stuck = accumulated_reward = 0.0

        while not done and stuck < 150:
            # Get move from NN
            outputs = clean_outputs(net.serial_activate(observation.flatten()))

            # Make move
            observation, reward, done, info = env.step(outputs)
            
            # Adds distance traveled left since last move
            accumulated_reward += reward

            # Check if Mario is progressing in level
            stuck += 1 if reward <= 0 else 0
            
        fitnesses.append(accumulated_reward)

    fitness = np.array(fitnesses).mean()
    print("Species fitness: %s" % str(fitness))
    return fitness

def clean_outputs(outputs):
    return [1 if sigmoid(b) > 0.5 else 0 for b in outputs]

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def worker_evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)
    return simulate_species(net, my_env, args.episodes)


def train_network(env):

    def evaluate_genome(g):
        net = nn.create_feed_forward_phenotype(g)
        return simulate_species(net, env, args.episodes)

    def eval_fitness(genomes):
        for g in genomes:
            fitness = evaluate_genome(g)
            g.fitness = fitness

    # Simulation
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'gym_config')

    # NEAT
    pop = population.Population(config_path)

    # Load Save File
    if args.saveFile:
        pop.load_checkpoint(args.saveFile)

    # Start simulation
    pe = parallel.ParallelEvaluator(args.numCores, worker_evaluate_genome)
    pop.run(pe.evaluate, args.generations)

    pop.save_checkpoint(args.saveFile)

    # Log statistics.
    statistics.save_stats(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    # Show output of the most fit genome against training data.
    winner = pop.statistics.best_genome()

    # Save best network
    import pickle
    with open('winner.pkl', 'wb') as output:
       pickle.dump(winner, output, 1)

my_env = gym.make(game_name)
train_network(my_env)
