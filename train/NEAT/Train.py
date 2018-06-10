import argparse
import gym
import ppaquette_gym_super_mario
import csv
import os
import pickle
import numpy as np
import json
from neat import nn, population, statistics, parallel
from neat.math_util import mean


# -----------------------------------------------------------------------------
#  GLOBALS
# -----------------------------------------------------------------------------


game_name = 'ppaquette/SuperMarioBros-1-1-Tiles-v0'
genome_infos = {}
smb_env = None
args = None


# -----------------------------------------------------------------------------
#  VINE LOGGING
# -----------------------------------------------------------------------------


def save_statistics(pop_size, generation):
    """Splits apart the file into generations and saves them"""
    with open(args.parallelLoggingFile) as file:
        contents = file.readlines()

        for n in range(0, args.generations):
            start_point = n * pop_size
            end_point = start_point + pop_size

            save_offspring_statistics(n + generation, contents[start_point:end_point])
            save_parent_statistics(n + generation, contents[start_point:end_point])


def save_offspring_statistics(generation, genomes):
    """Save offspring statistics"""
    path = args.snapshotsDir + "/snapshot_gen_{:04}/".format(int(generation))
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_offspring_{:04}.dat".format(int(generation))
    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')

        for genome in genomes:
            genome_dict = json.loads(genome)
            row = np.hstack(("{:.6f}".format(genome_dict['score']), "{:.8f}".format(genome_dict['time']), "{:.6f}".format(genome_dict['fitness'])))
            writer.writerow(row)

    print('Created snapshot:' + filename)


def save_parent_statistics(generation, genomes):
    """Save parent statistics"""
    path = args.snapshotsDir + "/snapshot_gen_{:04}/".format(generation + 1)
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_parent_{:04}.dat".format(generation + 1)    
    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')

        cuml_score = 0
        cuml_time = 0
        cuml_fitness = 0

        for genome in genomes:
            genome_dict = json.loads(genome)
            cuml_score += genome_dict['score']
            cuml_time += genome_dict['time']
            cuml_fitness += genome_dict['fitness']

        row = np.hstack(("{:.6f}".format(cuml_score / len(genomes)), "{:.8f}".format(cuml_time / len(genomes)), "{:.6f}".format(cuml_fitness / len(genomes))))
        writer.writerow(row)

    print('Created parent snapshot:' + filename)


# -----------------------------------------------------------------------------
#  NEAT TRAINING
# -----------------------------------------------------------------------------


def simulate_genome(net, env, episodes=1):
    """Run the simulation"""
    fitnesses = []

    for runs in range(episodes):  
        if args.v:
            print('Running episode: %s' % str(runs))   
            
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
    
    if args.v:
        print("Genome fitness: %s" % str(fitness))

    env.close()

    return fitness, info


def clean_outputs(outputs):
    """Produce valid output from neural network"""
    return [1 if sigmoid(b) > 0.5 else 0 for b in outputs]


def sigmoid(x):
    """Sigmoid function"""
    return (1 / (1 + np.exp(-x)))


def worker_evaluate_genome(g):
    """Evalute genome function for multi-threading"""
    net = nn.create_feed_forward_phenotype(g)
    fitness, info = simulate_genome(net, smb_env, args.episodes)

    with open(args.parallelLoggingFile, 'a') as file:
        if fitness <= 0:
            fitness = 1

        file.write(
            json.dumps({
                'fitness': fitness,
                'score': info['score'],
                'time': info['time']
            }) + "\n"
        )

    return fitness


def train_network(env):
    """Train the NEAT network"""
    pop = population.Population(args.configFile)
    gen = 0

    # Load Save File
    if args.saveFile and os.path.exists(args.saveFile):
        pop.load_checkpoint(args.saveFile)
        gen = pop.generation

    # Clear parallel logging file
    open(args.parallelLoggingFile, 'w').close()

    if not args.playBest:
        pe = parallel.ParallelEvaluator(args.numCores, worker_evaluate_genome)       
        pop.run(pe.evaluate, args.generations) 
        pop.save_checkpoint(args.saveFile)

        print("\n ****** Training output ****** \n")
        print("Number of evaluations: {0}".format(pop.total_evaluations))

        print("Saving VINE statistics into: {0}".format(args.snapshotsDir))
        save_statistics(pop.config.pop_size, gen)

        with open(args.saveFile + '.pkl', 'wb') as output:
            print("Saving best genome into: {0}.pkl".format(args.saveFile))
            pickle.dump(pop.statistics.best_genome(), output, 1)
    else: 
        winner = pickle.load(open(args.saveFile + '.pkl', 'rb'))
        winner_net = nn.create_feed_forward_phenotype(winner)
        simulate_species(winner_net, env, 1)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mario NEAT Trainer')
    parser.add_argument('--config-file', dest="configFile", type=str, default="/opt/train/NEAT/gym_config",
                        help="The path to the NEAT parameter config file to use")
    parser.add_argument('--episodes', type=int, default=1,
                        help="The number of times to run a single genome. This takes the fitness score from the mean of all the runs")
    parser.add_argument('--generations', type=int, default=10,
                        help="The number of generations to evolve the network")
    parser.add_argument('--save-file', dest="saveFile", type=str, default="/opt/train/NEAT/neat_network",
                        help="Uses a checkpoint to start the simulation")
    parser.add_argument('--play-best', dest="playBest", action='store_true',
                        help="Play the best of a trained network")
    parser.add_argument('--num-cores', dest="numCores", type=int, default=1,
                        help="The number of cores on your computer for parallel execution")
    parser.add_argument('--parallel-logging-file', dest="parallelLoggingFile", type=str, default="/opt/train/NEAT/parallel_info.ndjson",
                        help="The file path to log all requisite information from every genome")
    parser.add_argument('--snapshots-dir', dest="snapshotsDir", type=str, default="/opt/train/NEAT/snapshots",
                        help="The snapshots directory for VINE logging")
    parser.add_argument('--display', dest="display", type=int, default=1,
                        help="The virtual display buffer to bind on.  Will only bind on positive integers")
    parser.add_argument('--v', action='store_true',
                        help="Shows fitness for each species")
    args = parser.parse_args()

    if args.display >= 1:
        os.environ["DISPLAY"] = ":{0}".format(args.display)

    smb_env = gym.make(game_name)
    train_network(smb_env)
