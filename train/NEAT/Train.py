import argparse
import gym
import ppaquette_gym_super_mario
import csv
import os
import pickle
import numpy as np
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


def master_extract_cloud_ga(population):
    """Save offspring statistics"""
    path = args.loggingDir + "/snapshot_gen_{:04}/".format(int(population.generation))
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_offspring_{:04}.dat".format(int(population.generation))
    with open(os.path.join(path, filename), 'w+') as file:        
        writer = csv.writer(file, delimiter=' ')
        for currentSpeciesResults in population.species:
            for member in currentSpeciesResults.members:                
                #if (result == "distance"):
                # {'level': 0, 'distance': 6, 'score': 0, 'coins': 0, 'time': 388, 'player_status': 0, 'life': 3}    
                if member in genome_infos:         
                    row = np.hstack(("{:.6f}".format(genome_infos[member].get('score')),"{:.8f}".format(genome_infos[member].get('time')),"{:.6f}".format(member.fitness)))
                    writer.writerow(row)
                else:
                    print('ERROR: could not find: genome' + member.ID)

    print('Created snapshot:' + filename)


def master_extract_parent(population):
    """Save parent statistics"""
    generationNumber = population.generation
    
    #We will set the winner genome of the generation as the next parent
    path = args.loggingDir + "/snapshot_gen_{:04}/".format(generationNumber +1)
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_parent_{:04}.dat".format(generationNumber +1)    
    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')
      
        #get adverage score
        advScore = 0
        for genomeInfo in genome_infos:
            advScore += genome_infos[genomeInfo].get('score')
        advScore = advScore / len(genome_infos)

        #get adverage time
        advTime = 0
        for genomeInfo in genome_infos:
            advTime += genome_infos[genomeInfo].get('time')
        advTime = advTime / len(genome_infos)

        fit_mean = mean([c.fitness for c in genome_infos.keys()])

        #np is NumPy
        #looks like VINE wants floating point values.        
        row = np.hstack(("{:.6f}".format(advScore),"{:.8f}".format(advTime),"{:.6f}".format(fit_mean)))
        writer.writerow(row)
       
    print('Created parent snapshot:' + filename)


# -----------------------------------------------------------------------------
#  NEAT TRAINING
# -----------------------------------------------------------------------------


def simulate_species(net, env, episodes=1):
    """Run the simulation"""
    fitnesses = []

    for runs in range(episodes):  
        if episodes > 1:
            print('Running episode: '+ str(runs))   
            
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
        print("Species fitness: %s" % str(fitness))

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
    fitness, info = simulate_species(net, smb_env, args.episodes)    
    return fitness

def train_network(env):


    def evaluate_genome(g):
        """Evaluate genome"""
        net = nn.create_feed_forward_phenotype(g)
        return simulate_species(net, env, args.episodes)


    def eval_fitness(genomes):
        """Evaluate fitness"""
        for g in genomes:            
            fitness, info = evaluate_genome(g)    
            genome_infos[g] = info
            g.fitness = fitness

        # Log vine results
        if args.vineLogging: 
            master_extract_cloud_ga(pop)
            master_extract_parent(pop)

    # Validate vine config
    if args.numCores > 1:
        print('VINE logging does not support multi-core atm. Turning off VINE logging.')
        args.vineLogging = False

    # NEAT
    pop = population.Population(args.configFile)

    # Load Save File
    if args.saveFile and os.path.exists(args.saveFile):
        pop.load_checkpoint(args.saveFile)

    if not args.playBest:
        # For VINE stop running in parallel
        if args.vineLogging or args.numCores == 1: 
            genome_infos.clear()
            pop.run(eval_fitness, args.generations)
        else:
            pe = parallel.ParallelEvaluator(args.numCores, worker_evaluate_genome)       
            pop.run(pe.evaluate, args.generations) 
       
        pop.save_checkpoint(args.saveFile)

        # Log statistics
        statistics.save_stats(pop.statistics)
        statistics.save_species_count(pop.statistics)
        statistics.save_species_fitness(pop.statistics)

        print('Number of evaluations: {0}'.format(pop.total_evaluations))

        # Show output of the most fit genome against training data
        winner = pop.statistics.best_genome()
        
        # Save best network
        with open(args.saveFile + '.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)
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
                        help="The number of times to run a single genome. This takes the fitness score from the worst run")
    parser.add_argument('--generations', type=int, default=10,
                        help="The number of generations to evolve the network")
    parser.add_argument('--save-file', dest="saveFile", type=str, default="neat_network",
                        help="Uses a checkpoint to start the simulation")
    parser.add_argument('--play-best', dest="playBest", action='store_true',
                        help="Play the best of a trained network")
    parser.add_argument('--num-cores', dest="numCores", type=int, default=1,
                        help="The number cores on your computer for parallel execution")
    parser.add_argument('--vine-logging', dest="vineLogging", action='store_true',
                        help="Log out fitness of patent and children generations for VINE")
    parser.add_argument('--logging-dir', dest="loggingDir", type=str, default="snapshots",
                        help="The directory to log into")
    parser.add_argument('--display', dest="display", type=int, default=1,
                        help="The virtual display buffer to bind to.  Will only bind on positive integers")
    parser.add_argument('--v', action='store_true',
                        help="Shows fitness for each species")
    args = parser.parse_args()

    if args.display >= 1:
        os.environ["DISPLAY"] = ":{0}".format(args.display)

    smb_env = gym.make(game_name)
    train_network(smb_env)
