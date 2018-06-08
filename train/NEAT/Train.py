import argparse
import gym
import ppaquette_gym_super_mario
import os
import numpy as np
from neat import nn, population, statistics, parallel

#save offspring stats
#ugly name to match VINE examples for now
def master_extract_cloud_ga(population):
    import csv
    import os

    path = "snapshots/snapshot_gen_{:04}/".format(int(population.generation))
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_offspring_{:04}.dat".format(int(population.generation))
    with open(os.path.join(path, filename), 'w+') as file:        
        writer = csv.writer(file, delimiter=' ')
        for currentSpeciesResults in population.species:
            for member in currentSpeciesResults.members:                
                #if (result == "distance"):
                # {'level': 0, 'distance': 6, 'score': 0, 'coins': 0, 'time': 388, 'player_status': 0, 'life': 3}    
                if member in genomeInfos:         
                    row = np.hstack(("{:.6f}".format(genomeInfos[member].get('score')),"{:.8f}".format(genomeInfos[member].get('time')),"{:.6f}".format(member.fitness)))
                    writer.writerow(row)
                else:
                    print('ERROR: could not find: genome' + member.ID)                
    print('Created snapshot:' + filename)

#save parent stats
#same ugly name issue for now
def master_extract_parent(population):
    import os
    import csv
    from neat.math_util import mean

    generationNumber = population.generation
    
    #We will set the winner genome of the generation as the next parent
    path = "snapshots/snapshot_gen_{:04}/".format(generationNumber +1)
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_parent_{:04}.dat".format(generationNumber +1)    
    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')
      
        #get adverage score
        advScore = 0
        for genomeInfo in genomeInfos:
            advScore += genomeInfos[genomeInfo].get('score')
        advScore = advScore / len(genomeInfos)

        #get adverage time
        advTime = 0
        for genomeInfo in genomeInfos:
            advTime += genomeInfos[genomeInfo].get('time')
        advTime = advTime / len(genomeInfos)

        fit_mean = mean([c.fitness for c in genomeInfos.keys()])

        #np is NumPy
        #looks like VINE wants floating point values.        
        row = np.hstack(("{:.6f}".format(advScore),"{:.8f}".format(advTime),"{:.6f}".format(fit_mean)))
        writer.writerow(row)
       
    print('Created parent snapshot:' + filename)

### User Params ###
# The name of the game to solve
game_name = 'ppaquette/SuperMarioBros-1-1-Tiles-v0'

#list of genome, info pairs
genomeInfos = {}

### End User Params ###

parser = argparse.ArgumentParser(description='Mario NEAT Trainer')
parser.add_argument('--episodes', type=int, default=1,
                    help="The number of times to run a single genome. This takes the fitness score from the worst run")
parser.add_argument('--generations', type=int, default=10,
                    help="The number of generations to evolve the network")
parser.add_argument('--save-file', dest="saveFile", type=str, default="network",
                    help="Uses a checkpoint to start the simulation")
parser.add_argument('--playBest', dest="playBest", type=str, default="network",
                    help="Play the best of a trained network")
parser.add_argument('--num-cores', dest="numCores", type=int, default=1,
                    help="The number cores on your computer for parallel execution")
parser.add_argument('-v',
                    help="Shows fitness for each species")
parser.add_argument('--logging-format', dest="loggingFormat", type=str, default="vine",
                    help="Log out fitness of patent and children generations for VINE")
args = parser.parse_args()

def simulate_species(net, env, episodes=1):
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

    return fitness, info

def clean_outputs(outputs):
    return [1 if sigmoid(b) > 0.5 else 0 for b in outputs]

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def worker_evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)
    
    fitness = simulate_species(net, my_env, args.episodes)    
    return fitness


def train_network(env):

    def evaluate_genome(g):
        net = nn.create_feed_forward_phenotype(g)
        return simulate_species(net, env, args.episodes)


    def eval_fitness(genomes):
        for g in genomes:            
            fitness, info = evaluate_genome(g)    
            genomeInfos[g] = info
            g.fitness = fitness

        #log results
        if args.loggingFormat == "vine": 

            #VINE log population
            master_extract_cloud_ga(pop)

            #VINE log adv. as parent
            master_extract_parent(pop)
            

    #validate vine config
    if args.numCores > 1:
        print('VINE logging does not support multi-core atm. Turning off VINE logging.')
        args.loggingFormat = ""

    # Simulation
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'gym_config')

    # NEAT
    pop = population.Population(config_path)

    # Load Save File
    if args.saveFile and os.path.exists(args.saveFile):
        pop.load_checkpoint(args.saveFile)

    if args.playBest is not None:
        # Start simulation

        # For VINE stop running in parallel
        if args.loggingFormat == "vine": 
            genomeInfos.clear()
            pop.run(eval_fitness, args.generations)
        
        else:
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
        with open(args.saveFile + '.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)
    else: 
        # Save best network
        import pickle
        winner = pickle.load( open(args.playBest + '.pkl', 'rb') )

        winner_net = nn.create_feed_forward_phenotype(winner)

        simulate_species(winner_net, env, 1)

my_env = gym.make(game_name)
train_network(my_env)
