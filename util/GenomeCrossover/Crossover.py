import argparse
import pickle
from random import randint
from neat.genome import DefaultGenome
from neat.six_util import iteritems


def crossover(genome1, genome2):
    """ Configure a new genome by crossover from two parent genomes. """
    assert isinstance(genome1.fitness, (int, float))
    assert isinstance(genome2.fitness, (int, float))
    if genome1.fitness > genome2.fitness:
        parent1, parent2 = genome1, genome2
    else:
        parent1, parent2 = genome2, genome1

    connections = {}
    nodes = {}

    # Inherit connection genes
    for key, cg1 in iteritems(parent1.connections):
        cg2 = parent2.connections.get(key)
        if cg2 is None:
            # Excess or disjoint gene: copy from the fittest parent.
            connections[key] = cg1.copy()
        else:
            # Homologous gene: combine genes from both parents.
            connections[key] = cg1.crossover(cg2)

    # Inherit node genes
    parent1_set = parent1.nodes
    parent2_set = parent2.nodes

    for key, ng1 in iteritems(parent1_set):
        ng2 = parent2_set.get(key)
        assert key not in nodes
        if ng2 is None:
            # Extra gene: copy from the fittest parent
            nodes[key] = ng1.copy()
        else:
            # Homologous gene: combine genes from both parents.
            nodes[key] = ng1.crossover(ng2)

    result_genome = DefaultGenome(randint(10000, 99999))
    result_genome.nodes = nodes
    result_genome.connections = connections

    # Set fitness to average of input genomes
    result_genome.fitness = ((genome1.fitness + genome2.fitness) / 2)

    return result_genome


def main():
    parser = argparse.ArgumentParser(description='Python NEAT Genome Crossover')
    parser.add_argument('--genome1-path', type=str, required=True,
                        help="The path to the first Genome to cross")
    parser.add_argument('--genome2-path', type=str, required=True,
                        help="The path to the second Genome to cross")
    parser.add_argument('--genome-out-path', type=str, required=True,
                        help="The output path for the resultant genome")
    args = parser.parse_args()

    # Load the pickled genomes
    genome1 = pickle.load(open(args.genome1_path, 'rb'))
    genome2 = pickle.load(open(args.genome2_path, 'rb'))

    # Cross the genomes together
    genome_out = crossover(genome1, genome2)

    # Pickle and dump the crossed genome to disk
    with open(args.genome_out_path, 'wb') as output:
        pickle.dump(genome_out, output, 1)


if __name__ == '__main__':
    main()
