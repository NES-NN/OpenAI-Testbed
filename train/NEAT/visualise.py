from __future__ import print_function

import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np

import json
import os
import csv

# -----------------------------------------------------------------------------
#  neat-python LOGGING
# -----------------------------------------------------------------------------

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()

# -----------------------------------------------------------------------------
#  VINE LOGGING
# -----------------------------------------------------------------------------
#TODO: I just sort of hacked these in, they need to be cleaned up

def save_statistics(stats, snapshotsDir='snapshots'):
    """Splits apart the file into generations and saves them"""
    with open(parallelLoggingFile) as file:
        contents = file.readlines()

        for n in range(0, gens):
            start_point = n * pop_size
            end_point = start_point + pop_size

            save_offspring_statistics(n + generation, contents[start_point:end_point], snapshotsDir)
            save_parent_statistics(n + generation, contents[start_point:end_point], snapshotsDir)


def save_offspring_statistics(generation, genomes, snapshotsDir):
    """Save offspring statistics"""
    path = snapshotsDir + "/snapshot_gen_{:04}/".format(int(generation))
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


def save_parent_statistics(generation, genomes, snapshotsDir):
    """Save parent statistics"""
    path = snapshotsDir + "/snapshot_gen_{:04}/".format(generation + 1)
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
