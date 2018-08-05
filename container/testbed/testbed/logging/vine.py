"""
    vine.py
"""

import os
import json
import csv
import logging
import numpy as np


def save_statistics(parallel_logging_file, generation_count, curr_generation,
                    pop_size, snapshots_dir='snapshots'):
    """Splits apart the file into generations and saves them"""
    with open(parallel_logging_file) as file:
        contents = file.readlines()

        for index in range(0, generation_count):
            start_point = index * pop_size
            end_point = start_point + pop_size

            save_offspring_statistics(index + curr_generation,
                                      contents[start_point:end_point], snapshots_dir)
            save_parent_statistics(index + curr_generation,
                                   contents[start_point:end_point], snapshots_dir)


def save_offspring_statistics(generation, genomes, snapshots_dir):
    """Save offspring statistics"""
    path = snapshots_dir + "/snapshot_gen_{:04}/".format(int(generation))
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_offspring_{:04}.dat".format(int(generation))
    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')

        for genome in genomes:
            genome_dict = json.loads(genome)
            row = np.hstack(
                (
                    "{:.6f}".format(genome_dict['score']),
                    "{:.8f}".format(genome_dict['time']),
                    "{:.6f}".format(genome_dict['fitness'])
                )
            )
            writer.writerow(row)

    logging.debug('Created snapshot: %s', filename)


def save_parent_statistics(generation, genomes, snapshots_dir):
    """Save parent statistics"""
    path = snapshots_dir + "/snapshot_gen_{:04}/".format(generation + 1)
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_parent_{:04}.dat".format(generation + 1)
    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')

        cumulative_score = 0
        cumulative_time = 0
        cumulative_fitness = 0

        for genome in genomes:
            genome_dict = json.loads(genome)
            cumulative_score += genome_dict['score']
            cumulative_time += genome_dict['time']
            cumulative_fitness += genome_dict['fitness']

        row = np.hstack(
            (
                "{:.6f}".format(cumulative_score / len(genomes)),
                "{:.8f}".format(cumulative_time / len(genomes)),
                "{:.6f}".format(cumulative_fitness / len(genomes))
            )
        )
        writer.writerow(row)

    logging.debug('Created parent snapshot: %s', filename)
