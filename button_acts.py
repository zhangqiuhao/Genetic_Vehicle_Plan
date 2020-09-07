import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import time
import numpy

from Genetic_Vehicle_Plan.GLOBAL_VARS import *
from Genetic_Vehicle_Plan.QueuingAlgorithm import Ui_Form
from Genetic_Vehicle_Plan.vpga import *
from Genetic_Vehicle_Plan.input import *


class button_load(QThread):
    signal = pyqtSignal(list)

    def __init__(self):
        super(button_load, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        site_data = []
        vehicle_data = []
        filename = SITE_DATA
        input_site = open(filename).readlines()
        for line in input_site:
            line = line.split(";")
            site_data.append(
                [line[0], line[2], line[6], line[8], str(int(line[9]) / 2), str(10), str(120), str(0.5)])

        filename = VEHICLE_DATA
        input_site = open(filename).readlines()
        for line in input_site:
            line = line.split("\n")[0].split(";")
            vehicle_data.append([line[0], line[1], line[2], line[3], line[4], line[5]])

        self.signal.emit([site_data, vehicle_data])


class button_run(QThread):
    signal = pyqtSignal(list, list)

    def __init__(self,
                 vehicle_data,
                 site_data,
                 num_generations,
                 num_parents_mating,
                 initial_population=None,
                 num_genes=None,
                 sol_per_pop=None,
                 parent_selection_type="sss",
                 keep_parents=-1,
                 K_tournament=3,
                 crossover_type="pmx",
                 crossover_probability=None,
                 mutation_type="random",
                 mutation_probability=None,
                 mutation_by_replacement=False,
                 mutation_percent_genes=10,
                 mutation_num_genes=None,
                 callback_generation=None,
                 verbose=False,
                 multi_processing=False,
                 process_bar=None):
        super(button_run, self).__init__()

        ga_instance = GA(vehicle_data=vehicle_data,
                         site_data=site_data,
                         num_generations=num_generations,
                         num_parents_mating=num_parents_mating,
                         initial_population=initial_population,
                         num_genes=num_genes,
                         sol_per_pop=sol_per_pop,
                         parent_selection_type=parent_selection_type,
                         keep_parents=keep_parents,
                         K_tournament=K_tournament,
                         crossover_type=crossover_type,
                         crossover_probability=crossover_probability,
                         mutation_type=mutation_type,
                         mutation_probability=mutation_probability,
                         mutation_by_replacement=mutation_by_replacement,
                         mutation_percent_genes=mutation_percent_genes,
                         mutation_num_genes=mutation_num_genes,
                         callback_generation=callback_generation,
                         verbose=verbose,
                         multi_processing=multi_processing,
                         process_bar=process_bar)

        print("Starting Genetic Algorithm")
        ga_instance.run()

        self.solution, self.solution_fitness, self.solution_idx, self.err, self.current_site_data, \
        self.current_vehicle_state, self.best_solutions_fitness \
            = ga_instance.best_solution()

        '''
        print(vehicle_data)
        self.vehicle_data = vehicle_data
        self.site_data = site_data
        self.num_generations = num_generations,
        self.num_parents_mating = num_parents_mating,
        self.initial_population = initial_population,
        self.num_genes = num_genes,
        self.sol_per_pop = sol_per_pop,
        self.parent_selection_type = parent_selection_type,
        self.keep_parents = keep_parents,
        self.K_tournament = K_tournament,
        self.crossover_type = crossover_type,
        self.crossover_probability = crossover_probability,
        self.mutation_type = mutation_type,
        self.mutation_probability = mutation_probability,
        self.mutation_by_replacement = mutation_by_replacement,
        self.mutation_percent_genes = mutation_percent_genes,
        self.mutation_num_genes = mutation_num_genes,
        self.callback_generation = callback_generation,
        self.verbose = verbose,
        self.multi_processing = multi_processing,
        self.process_bar = process_bar
        '''

    def __del__(self):
        self.wait()

    def run(self):
        self.signal.emit([self.solution, self.solution_fitness, self.solution_idx, self.err, self.current_site_data,
                          self.current_vehicle_state, self.best_solutions_fitness])
