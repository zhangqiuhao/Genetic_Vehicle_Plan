import numpy
import random
import matplotlib.pyplot
import pickle
import time
from Genetic_Vehicle_Plan.vehicle_timetable import *
import multiprocessing

CORE_NUMBER = 6


class GA:
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

        """
        The constructor of the GA class accepts all parameters required to create an instance of the GA class. It validates such parameters.

        num_generations: Number of generations.
        num_parents_mating: Number of solutions to be selected as parents in the mating pool.

        initial_population: A user-defined initial population. It is useful when the user wants to start the generations with a custom initial population. It defaults to None which means no initial population is specified by the user. In this case, PyGAD creates an initial population using the 'sol_per_pop' and 'num_genes' parameters. An exception is raised if the 'initial_population' is None while any of the 2 parameters ('sol_per_pop' or 'num_genes') is also None.
        sol_per_pop: Number of solutions in the population.
        num_genes: Number of parameters in the function.

        init_range_low: The lower value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20 and higher.
        init_range_high: The upper value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20.
        # It is OK to set the value of any of the 2 parameters ('init_range_high' and 'init_range_high') to be equal, higher or lower than the other parameter (i.e. init_range_low is not needed to be lower than init_range_high).

        parent_selection_type: Type of parent selection.
        keep_parents: If 0, this means the parents of the current population will not be used at all in the next population. If -1, this means all parents in the current population will be used in the next population. If set to a value > 0, then the specified value refers to the number of parents in the current population to be used in the next population. In some cases, the parents are of high quality and thus we do not want to loose such some high quality solutions. If some parent selection operators like roulette wheel selection (RWS), the parents may not be of high quality and thus keeping the parents might degarde the quality of the population.
        K_tournament: When the value of 'parent_selection_type' is 'tournament', the 'K_tournament' parameter specifies the number of solutions from which a parent is selected randomly.

        crossover_type: Type of the crossover operator. If  crossover_type=None, then the crossover step is bypassed which means no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
        crossover_probability: The probability of selecting a solution for the crossover operation. If the solution probability is <= crossover_probability, the solution is selected. The value must be between 0 and 1 inclusive.

        mutation_type: Type of the mutation operator. If mutation_type=None, then the mutation step is bypassed which means no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
        mutation_probability: The probability of selecting a gene for the mutation operation. If the gene probability is <= mutation_probability, the gene is selected. The value must be between 0 and 1 inclusive. If specified, then no need for the parameters mutation_percent_genes, mutation_num_genes, random_mutation_min_val, and random_mutation_max_val.

        mutation_by_replacement: An optional bool parameter. It works only when the selected type of mutation is random (mutation_type="random"). In this case, setting mutation_by_replacement=True means replace the gene by the randomly generated value. If False, then it has no effect and random mutation works by adding the random value to the gene.

        mutation_percent_genes: Percentage of genes to mutate which defaults to 10%. This parameter has no action if the parameter mutation_num_genes exists.
        mutation_num_genes: Number of genes to mutate which defaults to None. If the parameter mutation_num_genes exists, then no need for the parameter mutation_percent_genes.

        callback_generation: If not None, then it accepts a function to be called after each generation. This function must accept a single parameter representing the instance of the genetic algorithm. If the function returned "stop", then the run() method stops without completing the generations.
        """

        if initial_population is None:
            if (sol_per_pop is None) or (num_genes is None):
                raise ValueError(
                    "Error creating the initial population\n\nWhen the parameter initial_population is None, then neither of the 2 parameters sol_per_pop and num_genes can be None at the same time.\nThere are 2 options to prepare the initial population:\n1) Create an initial population and assign it to the initial_population parameter. In this case, the values of the 2 parameters sol_per_pop and num_genes will be deduced.\n2) Allow the genetic algorithm to create the initial population automatically by passing valid integer values to the sol_per_pop and num_genes parameters.")
            elif (type(sol_per_pop) is int) and (type(num_genes) in [int, list, numpy.ndarray]):
                # Validating the number of solutions in the population (sol_per_pop)
                if sol_per_pop <= 0:
                    self.valid_parameters = False
                    raise ValueError(
                        "The number of solutions in the population (sol_per_pop) must be > 0 but {sol_per_pop} found. \nThe following parameters must be > 0: \n1) Population size (i.e. number of solutions per population) (sol_per_pop).\n2) Number of selected parents in the mating pool (num_parents_mating).\n".format(
                            sol_per_pop=sol_per_pop))
                # Validating the number of gene.
                if type(num_genes) is int:
                    if num_genes <= 0:
                        self.valid_parameters = False
                        raise ValueError(
                            "Number of genes cannot be <= 0 but {num_genes} found.\n".format(num_genes=num_genes))
                    self.num_genes = num_genes  # Number of genes in the solution.
                else:
                    for _ in num_genes:
                        if type(_) is int and _ <= 0:
                            self.valid_parameters = False
                            raise ValueError(
                                "Number of genes must be int and > 0 but {num_genes} found.\n".format(
                                    num_genes=num_genes))
                    self.num_genes = sum(num_genes)  # Number of genes in the solution.
                # When initial_population=None and the 2 parameters sol_per_pop and num_genes have valid integer values, then the initial population is created.
                # Inside the initialize_population() method, the initial_population attribute is assigned to keep the initial population accessible.
                self.sol_per_pop = sol_per_pop  # Number of solutions in the population.
                self.initialize_population()
            else:
                raise TypeError(
                    "The expected type of both the sol_per_pop and num_genes parameters is int but {sol_per_pop_type} and {num_genes_type} found.".format(
                        sol_per_pop_type=type(sol_per_pop), num_genes_type=type(num_genes)))
        elif numpy.array(initial_population).ndim != 2:
            raise ValueError(
                "A 2D list is expected to the initial_population parameter but a {initial_population_ndim}-D list found.".format(
                    initial_population_ndim=numpy.array(initial_population).ndim))
        else:
            self.initial_population = numpy.array(initial_population)
            self.population = self.initial_population  # A NumPy array holding the initial population.
            self.num_genes = self.initial_population.shape[1]  # Number of genes in the solution.
            self.sol_per_pop = self.initial_population.shape[0]  # Number of solutions in the population.
            self.pop_size = (self.sol_per_pop, self.num_genes)  # The population size.

        # Validating the number of parents to be selected for mating (num_parents_mating)
        if num_parents_mating <= 0:
            self.valid_parameters = False
            raise ValueError(
                "The number of parents mating (num_parents_mating) parameter must be > 0 but {num_parents_mating} found. \nThe following parameters must be > 0: \n1) Population size (i.e. number of solutions per population) (sol_per_pop).\n2) Number of selected parents in the mating pool (num_parents_mating).\n".format(
                    num_parents_mating=num_parents_mating))

        # Validating the number of parents to be selected for mating: num_parents_mating
        if num_parents_mating > self.sol_per_pop:
            self.valid_parameters = False
            raise ValueError(
                "The number of parents to select for mating ({num_parents_mating}) cannot be greater than the number of solutions in the population ({sol_per_pop}) (i.e., num_parents_mating must always be <= sol_per_pop).\n".format(
                    num_parents_mating=num_parents_mating, sol_per_pop=self.sol_per_pop))

        self.num_parents_mating = num_parents_mating

        # crossover: Refers to the method that applies the crossover operator based on the selected type of crossover in the crossover_type property.
        # Validating the crossover type: crossover_type
        if crossover_type == "pmx":
            self.crossover = self.partial_mapped_crossover
        elif crossover_type == "ox":
            self.crossover = self.order_crossover
        elif crossover_type == "cx":
            self.crossover = self.cycle_crossover
        elif crossover_type is None:
            self.crossover = None
        else:
            self.valid_parameters = False
            raise ValueError(
                "Undefined crossover type. \nThe assigned value to the crossover_type ({crossover_type}) argument does not refer to one of the supported crossover types which are: \n-single_point (for single point crossover)\n-two_points (for two points crossover)\n-uniform (for uniform crossover).\n".format(
                    crossover_type=crossover_type))

        self.crossover_type = crossover_type

        if crossover_probability is None:
            self.crossover_probability = None
        elif type(crossover_probability) in [int, float]:
            if 0 <= crossover_probability <= 1:
                self.crossover_probability = crossover_probability
            else:
                self.valid_parameters = False
                raise ValueError(
                    "The value assigned to the 'crossover_probability' parameter must be between 0 and 1 inclusive but {crossover_probability_value} found.".format(
                        crossover_probability_value=crossover_probability))
        else:
            self.valid_parameters = False
            raise ValueError(
                "Unexpected type for the 'crossover_probability' parameter. Float is expected by {crossover_probability_type} found.".format(
                    crossover_probability_type=type(crossover_probability)))

        # mutation: Refers to the method that applies the mutation operator based on the selected type of mutation in the mutation_type property.
        # Validating the mutation type: mutation_type
        if mutation_type == "random":
            self.mutation = self.random_mutation
        elif mutation_type == "swap":
            self.mutation = self.swap_mutation
        elif mutation_type == "reverse":
            self.mutation = self.reverse_mutation
        elif mutation_type is None:
            self.mutation = None
        else:
            self.valid_parameters = False
            raise ValueError(
                "Undefined mutation type. \nThe assigned value to the mutation_type argument ({mutation_type}) does not refer to one of the supported mutation types which are: \n-random (for random mutation)\n-swap (for swap mutation)\n-reverse (for reverse mutation).\n".format(
                    mutation_type=mutation_type))

        self.mutation_type = mutation_type

        if mutation_probability is None:
            self.mutation_probability = None
        elif type(mutation_probability) in [int, float]:
            if 0 <= mutation_probability <= 1:
                self.mutation_probability = mutation_probability
            else:
                self.valid_parameters = False
                raise ValueError(
                    "The value assigned to the 'mutation_probability' parameter must be between 0 and 1 inclusive but {mutation_probability_value} found.".format(
                        mutation_probability_value=mutation_probability))
        else:
            self.valid_parameters = False
            raise ValueError(
                "Unexpected type for the 'mutation_probability' parameter. Float is expected by {mutation_probability_type} found.".format(
                    mutation_probability_type=type(mutation_probability)))

        # Number of genes or percentage of genes to be mutated can be defined when mutation type is not None
        if self.mutation_type is not None:
            if mutation_num_genes is None:
                if mutation_percent_genes < 0 or mutation_percent_genes > 100:
                    self.valid_parameters = False
                    raise ValueError(
                        "The percentage of selected genes for mutation (mutation_percent_genes) must be >= 0 and <= 100 inclusive but ({mutation_percent_genes}) found.\n".format(
                            mutation_percent_genes=mutation_percent_genes))
                else:
                    # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                    if mutation_num_genes is None:
                        mutation_num_genes = numpy.uint32((mutation_percent_genes * self.num_genes) / 100)
                        # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                        if mutation_num_genes == 0:
                            mutation_num_genes = 1
            elif mutation_num_genes <= 0:
                self.valid_parameters = False
                raise ValueError(
                    "The number of selected genes for mutation (mutation_num_genes) cannot be <= 0 but {mutation_num_genes} found.\n".format(
                        mutation_num_genes=mutation_num_genes))
            elif mutation_num_genes > self.num_genes:
                self.valid_parameters = False
                raise ValueError(
                    "The number of selected genes for mutation (mutation_num_genes) ({mutation_num_genes}) cannot be greater than the number of genes ({num_genes}).\n".format(
                        mutation_num_genes=mutation_num_genes, num_genes=self.num_genes))
            elif type(mutation_num_genes) is not int:
                self.valid_parameters = False
                raise ValueError(
                    "The number of selected genes for mutation (mutation_num_genes) must be a positive integer >= 1 but {mutation_num_genes} found.\n".format(
                        mutation_num_genes=mutation_num_genes))
        else:
            pass

        if not (type(mutation_by_replacement) is bool):
            self.valid_parameters = False
            raise TypeError(
                "The expected type of the 'mutation_by_replacement' parameter is bool but {mutation_by_replacement_type} found.".format(
                    mutation_by_replacement_type=type(mutation_by_replacement)))

        self.mutation_by_replacement = mutation_by_replacement

        if self.mutation_type != "random" and self.mutation_by_replacement:
            print(
                "Warning: The mutation_by_replacement parameter is set to True while the mutation_type parameter is not set to random but {mut_type}. Note that the mutation_by_replacement parameter has an effect only when mutation_type='random'.".format(
                    mut_type=mutation_type))

        if (self.mutation_type is None) and (self.crossover_type is None):
            print(
                "Warning: the 2 parameters mutation_type and crossover_type are None. This disables any type of evolution the genetic algorithm can make. As a result, the genetic algorithm cannot find a better solution that the best solution in the initial population.")

        # select_parents: Refers to a method that selects the parents based on the parent selection type specified in the parent_selection_type attribute.
        # Validating the selected type of parent selection: parent_selection_type
        if parent_selection_type == "sss":
            self.select_parents = self.steady_state_selection
        elif parent_selection_type == "rws":
            self.select_parents = self.roulette_wheel_selection
        elif parent_selection_type == "sus":
            self.select_parents = self.stochastic_universal_selection
        elif parent_selection_type == "random":
            self.select_parents = self.random_selection
        elif parent_selection_type == "tournament":
            self.select_parents = self.tournament_selection
        elif parent_selection_type == "rank":
            self.select_parents = self.rank_selection
        else:
            self.valid_parameters = False
            raise ValueError(
                "Undefined parent selection type: ({parent_selection_type}). \nThe assigned value to the parent_selection_type argument does not refer to one of the supported parent selection techniques which are: \n-sss (for steady state selection)\n-rws (for roulette wheel selection)\n-sus (for stochastic universal selection)\n-rank (for rank selection)\n-random (for random selection)\n-tournament (for tournament selection).\n".format(
                    parent_selection_type=parent_selection_type))

        if parent_selection_type == "tournament":
            if K_tournament > self.sol_per_pop:
                K_tournament = self.sol_per_pop
                print(
                    "Warning: K of the tournament selection ({K_tournament}) should not be greater than the number of solutions within the population ({sol_per_pop}).\nK will be clipped to be equal to the number of solutions in the population (sol_per_pop).\n".format(
                        K_tournament=K_tournament, sol_per_pop=self.sol_per_pop))
            elif K_tournament <= 0:
                self.valid_parameters = False
                raise ValueError("K of the tournament selection cannot be <=0 but {K_tournament} found.\n".format(
                    K_tournament=K_tournament))
        self.K_tournament = K_tournament

        # Validating the number of parents to keep in the next population: keep_parents
        if keep_parents > self.sol_per_pop or keep_parents > self.num_parents_mating or keep_parents < -1:
            self.valid_parameters = False
            raise ValueError(
                "Incorrect value to the keep_parents parameter: {keep_parents}. \nThe assigned value to the keep_parent parameter must satisfy the following conditions: \n1) Less than or equal to sol_per_pop\n2) Less than or equal to num_parents_mating\n3) Greater than or equal to -1.".format(
                    keep_parents=keep_parents))
        self.keep_parents = keep_parents

        if self.keep_parents == -1:  # Keep all parents in the next population.
            self.num_offspring = self.sol_per_pop - self.num_parents_mating
        elif self.keep_parents == 0:  # Keep no parents in the next population.
            self.num_offspring = self.sol_per_pop
        elif self.keep_parents > 0:  # Keep the specified number of parents in the next population.
            self.num_offspring = self.sol_per_pop - self.keep_parents

        # Check if the callback_generation exists.
        if not (callback_generation is None):
            # Check if the callback_generation is a function.
            if callable(callback_generation):
                # Check if the callback_generation function accepts only a single paramater.
                if callback_generation.__code__.co_argcount == 1:
                    self.callback_generation = callback_generation
                else:
                    self.valid_parameters = False
                    raise ValueError(
                        "The callback_generation function must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed callback_generation function named '{funcname}' accepts {argcount} argument(s).".format(
                            funcname=callback_generation.__code__.co_name,
                            argcount=callback_generation.__code__.co_argcount))
            else:
                self.valid_parameters = False
                raise ValueError(
                    "The value assigned to the 'callback_generation' parameter is expected to be of type function but {callback_generation_type} found.".format(
                        callback_generation_type=type(callback_generation)))
        else:
            self.callback_generation = None

        # The number of completed generations.
        self.generations_completed = 0

        # At this point, all necessary parameters validation is done successfully and we are sure that the parameters are valid.
        self.valid_parameters = True  # Set to True when all the parameters passed in the GA class constructor are valid.

        # Parameters of the genetic algorithm.
        self.num_generations = abs(num_generations)
        self.parent_selection_type = parent_selection_type

        # Parameters of the mutation operation.
        self.mutation_percent_genes = mutation_percent_genes
        self.mutation_num_genes = mutation_num_genes

        # Verbose mode
        self.verbose = verbose
        self.multi_processing = multi_processing

        # Even such this parameter is declared in the class header, it is assigned to the object here to access it after saving the object.
        self.best_solutions_fitness = []  # A list holding the fitness value of the best solution for each generation.
        self.best_solution_generation = -1  # The generation number at which the best fitness value is reached. It is only assigned the generation number after the `run()` method completes. Otherwise, its value is -1.

        # Get relation between site number and site name
        self.process_bar = process_bar[0]
        self.timer = process_bar[1]
        self.vehicle_data = vehicle_data
        self.site_data = site_data
        self.relation = {}
        _tmp, __ = 0, 0
        for _ in site_data:
            _tmp += _.n_deliver
            self.relation[_.order_num] = range(__, _tmp)
            __ = _tmp
        self.create_t_table = create_time_table(self.site_data, self.vehicle_data, self.relation)

    # Initialize and check the population for its validity
    def create_individual(self):
        """create one individual"""
        # initial a list from 0~n-1, n is the number of the requested cars
        new_ind = list(range(self.num_genes))
        while True:
            # randomly shuffle the list to create new individuals
            random.shuffle(new_ind)
            # check the individual ligity
            if self.check_list_validity(new_ind):
                break
        return new_ind

    def initialize_population(self):
        """Initialize population"""
        # The population will have sol_per_pop chromosome where each chromosome has num_genes genes.
        self.pop_size = (self.sol_per_pop, self.num_genes)
        self.population = []
        for _ in range(self.sol_per_pop):
            new_ind = self.create_individual()
            self.population.append(new_ind)
        self.population = numpy.asarray(self.population)
        # Keeping the initial population in the initial_population attribute.
        self.initial_population = self.population.copy()

    def check_list_validity(self, my_list):
        seen = []
        for number in my_list:
            if number >= 0:
                if number in seen:
                    return False
                else:
                    seen.append(number)
        return True

    def check_matrix_validity(self, my_matrix):
        for my_list in my_matrix:
            if not self.check_list_validity(my_list):
                raise Exception('Existing same car number', my_list)
        return True

    def cal_pop_fitness(self):
        """
        Calculating the fitness values of all solutions in the current population.
        It returns:
            -fitness: An array of the calculated fitness values.
        """

        if not self.valid_parameters:
            raise ValueError(
                "ERROR calling the cal_pop_fitness() method: \nPlease check the parameters passed while creating an instance of the GA class.\n")

        err = []
        if self.multi_processing:
            p = multiprocessing.Pool(CORE_NUMBER)
            pop_fitness = p.map(self.process, self.population)
            p.close()
            p.join()
        else:
            pop_fitness = []
            for sol in self.population:
                sum_err, err = self.process(sol)
                pop_fitness.append(sum_err)
        return pop_fitness, err

    def process(self, sol):
        sum_err, err = self.cal_sol_error(sol)
        if self.multi_processing:
            return sum_err
        else:
            return sum_err, err

    def cal_sol_error(self, sol):
        current_site_data, current_vehicle_state = self.create_t_table.run(sol)
        # Calculating the fitness value of each solution in the current population.
        err = []
        sum_err = []
        for sites in current_site_data:
            err_site = []
            for idx_i in range(len(sites.ideal_UT)):
                idx = len(sites.DT) - len(sites.ideal_UT) + idx_i
                if sites.DT[idx]['name'] != sites.ideal_UT[idx_i]['name']:
                    print('Car_name not match')
                else:
                    # Calculate the time difference between the vehicle arrive time and its ideal pump time
                    # After_or_before = 1 when the arriving time is after the ideal time, which is delay
                    # After_or_before = -1 when the arriving time is before the ideal time, which is lost
                    diff = sites.DT[idx]['time'] - sites.ideal_UT[idx_i]['time']
                    if diff.days < 0:
                        after_or_before = -1
                    else:
                        after_or_before = 1
                    diff = abs(diff).days * 24 * 60 + abs(diff).seconds / 60
                    err_site.append(after_or_before * max(0, diff - sites.t_buffer))
            err.append(err_site)
            sum_err.append(numpy.sum(numpy.abs(numpy.array(err_site))))
        return 1 / numpy.sum(sum_err), err

    def run(self):
        """
        Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """
        start_time = time.time()
        if not self.valid_parameters:
            raise ValueError(
                "ERROR calling the run() method: \nThe run() method cannot be executed with invalid parameters. Please check the parameters passed while creating an instance of the GA class.\n")

        for generation in range(self.num_generations):
            # Measuring the fitness of each chromosome in the population.
            fitness, _ = self.cal_pop_fitness()

            # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
            self.best_solutions_fitness.append(numpy.max(fitness))

            # Selecting the best parents in the population for mating.
            parents = self.select_parents(fitness, num_parents=self.num_parents_mating)

            # If self.crossover_type=None, then no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
            if self.crossover_type is None:
                if self.num_offspring <= self.keep_parents:
                    offspring_crossover = parents[0:self.num_offspring]
                else:
                    offspring_crossover = numpy.concatenate(
                        (parents, self.population[0:(self.num_offspring - parents.shape[0])]))
            else:
                # Generating offspring using crossover.
                offspring_crossover = self.crossover(parents, offspring_size=(self.num_offspring, self.num_genes))

            # If self.mutation_type=None, then no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
            if self.mutation_type is None:
                offspring_mutation = offspring_crossover
            else:
                # Adding some variations to the offspring using mutation.
                offspring_mutation = self.mutation(offspring_crossover)

            if self.keep_parents == 0:
                self.population = offspring_mutation
            elif self.keep_parents == -1:
                # Creating the new population based on the parents and offspring.
                self.population[0:parents.shape[0], :] = parents
                self.population[parents.shape[0]:, :] = offspring_mutation
            elif self.keep_parents > 0:
                parents_to_keep = self.steady_state_selection(fitness, num_parents=self.keep_parents)
                self.population[0:parents_to_keep.shape[0], :] = parents_to_keep
                self.population[parents_to_keep.shape[0]:, :] = offspring_mutation

            self.generations_completed = generation + 1  # The generations_completed attribute holds the number of the last completed generation.

            # If the callback_generation attribute is not None, then call the callback function after the generation.
            if not (self.callback_generation is None):
                r = self.callback_generation(self)
                if type(r) is str and r.lower() == "stop":
                    break

            self.process_bar.setValue((generation + 1) / self.num_generations * 100)
            self.timer.setText("用时: {:.1f}秒，进度： ".format(time.time() - start_time))
            if self.verbose:
                print("\rCompleted the {generation}/{num_generations} generations"
                      .format(generation=generation + 1, num_generations=self.num_generations), end="")

        self.best_solution_generation = \
            numpy.where(
                numpy.array(self.best_solutions_fitness) == numpy.max(numpy.array(self.best_solutions_fitness)))[0][
                0]
        # After the run() method completes, the run_completed flag is changed from False to True.
        self.run_completed = True  # Set to True only after the run() method completes gracefully.
        print("\nComplete")

    # This is the part for parents choosing
    def steady_state_selection(self, fitness, num_parents):
        """
        Selects the parents using the steady-state selection technique.
        Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        # Selecting num_parents individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness[parent_num], :]
        return parents

    def rank_selection(self, fitness, num_parents):
        """
        Selects the parents using the rank selection technique.
        Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :]
        return parents

    def random_selection(self, fitness, num_parents):
        """
        Selects the parents randomly. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        parents = numpy.empty((num_parents, self.population.shape[1]))

        rand_indices = numpy.random.randint(low=0, high=fitness.shape[0], size=num_parents)

        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[rand_indices[parent_num], :]
        return parents

    def tournament_selection(self, fitness, num_parents):
        """
        Selects the parents using the tournament selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            rand_indices = numpy.random.randint(low=0, high=len(fitness), size=self.K_tournament)
            K_fitnesses = fitness[rand_indices]
            selected_parent_idx = numpy.where(K_fitnesses == numpy.max(K_fitnesses))[0][0]
            parents[parent_num, :] = self.population[rand_indices[selected_parent_idx], :]
        return parents

    def roulette_wheel_selection(self, fitness, num_parents):
        """
        Selects the parents using the roulette wheel selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sum = numpy.sum(fitness)
        probes = fitness / fitness_sum
        probes_start = numpy.zeros(probes.shape,
                                   dtype=numpy.float)  # An array holding the start values of the ranges of probabilities.
        probes_end = numpy.zeros(probes.shape,
                                 dtype=numpy.float)  # An array holding the end values of the ranges of probabilities.

        curr = 0.0
        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probes.shape[0]):
            min_probes_idx = numpy.where(probes == numpy.min(probes))[0][0]
            probes_start[min_probes_idx] = curr
            curr = curr + probes[min_probes_idx]
            probes_end[min_probes_idx] = curr
            probes[min_probes_idx] = 99999999999

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            rand_prob = numpy.random.rand()
            for idx in range(probes.shape[0]):
                if probes_start[idx] <= rand_prob < probes_end[idx]:
                    parents[parent_num, :] = self.population[idx, :]
                    break
        return parents

    def stochastic_universal_selection(self, fitness, num_parents):
        """
        Selects the parents using the stochastic universal selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sum = numpy.sum(fitness)
        probes = fitness / fitness_sum
        probes_start = numpy.zeros(probes.shape,
                                   dtype=numpy.float)  # An array holding the start values of the ranges of probabilities.
        probes_end = numpy.zeros(probes.shape,
                                 dtype=numpy.float)  # An array holding the end values of the ranges of probabilities.

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probes.shape[0]):
            min_probes_idx = numpy.where(probes == numpy.min(probes))[0][0]
            probes_start[min_probes_idx] = curr
            curr = curr + probes[min_probes_idx]
            probes_end[min_probes_idx] = curr
            probes[min_probes_idx] = 99999999999

        pointers_distance = 1.0 / self.num_parents_mating  # Distance between different pointers.
        first_pointer = numpy.random.uniform(low=0.0, high=pointers_distance, size=1)  # Location of the first pointer.

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            rand_pointer = first_pointer + parent_num * pointers_distance
            for idx in range(probes.shape[0]):
                if probes_start[idx] <= rand_pointer < probes_end[idx]:
                    parents[parent_num, :] = self.population[idx, :]
                    break
        return parents

    # The is the part for crossover
    def shift_list(self, my_list, shift):
        temp = my_list[0:shift]
        init = 0
        for i in range(shift, len(my_list)):
            my_list[init] = my_list[i]
            init += 1
        my_list[len(my_list) - shift:] = temp
        return my_list

    def create_cycle(self, parent1, parent2, index, init_index, my_list):
        value_in_list1 = parent1[index]
        index_in_list2 = parent2.index(value_in_list1)
        while init_index != index_in_list2:
            my_list.append(index_in_list2)
            self.create_cycle(parent1, parent2, index_in_list2, init_index, my_list)
            return my_list
        return []

    def partial_mapped_crossover(self, parents, offspring_size, shift=False):
        """
        Applies the Partial_Mapped Crossover.
        The crossover takes place at the same period of genes from both parents.
        There might exist identical car number, thus legitimate should be taken place
        """
        offspring = numpy.full(offspring_size, -1, dtype=int)
        for k in range(offspring_size[0]):
            # If the chromosome has only a single gene. In this case, this gene is copied from the second parent.
            if parents.shape[1] == 1:
                crossover_point1 = 0
            else:
                crossover_point1 = numpy.random.randint(low=0, high=numpy.ceil(parents.shape[1] / 2 + 1), size=1)[0]

            crossover_point2 = crossover_point1 + \
                               numpy.random.randint(low=1, high=numpy.ceil(parents.shape[1] / 2), size=1)[0]

            if self.crossover_probability is not None:
                probes = numpy.random.random(size=parents.shape[0])
                indices = numpy.where(probes <= self.crossover_probability)[0]

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(set(indices), 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k + 1) % parents.shape[0]
            # The genes from the beginning of the chromosome up to the first point are copied from the first parent.
            offspring[k, 0:crossover_point1] = parents[parent1_idx, 0:crossover_point1]
            # The genes from the second point up to the end of the chromosome are copied from the first parent.
            offspring[k, crossover_point2:] = parents[parent1_idx, crossover_point2:]
            # Legitimate gene fragments without same number
            pmx_pair = numpy.transpose([parents[parent1_idx, crossover_point1:crossover_point2],
                                        parents[parent2_idx, crossover_point1:crossover_point2]])

            gene = offspring[k, :]
            gene = gene.tolist()
            temp = gene.copy()
            legit = False
            while not legit:
                for crossover_value in pmx_pair:
                    try:
                        gene[gene.index(crossover_value[1])] = crossover_value[0]
                    except ValueError:
                        pass
                temp = gene.copy()
                # The genes between the 2 points are copied from the second parent.
                temp[crossover_point1:crossover_point2] = parents[parent2_idx, crossover_point1:crossover_point2]
                legit = self.check_list_validity(temp)

            if shift:
                shift_size = numpy.random.randint(low=0, high=numpy.ceil(parents.shape[1] / 2 + 1), size=1)[0]
                temp = self.shift_list(temp, shift_size)

            offspring[k, :] = temp
        self.check_matrix_validity(offspring)
        return offspring

    def order_crossover(self, parents, offspring_size):
        self.partial_mapped_crossover(parents, offspring_size, shift=True)

    def cycle_crossover(self, parents, offspring_size):
        """
        Applies the cycle Crossover.
        """
        offspring = numpy.zeros(offspring_size, dtype=int)
        for k in range(offspring_size[0]):
            if parents.shape[1] == 1:
                start_point = 0
            else:
                start_point = numpy.random.randint(low=0, high=numpy.ceil(parents.shape[1]), size=1)[0]

            if self.crossover_probability is not None:
                probes = numpy.random.random(size=parents.shape[0])
                indices = numpy.where(probes <= self.crossover_probability)[0]

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(set(indices), 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k + 1) % parents.shape[0]

            # create the cycle
            parent1 = parents[parent1_idx].tolist()
            parent2 = parents[parent2_idx].tolist()
            cycle = self.create_cycle(parent1, parent2, start_point, init_index=start_point, my_list=[start_point])
            for index in cycle:
                parent1[index] = parent2[index]
            offspring[k] = parent1
        self.check_matrix_validity(offspring)
        return offspring

    # This is the part for mutation
    def random_mutation(self, offspring):
        """
        Applies the random mutation which randomly changes a whole individual
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        if self.mutation_probability is not None:
            probes = numpy.random.random(size=offspring.shape[0])
            indices = numpy.where(probes <= self.mutation_probability)[0]

            for idx in indices:
                offspring[idx] = self.create_individual()
                if self.verbose:
                    print('Mutated at the {idx} solution'.format(idx=idx))
        return offspring

    def swap_mutation(self, offspring):
        """
        Applies the swap mutation which changes two genes randomly
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """
        offspring_length = offspring.shape[1]
        if self.mutation_probability is not None:
            for gene_idx, gene in enumerate(offspring):
                probes = numpy.random.random(size=offspring_length)
                indices = numpy.where(probes <= self.mutation_probability)[0]
                for idx in indices:
                    mutation_position = numpy.random.randint(low=1, high=offspring_length - 1, size=1)[0]
                    temp = gene[idx]
                    gene[idx] = gene[mutation_position % offspring_length]
                    gene[mutation_position] = temp
                    if self.verbose:
                        print('Mutated at the {gene_idx} solution, the {idx} and {mutation_position} gene'.format(
                            gene_idx=gene_idx, idx=idx, mutation_position=mutation_position))
        return offspring

    def reverse_mutation(self, offspring):
        """
        Applies the reverse mutation which reverses the individual
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        if self.mutation_probability is not None:
            probes = numpy.random.random(size=offspring.shape[0])
            indices = numpy.where(probes <= self.mutation_probability)[0]

            for idx in indices:
                offspring[idx] = numpy.flipud(offspring[idx])
                if self.verbose:
                    print('Mutated at the {idx} solution'.format(idx=idx))
        return offspring

    def best_solution(self):

        """
        Returns information about the best solution found by the genetic algorithm. Can only be called after completing at least 1 generation.
        If no generation is completed (at least 1), an exception is raised. Otherwise, the following is returned:
            -best_solution: Best solution in the current population.
            -best_solution_fitness: Fitness value of the best solution.
            -best_match_idx: Index of the best solution in the current population.
        """

        if self.generations_completed < 1:
            raise RuntimeError(
                "The best_solution() method can only be called after completing at least 1 generation but {generations_completed} is completed.".format(
                    generations_completed=self.generations_completed))

        #        if self.run_completed == False:
        #            raise ValueError("Warning calling the best_solution() method: \nThe run() method is not yet called and thus the GA did not evolve the solutions. Thus, the best solution is retireved from the initial random population without being evolved.\n")

        # Getting the best solution after finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        self.multi_processing = False
        fitness, err = self.cal_pop_fitness()
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = numpy.where(fitness == numpy.max(fitness))[0][0]

        best_solution = self.population[best_match_idx, :]
        best_fitness = 1 / fitness[best_match_idx]
        current_site_data, current_vehicle_state = self.create_t_table.run(best_solution)
        return best_solution, best_fitness, best_match_idx, err, current_site_data, current_vehicle_state, self.best_solutions_fitness
