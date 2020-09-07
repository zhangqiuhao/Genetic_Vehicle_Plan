from Genetic_Vehicle_Plan.vpga import *
from Genetic_Vehicle_Plan.input import *
import time
import numpy

# N = int, number of sites
# M = int, number of vehicles

# Vehicle data
vehicle_data = input_vehicle_data('data/vehicle_state.txt')  # Cubic able to be carried by vehicles
average_vehicle_cubic = numpy.average(
    numpy.asarray([int(x.cubic) for x in vehicle_data]))  # The number of vehicles according to its cubic

# Input site data
site_data = input_site_data('data/2020-08-31.csv', average_vehicle_cubic)
num_genes = numpy.asarray([int(x.n_deliver) for x in site_data])
cubic_total = sum(numpy.asarray([int(x.demand_cubic) for x in site_data]))
print('There are {num_car} vehicles available, the total planned cubic is {cubic_total}'
      .format(num_car=len(vehicle_data), cubic_total=cubic_total))

# Camp data
cubic_mixing_machines = numpy.asarray([3, 4.5])  # The cubic of the mixing machines
mixing_time = ['C20:30', 'C30:30', 'C40:30', 'C50:40']

ga_instance = GA(vehicle_data=vehicle_data,
                 site_data=site_data,
                 num_generations=10,
                 num_parents_mating=15,
                 num_genes=num_genes,
                 sol_per_pop=100,
                 parent_selection_type="rank",
                 crossover_type="cx",
                 crossover_probability=0.6,
                 mutation_type="swap",
                 mutation_probability=0.001,
                 verbose=False,
                 multi_processing=True)

print("Starting Genetic Algorithm")
start_time = time.time()
ga_instance.run()
print(time.time() - start_time)

solution, solution_fitness, solution_idx, err = ga_instance.best_solution()
positives = [y for x in err for y in x if y >= 0]
negatives = [y for x in err for y in x if y < 0]
print(solution)
print(solution_fitness)
print(solution_idx)
print(err)
print("The total delay on site: {delay}, total lost for camp: {lost}".format(delay=sum(positives), lost=sum(negatives)))

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_result()
