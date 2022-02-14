__version__ = '5'
__author__ = 'Akram Kalaee'

import coverage
from datetime import datetime
import os
import time
import csv
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import util as util
import clustering
import scipy.spatial.distance as dist
import coverage_analysis
import textdistance
import decision_tree


def get_satisfactory_status(testfunc, args):
    # return testfunc(*args)
    x = testfunc(*args)
    if x == 'answer':
        return True
    else:
        return False


class domain_generation(object):
    def __init__(self, global_config, GA_config, clustering_config, test_config):
        self.test_budget_type = test_config['test_budget_type']
        self.test_sampling_budget = test_config['test_sampling_budget']
        self.test_time_budget = test_config['test_time_budget']

        self.boundary_lower_val = global_config['boundary_lower_val']
        self.boundary_upper_val = global_config['boundary_upper_val']
        self.riemann_delta = global_config['riemann_delta']
        self.algorithm = global_config['algorithm']
        self.test_budget = global_config['test_budget']
        self.time_budget = global_config['time_budget']
        self.sampling_budget = global_config['sampling_budget']

        self.eps = clustering_config['eps']
        self.min_samples = clustering_config['min_samples']

        try:
            self.sut = getattr(global_config['SUT'], "main")
            # print('sut : ', self.sut)
        except:
            self.sut = None

        self.dimensions = util.get_args_number(self.sut)
        self.parameters = util.get_args(self.sut)
        self.eps = clustering_config['eps']
        self.min_samples = clustering_config['min_samples']
        self.output_file_name = os.path.basename(global_config['SUT'].__file__).split('.')[0]
        self.target_path = global_config['target_path']

        self.sub_domain_number = 0
        self.invalid_samples = 0
        self.execution_time = 0
        self.fitness_call_number = 0
        self.pc_len = 3
        self.test_data_generation_time = 0
        self.clustering_time = 0
        self.preparing_domains_time = 0

        self.dir_name = ''
        self.samples = 0
        self.test_data = []

        if self.algorithm != 'RT':
            self.population_size = GA_config['population']
            self.mutation_rate = GA_config['mutation_rate']
            self.crossover_rate = GA_config['crossover_rate']
            self.iteration = GA_config['iteration']
            self.tournament_size = GA_config['tournament_size']

    def random_testing(self):
        self.create_log()

        test_data_pool = []
        i = 0
        start_time = time.time()

        while (self.test_budget_type == 'sampling' and len(test_data_pool) < self.test_sampling_budget) or (
                self.test_budget_type == 'time' and (time.time() - start_time) / 60 < self.test_time_budget):
            test_data = [random.uniform(self.boundary_lower_val, self.boundary_upper_val) for j in
                         range(self.dimensions)]
            if self.evaluate_test_data(test_data):
                test_data_pool.append(test_data)
                # print(test_data)
            else:
                self.invalid_samples += 1
                # print('Not right')
            i += 1

        self.test_data_generation_time = time.time() - start_time
        self.samples = len(test_data_pool)
        log_address = self.get_test_fitness(self.sut, test_data_pool[0])
        self.target_path = log_address

        print("time: ", self.test_data_generation_time)
        print("Samples: ", self.samples)

        return test_data_pool

    def evaluate_test_data(self, test_data):
        return get_satisfactory_status(self.sut, test_data)

    def create_initial_population(self):
        return [[random.uniform(self.boundary_lower_val, self.boundary_upper_val) for j in range(self.dimensions)] for i
                in range(self.population_size)]

    def get_test_fitness(self, testfunc, args):
        date_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        cov = coverage.Coverage(
            data_file='{}/.coverage'.format(self.dir_name), include="*SUT*",
            branch=True, data_suffix='{}'.format(date_time)
        )
        cov.start()

        testfunc(*args)

        cov.stop()
        cov.save()

        cov_path = r'{}/.coverage.{}'.format(self.dir_name, date_time)
        traversed_path = coverage_analysis.coverage_report(cov_path)

        return traversed_path

    def get_fitness(self, testfunc, args):
        date_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        cov = coverage.Coverage(
            data_file='{}/.coverage'.format(self.dir_name), include="*SUT*",
            branch=True, data_suffix='{}'.format(date_time)
        )
        cov.start()

        testfunc(*args)

        cov.stop()
        cov.save()

        cov_path = r'{}/.coverage.{}'.format(self.dir_name, date_time)
        traversed_path = coverage_analysis.coverage_report(cov_path)
        # xx

        path1, path2 = coverage_analysis.convert_coverage_report_to_vector(self.target_path, traversed_path)

        print("path1: {}, path2: {}".format(path1, path2))
        # Change
        # jaccard_distance = dist.jaccard(path1, path2)
        cosine_distance = dist.cosine(path1, path2)
        # hamming = textdistance.damerau_levenshtein
        # [1,0,0,0,1,1,0]
        # [1,0,1,1,0,1,0]
        # test = dist.
        fitness = 1
        leven_distance = textdistance.levenshtein.normalized_distance(path1, path2)

        if leven_distance == 0:
            fitness = 0
        else:
            fitness = leven_distance

        # if cosine_distance == 0 and jaccard_distance == 0:
        #     fitness = 0
        # elif jaccard_distance <= 0.3:
        #     fitness = cosine_distance

        # print("test data: {}, jaccard: {}, cosine: {}  --> {}".format(args, jaccard_distance, cosine_distance, fitness))
        print(
            "test data: {}, levenshtein_distance: {}, cosine: {}  --> {}".format(args, leven_distance, cosine_distance,
                                                                                 fitness))
        # print("test data: {}, levenshtein_distance {}  --> {}".format(args, leven_distance, fitness))

        return fitness

    def evaluate_population(self, population):
        fitness = [self.get_fitness(self.sut, individual) for individual in population]
        self.fitness_call_number += len(population)
        return list(zip(population, fitness))

    def selection(self, evaluated_population, tournament_size):
        competition = random.sample(evaluated_population, tournament_size)
        winner = min(competition, key=lambda item: item[1])
        return winner[:]

    def crossover(self, parent1, parent2):
        pos = random.randint(1, len(parent1))
        offspring1 = parent1[:pos] + parent2[pos:]
        offspring2 = parent2[:pos] + parent1[pos:]
        return offspring1, offspring2

    def mutate(self, chromosome):
        mutated = chromosome[:]
        for pos in range(len(mutated)):
            if random.random() < self.mutation_rate:
                while True:
                    new_value = random.uniform(self.boundary_lower_val, self.boundary_upper_val)
                    if mutated[pos] != new_value:
                        break
                mutated[pos] = new_value

        return mutated

    def genetic_algorithm(self):
        test_data_pool = []
        generation = 0
        print('Initialize population...')
        population = self.create_initial_population()
        print('pop: ', population)
        print("Initialized DOMAIN_GENERATION SUCCESS!\n")
        evaluated_population = self.evaluate_population(population)
        print('evaluated population: ', evaluated_population)

        for i in range(6):
            feature = []
            features = []
            labels = []
            fitness = []
            for individual in evaluated_population:
                for item in individual[0]:
                    feature.append(item)
                features.append(feature)
                fitness.append(individual[1])

                # Set Label
                if individual[1] == 0:
                    labels.append(0)
                    if individual[0] not in test_data_pool:
                        test_data_pool.append(individual[0])
                else:
                    labels.append(1)

                feature = []

            # svm or dt
            # csv_address = self.dir_name + '/test_data_' + self.output_file_name + '.csv'
            print('features: ', features)
            # final_labels = np.ndarray(labels)

            evaluated_population = decision_tree.make_tree(features, labels)
            # min distance
            print('eval_original: ', evaluated_population)
            # update fitness

            # **************

            start_time = time.time()

            while (self.test_budget == 'time' and (time.time() - start_time) / 60 < self.time_budget) or (
                    self.test_budget != 'time' and generation < self.iteration):
                print("\n*** Iteration {}  ====> ".format(generation))
                new_population = []
                while len(new_population) < len(population):
                    # Selection
                    parent1 = self.selection(evaluated_population, self.tournament_size)
                    parent2 = self.selection(evaluated_population, self.tournament_size)
                    print('parent1: ', parent1)
                    print('parent2: ', parent2)
                    # Crossover
                    if random.random() < self.crossover_rate:
                        offspring1, offspring2 = self.crossover(parent1[0], parent2[0])
                    else:
                        offspring1 = parent1[0]
                        offspring2 = parent2[0]

                    # Mutation
                    offspring1 = self.mutate(offspring1)
                    offspring2 = self.mutate(offspring2)

                    new_population.append(offspring1)
                    new_population.append(offspring2)

                generation += 1
                population = new_population

                print('Evaluation of generation...')
                evaluated_population = self.evaluate_population(population)

            feature = []
            features = []
            labels = []
            fitness = []

            for individual in evaluated_population:
                for item in individual[0]:
                    feature.append(item)
                features.append(feature)
                fitness.append(individual[1])

                # Set Label
                if individual[1] == 0:
                    labels.append(0)
                    if individual[0] not in test_data_pool:
                        test_data_pool.append(individual[0])
                else:
                    labels.append(1)

                feature = []

            # svm or dt
            # csv_address = self.dir_name + '/test_data_' + self.output_file_name + '.csv'
            print('features: ', features)
            # final_labels = np.ndarray(labels)

            evaluated_population = decision_tree.make_tree(features, labels)
            # svm or dt

            # min distance

            # update fitness

            # **************

            print("Done SUCCESS!")

            self.test_data_generation_time = time.time() - start_time
            print("#test_data: {}".format(len(test_data_pool)))
            self.test_data = test_data_pool

            # arr = np.concatenate((features, labels), axis=1)
            # print('arr: ', arr)
            # print(arr)

        return test_data_pool, evaluated_population

    def prepare_sub_domains(self, test_suite):

        dims_low_bound = []
        dims_high_bound = []
        dims_position = []
        sub_domains = []

        for i in range(self.dimensions):
            dims_low_bound.append(min([point[i] for point in test_suite]))
            dims_high_bound.append(max([point[i] for point in test_suite]))
            dims_position.append(dims_low_bound[i])

        delta = self.riemann_delta
        sub_domain_number = math.ceil(math.fabs(dims_high_bound[0] - dims_low_bound[0]) / delta)

        dims_size = [0 for i in range(self.dimensions)]  # sub_domain_number
        dims_size[0] = delta

        for i in range(sub_domain_number):
            if i == sub_domain_number and dims_position[0] + delta > dims_high_bound[0]:
                delta = math.fabs(dims_high_bound[0] - dims_position[0])
            ######################################
            sub_df = [coord for coord in test_suite if dims_position[0] <= coord[0] <= dims_position[0] + delta]
            sub_df = np.matrix(sub_df)
            n_clusters_, partitions = clustering.clustering_test_data_hdbscan(sub_df)

            ######################################## sub-domains
            for _points in partitions:
                sub_domain = []

                for i in range(self.dimensions):
                    min_i = min([point[i] for point in _points])
                    max_i = max([point[i] for point in _points])

                    dims_size[i] = math.fabs(max_i - min_i)
                    dims_position[i] = min_i
                    lower_bound = dims_position[i]
                    upper_bound = dims_position[i] + dims_size[i]
                    sub_domain.append([lower_bound, upper_bound])

                sub_domains.append(sub_domain)

            dims_position[0] = dims_position[0] + delta

        util.save_domains(self.dir_name, sub_domains, self.dimensions, self.output_file_name)

        return sub_domains

    def pltDomains(self, data_points, sub_domains, title):
        x_set, y_set, z_set = [], [], []

        for point in data_points:
            x_set.append(point[0])
            y_set.append(point[1])

        fig1 = plt.figure()
        # plot 2-d or 3-d data points
        if self.dimensions == 2:
            ax1 = fig1.add_subplot(111)
            ax1.set_title(title)
            ax1.scatter(x_set, y_set, s=0.5, zorder=10)
            ax1.set(xlabel='X', ylabel='Y')
        elif self.dimensions == 3:
            for point in data_points:
                z_set.append(point[2])

            ax1 = fig1.gca(projection='3d')
            ax1.set_title(title)
            ax1.scatter(x_set, y_set, z_set, s=0.5, zorder=10)
            ax1.set(xlabel='X', ylabel='Y', zlabel='Z')

        ######################################## sub-domains

        for idx, sub_domain in enumerate(sub_domains):
            # print(sub_domain)
            x = sub_domain[0][0]
            y = sub_domain[1][0]
            width = sub_domain[0][1] - x
            height = sub_domain[1][1] - y

            if self.dimensions == 2:
                ax1.add_patch(patches.Rectangle(xy=(x, y), width=width, height=height, edgecolor='r', facecolor='none',
                                                linewidth=2))
            elif self.dimensions == 3:
                z = sub_domain[2][0]
                length = sub_domain[2][1] - sub_domain[2][0]
                positions = [(x, y, z)]
                sizes = [(width, height, length)]
                colors = ["hotpink"]
                pc = util.plotCubeAt2(positions, sizes, colors=colors, edgecolor="k")
                ax1.add_collection3d(pc)

        if not os.path.exists('images'):
            os.makedirs('images')
        plt.savefig('{}/domains_{}.png'.format(self.dir_name, self.output_file_name))
        util.report('Plot saved', '')

        plt.show(block=True)

    def pltPoints(self, data_points, title, block=False):
        x_set, y_set, z_set = [], [], []

        for point in data_points:
            x_set.append(point[0])
            y_set.append(point[1])

        fig1 = plt.figure()
        # plot 2-d or 3-d data points
        if self.dimensions == 2:
            ax1 = fig1.add_subplot(111)
            ax1.set_title(title)
            ax1.scatter(x_set, y_set, s=1, zorder=10)
            ax1.set(xlabel='X', ylabel='Y')
        elif self.dimensions == 3:
            for point in data_points:
                z_set.append(point[2])

            ax1 = fig1.gca(projection='3d')
            ax1.set_title(title)
            ax1.scatter(x_set, y_set, z_set, s=1, zorder=10)
            ax1.set(xlabel='X', ylabel='Y', zlabel='Z')

        ######################################## sub-domains

        plt.savefig('{}/test-data-{}.png'.format(self.dir_name, self.output_file_name))
        util.report('Plot was saved', '')

        plt.show(block=block)

    def get_dir_name(self):
        dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_name = '{}'.format('log')

    def create_log(self):
        dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_name = '{}'.format('log')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        dir_name = 'log/{}'.format(dt)
        os.makedirs(dir_name)
        self.dir_name = dir_name

    def run(self):
        # self.create_log()

        print('Generating test data...')
        test_suite, population = self.genetic_algorithm()

        print('sut1: ', self.sut)
        if len(test_suite) == 0:
            print('Test data not found! Retry again or modify the configurations.')
        else:
            # util.save_test_suite(self.dir_name, test_suite, self.parameters, self.output_file_name)
            x = []
            for item in self.parameters:
                x.append(item)
            x.append('fitness')
            x.append('label')
            print('Pop: ', population)
            # util.save_test_suite(self.dir_name, population, x, self.output_file_name)
            util.report('Test data was generated and saved', "%.2f (sec)" % (self.test_data_generation_time))

            self.pltPoints(test_suite, 'Plotting test data', False)

            # csv_address = self.dir_name + '/test_data_' + self.output_file_name + '.csv'
            # decision_tree.make_tree(csv_address)

            print('Clustering test data...')
            test_suite.sort()

            start_time = time.time()
            print('Preparing sub_domains...')
            start_time = time.time()
            sub_domains = self.prepare_sub_domains(test_suite)
            self.preparing_domains_time = time.time() - start_time
            util.report('Sub_domains was saved', "%.2f (sec)" % (self.preparing_domains_time))
            self.sub_domain_number = len(sub_domains)

            print('Plotting sub_domains...')
            plot_title = 'riemann_delta= {} \n clustering config: eps= {}, min_samples={} \n' \
                         ' genetic config: population= {}, iteration_no= {}'.format(
                self.riemann_delta, self.eps, self.min_samples, self.population_size, self.iteration)

            self.pltDomains(test_suite, sub_domains, plot_title)
