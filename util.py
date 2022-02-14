__version__ = '2'
__author__ = 'Akram Kalaee'

import ast
import datetime
import inspect
import os
import csv
from itertools import islice
import ntpath
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_branch_number(func):
    return inspect.getsource(func).count('evaluate_condition')


def get_args_number(func):
    return func.__code__.co_argcount


def get_args(func):
    return func.__code__.co_varnames


def cuboid_data2(o, size=(1, 1, 1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X


def plotCubeAt2(positions, sizes=None, colors=None, **kwargs):
    if not isinstance(colors, (list, np.ndarray)): colors = ["C0"] * len(positions)
    if not isinstance(sizes, (list, np.ndarray)): sizes = [(1, 1, 1)] * len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data2(p, size=s))
    return Poly3DCollection(np.concatenate(g), facecolors=np.repeat(colors, 6), alpha=0.7, linewidth=2, zorder=0,
                            **kwargs)


def save_domains(path, sub_domains, dimensions, sut):
    dt = datetime.datetime.now().strftime('_date_%Y-%m-%d_%H-%M-%S')
    # Create a directory and an empty csv file within to save mode csv log.
    dir_name = '{}'.format(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    ind = ', '.join(['dim{}_lower, dim{}_upper'.format(dim, dim) for dim in range(dimensions)])
    ind += '\n'
    for domain in sub_domains:
        # print(domain)
        ind += ', '.join(['{}, {}'.format(domain[dim][0], domain[dim][1]) for dim in range(dimensions)])
        ind += str('\n')
    with open(dir_name + '/domains_' + sut + str('.csv'), mode='w', encoding='utf8') as f:
        f.write(ind)


def save_test_suite(path, test_suite, parameters, sut):
    dt = datetime.datetime.now().strftime('_date_%Y-%m-%d_%H-%M-%S')
    # Create a directory and an empty csv file within to save mode csv log.
    dir_name = '{}'.format(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    ind = ','.join(['{}'.format(param) for param in parameters])
    ind += '\n'
    with open(dir_name + '/test_data_' + sut + str('.csv'), mode='w', encoding='utf8') as f:
        f.write(ind)
        for test_data in test_suite:
            ind = ', '.join(['{}'.format(test_data[dim]) for dim, param in enumerate(parameters)])
            ind += str('\n')
            f.write(ind)


def read_test_suite(path):
    # with open(path , mode='r') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #     fieldnames = csv_reader.fieldnames
    #     line_count = 0
    #     test_suite = []
    #     #csv_reader = csv_reader.pop(0)
    #     for row in csv_reader:
    #         current_list = []
    #         current_list = [row[param] for param in fieldnames]
    #         test_suite.append(current_list)
    test_suite = []
    with open(path, mode='r') as csv_file:
        for row in islice(csv_file, 1, None):
            values = list(row.split(','))
            for i, value in enumerate(values):
                nValue = ast.literal_eval(value.strip())
                values[i] = nValue
            test_suite.append(values)
    return test_suite


def report(message, info):
    print('=====================================')
    print('      {}!   {}      '.format(message, info))
    print('=====================================\n')
