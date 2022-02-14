__version__ = '5'
__author__ = 'Akram Kalaee'

from domain_generation import domain_generation
from config import clustering_config as clustering_config, genetic_algorithm_config as GA_config, \
    global_config as global_config, test_config

from staticfg import CFGBuilder
import PrimePath as pp
import graphviz as gv


def main():
    # create_cfg()

    # generate_prime_path()

    ga_solver = domain_generation(global_config, GA_config, clustering_config, test_config)

    ga_solver.random_testing()

    ga_solver.run()


def create_cfg():
    cfg = CFGBuilder().build_from_file('sut_1.py', 'SUT/sut_1.py')
    graph = cfg.build_visual('exampleCFG', 'pdf', True, True)
    nodes = graph[0]
    edges = graph[1]
    end_nodes = graph[2]

    node_temp = []
    node_temp_items = []
    edges.sort(key=lambda x: x[0])

    for node in nodes:
        flag = False
        for edge in edges:
            if node == edge[0]:
                node_temp_items.append(edge[1])
                flag = True
        if not flag:
            node_temp_items.append(-1)
        node_temp.append(node_temp_items)
        node_temp_items = []

    f = open("assets/testcase2", "w")
    f.write('# Nodes(must identified by int nums)\n')
    for node in nodes:
        f.write(str(node) + ' ')
    f.write('\n# Start Nodes\n')
    f.write(str(nodes[0]) + '\n')
    f.write('# End Nodes\n')
    for end_node in end_nodes:
        f.write(str(end_node) + ' ')
    f.write('\n# Edges(''-1'' means the node has no out-degree)\n')

    for item in node_temp:
        for i in item:
            f.write(str(i) + ' ')
        f.write('\n')

    f.close()

    print('Nodes: ', nodes)
    print('Edges: ', edges)
    print('End Nodes: ', end_nodes)


def generate_prime_path():
    prime_reader = pp.readGraphFromFile('assets/testcase2')
    prime_paths = pp.findPrimePaths(prime_reader)
    for path in prime_paths:
        print('Prime Path: ', str(path))


if __name__ == '__main__':
    main()
