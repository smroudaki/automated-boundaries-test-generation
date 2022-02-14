__version__ = '5'
__author__ = 'Akram Kalaee'

from SUT import sut_1, sut_2

test_config = {
    'test_budget_type': 'sampling',  # [sampling, time]
    'test_sampling_budget': 1,
    'test_time_budget': 0.1,  # minutes
}

global_config = {
    'SUT': sut_1,
    'target_path': r"assets\target3.txt",
    'boundary_lower_val': -10,
    'boundary_upper_val': 300,
    'riemann_delta': 0.5,
    'test_budget': 'time',  # [sampling, time]
    'sampling_budget': 100,  # 1000000,
    'time_budget': 0.1,  # minutes
    'algorithm': 'GA',
}

clustering_config = {
    'clustering_algorithm': 'hdbscan',
    'min_samples': 5,
    'eps': 0.001,
}

genetic_algorithm_config = {
    'crossover_rate': 0.7,
    'mutation_rate': 0.5,
    'tournament_size': 10,
    'population': 500,
    'iteration': 50,
}
