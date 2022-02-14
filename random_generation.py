import random
import time


def random_testing(self):
    test_data_pool = []
    i = 0
    start_time = time.time()

    while (self.test_budget == 'sampling' and i <= self.sampling_budget) or (
            self.test_budget == 'time' and (time.time() - start_time) / 60 < self.time_budget):
        test_data = [random.uniform(self.boundary_lower_val, self.boundary_upper_val) for j in range(self.dimensions)]
        if self.evaluate_test_data(test_data):
            test_data_pool.append(test_data)
        # print(test_data)
        else:
            self.invalid_samples += 1
        i += 1

    self.test_data_generation_time = time.time() - start_time
    self.samples = len(test_data_pool)

    return test_data_pool


def evaluate_test_data(self, test_data):
    return sut.get_satisfactory_status(self.sut, test_data)


def get_satisfactory_status(testfunc, args):
    return testfunc(*args)
