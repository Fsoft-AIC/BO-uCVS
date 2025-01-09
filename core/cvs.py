from abc import ABC, abstractmethod
import numpy as np
import torch

from core.utils import powerset


def get_eps_schedule(
    id, costs, control_sets, random_sets, variances, lengthscales, budget
):
    if id == 0:
        return None  # placeholder
    else:
        raise NotImplementedError

class NormCost(ABC):
    def __init__(self, mean, var=0.01):
        super().__init__()
        self.mean = mean
        self.var =  var
        self.a_min = 0.001
        self.a_max = 2

    def sample(self, i):
        return np.clip(self.mean[i] + self.var * np.random.randn(),a_max=self.a_max,a_min=self.a_min)

class EpsilonSchedule(ABC):
    def __init__(self):
        super().__init__()
        self.last_eps = None
        self.t = 0

    @abstractmethod
    def next(self):
        pass

    def update(self):
        self.t += 1


class LinearSchedule(EpsilonSchedule):
    def __init__(self, start_eps, cutoff_iter):
        super().__init__()
        self.start_eps = start_eps
        self.cutoff_iter = cutoff_iter

    def next(self):
        eps = self.start_eps * np.maximum(1 - self.t / self.cutoff_iter, 0.0)
        self.last_eps = eps
        return eps


def get_control_sets(dims, control_id):
    if dims == 3:
        if control_id == 0:
            inter = powerset(np.arange(3))
        else:
            raise NotImplementedError
    elif dims == 5:
        if control_id == 0:
            inter = [
                (0, 1),
                (2, 3),
                (3, 4),
                (0, 1, 2),
                (1, 2, 3),
                (2, 3, 4),
                (0, 1, 2, 3, 4),
            ]
        elif control_id == 1:
            inter = [(3, 4), (1, 4), (0, 3), (1, 2), (2, 4), (0, 1), (2, 3)]
        else:
            raise NotImplementedError
    elif dims == 6:
        if control_id == 0:
            inter = [
                (0, 1),
                (2, 3),
                (4, 5),
                (0, 1, 2, 3),
                (1, 2, 3, 4),
                (2, 3, 4, 5),
                (0, 1, 2, 3, 4, 5),
            ]
        elif control_id == 1:
            inter = [
                (0, 1),
                (2, 3),
                (4, 5),
                (0, 1, 2, 3),
                (1, 2, 3, 4),
                (2, 3, 4, 5),
                (0, 1, 2, 4, 5),
            ]
        else:
            raise NotImplementedError
    elif dims == 8:
        if control_id == 0:
            inter = [
                (0, 1, 2),
                (3, 4, 5),
                (5, 6, 7),
                (0, 1, 2, 3, 4),
                (2, 3, 4, 5, 6),
                (3, 4, 5, 6, 7),
                (0, 1, 2, 3, 4, 5, 6, 7),
            ]
        else:
            raise NotImplementedError
    elif dims == 12:
        if control_id == 0:
            inter = [
                (0, 1, 2),
                (3, 4, 5),
                (6, 7, 8),
                (9, 10, 11),
                (0, 1, 2, 3, 4, 5),
                (6, 7, 8, 9, 10, 11),
                (0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11),
            ]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    control_sets = [np.array(tup) for tup in inter]
    return control_sets


def get_costs(cost_id):
    if cost_id == 0:
       return np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 1.0])
    elif cost_id == 1:
        return np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 1.0])
    elif cost_id == 2:
        return np.array([0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 1.0])
    if cost_id == 3:
        costs = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 1.0])
    elif cost_id == 4:
        costs = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 1.0])
    elif cost_id == 5:
        costs = np.array([0.3, 0.3, 0.3, 0.8, 0.8, 0.8, 1.0])
    else:
        raise NotImplementedError

    return NormCost(mean=costs)


def get_random_sets(dims, control_sets):
    random_sets = []
    for control_set in control_sets:
        random_sets.append(np.setdiff1d(np.arange(dims), control_set))

    return random_sets


def get_control_sets_and_costs(dims, control_sets_id, costs_id):
    control_sets = get_control_sets(dims=dims, control_id=control_sets_id)

    random_sets = get_random_sets(dims=dims, control_sets=control_sets)

    for i in range(len(control_sets)):
        assert np.allclose(
            np.sort(np.concatenate([control_sets[i], random_sets[i]])), np.arange(dims)
        )

    costs = get_costs(cost_id=costs_id)

    return control_sets, random_sets, costs

