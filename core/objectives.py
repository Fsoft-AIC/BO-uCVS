from botorch import test_functions
from botorch.models import SingleTaskGP
import pickle
import torch

from core.gp import sample_gp_prior
from core.utils import maximize_fn
from data.plant.plant_funcs import create_leaf_max_area_func
import os


def get_objective(objective_name, noise_std, is_input_transform, kernel, dims):
    """
    Get objective function, bounds and its max function value (for regret).
    :param objective_name: str.
    :param noise_std: float.
    :param is_input_transform: bool. Set to True to transform the domain to the unit hypercube.
    :return: objective function Callable that takes in arrays of shape (..., d) and returns an array of shape (..., 1),
    bounds with shape (2, d), optimal function value.
    """
    data_dir = "data/"
    if objective_name == "gpsample":
        bounds = torch.stack([torch.zeros(dims), torch.ones(dims)])

        obj_func = sample_gp_prior(kernel=kernel, bounds=bounds, num_points=1000)

        _, opt_val = maximize_fn(
            f=obj_func,
            bounds=bounds,
            mode="L-BFGS-B",
            n_warmup=10000,
        )
    elif objective_name == "extendackley":
        neg_obj = test_functions.Ackley(dim=6, negate=True)
        bounds = torch.cat([neg_obj.bounds.to(dtype=torch.double)] * 2,1)
        unsqueezed_obj = lambda x: neg_obj(x[:,:6]).unsqueeze(-1)
        if is_input_transform:
            obj_func = input_transform_wrapper(obj_func=unsqueezed_obj, bounds=bounds)
            bounds = torch.stack(
                [torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])]
            )
        else:
            obj_func = unsqueezed_obj
        opt_val = neg_obj.optimal_value
    elif objective_name == "extendlevy":
        neg_obj = test_functions.Levy(dim=6, negate=True)
        bounds = torch.cat([neg_obj.bounds.to(dtype=torch.double)] * 2,1)
        unsqueezed_obj = lambda x: neg_obj(x[:,:6]).unsqueeze(-1)
        if is_input_transform:
            obj_func = input_transform_wrapper(obj_func=unsqueezed_obj, bounds=bounds)
            bounds = torch.stack(
                [torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])]
            )
        else:
            obj_func = unsqueezed_obj
        opt_val = neg_obj.optimal_value
    elif objective_name == "hartmann":
        neg_obj = test_functions.Hartmann(dim=3, negate=True)
        bounds = neg_obj.bounds.to(dtype=torch.double)
        unsqueezed_obj = lambda x: neg_obj(x).unsqueeze(-1)
        if is_input_transform:
            obj_func = input_transform_wrapper(obj_func=unsqueezed_obj, bounds=bounds)
            bounds = torch.stack(
                [torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])]
            )
        else:
            obj_func = unsqueezed_obj
        opt_val = neg_obj.optimal_value
    elif objective_name == "Hartmann6":
        neg_obj = test_functions.Hartmann(dim=6, negate=True)
        bounds = torch.cat([neg_obj.bounds.to(dtype=torch.double)] * 2,1)
        unsqueezed_obj = lambda x: neg_obj(x[:,:6]).unsqueeze(-1)
        if is_input_transform:
            obj_func = input_transform_wrapper(obj_func=unsqueezed_obj, bounds=bounds)
            bounds = torch.stack(
                [torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])]
            )
        else:
            obj_func = unsqueezed_obj
        opt_val = neg_obj.optimal_value
    elif objective_name == "extendhartmann":
        neg_obj = test_functions.Hartmann(dim=3, negate=True)
        bounds = torch.cat([neg_obj.bounds.to(dtype=torch.double)] * 2,1)
        unsqueezed_obj = lambda x: neg_obj(x[:,:3]).unsqueeze(-1)
        if is_input_transform:
            obj_func = input_transform_wrapper(obj_func=unsqueezed_obj, bounds=bounds)
            bounds = torch.stack(
                [torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])]
            )
        else:
            obj_func = unsqueezed_obj
        opt_val = neg_obj.optimal_value
    elif objective_name == "maxhartmann":
        neg_obj = test_functions.Hartmann(dim=3, negate=True)
        bounds = torch.cat([neg_obj.bounds.to(dtype=torch.double)] * 2,1)
        unsqueezed_obj = lambda x: torch.max(torch.cat([neg_obj(x[:,:3]).unsqueeze(-1), neg_obj(x[:,3:6]).unsqueeze(-1)],1),dim = 1,keepdim = True )[0]
        if is_input_transform:
            obj_func = input_transform_wrapper(obj_func=unsqueezed_obj, bounds=bounds)
            bounds = torch.stack(
                [torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])]
            )
        else:
            obj_func = unsqueezed_obj

        opt_val = neg_obj.optimal_value
    elif objective_name == "triplebranin":
        neg_obj = test_functions.Branin(negate=True)
        bounds = torch.cat([neg_obj.bounds.to(dtype=torch.double)]*3,1)
        unsqueezed_obj = lambda x: -((-(neg_obj(x[:,:2]).unsqueeze(-1) - neg_obj.optimal_value) \
            * (neg_obj(x[:,2:4]).unsqueeze(-1) - neg_obj.optimal_value) * \
            (neg_obj(x[:,4:6]).unsqueeze(-1) - neg_obj.optimal_value)) ** (1./3) - 44.81 - neg_obj.optimal_value) / 51.95
        if is_input_transform:
            obj_func = input_transform_wrapper(obj_func=unsqueezed_obj, bounds=bounds)
            bounds = torch.stack(
                [torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])]
            )
        else:
            obj_func = unsqueezed_obj

        opt_val = (44.81 + neg_obj.optimal_value) / 51.95

    elif objective_name == "plant":
        bounds = torch.tensor(
            [[0, 7.7], [0, 3.5], [0, 10.4], [8.9, 11.3], [2.5, 6.5]], dtype=torch.double
        ).T
        leafarea_meanvar_func = create_leaf_max_area_func(standardize=True)
        obj_func = lambda x: torch.tensor(
            leafarea_meanvar_func(x.numpy())[0], dtype=torch.double
        )
        if is_input_transform:
            obj_func = input_transform_wrapper(obj_func=obj_func, bounds=bounds)
            bounds = torch.stack(
                [torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])]
            )
        if os.path.isfile(data_dir + "plant/plant_opt_val.pickle"):
            opt_val =pickle.load(
                open(data_dir + "plant/plant_opt_val.pickle", "rb")
            )
        else:
            _, opt_val = maximize_fn(f=obj_func, n_warmup=10000, bounds=bounds)
            with open(data_dir + "plant/plant_opt_val.pickle", 'wb') as file:
                pickle.dump(opt_val,file)
    elif objective_name == "airfoil":
        bounds = torch.stack([torch.zeros(dims), torch.ones(dims)])

        X, y, state_dict = pickle.load(
            open(data_dir + "airfoil/airfoil_X_Y_statedict.p", "rb")
        )
        model = SingleTaskGP(train_X=X, train_Y=y)
        model.load_state_dict(state_dict)

        obj_func = lambda x: model.posterior(x).mean

        opt_val = None
    else:
        raise Exception("Incorrect obj_name passed to get_objective")

    noisy_obj_func = noisy_wrapper(obj_func=obj_func, noise_std=noise_std)
    return obj_func, noisy_obj_func, opt_val, bounds


def input_transform_wrapper(obj_func, bounds):
    """
    Wrapper around an existing objective function. Changes the bounds of the objective function to be the d-dim
    unit hypercube [0, 1]^d.
    :param obj_func: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    :param bounds: array of shape (2, d).
    :return: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    """
    return lambda x: obj_func(input_transform(x, bounds))


def input_transform(x, bounds):
    return x * (bounds[1] - bounds[0]) + bounds[0]


def noisy_wrapper(obj_func, noise_std):
    """
    Wrapper around an existing objective function. Turns a noiseless objective function into a noisy one.
    :param obj_func: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    :param noise_std: float.
    :return: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    """
    return lambda x: obj_func(x) 
