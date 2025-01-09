from filelock import FileLock
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints.constraints import Interval
from gpytorch.kernels import ScaleKernel, RBFKernel
import matplotlib
import numpy as np
from pathlib import Path
import pickle
import torch
import os

from core.acquisitions import get_acquisition
from core.dists import get_dists_and_samples, get_marginal_var
from core.objectives import get_objective
from core.optimization import bo_loop, modified_bo_loop
from core.cvs import get_control_sets_and_costs, get_eps_schedule
from core.regret import get_regret, plot_regret
from core.utils import log, uniform_samples


def main(
    obj_name,
    acq_name,
    eps_schedule_id,
    costs_id,
    var_id,
    budget,
    seed,
    control_sets_id,
    dims,
    noise_std,
    init_lengthscale,
    n_init_points,
    explore_start_budget = None,
):
    args = dict(locals().items())
    log(f"Running with parameters {args}")

    log(f"======== NEW RUN ========")
    log(
        (
            f"{obj_name}, {acq_name}, eps_sched:{eps_schedule_id}, cost_id:{costs_id}"
            f", var_id:{var_id}, C:{budget}, seed:{seed}"
        )
    )
    torch.manual_seed(seed)

    # Directory for saving results
    base_dir = "results/" + obj_name + "/"
    pickles_save_dir = base_dir + "pickles/"
    figures_save_dir = base_dir + "figures/"
    inter_save_dir = base_dir + "inter/"
    Path(pickles_save_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_save_dir).mkdir(parents=True, exist_ok=True)
    Path(inter_save_dir).mkdir(parents=True, exist_ok=True)
    filename = (
        f"{obj_name}_{acq_name}_es{eps_schedule_id}_c{costs_id}"
        f"_var{var_id}_C{budget}_seed{seed}"
    )
    filename = filename.replace(".", ",")
    # if os.path.isfile(pickles_save_dir + f"{filename}.p"):
    #     return

    if explore_start_budget is None:
        explore_start_budget = budget * 0.6
    # Objective function
    if obj_name == "gpsample":  # If sampling from GP, we need to define kernel first
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dims))
        kernel.outputscale = 1.0
        kernel.base_kernel.lengthscale = init_lengthscale
    else:
        kernel = None

    obj_func, noisy_obj_func, opt_val_det, bounds = get_objective(
        objective_name=obj_name,
        noise_std=noise_std,
        is_input_transform=True,
        kernel=kernel,
        dims=dims,
    )
    log(f"opt_val: {opt_val_det}")

    # Initialize state
    log("Starting new run from iter 0")
    start_iter = 0
    state_dict = None
    # Initial data
    init_X = uniform_samples(bounds=bounds, n_samples=n_init_points)
    with torch.no_grad():
        init_y = noisy_obj_func(init_X)

    # GP parameters
    if obj_name != "gpsample":
        dims = bounds.shape[-1]
        lengthscale_constraint = Interval(1e-2, 1e+4)
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dims, lengthscale_constraint = lengthscale_constraint))
        kernel.outputscale = 1.0
        kernel.base_kernel.lengthscale = init_lengthscale

    likelihood = GaussianLikelihood()
    likelihood.noise = noise_std**2

    # Control/random sets and costs
    control_sets, random_sets, costs = get_control_sets_and_costs(
        dims=dims, control_sets_id=control_sets_id, costs_id=costs_id
    )
    marginal_var = get_marginal_var(var_id=var_id)
    all_dists, all_dists_samples = get_dists_and_samples(
        dims=dims, variance=marginal_var
    )
    variances = marginal_var * np.ones(dims, dtype=np.double)
    lengthscales = init_lengthscale * np.ones(dims, dtype=np.double)
    eps_schedule = get_eps_schedule(
        id=eps_schedule_id,
        costs=costs,
        control_sets=control_sets,
        random_sets=random_sets,
        variances=variances,
        lengthscales=lengthscales,
        budget=budget,
    )

    acquisition = get_acquisition(
        acq_name=acq_name, eps_schedule_id=eps_schedule_id, costs=costs
    )
    # Optimization loop
    (
        final_X,
        final_y,
        control_set_idxs,
        control_queries,
        T,
        train_costs,
    ) = modified_bo_loop(
        train_X= init_X,
        train_y= init_y,
        likelihood = likelihood,
        kernel = kernel,
        noisy_obj_func = noisy_obj_func,
        explore_start_budget = explore_start_budget,
        budget = budget,
        acquisition= None,
        bounds=bounds,
        all_dists=all_dists,
        control_sets=control_sets,
        random_sets=random_sets,
        all_dists_samples=all_dists_samples,
        costs=costs,
        eps_schedule=eps_schedule,
        file_path = pickles_save_dir + "raw_" + f"{filename}.pkl",
    )
    print("final y:")
    print(torch.squeeze(final_y))
    print("train_cost:")
    print(train_costs)
    print("control_set_idxs:")
    print(control_set_idxs)
    print("count of each control set played:")
    # Regret
    log("Calculating regret")
    with open(pickles_save_dir + "raw_" + f"{filename}.pkl", 'wb') as file:
        pickle.dump((final_X,final_y,control_set_idxs,control_queries,T,train_costs),file)
    simple_regret, cumu_regret, cs_cumu_regret, cost_per_iter = get_regret(
        control_set_idxs=control_set_idxs,
        control_queries=control_queries,
        obj_func=obj_func,
        control_sets=control_sets,
        random_sets=random_sets,
        all_dists_samples=all_dists_samples,
        bounds=bounds,
        costs=train_costs,
        opt_val_det = opt_val_det,
    )

    plot_regret(
        regret=cumu_regret,
        cost_per_iter=cost_per_iter,
        x_axis="T",
        num_iters=T,
        save=True,
        save_dir=figures_save_dir,
        filename=filename + "_T",
    )

    plot_regret(
        regret=cs_cumu_regret,
        cost_per_iter=cost_per_iter,
        x_axis="C",
        num_iters=T,
        save=True,
        save_dir=figures_save_dir,
        filename=filename + "_C",
    )

    plot_regret(
        regret=simple_regret,
        cost_per_iter=cost_per_iter,
        x_axis="C",
        num_iters=T,
        save=True,
        save_dir=figures_save_dir,
        filename=filename + "_Csimple",
    )

    # Save results
    pickle.dump(
        (
            final_X,
            final_y,
            control_set_idxs,
            control_queries,
            all_dists_samples,
            simple_regret,
            cumu_regret,
            cs_cumu_regret,
            cost_per_iter,
            T,
            args,
        ),
        open(pickles_save_dir + f"{filename}.p", "wb"),
    )

    print("cumu_regret:")
    print(cumu_regret)
    print(np.histogram(control_set_idxs, bins=np.arange(len(control_sets) + 1))[0])

    log(f"Completed run with parameters {args}")

seeds = [5]
obj_names = ["plant"]
costs_ids = [5]
var_ids = [4]
## c=4, var=3, seed =4, obj_name = airfoil
for seed in seeds:
    for obj_name in obj_names:
        for costs_id in costs_ids:
            for var_id in var_ids:
                if obj_name == "gpsample":
                    control_sets_id = 0
                    dims = 3
                    noise_std = 0.01
                    init_lengthscale = 0.1
                    n_init_points = 5
                    budget = 100
                elif obj_name == "extendackley":
                    control_sets_id = 0
                    dims = 12
                    noise_std = 0.01
                    init_lengthscale = 0.05
                    n_init_points = 5
                    budget = 100
                elif obj_name == "extendlevy":
                    control_sets_id = 0
                    dims = 12
                    noise_std = 0.01
                    init_lengthscale = 0.2
                    n_init_points = 5
                    budget = 100
                elif obj_name == "extendhartmann":
                    control_sets_id = 0
                    dims = 6
                    noise_std = 0.01
                    init_lengthscale = 0.2
                    n_init_points = 5
                    budget = 100
                elif obj_name == "hartmann":
                    control_sets_id = 0
                    dims = 3
                    noise_std = 0.01
                    init_lengthscale = 0.1
                    n_init_points = 5
                    budget = 100
                elif obj_name == "maxhartmann":
                    control_sets_id = 0
                    dims = 3
                    noise_std = 0.01
                    init_lengthscale = 0.1
                    n_init_points = 5
                    budget = 100
                elif obj_name == "plant":
                    control_sets_id = 0
                    dims = 5
                    noise_std = 0.01
                    init_lengthscale = 0.1
                    n_init_points = 5
                    budget = 100
                elif obj_name == "airfoil":
                    control_sets_id = 1
                    dims = 5
                    noise_std = 0.01
                    init_lengthscale = 0.2
                    n_init_points = 5
                    budget = 100
                elif obj_name == "triplebranin":
                    control_sets_id = 0
                    dims = 6
                    noise_std = 0.01
                    init_lengthscale = 0.2
                    n_init_points = 5
                    budget = 100
                elif obj_name == "Hartmann6":
                    control_sets_id = 0
                    dims = 12
                    noise_std = 0.01
                    init_lengthscale = 0.05
                    n_init_points = 5
                    budget = 100
                else:
                    raise NotImplementedError
                main(
                    obj_name = obj_name,
                    acq_name = "proposed",
                    eps_schedule_id = 0,
                    costs_id=costs_id,
                    var_id=var_id,
                    budget=budget,
                    seed=seed,
                    control_sets_id=control_sets_id,
                    dims=dims,
                    noise_std=0.01,
                    init_lengthscale=0.2,
                    n_init_points=5,
                )