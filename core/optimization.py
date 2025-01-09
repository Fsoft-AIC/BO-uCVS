import numpy as np
import torch
import pickle
from copy import deepcopy

from core.dists import sample_from_random_sets
from core.gp import ExactGPModel
from core.utils import log
from core.acquisitions import UCB_one_subset, Exploit_CVS, LCB_costs


def bo_loop(
    train_X,
    train_y,
    likelihood,
    kernel,
    noisy_obj_func,
    start_iter,
    budget,
    acquisition,
    bounds,
    all_dists,
    control_sets,
    random_sets,
    all_dists_samples,
    costs,
    eps_schedule,
):
    control_set_idxs = []
    control_queries = []
    all_eps = []
    train_costs = [[]]*len(control_sets)
    train_costs_flat = []

    t = 1
    while budget>0:
        log(f"t = {t}, budget remaining :{budget}")

        gp = ExactGPModel(train_X, torch.squeeze(train_y), kernel, likelihood)
        # gp.fit(known_lengthscale=True)
        gp.eval()
        print(gp.covar_module.base_kernel.lengthscale.detach())
        print(gp.covar_module.outputscale.detach())
        if isinstance(costs, np.ndarray) or isinstance(costs, list):
            lcb_costs = costs
        else: 
            lcb_costs = LCB_costs(train_costs, t)
        if t <400:
            control_set_idx, control_query = acquisition.acquire(
                train_X=train_X,
                train_y=train_y,
                gp=gp,
                all_dists_samples=all_dists_samples,
                control_sets=control_sets,
                random_sets=random_sets,
                bounds=bounds,
                eps_schedule=eps_schedule,
                costs=lcb_costs,
            )
        else:
            control_set = [int(i) for i in control_sets[control_set_idx]]
            control_query = torch.clamp(control_query + torch.randn(len(control_set)) / (bounds[1][control_set]-bounds[0][control_set])
                                                ,min=bounds[0][control_set],max=bounds[1][control_set])
        log(f"Control set chosen: {control_set_idx}")

        # Exit condition
        if isinstance(costs, np.ndarray) or isinstance(costs, list):
            cost = costs[control_set_idx]
        else:
            cost = costs.sample(control_set_idx)

        # Observation
        dims = bounds.shape[-1]
        if len(control_sets[control_set_idx]) == dims:
            x_t = control_query
        else:
            control_set = control_sets[control_set_idx]
            random_set = random_sets[control_set_idx]
            cat_idxs = np.concatenate([control_set, random_set])
            order_idxs = np.array(
                [np.where(cat_idxs == j)[0][0] for j in np.arange(len(cat_idxs))]
            )
            random_query = sample_from_random_sets(
                all_dists=all_dists, random_set=random_set
            )
            unordered_x_t = torch.cat([control_query, random_query], dim=-1)
            x_t = unordered_x_t[:, order_idxs]
        with torch.no_grad():
            y_t = noisy_obj_func(x_t)  # (1 ,1)

        # Update datasets
        train_X = torch.cat([train_X, x_t])
        train_y = torch.cat([train_y, y_t])
        control_set_idxs.append(control_set_idx)
        control_queries.append(control_query)

        train_costs[control_set_idx].append(cost)
        train_costs_flat.append(cost)

        # Epsilon schedule management
        if eps_schedule is not None:
            eps_schedule.update()
            all_eps.append(eps_schedule.last_eps)

        # Loop management
        t += 1
        budget = budget - cost

    return train_X, train_y, control_set_idxs, control_queries, t-1, train_costs_flat


def modified_bo_loop(
    train_X,
    train_y,
    likelihood,
    kernel,
    noisy_obj_func,
    explore_start_budget,
    budget,
    acquisition,
    bounds,
    all_dists,
    control_sets,
    random_sets,
    all_dists_samples,
    costs,
    eps_schedule,
    file_path = None
):
    control_set_idxs = []
    control_queries = []
    train_costs = [[] for _ in range(len(control_sets))]
    train_costs_flat = []
    old_kernel = deepcopy(kernel)
    old_likelihood = deepcopy(likelihood)
    t = 1
    exploit_start_budget = budget - explore_start_budget

    while budget > 0:
        log(f"t = {t}, budget remaining :{budget}")
        if budget >= exploit_start_budget:
            gp = ExactGPModel(train_X, torch.squeeze(train_y), kernel, likelihood)
            # gp.fit(known_lengthscale=(t<=120))
            print(gp.covar_module.base_kernel.lengthscale.detach())
            gp.eval()
            control_set_idx = t % len(control_sets)
            acquisition =  UCB_one_subset(beta = 2.)
            control_query = acquisition.acquire(gp=gp,
                        control_sets=control_sets,
                        random_sets = random_sets,
                        all_dists_samples=all_dists_samples,
                        bounds=bounds,
                        index = control_set_idx)

        else:
            if isinstance(costs, np.ndarray) or isinstance(costs, list):
                lcb_costs = costs
            else: 
                lcb_costs = LCB_costs(train_costs, t)
            if t < 450:
                gp = ExactGPModel(train_X, torch.squeeze(train_y), kernel, likelihood)
                gp.fit()
                gp.eval()
                acquisition =  Exploit_CVS(beta = 1.96)
                control_set_idx, control_query = acquisition.acquire(
                    train_X=train_X,
                    train_y=train_y,
                    gp=gp,
                    all_dists_samples=all_dists_samples,
                    control_sets=control_sets,
                    random_sets=random_sets,
                    bounds=bounds,
                    eps_schedule=eps_schedule,
                    costs=lcb_costs,
                )
            else:
                acquisition =  UCB_one_subset(beta = 2.)
                control_set_idx = np.random.choice(len(control_sets))
                control_query = acquisition.acquire(gp=gp,
                        control_sets=control_sets,
                        random_sets = random_sets,
                        all_dists_samples=all_dists_samples,
                        bounds=bounds,
                        index = control_set_idx)

        log(f"Control set chosen: {control_set_idx}")

        # Observation
        if isinstance(costs, np.ndarray) or isinstance(costs, list):
            cost = costs[control_set_idx]
        else:
            cost = costs.sample(control_set_idx)
        dims = bounds.shape[-1]
        if len(control_sets[control_set_idx]) == dims:
            x_t = control_query
        else:
            control_set = control_sets[control_set_idx]
            random_set = random_sets[control_set_idx]
            cat_idxs = np.concatenate([control_set, random_set])
            order_idxs = np.array(
                [np.where(cat_idxs == j)[0][0] for j in np.arange(len(cat_idxs))]
            )
            random_query = sample_from_random_sets(
                all_dists=all_dists, random_set=random_set
            )
            unordered_x_t = torch.cat([control_query, random_query], dim=-1)
            x_t = unordered_x_t[:, order_idxs]
        with torch.no_grad():
            y_t = noisy_obj_func(x_t)  # (1 ,1)
        

        # Update datasets
        train_X = torch.cat([train_X, x_t])
        train_y = torch.cat([train_y, y_t])
        control_set_idxs.append(control_set_idx)
        control_queries.append(control_query)

        # Save real cost
        train_costs[control_set_idx].append(cost)
        train_costs_flat.append(cost)

        # Restart initial kernel and likelihood
        kernel = deepcopy(old_kernel)
        likelihood = deepcopy(old_likelihood)
        
        # Loop management
        budget = budget - cost
        t += 1
        if (file_path is not None) and (t%10==1):
            log("Save to " + file_path)
            with open(file_path, 'wb') as file:
                pickle.dump((train_X,train_y,control_set_idxs,control_queries,t-1,train_costs_flat),file)

    return train_X, train_y, control_set_idxs, control_queries, t-1, train_costs_flat
