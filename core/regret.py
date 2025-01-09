import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

from core.dists import get_opt_queries_and_vals, maximize_fn
from core.utils import expectation_det


def interpolate_regret(regrets, all_cost_per_iter_cumusums):
    cost_axis = np.sort(np.concatenate(all_cost_per_iter_cumusums))
    cost_axis = np.unique(cost_axis)  # remove duplicates
    new_regrets = np.zeros((len(regrets), len(cost_axis)))
    for i, regret in enumerate(regrets):
        cost_cumusum = all_cost_per_iter_cumusums[i]
        assert len(cost_cumusum) == len(regret)
        for j, cost in enumerate(cost_axis):
            where = np.where(cost_cumusum == cost)[0]
            if len(where) != 0:
                assert len(where) == 1
                new_regrets[i, j] = regret[where[0]]
            else:
                if j != 0:
                    new_regrets[i, j] = new_regrets[i, j - 1]
                else:
                    new_regrets[i, j] = regret[0]

    return new_regrets, cost_axis


def get_regret(
    control_set_idxs,
    control_queries,
    obj_func,
    control_sets,
    random_sets,
    all_dists_samples,
    bounds,
    costs,
    opt_val_det=None,
):
    dims = bounds.shape[-1]
    if opt_val_det is not None:
        max_val = opt_val_det
    else:
        skip_expectations = False
        for control_set in control_sets:
            if len(control_set)==dims:
                _, max_val = maximize_fn(
                    f=obj_func,
                    bounds=bounds,
                    mode="L-BFGS-B",
                    n_warmup=10000,
                )
                skip_expectations = True
        if not skip_expectations:
            _, opt_vals = get_opt_queries_and_vals(
                f=obj_func,
                control_sets=control_sets,
                random_sets=random_sets,
                all_dists_samples=all_dists_samples,
                bounds=bounds,
                max_mode="L-BFGS-B",
                n_warmup = 100,
                n_iter = 2,
            )
            max_val = torch.max(opt_vals)
    iter_vals = []
    cost_per_iter = []
    for t in range(len(control_queries)):
        i = control_set_idxs[t]
        control_set = control_sets[i]

        if len(control_set) == dims:
            val = obj_func(control_queries[t])
        else:
            random_set = random_sets[i]
            cat_idxs = np.concatenate([control_set, random_set])
            order_idxs = np.array(
                [np.where(cat_idxs == j)[0][0] for j in np.arange(len(cat_idxs))]
            )
            with torch.no_grad():
                val = expectation_det(
                    f=obj_func,
                    x_control=control_queries[t],
                    random_dists_samples=all_dists_samples[:, random_set],
                    order_idxs=order_idxs,
                )
        iter_vals.append(val)
        cost_per_iter.append(costs[t])
    iter_vals = torch.cat(iter_vals, dim=0).squeeze(-1)  # (num_iters)

    simple_regret = (max_val - torch.cummax(iter_vals, dim=0)[0]).cpu().detach().numpy()
    cumu_regret = torch.cumsum(max_val - iter_vals, dim=0).cpu().detach().numpy()
    cost_per_iter = np.array(cost_per_iter)
    cs_cumu_regret = (
        torch.cumsum(
            torch.tensor(cost_per_iter, dtype=torch.double) * (max_val - iter_vals),
            dim=0,
        )
        .cpu()
        .detach()
        .numpy()
    )

    return simple_regret, cumu_regret, cs_cumu_regret, cost_per_iter


def get_cumu_regret(X, obj_func, opt_val):
    sample_regret = opt_val - obj_func(X).squeeze(-1)  # (num_iters, )
    return torch.cumsum(sample_regret, dim=0).cpu().detach().numpy()


def get_simple_regret(X, obj_func, opt_val):
    return (
        opt_val - torch.cummax(obj_func(X).squeeze(-1), dim=0)[0].cpu().detach().numpy()
    )


def plot_regret(
    regret,
    cost_per_iter,
    x_axis,
    num_iters,
    title="",
    save=False,
    save_dir="",
    filename="",
    show_plot=False,
):
    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle(title, size=12)
    fig.set_size_inches(12, 6)
    fig.set_dpi(200)

    if x_axis == "T":
        ax1.plot(np.arange(num_iters), regret)
    elif x_axis == "C":
        ax1.plot(np.cumsum(cost_per_iter), regret)

    ax1.axhline(y=0, xmax=num_iters, color="grey", alpha=0.5, linestyle="--")
    ax1.set_xlabel(x_axis)
    plt.grid()
    fig.tight_layout()
    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + filename, bbox_inches="tight")
    if show_plot:
        plt.show()


def first_index_of_conv(control_set_idxs, best_idx):
    v = len(control_set_idxs)
    first_index = v - 1
    for j in range(1, v + 1):
        if control_set_idxs[v - j] == best_idx:
            first_index = v - j
        else:
            break
    return first_index


def get_sample_regret_from_cumu(cumu_regret):
    sample_regret = np.zeros(len(cumu_regret))
    sample_regret[0] = cumu_regret[0]
    for j in range(1, len(cumu_regret)):
        sample_regret[j] = cumu_regret[j] - cumu_regret[j - 1]
    return sample_regret
