import datetime
from itertools import combinations
import numpy as np
from scipy.optimize import minimize
import torch
import time


def log(msg):
    print(str(datetime.datetime.now()) + " - " + msg)


def uniform_samples(bounds, n_samples):
    low = bounds[0]
    high = bounds[1]
    d = len(low)
    return torch.rand(size=(n_samples, d), dtype=torch.double) * (high - low) + low


def get_missing_idxs(n_idxs, given_idxs):
    all_idxs = torch.arange(n_idxs)
    combined = torch.cat((all_idxs, given_idxs))
    uniques, counts = combined.unique(return_counts=True)
    return torch.sort(uniques[counts == 1])[0]


def minmax_normalize(arr):
    b = torch.max(arr)
    a = torch.min(arr)
    return (arr - a) / (b - a)


def maximize_fn(f, bounds, mode="L-BFGS-B", n_warmup=100, n_iter=2):
    """
    Approximately maximizes a function f using sampling + L-BFGS-B method adapted from
    https://github.com/fmfn/BayesianOptimization.
    :param f: Callable that takes in an array of shape (n, d) and returns an array of shape (n, 1).
    :param bounds: Array of shape (2, d). Lower and upper bounds of each variable.
    :param n_warmup: int. Number of random samples.
    :param n_iter: int. Number of L-BFGS-B starting points.
    :param n_iter_direct:
    :return: (Array of shape (d,), max_val).
    """
    neg_func_squeezed = lambda x: np.squeeze(
        -f(torch.tensor(x[None, :])).cpu().detach().numpy()
    )

    if mode == "L-BFGS-B":
        # log("Starting random sampling")
        # Random sampling
        start = time.process_time()
        x_tries = uniform_samples(bounds=bounds, n_samples=n_warmup)
        f_x = np.zeros((x_tries.shape[0],1))
        for i in range(x_tries.shape[0]):
            with torch.no_grad():
                f_x = torch.squeeze(f(x_tries[i:i+1]), dim=1).cpu().detach().numpy()
        x_max = x_tries[np.argmax(f_x)]
        f_max = np.max(f_x)
        elapsed_time = time.process_time() - start


        # log("Starting L-BFGS-B")
        # L-BFGS-B
        print(f"n_iter : {n_iter}, n_warmup : {n_warmup}")
        x_seeds = uniform_samples(bounds=bounds, n_samples=n_iter - 1)
        x_seeds = torch.cat([x_seeds, x_max[None, :]], dim=0)
        x_seeds_np = x_seeds.cpu().detach().numpy()
        for x_try in x_seeds_np:
            if elapsed_time >1000:
                break
            res = minimize(
                fun=neg_func_squeezed,
                x0=x_try,
                bounds=bounds.T,
                method="L-BFGS-B",
            )
            elapsed_time = time.process_time() - start
            if not res.success:
                continue
            if -res.fun >= f_max:
                x_max = res.x
                f_max = -res.fun
    # elif mode == "DIRECT":
    #     res = direct_minimize(
    #         func=neg_func_squeezed, bounds=bounds.T, algmethod=1, maxT=n_iter_direct
    #     )
    #     if not res.success:
    #         raise Exception("DIRECT failed in maximize_fn")
    #     x_max = res.x
    #     f_max = -res.fun
    else:
        raise NotImplementedError

    f_argmax = np.clip(x_max, bounds[0], bounds[1])
    return torch.tensor(f_argmax), f_max


def expectation_det(f, x_control, random_dists_samples, order_idxs):
    """
    Approximates the expectation of f given some deterministic variables (x_control) and some random
    variables (random_set_idxs) with deterministically chosen Monte Carlo samples.
    :param f: Callable that takes in an array of shape (n, d) and returns an array of shape (n, 1).
    :param x_control: array of shape (b, |control_set|, ).
    :param random_dists_samples: array of shape (n_samples, |random_set|).
    :param order_idxs: int array of shape (d, ).
    :return: array of shape (b, 1).
    """
    d = len(order_idxs)
    n_samples, d_r = random_dists_samples.shape
    b, d_c = x_control.shape

    X_c = x_control[:, None, :].repeat(1, n_samples, 1)  # (b, n_samples, d_c)
    X_r = random_dists_samples[None, :, :].repeat(b, 1, 1)  # (b, n_samples, d_r)
    X_unordered = torch.cat([X_c, X_r], dim=-1)
    X = X_unordered[:, :, order_idxs]  # (b, n_samples, d)

    X_reshaped = X.reshape((-1, d))  # (b * n_samples, d)
    f_vals_reshaped = f(X_reshaped)  # (b * n_samples)
    f_vals = f_vals_reshaped.reshape((b, n_samples))

    return torch.mean(f_vals, dim=-1, keepdim=True)


def expectation(f, x_control, random_set_idxs, all_dists, n_samples, batch_size=-1):
    """
    Approximates the expectation of f given some deterministic variables (x_control) and some random
    variables (random_set_idxs) with random Monte Carlo sampling.
    :param f: Callable that takes in an array of shape (n, d) and returns an array of shape (n, 1).
    :param x_control: array of shape (b, |control_set|, ).
    :param random_set_idxs: array of shape (|random_set|, ). Indices indicating which variables
    are to be random.
    :param all_dists: list of Distributions of length d.
    :param n_samples: int.
    :param batch_size: If -1, evaluates f(X) at once where X has b * n_samples points. Otherwise,
    breaks X up into batches to evaluate f(X) in a slower but more memory efficient
    manner.
    :return: array of shape (b, 1).
    """
    d = len(all_dists)
    b, d_c = x_control.shape
    assert d_c + len(random_set_idxs) == d
    assert batch_size <= b * n_samples

    X = torch.zeros((b, n_samples, d), dtype=torch.double)

    # Get samples
    control_counter = 0
    for j in range(d):
        if j in random_set_idxs:
            X[:, :, j] = all_dists[j].sample(n_samples=n_samples)[None, :]
        else:
            X[:, :, j] = x_control[:, control_counter][:, None]
            control_counter += 1

    X_reshaped = X.reshape((-1, d))  # (b * n_samples, d)
    if batch_size == -1:
        f_vals_reshaped = f(X_reshaped)  # (b * n_samples)
    else:
        f_vals_reshaped = batch_evaluate(f=f, X=X_reshaped, batch_size=batch_size)
    f_vals = f_vals_reshaped.reshape((b, n_samples))

    return torch.mean(f_vals, dim=-1, keepdim=True)


def batch_evaluate(f, X, batch_size):
    n = X.shape[0]
    ret = torch.zeros((n, 1))

    start = 0
    end = start + batch_size
    while end < n:
        ret[start:end] = f(X[start:end])
        start = end
        end = end + batch_size

    # final batch
    ret[start:n] = f(X[start:n])

    return ret


def powerset(arr):
    pset = []
    for r in range(1, len(arr) + 1):
        pset += list(combinations(arr, r))
    return pset


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
