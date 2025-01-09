from abc import ABC, abstractmethod
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.kernels import RFFKernel
import numpy as np
import time
import torch

from core.dists import get_opt_queries_and_vals
from core.gp import PosteriorModel
from core.utils import maximize_fn


def get_acquisition(acq_name, eps_schedule_id, costs):
    if acq_name == "ucb-cvs":
        return UCB_CVS(beta=2.0)
    elif acq_name == "etc":
        # For ETC, we cannot define the required eps_schedule a priori,
        # so we do it like this
        if eps_schedule_id == 0:
            target_num_plays = np.array([50, 50])
        elif eps_schedule_id == 1:
            target_num_plays = np.array([100, 100])
        elif eps_schedule_id == 2:
            # Adaptive
            cheap = costs[0]
            assert cheap == costs[1]
            mid = costs[3]
            assert mid == costs[4]
            target_num_plays = np.array([int(4 / cheap), int(4 / mid)])
        else:
            raise NotImplementedError

        return ETC_UCB_CVS(
            beta=2.,
            grouped_control_set_idxs=np.array([[4,0,1,2],[5,3,6]]),
            target_num_plays=target_num_plays,
        )
    elif acq_name == "ucb":
        return UCB_PSQ(beta=2.0)
    elif acq_name == "ts":
        return TS_PSQ(n_features=1024)
    elif acq_name == "ei":
        return EI_CVnaive()
    elif acq_name == "ucb-naive":
        return UCB_CVS_naive(beta=2.0)
    elif acq_name == "proposed":
        return None
    elif acq_name == "ts-naive":
        return TS_PSQ_CVnaive(n_features=1024)
    else:
        raise Exception("Incorrect acq_name passed to get_acquisition")


class Acquisition(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        pass


class EI_CVnaive(Acquisition):
    """
    WARNING: as implemented, assumes the full query control set is available.
    """

    def __init__(self):
        super().__init__()

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        q = len(train_X)
        post_model = PosteriorModel(gp)
        sampler = SobolQMCNormalSampler(32)

        opt_improvements = []
        opt_queries = []
        dims = bounds.shape[-1]
        for i in range(len(control_sets)):
            if len(control_sets[i]) == dims:  # full query control set

                def ei(X):
                    posterior = post_model.posterior(torch.cat([train_X, X], dim=0))
                    samples = sampler(posterior)
                    best = torch.max(samples[:, :q], dim=1, keepdim=True)[0]
                    inter1 = torch.clamp_min(input=samples[:, q:] - best, min=0.0)
                    return torch.mean(inter1, dim=0)

                ret_query, ret_improvement = maximize_fn(
                    f=ei,
                    bounds=bounds,
                    n_warmup=2000,
                )
            else:
                control_set = control_sets[i]
                random_set = random_sets[i]
                random_dists_samples = all_dists_samples[
                    :, random_set
                ]  # (n_samples, d_r)

                cat_idxs = np.concatenate([control_set, random_set])
                order_idxs = np.array(
                    [np.where(cat_idxs == j)[0][0] for j in np.arange(len(cat_idxs))]
                )

                def ei(x_control):  # (b, |control_set|, )
                    n_samples, d_r = random_dists_samples.shape
                    b, d_c = x_control.shape

                    X_c = x_control[:, None, :].repeat(
                        1, n_samples, 1
                    )  # (b, n_samples, d_c)
                    X_r = random_dists_samples[None, :, :].repeat(
                        b, 1, 1
                    )  # (b, n_samples, d_r)
                    X_unordered = torch.cat([X_c, X_r], dim=-1)
                    X = X_unordered[:, :, order_idxs]  # (b, n_samples, d)
                    train_X_rep = train_X[None, :, :].repeat(
                        b, 1, 1
                    )  # (b, n_train_X, d)
                    # (b, n_train_X + n_samples, d)
                    posterior = post_model.posterior(torch.cat([train_X_rep, X], dim=1))
                    samples = sampler(
                        posterior
                    )  # (f_samples, b, n_train_X + n_samples, 1)
                    inter1 = torch.mean(samples[:, :, q:], dim=2)  # (f_samples, b, 1)
                    best = torch.max(samples[:, :, :q], dim=2)[0]  # (f_samples, b, 1)
                    inter2 = torch.clamp_min(input=inter1 - best, min=0.0)
                    return torch.mean(inter2, dim=0)  # (b, 1)

                ret_query, ret_improvement = maximize_fn(
                    f=ei,
                    bounds=bounds[:, control_set],
                )

            opt_queries.append(ret_query[None, :])
            opt_improvements.append(ret_improvement)

        opt_improvements = torch.tensor(opt_improvements)
        print(opt_improvements)
        cost_weighted_improvements = opt_improvements / costs
        print(cost_weighted_improvements)
        ret_control_idx = torch.argmax(cost_weighted_improvements).item()

        ret_query = opt_queries[ret_control_idx]

        return ret_control_idx, ret_query


class TS_PSQ(Acquisition):
    """
    Adapted from https://botorch.org/tutorials/thompson_sampling.
    """

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        start = time.process_time()
        dims = bounds.shape[-1]

        # Thompson sampling via random Fourier features
        rff_k = RFFKernel(num_samples=self.n_features, num_dims=dims, ard_num_dims=dims)
        rff_k.lengthscale = gp.covar_module.base_kernel.lengthscale
        rff_k.randn_weights = rff_k.randn_weights.double()

        def featurize(X):
            return rff_k._featurize(X, normalize=True) * torch.sqrt(
                gp.covar_module.outputscale
            )

        Phi = featurize(train_X)
        s2 = gp.likelihood.noise
        PPinv = torch.linalg.inv(Phi @ Phi.T + s2 * torch.eye(Phi.shape[0]))
        mean = Phi.T @ PPinv @ train_y
        cov = torch.eye(Phi.shape[-1]) - Phi.T @ PPinv @ Phi
        L = torch.linalg.cholesky(cov)
        theta = mean + L @ torch.randn(size=mean.size(), dtype=torch.double)

        def f_sample(X):
            return featurize(X) @ theta

        skip_expectations = False
        for i, control_set in enumerate(control_sets):
            if len(control_set) == dims:
                skip_expectations = True
                ret_control_idx = i
                ret_query, _ = maximize_fn(
                    f=f_sample,
                    bounds=bounds,
                    mode="L-BFGS-B",
                    n_warmup=2000,
                )
                ret_query = ret_query[None, :]
                break

        if not skip_expectations:
            opt_queries, opt_vals = get_opt_queries_and_vals(
                f=f_sample,
                control_sets=control_sets,
                random_sets=random_sets,
                all_dists_samples=all_dists_samples,
                bounds=bounds,
                max_mode="L-BFGS-B",
            )

            ret_control_idx = torch.argmax(opt_vals).item()
            ret_query = opt_queries[ret_control_idx]

        elapsed_time = time.process_time() - start
        print(f"Elapsed CPU time: {elapsed_time}")
        return ret_control_idx, ret_query


class TS_PSQ_CVnaive(Acquisition):
    """
    Adapted from https://botorch.org/tutorials/thompson_sampling.
    """

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        dims = bounds.shape[-1]

        # Thompson sampling via random Fourier features
        rff_k = RFFKernel(num_samples=self.n_features, num_dims=dims, ard_num_dims=dims)
        rff_k.lengthscale = gp.covar_module.base_kernel.lengthscale
        rff_k.randn_weights = rff_k.randn_weights.double()

        def featurize(X):
            return rff_k._featurize(X, normalize=True) * torch.sqrt(
                gp.covar_module.outputscale
            )

        Phi = featurize(train_X)
        s2 = gp.likelihood.noise
        PPinv = torch.linalg.inv(Phi @ Phi.T + s2 * torch.eye(Phi.shape[0]))
        mean = Phi.T @ PPinv @ train_y
        cov = torch.eye(Phi.shape[-1]) - Phi.T @ PPinv @ Phi
        L = torch.linalg.cholesky(cov)
        theta = mean + L @ torch.randn(size=mean.size(), dtype=torch.double)

        def f_sample(X):
            return featurize(X) @ theta

        opt_queries, opt_vals = get_opt_queries_and_vals(
            f=f_sample,
            control_sets=control_sets,
            random_sets=random_sets,
            all_dists_samples=all_dists_samples,
            bounds=bounds,
            max_mode="L-BFGS-B",
        )

        vals_per_cost = opt_vals / costs

        ret_control_idx = torch.argmax(vals_per_cost).item()
        ret_query = opt_queries[ret_control_idx]

        return ret_control_idx, ret_query


class UCB_PSQ(Acquisition):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        def ucb(X):
            f_preds = gp(X)
            mean = f_preds.mean
            variance = f_preds.variance
            return (mean + self.beta * torch.sqrt(variance))[:, None]

        dims = bounds.shape[-1]
        skip_expectations = False

        for i, control_set in enumerate(control_sets):
            if len(control_set) == dims:
                skip_expectations = True
                ret_control_idx = i
                ret_query, _ = maximize_fn(
                    f=ucb,
                    bounds=bounds,
                    mode="L-BFGS-B",
                    n_warmup=2000,
                )
                ret_query = ret_query[None, :]
                break

        if not skip_expectations:
            opt_queries, opt_vals = get_opt_queries_and_vals(
                f=ucb,
                control_sets=control_sets,
                random_sets=random_sets,
                all_dists_samples=all_dists_samples,
                bounds=bounds,
                max_mode="L-BFGS-B",
            )
            ret_control_idx = torch.argmax(opt_vals).item()
            ret_query = opt_queries[ret_control_idx]

        return ret_control_idx, ret_query


class UCB_CVS(Acquisition):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        def ucb(X):
            f_preds = gp(X)
            mean = f_preds.mean
            variance = f_preds.variance
            return (mean + self.beta * torch.sqrt(variance))[:, None]

        eps = eps_schedule.next()

        opt_queries, opt_vals = get_opt_queries_and_vals(
            f=ucb,
            control_sets=control_sets,
            random_sets=random_sets,
            all_dists_samples=all_dists_samples,
            bounds=bounds,
            max_mode="L-BFGS-B",
        )

        max_val = torch.max(opt_vals)
        ret_control_idx = None

        for i in range(len(opt_vals)):
            if opt_vals[i] + eps >= max_val:
                if ret_control_idx is None or (
                    costs[i] <= costs[ret_control_idx]
                    and opt_vals[i] >= opt_vals[ret_control_idx]
                ):
                    ret_control_idx = i

        if ret_control_idx is None:
            ret_control_idx = torch.argmax(opt_vals).item()

        ret_query = opt_queries[ret_control_idx]

        return ret_control_idx, ret_query


class ETC_UCB_CVS(Acquisition):
    def __init__(self, beta, grouped_control_set_idxs, target_num_plays):
        """

        :param beta:
        :param grouped_control_set_idxs: e.g. [[0, 1, 2], [3, 4, 5]]
        :param target_num_plays: [100, 100]
        """
        super().__init__()
        self.beta = beta
        self.grouped_control_set_idxs = grouped_control_set_idxs
        self.target_num_plays = target_num_plays
        self.current_num_plays = np.zeros(len(target_num_plays))

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        start = time.process_time()

        def ucb(X):
            f_preds = gp(X)
            mean = f_preds.mean
            variance = f_preds.variance
            return (mean + self.beta * torch.sqrt(variance))[:, None]

        indices = np.arange(len(control_sets))
        for group in range(len(self.target_num_plays)):
            if self.current_num_plays[group] < self.target_num_plays[group]:
                indices = self.grouped_control_set_idxs[group]
                self.current_num_plays[group] += 1
                break

        dims = bounds.shape[-1]
        skip_expectations = False
        for idx in indices:
            if len(control_sets[idx]) == dims:
                ret_control_idx = idx
                ret_query, opt_val = maximize_fn(
                    f=ucb,
                    bounds=bounds,
                    mode="L-BFGS-B",
                    n_warmup=2000,
                )
                ret_query = ret_query[None, :]
                skip_expectations = True
                break

        if not skip_expectations:
            opt_queries, opt_vals = get_opt_queries_and_vals(
                f=ucb,
                control_sets=control_sets,
                random_sets=random_sets,
                all_dists_samples=all_dists_samples,
                bounds=bounds,
                max_mode="L-BFGS-B",
                indices=indices,
            )
            ret_control_idx = indices[torch.argmax(opt_vals).item()]
            ret_query = opt_queries[torch.argmax(opt_vals).item()]
            opt_val = opt_vals.max()
        print("=============")
        print(opt_val)

        elapsed_time = time.process_time() - start
        print(f"Elapsed CPU time: {elapsed_time}")
        return ret_control_idx, ret_query


class UCB_CVS_naive(Acquisition):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        def ucb(X):
            f_preds = gp(X)
            mean = f_preds.mean
            variance = f_preds.variance
            return (mean + self.beta * torch.sqrt(variance))[:, None]

        opt_queries, opt_vals = get_opt_queries_and_vals(
            f=ucb,
            control_sets=control_sets,
            random_sets=random_sets,
            all_dists_samples=all_dists_samples,
            bounds=bounds,
            max_mode="L-BFGS-B",
        )

        vals_per_cost = opt_vals / costs

        ret_control_idx = torch.argmax(vals_per_cost).item()
        ret_query = opt_queries[ret_control_idx]

        return ret_control_idx, ret_query
    
class UCB_one_subset(Acquisition):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def acquire(self,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        index,):
        dims = bounds.shape[-1]

        def ucb(X):
            f_preds = gp(X)
            mean = f_preds.mean
            variance = f_preds.variance
            return (mean + self.beta * torch.sqrt(variance))[:, None]
        if len(control_sets[index]) == dims:
            ret_query, opt_vals = maximize_fn(
                f=ucb,
                bounds=bounds,
                mode="L-BFGS-B",
                n_warmup=2000,
            )
            ret_query = ret_query[None, :]
            return ret_query

        opt_queries, opt_vals = get_opt_queries_and_vals(
            f=ucb,
            control_sets=control_sets,
            random_sets=random_sets,
            all_dists_samples=all_dists_samples,
            bounds=bounds,
            max_mode="L-BFGS-B",
            indices = [index]
        )
        return opt_queries[0]

class Exploit_CVS(Acquisition):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        def lcb(X):
            f_preds = gp(X)
            mean = f_preds.mean
            variance = f_preds.variance
            return (mean - 0.3* torch.sqrt(variance))[:, None]
        def ucb(X):
            f_preds = gp(X)
            mean = f_preds.mean
            variance = f_preds.variance
            return (mean + self.beta * torch.sqrt(variance))[:, None]
        skip_expectations = False
        dims = bounds.shape[-1]

        indices = np.arange(len(control_sets))
        for idx in indices:
            if len(control_sets[idx]) == dims:
                ret_control_idx = idx
                ret_query, Lcb_max = maximize_fn(
                    f=lcb,
                    bounds=bounds,
                    mode="L-BFGS-B",
                    n_warmup=1500,
                )
                skip_expectations = True
                Lcb_max = Lcb_max if torch.max(train_y) < Lcb_max else torch.max(train_y)

        print(skip_expectations)    
        if not skip_expectations:
            opt_queries, opt_Lcb = get_opt_queries_and_vals(
                f=lcb,
                control_sets=control_sets,
                random_sets=random_sets,
                all_dists_samples=all_dists_samples,
                bounds=bounds,
                max_mode="L-BFGS-B",
            )
            ret_control_idx = torch.argmax(opt_Lcb).item()
            ret_query = opt_queries[ret_control_idx]
            Lcb_max = opt_Lcb[ret_control_idx].item()

        opt_queries, opt_vals = get_opt_queries_and_vals(
            f=ucb,
            control_sets=control_sets,
            random_sets=random_sets,
            all_dists_samples=all_dists_samples,
            bounds=bounds,
            max_mode="L-BFGS-B",)

        ret_control_idx = None

        for i in range(len(opt_vals)):
            if opt_vals[i]  >= Lcb_max:
                if ret_control_idx is None or (
                    costs[i] <= costs[ret_control_idx]
                ):
                    ret_control_idx = i

        if ret_control_idx is None:
            ret_control_idx = torch.argmax(opt_vals).item()
        print(f"max lcb = {Lcb_max}")
        print(f"max ucb = {opt_vals}")

        ret_query = opt_queries[ret_control_idx]

        return ret_control_idx, ret_query

def LCB_costs(cost_samples, max_iter):
    n = len(cost_samples)
    lcb_costs = np.zeros(n)
    for i in range(n):
        k = len(cost_samples[i])
        if k<1:
            lcb_costs[i] = -np.inf
        else:
            mean = np.mean(cost_samples[i])
            lcb_costs[i] = mean - np.sqrt(2*np.log(max_iter)/k)
    return lcb_costs
