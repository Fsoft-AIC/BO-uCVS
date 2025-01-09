from botorch.models.utils import gpt_posterior_settings
from botorch.posteriors import GPyTorchPosterior
import gpytorch
import torch
from torch.nn import Module

from core.utils import uniform_samples


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        # self.mean_y = train_y.mean()
        # self.std_y = train_y.std()
        
        self.train_x = train_x
        self.train_y = train_y

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) 
        # mean_x = self.mean_module(x)
        # covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    def fit(self, known_lengthscale = False):
        # Find optimal model hyperparameters
        self.train()
        self.likelihood.train()

        def param_train():
            for name, param in self.named_parameters():
                if name not in ["covar_module.base_kernel.raw_lengthscale"]:
                    yield param
                else:
                    if not known_lengthscale:
                        print("=====Train lengscale=====")
                        yield param

        # Use the adam optimizer
        optimizer = torch.optim.Adam([{"params": param_train()}], lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(10):
            # Zero gradients from previous iteration
            with gpytorch.settings.cholesky_jitter(1e-2):
                optimizer.zero_grad()
                # Output from model
                output = self(self.train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, self.train_y)
                loss.backward()
                optimizer.step()

class PosteriorModel(Module):
    def __init__(self, custom_model):
        super().__init__()
        self.custom_model = custom_model
        self.num_outputs = 1

    def posterior(self, X):
        self.custom_model.eval()
        with gpt_posterior_settings():
            mvn = self.custom_model(X)
        posterior = GPyTorchPosterior(mvn=mvn)
        return posterior


def sample_gp_prior(kernel, bounds, num_points, jitter=1e-06):
    """
    Sample a random function from a GP prior with mean 0 and covariance specified by a kernel.
    :param kernel: a GPyTorch kernel.
    :param bounds: array of shape (2, num_dims).
    :param num_points: int.
    :param jitter: float.
    :return: Callable that takes in an array of shape (n, N) and returns an array of shape (n, 1).
    """
    points = uniform_samples(bounds=bounds, n_samples=num_points)
    cov = kernel(points).evaluate() + jitter * torch.eye(num_points)
    f_vals = torch.distributions.MultivariateNormal(
        torch.zeros(num_points, dtype=torch.double), cov
    ).sample()[:, None]

    L = torch.linalg.cholesky(cov)
    L_bs_f = torch.linalg.solve_triangular(L, f_vals, upper=False)
    LT_bs_L_bs_f = torch.linalg.solve_triangular(L.T, L_bs_f, upper=True)
    return lambda x: kernel(x, points).evaluate() @ LT_bs_L_bs_f
