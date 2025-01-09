import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import pkgutil


def init_heteroscedastic_gp(num_inducing, d):
    """
    Initializes default heteroscedastic GP, so that we can load the parameters later.
    :param num_inducing:
    :param d:
    :return:
    """
    likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
        distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
        scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
    )

    kernels = [
        gpf.kernels.SquaredExponential(lengthscales=np.ones(d)),
        gpf.kernels.SquaredExponential(lengthscales=np.ones(d)),
    ]

    kernel = gpf.kernels.SeparateIndependent(kernels)

    Z = tf.zeros((num_inducing, d))
    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
        [
            gpf.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
            gpf.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
        ]
    )

    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing_variable,
        num_latent_gps=likelihood.latent_dim,
    )
    return model


def create_leaf_max_area_func(standardize=False):
    NUM_INDUCING = 464
    DIMS = 5
    LEAF_MAX_AREA_DATA_MEAN = 108.59035777465141
    LEAF_MAX_AREA_DATA_STD = 107.55729094567415

    dict_name = "gp_leaf_dict.p"
    gp = init_heteroscedastic_gp(num_inducing=NUM_INDUCING, d=DIMS)
    data = pkgutil.get_data(__name__, dict_name)
    param_dict = pickle.loads(data)
    gpf.utilities.multiple_assign(gp, param_dict)

    def leaf_max_area_func(X):
        """
        Returns the predictive mean and variance of the maximum leaf area.
        :param X: Array of shape (num_preds, 5). Columns are Ca [0, 7.7] log uM, B [0, 3.5] log uM,
        NH3 [0, 10.4] log uM, K  [8.9, 11.3] log uM, pH [2.5, 6.5].
        :return: Tuple (array of shape (num_preds, 1), array of shape (num_preds, 1). First element is mean, second
        is variance.
        """
        mean, var = gp.predict_y(X)
        return mean.numpy() * (
            LEAF_MAX_AREA_DATA_STD
        ) + LEAF_MAX_AREA_DATA_MEAN, var.numpy() * (LEAF_MAX_AREA_DATA_STD**2)

    def leaf_max_area_func_standardized(X):
        """
        Returns the predictive mean and variance of the maximum leaf area standardized.
        :param X: Array of shape (num_preds, 5). Columns are Ca [0, 7.7] log uM, B [0, 3.5] log uM,
        NH3 [0, 10.4] log uM, K  [8.9, 11.3] log uM, pH [2.5, 6.5].
        :return: Tuple (array of shape (num_preds, 1), array of shape (num_preds, 1). First element is mean, second
        is variance.
        """
        mean, var = gp.predict_y(X)
        return mean.numpy(), var.numpy()

    if not standardize:
        return leaf_max_area_func
    else:
        return leaf_max_area_func_standardized
