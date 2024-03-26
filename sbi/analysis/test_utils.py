from typing import List, Union

import numpy as np


def PP_vals(samples: np.ndarray, alphas: Union[List, np.ndarray]) -> np.ndarray:
    """Computes the PP-values: empirical c.d.f. of a random variable.
    Used for Probability - Probabiity (P-P) plots.

    Args:
        samples: samples from the random variable, of shape (n_samples, dim).
        alphas: alpha values to evaluate the c.d.f, of shape (n_alphas,).

    Returns:
        pp_vals: empirical c.d.f. values for each alpha, of shape (n_alphas,).
    """
    pp_vals = [(samples <= alpha).mean() for alpha in alphas]
    return np.array(pp_vals)


def get_probas_per_marginal(probas: np.ndarray, samples: np.ndarray) -> dict:
    """Associates the given probabilities with each marginal dimension
    of the samples.
    Used for customized pairplots of the `samples` with `probas`
    as weights for the colormap.

    Args:
        probas: weights to associate with the samples, of shape (n_samples,).
        samples: samples to extract the marginals, of shape (n_samples, dim).

    Returns:
        dicts: dictionary with keys as the marginal dimensions and values as
            dictionaries with items:
            - "s" (resp. "s_1", "s_2"): 1D (resp. 2D) marginal samples.
            - "probas": weights associated with the samples.
    """
    dim = samples.shape[-1]
    dicts = {}
    for i in range(dim):
        samples_i = samples[:, i].reshape(-1, 1)
        dict_i = {"probas": probas}
        dict_i["s"] = samples_i[:, 0]
        dicts[f"{i}"] = dict_i

        for j in range(i + 1, dim):
            samples_ij = samples[:, [i, j]]
            dict_ij = {"probas": probas}
            dict_ij["s_1"] = samples_ij[:, 0]
            dict_ij["s_2"] = samples_ij[:, 1]
            dicts[f"{i}_{j}"] = dict_ij
    return dicts
