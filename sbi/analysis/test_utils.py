import numpy as np
import pandas as pd


def PP_vals(RV_samples, alphas):
    """Compute the PP-values: empirical c.d.f. of a random variable (RV).
    Used for Probability - Probabiity (P-P) plots.

    Args:
        RV_samples (np.array): samples from the random variable.
        alphas (list, np.array): alpha values to evaluate the c.d.f.

    Returns:
        pp_vals (list): empirical c.d.f. values for each alpha.
    """
    pp_vals = [np.mean(RV_samples <= alpha) for alpha in alphas]
    return pp_vals


# for L-C2ST diagnostics
def compute_dfs_with_probas_marginals(probas, P_eval):
    """Compute dataframes with predicted class probabilities for each
    (1d and 2d) marginal sample of the density estimator.
    Used in `sbi/analysis/plots/pairplot_with_proba_intensity`.

    Args:
        probas (np.array): predicted class probabilities on test data.
        P_eval (torch.Tensor): corresponding sample from the density estimator
            (test data directly or transformed test data in the case of a
            normalizing flow density estimator).

    Returns:
        dfs (dict of pd.DataFrames): dict of dataframes if predicted probabilities
        for each marginal dimension (keys).
    """
    dim = P_eval.shape[-1]
    dfs = {}
    for i in range(dim):
        P_i = P_eval[:, i].numpy().reshape(-1, 1)
        df = pd.DataFrame({"probas": probas})
        df["z"] = P_i[:, 0]
        dfs[f"{i}"] = df

        for j in range(i + 1, dim):
            P_ij = P_eval[:, [i, j]].numpy()
            df = pd.DataFrame({"probas": probas})
            df["z_1"] = P_ij[:, 0]
            df["z_2"] = P_ij[:, 1]
            dfs[f"{i}_{j}"] = df
    return dfs
