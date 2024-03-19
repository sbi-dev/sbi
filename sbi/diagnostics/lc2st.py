from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from torch import Tensor
from tqdm import tqdm


class LC2ST:
    def __init__(
        self,
        thetas: Tensor,
        xs: Tensor,
        posterior_samples: Tensor,
        seed: int = 1,
        n_folds: int = 5,
        classifier: str = "mlp",
        z_score: bool = False,
        clf_class: BaseEstimator = None,
        clf_kwargs: Dict[str, Any] = None,
    ) -> None:
        """
        L-C2ST: Local Classifier Two-Sample Test
        -----------------------------------------
        Implementation based on the official code from [1] and the exisiting C2ST
        metric [2], using scikit-learn classifiers.

        TODO: Explain the method.
        TODO: Add version for normalizing flows.

        Args:
            thetas: Samples from the prior.
                Shape (n_samples, dim_theta)
            xs: Corresponding simulated data.
                Shape (n_samples, dim_x)
            posterior_samples: Samples from the estiamted posterior.
                Shape (n_samples, dim_theta)
            seed: Seed for the sklearn classifier and the KFold cross validation.
                Defaults to 1.
            n_folds: Number of folds for the cross-validation.
                Defaults to 5.
            z_score: Whether to z-score to normalize the data.
                Defaults to False.
            classifier: Classification architecture to use.
                Possible values: "rf" or "mlp", defaults to "mlp".
            clf_class: Custom sklearn classifier class.
                Defaults to None.
            clf_kwargs: Custom kwargs for the sklearn classifier.
                Defaults to None.

        References:
        [1] : https://arxiv.org/abs/2306.03580, https://github.com/JuliaLinhart/lc2st
        [2] : https://github.com/sbi-dev/sbi/blob/main/sbi/utils/metrics.py
        """

        assert (
            thetas.shape[0] == xs.shape[0] == posterior_samples.shape[0]
        ), "Number of samples must match"

        self.P = posterior_samples
        self.x_P = xs
        self.Q = thetas
        self.x_Q = xs
        self.seed = seed
        self.n_folds = n_folds

        # initialize classifier
        if "mlp" in classifier.lower():
            ndim = thetas.shape[-1]
            self.clf_class = MLPClassifier
            self.clf_kwargs = {
                "activation": "relu",
                "hidden_layer_sizes": (10 * ndim, 10 * ndim),
                "max_iter": 1000,
                "solver": "adam",
                "early_stopping": True,
                "n_iter_no_change": 50,
            }
        elif "rf" in classifier.lower():
            self.clf_class = RandomForestClassifier
            self.clf_kwargs = {}
        else:
            if clf_class is None:
                raise ValueError(
                    "Please provide a valid sklearn classifier class and kwargs."
                )
            self.clf_class = clf_class
            self.clf_kwargs = clf_kwargs

        # initialize, will be set after training
        self.trained_clfs = None
        self.trained_clfs_null = None

        # z-score normalization parameters
        self.z_score = z_score
        self.P_mean = torch.mean(self.P, dim=0)
        self.P_std = torch.std(self.P, dim=0)
        self.x_P_mean = torch.mean(self.x_P, dim=0)
        self.x_P_std = torch.std(self.x_P, dim=0)

        # TODO: case of normalizing flows
        self.null_distribution = None

    def train(self, verbosity: int = 0) -> List[Any]:
        """Returns the classifiers trained on observed data."""

        # prepare data
        if self.z_score:
            self.P = (self.P - self.P_mean) / self.P_std
            self.Q = (self.Q - self.P_mean) / self.P_std
            self.x_P = (self.x_P - self.x_P_mean) / self.x_P_std
            self.x_Q = (self.x_Q - self.x_P_mean) / self.x_P_std

        # initialize classifier
        clf = self.clf_class(**self.clf_kwargs)

        # cross-validation
        if self.n_folds > 1:
            trained_clfs = []
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            cv_splits = kf.split(self.P)
            for train_idx, _ in tqdm(
                cv_splits, desc="Cross-validation", disable=verbosity < 1
            ):
                # get train split
                P_train, Q_train = self.P[train_idx], self.Q[train_idx]
                x_P_train, x_Q_train = self.x_P[train_idx], self.x_Q[train_idx]

                # train classifier
                clf_n = train_lc2st(P_train, Q_train, x_P_train, x_Q_train, clf)

                trained_clfs.append(clf_n)
        else:
            # train single classifier
            clf = train_lc2st(self.P, self.Q, self.x_P, self.x_Q, clf)
            trained_clfs = [clf]

        # set trained classifiers
        self.trained_clfs = trained_clfs
        return trained_clfs

    def scores(
        self,
        P_eval: Tensor,
        x_eval: Tensor,
        return_probas: bool = False,
    ) -> np.ndarray:
        """Compute the L-C2ST scores for the observed data.

        Args:
            P_eval: Samples from P conditioned on the observation `x_eval`.
                Shape (n_samples, dim_theta)
            x_eval: Observation.
                Shape (n_samples, dim_x)
            return_probas: Whether to return the predicted probabilities of being in P.
                Defaults to False.

        Returns: one of
            scores: L-C2ST scores at `x_eval`.
            (probas, scores): Predicted probabilities and L-C2ST scores at `x_eval`.
        """
        assert self.trained_clfs is not None, "You need to train the classifiers first"

        # prepare data
        if self.z_score:
            P_eval = (P_eval - self.P_mean) / self.P_std
            x_eval = (x_eval - self.x_P_mean) / self.x_P_std

        probas, scores = [], []

        # evaluate classifiers
        for clf in self.trained_clfs:
            proba, score = eval_lc2st(P_eval, x_eval, clf, return_proba=True)
            probas.append(proba)
            scores.append(score)
        probas, scores = np.array(probas), np.array(scores)

        if return_probas:
            return probas, scores
        else:
            return scores

    def statistic(
        self,
        P_eval: Tensor,
        x_eval: Tensor,
    ) -> float:
        """Computes the L-C2ST statistic for the observed data.

        Args:
            P_eval: Samples from P (class 0) conditioned on the observation `x_eval`.
                Shape (n_samples, dim_theta)
            x_eval: Observation.
                Shape (n_samples, dim_x)

        Returns:
            L-C2ST statistic at `x_eval`
        """
        return self.scores(P_eval, x_eval).mean()

    def p_value(
        self,
        P_eval: Tensor,
        x_eval: Tensor,
        **kwargs,
    ):
        """Computes the p-value for L-C2ST.

        Args:
            P_eval: Samples from P (class 0) conditioned on the observation `x_eval`.
                Shape (n_samples, dim_theta)
            x_eval: Observation.
                Shape (n_samples, dim_x)
            kwargs: Additional arguments for `compute_stats_null` (n_trials, etc.)

        Returns:
            p-value for L-C2ST at `x_eval`
        """
        stat_data = self.statistic(P_eval, x_eval)
        stats_null = self.compute_stats_null(
            P_eval, x_eval, return_probas=False, **kwargs
        )
        return (stat_data < stats_null).mean()

    def reject(
        self,
        P_eval: Tensor,
        x_eval: Tensor,
        alpha: float = 0.05,
        **kwargs,
    ):
        """Computes the test result for L-C2ST at a given significance level.

        Args:
            P_eval: Samples from P (class 0) conditioned on the observation `x_eval`.
                Shape (n_samples, dim_theta)
            x_eval: Observation.
                Shape (n_samples, dim_x)
            alpha: Significance level.
                Defaults to 0.05.
            kwargs: Additional arguments for `compute_stats_null` (n_trials, etc.)

        Returns:
            True if the null hypothesis is rejected, False otherwise.
        """
        return self.p_value(P_eval, x_eval, **kwargs) < alpha

    def compute_stats_null(
        self,
        P_eval: Tensor,
        x_eval: Tensor,
        n_trials: int = 100,
        permutation: bool = True,
        return_probas: bool = False,
        verbosity: int = 1,
    ):
        """Compute the L-C2ST scores under the null hypothesis (H0).
        Saves the trained classifiers for the null distribution.

        Args:
            P_eval: Samples from P (class 0) conditioned on the observation `x_eval`.
                Shape (n_samples, dim_theta)
            x_eval: Observation.
                Shape (n_samples, dim_x)
            n_trials: Number of trials for the permutation test.
                Defaults to 100.
            permutation: Whether to use the permutation method for (H0).
                Defaults to True.
            return_probas: Whether to return the predicted probabilities of being in P.
                Defaults to False.
            verbosity: Verbosity level, defaults to 1.

        Returns: one of
            scores: L-C2ST scores under (H0).
            (probas, scores): Predicted probabilities and L-C2ST scores under (H0).
        """

        # initialize classifier
        clf = self.clf_class(**self.clf_kwargs)

        if self.trained_clfs_null is not None:
            assert (
                len(self.trained_clfs_null) == n_trials
            ), " You need one classifier per trial"

        trained_clfs_null = []
        probas_null, stats_null = [], []
        for t in tqdm(
            range(n_trials),
            desc=f"Computing T under (H0) - permutation = {permutation}",
            disable=verbosity < 1,
        ):
            # prepare data
            if permutation:
                joint_P = torch.cat([self.P, self.x_P], dim=1)
                joint_Q = torch.cat([self.Q, self.x_Q], dim=1)
                # permute data (same as permuting the labels)
                joint_P_perm, joint_Q_perm = permute_data(joint_P, joint_Q, seed=t)
                # extract the permuted P and Q and x
                P_t, x_P_t = (
                    joint_P_perm[:, : self.P.shape[-1]],
                    joint_P_perm[:, self.P.shape[1] :],
                )
                Q_t, x_Q_t = (
                    joint_Q_perm[:, : self.Q.shape[-1]],
                    joint_Q_perm[:, self.Q.shape[1] :],
                )

                P_eval_t = P_eval

                if self.z_score:
                    P_t = (P_t - self.P_mean) / self.P_std
                    Q_t = (Q_t - self.P_mean) / self.P_std
                    x_P_t = (x_P_t - self.x_P_mean) / self.x_P_std
                    x_Q_t = (x_Q_t - self.x_P_mean) / self.x_P_std

                    P_eval_t = (P_eval - self.P_mean) / self.P_std
                    x_eval = (x_eval - self.x_P_mean) / self.x_P_std
            else:
                assert (
                    self.null_distribution is not None
                ), "You need to provide a null distribution"
                P_t = self.null_distribution.sample((self.P.shape[0],))
                Q_t = self.null_distribution.sample((self.P.shape[0],))
                x_P_t, x_Q_t = self.x_P, self.x_Q

                P_eval_t = self.null_distribution.sample((P_eval.shape[0],))

                if self.z_score:
                    P_mean, P_std = torch.mean(P_t, dim=0), torch.std(P_t, dim=0)
                    P_t = (P_t - P_mean) / P_std
                    Q_t = (Q_t - P_mean) / P_std
                    x_P_t = (x_P_t - self.x_P_mean) / self.x_P_std
                    x_Q_t = (x_Q_t - self.x_P_mean) / self.x_P_std

                    P_eval_t = (P_eval_t - P_mean) / P_std

            # train and evaluate
            if self.trained_clfs_null is not None:
                clf_t = self.trained_clfs_null[t]
            else:
                clf_t = train_lc2st(P_t, Q_t, x_P_t, x_Q_t, clf)
            proba, score = eval_lc2st(P_eval_t, x_eval, clf_t, return_proba=True)
            probas_null.append(proba)
            stats_null.append(score.mean())
            trained_clfs_null.append(clf_t)

        self.trained_clfs_null = trained_clfs_null

        probas_null, stats_null = np.array(probas_null), np.array(stats_null)

        if return_probas:
            return probas_null, stats_null
        else:
            return stats_null


def train_lc2st(P: Tensor, Q: Tensor, x_P: Tensor, x_Q: Tensor, clf: Any) -> Any:
    # cpu and numpy
    P = P.cpu().numpy()
    Q = Q.cpu().numpy()
    x_P = x_P.cpu().numpy()
    x_Q = x_Q.cpu().numpy()

    # concatenate to get joint data
    joint_P = np.concatenate([P, x_P], axis=1)
    joint_Q = np.concatenate([Q, x_Q], axis=1)

    # prepare data
    data = np.concatenate((joint_P, joint_Q))
    # labels
    target = np.concatenate((
        np.zeros((joint_P.shape[0],)),
        np.ones((joint_Q.shape[0],)),
    ))

    # train classifier
    clf_ = clone(clf)
    clf_.fit(data, target)

    return clf_


def eval_lc2st(
    P: Tensor, observation: Tensor, clf: Any, return_proba: bool = False
) -> Tensor:
    # cpu and numpy
    P = P.cpu().numpy()
    observation = observation.cpu().numpy()

    # concatenate to get joint data
    joint_P = np.concatenate([P, observation.repeat(len(P), 1)], axis=1)

    # evaluate classifier
    # probability of being in P (class 0)
    proba = clf.predict_proba(joint_P)[:, 0]
    # mean squared error between proba and dirac at 0.5
    score = ((proba - [0.5] * len(proba)) ** 2).mean()

    if return_proba:
        return proba, score
    else:
        return score


def permute_data(P, Q, seed=1):
    """Permute the concatenated data [P,Q] to create null-hyp samples.

    Args:
        P (torch.Tensor): data of shape (n_samples, dim)
        Q (torch.Tensor): data of shape (n_samples, dim)
        seed (int, optional): random seed. Defaults to 42.
    """
    # set seed
    torch.manual_seed(seed)
    # check inputs
    assert P.shape[0] == Q.shape[0]

    n_samples = P.shape[0]
    X = torch.cat([P, Q], dim=0)
    X_perm = X[torch.randperm(n_samples * 2)]
    return X_perm[:n_samples], X_perm[n_samples:]
