# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
        num_folds: int = 1,
        num_ensemble: int = 1,
        classifier: str = "mlp",
        z_score: bool = False,
        clf_class: Optional[Any] = None,
        clf_kwargs: Optional[Dict[str, Any]] = None,
        num_trials_null: int = 100,
        permutation: bool = True,
    ) -> None:
        """
        L-C2ST: Local Classifier Two-Sample Test
        -----------------------------------------
        Implementation based on the official code from [1] and the exisiting C2ST
        metric [2], using scikit-learn classifiers.

        L-C2ST tests the local consistency of a posterior estimator q w.r.t. to the true
        posterior p, at fixed observation `x_o`, i.e. whether the following null
        hypothesis holds: $H_0(x_o) := q(\theta | x_o) = p(\theta | x_o)$.

        1. Trains a classifier to distinguish between samples from two joint
        distributions [theta_p, x_p] and [theta_q, x_q] and evaluates the L-C2ST
        statistic at a given observation `x_o`.
        2. The L-C2ST statistic is the mean squared error between the predicted
        probabilities of being in p (class 0) and a Dirac at 0.5, which corresponds to
        the chance level of the classifier, unable to distinguish between p and q.
        - If `num_ensemble`>1, the average prediction over all classifiers is used.
        - If `num_folds`>1 the average statistic over all cv-folds is used.

        To evaluate the test, steps 1 and 2 are performed over multiple trials under the
        null hypothesis (H0). If the null distribution is not known, it is estimated
        using the permutation method, i.e. by training the classifier on the permuted
        data. The statistics obtained under (H0) is then compared to the one obtained
        on observed data to compute the p-value, used to decide whether to reject (H0)
        or not.


        Args:
            thetas: Samples from the prior, of shape (sample_size, dim).
            xs: Corresponding simulated data, of shape (sample_size, dim_x).
            posterior_samples: Samples from the estiamted posterior,
                of shape (sample_size, dim)
            seed: Seed for the sklearn classifier and the KFold cross validation,
                defaults to 1.
            num_folds: Number of folds for the cross-validation,
                defaults to 1 (no cross-validation).
                This is useful to reduce variance coming from the data.
            num_ensemble: Number of classifiers for ensembling, defaults to 1.
                This is useful to reduce variance coming from the classifier.
            z_score: Whether to z-score to normalize the data, defaults to False.
            classifier: Classification architecture to use,
                possible values: "random_forest" or "mlp", defaults to "mlp".
            clf_class: Custom sklearn classifier class, defaults to None.
            clf_kwargs: Custom kwargs for the sklearn classifier, defaults to None.
            num_trials_null: Number of trials to estimate the null distribution,
                defaults to 100.
            permutation: Whether to use the permutation method for the null hypothesis,
                defaults to True.

        References:
        [1] : https://arxiv.org/abs/2306.03580, https://github.com/JuliaLinhart/lc2st
        [2] : https://github.com/sbi-dev/sbi/blob/main/sbi/utils/metrics.py
        """

        assert (
            thetas.shape[0] == xs.shape[0] == posterior_samples.shape[0]
        ), "Number of samples must match"

        # set observed data for classification
        self.theta_p = posterior_samples
        self.x_p = xs
        self.theta_q = thetas
        self.x_q = xs

        # z-score normalization parameters
        self.z_score = z_score
        self.theta_p_mean = torch.mean(self.theta_p, dim=0)
        self.theta_p_std = torch.std(self.theta_p, dim=0)
        self.x_p_mean = torch.mean(self.x_p, dim=0)
        self.x_p_std = torch.std(self.x_p, dim=0)

        # set parameters for classifier training
        self.seed = seed
        self.num_folds = num_folds
        self.num_ensemble = num_ensemble

        # initialize classifier
        if "mlp" in classifier.lower():
            ndim = thetas.shape[-1]
            self.clf_class = MLPClassifier
            if clf_kwargs is None:
                self.clf_kwargs = {
                    "activation": "relu",
                    "hidden_layer_sizes": (10 * ndim, 10 * ndim),
                    "max_iter": 1000,
                    "solver": "adam",
                    "early_stopping": True,
                    "n_iter_no_change": 50,
                }
        elif "random_forest" in classifier.lower():
            self.clf_class = RandomForestClassifier
            if clf_kwargs is None:
                self.clf_kwargs = {}
        elif "custom":
            if clf_class is None or clf_kwargs is None:
                raise ValueError(
                    "Please provide a valid sklearn classifier class and kwargs."
                )
            self.clf_class = clf_class
            self.clf_kwargs = clf_kwargs
        else:
            raise NotImplementedError

        # initialize classifiers, will be set after training
        self.trained_clfs = None
        self.trained_clfs_null = None

        # parameters for the null hypothesis testing
        self.num_trials_null = num_trials_null
        self.permutation = permutation
        # can be specified if known and independent of x (see `LC2ST-NF`)
        self.null_distribution = None

    def _train(
        self,
        theta_p: Tensor,
        theta_q: Tensor,
        x_p: Tensor,
        x_q: Tensor,
        verbosity: int = 0,
    ) -> List[Any]:
        """Returns the classifiers trained on observed data.

        Args:
            theta_p: Samples from P, of shape (sample_size, dim).
            theta_q: Samples from Q, of shape (sample_size, dim).
            x_p: Observations corresponding to P, of shape (sample_size, dim_x).
            x_q: Observations corresponding to Q, of shape (sample_size, dim_x).
            verbosity: Verbosity level, defaults to 0.

        Returns:
            List of trained classifiers for each cv fold.
        """

        # prepare data

        if self.z_score:
            theta_p = (theta_p - self.theta_p_mean) / self.theta_p_std
            theta_q = (theta_q - self.theta_p_mean) / self.theta_p_std
            x_p = (x_p - self.x_p_mean) / self.x_p_std
            x_q = (x_q - self.x_p_mean) / self.x_p_std

        # initialize classifier
        clf = self.clf_class(**self.clf_kwargs or {})

        if self.num_ensemble > 1:
            clf = EnsembleClassifier(clf, self.num_ensemble, verbosity=verbosity)

        # cross-validation
        if self.num_folds > 1:
            trained_clfs = []
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            cv_splits = kf.split(theta_p.numpy())
            for train_idx, _ in tqdm(
                cv_splits, desc="Cross-validation", disable=verbosity < 1
            ):
                # get train split
                theta_p_train, theta_q_train = theta_p[train_idx], theta_q[train_idx]
                x_p_train, x_q_train = x_p[train_idx], x_q[train_idx]

                # train classifier
                clf_n = train_lc2st(
                    theta_p_train, theta_q_train, x_p_train, x_q_train, clf
                )

                trained_clfs.append(clf_n)
        else:
            # train single classifier
            clf = train_lc2st(theta_p, theta_q, x_p, x_q, clf)
            trained_clfs = [clf]

        return trained_clfs

    def get_scores(
        self,
        theta_o: Tensor,
        x_o: Tensor,
        trained_clfs: List[Any],
        return_probs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Computes the L-C2ST scores given the trained classifiers:
        Mean squared error (MSE) between 0.5 and the predicted probabilities
        of being in class 0 over the dataset (`theta_o`, `x_o`).

        Args:
            theta_o: Samples from the posterior conditioned on the observation `x_o`,
                of shape (sample_size, dim).
            x_o: The observation, of shape (,dim_x).
            trained_clfs: List of trained classifiers, of length `num_folds`.
            return_probs: Whether to return the predicted probabilities of being in P,
                defaults to False.

        Returns: one of
            scores: L-C2ST scores at `x_o`, of shape (`num_folds`,).
            (probs, scores): Predicted probabilities and L-C2ST scores at `x_o`,
                each of shape (`num_folds`,).
        """
        # prepare data
        if self.z_score:
            theta_o = (theta_o - self.theta_p_mean) / self.theta_p_std
            x_o = (x_o - self.x_p_mean) / self.x_p_std

        probs, scores = [], []

        # evaluate classifiers
        for clf in trained_clfs:
            proba, score = eval_lc2st(theta_o, x_o, clf, return_proba=True)
            probs.append(proba)
            scores.append(score)
        probs, scores = np.array(probs), np.array(scores)

        if return_probs:
            return probs, scores
        else:
            return scores

    def train_on_observed_data(
        self, seed: Optional[int] = None, verbosity: int = 1
    ) -> Union[None, List[Any]]:
        """Trains the classifier on the observed data.
        Saves the trained classifier(s) as a list of length `num_folds`.

        Args:
            seed: random state of the classifier, defaults to None.
            verbosity: Verbosity level, defaults to 1.
        """
        # set random state
        if seed is not None:
            if "random_state" in self.clf_kwargs:
                print("WARNING: changing the random state of the classifier.")
            self.clf_kwargs["random_state"] = seed  # type: ignore

        # train the classifier
        trained_clfs = self._train(
            self.theta_p, self.theta_q, self.x_p, self.x_q, verbosity=verbosity
        )
        self.trained_clfs = trained_clfs

    def get_statistic_on_observed_data(
        self,
        theta_o: Tensor,
        x_o: Tensor,
    ) -> float:
        """Computes the L-C2ST statistics for the observed data:
        Mean over all cv-scores.

        Args:
            theta_o: Samples from the posterior conditioned on the observation `x_o`,
                of shape (sample_size, dim).
            x_o: The observation, of shape (, dim_x)

        Returns:
            L-C2ST statistic at `x_o`.
        """
        assert (
            self.trained_clfs is not None
        ), "No trained classifiers found. Run `train_on_observed_data` first."
        _, scores = self.get_scores(
            theta_o=theta_o,
            x_o=x_o,
            trained_clfs=self.trained_clfs,
            return_probs=True,
        )
        return scores.mean()

    def p_value(
        self,
        theta_o: Tensor,
        x_o: Tensor,
    ) -> float:
        r"""Computes the p-value for L-C2ST.

        The p-value is the proportion of times the L-C2ST statistic under the null
        hypothesis is greater than the L-C2ST statistic at the observation `x_o`.
        It is computed by taking the empirical mean over statistics computed on
        several trials under the null hypothesis: $1/H \sum_{h=1}^{H} I(T_h < T_o)$.

        Args:
            theta_o: Samples from the posterior conditioned on the observation `x_o`,
                of dhape (sample_size, dim).
            x_o: The observation, of shape (, dim_x).

        Returns:
            p-value for L-C2ST at `x_o`.
        """
        stat_data = self.get_statistic_on_observed_data(theta_o=theta_o, x_o=x_o)
        _, stats_null = self.get_statistics_under_null_hypothesis(
            theta_o=theta_o, x_o=x_o, return_probs=True, verbosity=0
        )
        return (stat_data < stats_null).mean()

    def reject_test(
        self,
        theta_o: Tensor,
        x_o: Tensor,
        alpha: float = 0.05,
    ) -> bool:
        """Computes the test result for L-C2ST at a given significance level.

        Args:
            theta_o: Samples from the posterior conditioned on the observation `x_o`,
                of shape (sample_size, dim).
            x_o: The observation, of shape (, dim_x).
            alpha: Significance level, defaults to 0.05.

        Returns:
            The L-C2ST result: True if rejected, False otherwise.
        """
        return self.p_value(theta_o=theta_o, x_o=x_o) < alpha

    def train_under_null_hypothesis(
        self,
        verbosity: int = 1,
    ) -> None:
        """Computes the L-C2ST scores under the null hypothesis (H0).
        Saves the trained classifiers for each null trial.

        Args:
            verbosity: Verbosity level, defaults to 1.
        """

        trained_clfs_null = {}
        for t in tqdm(
            range(self.num_trials_null),
            desc=f"Training the classifiers under H0, permutation = {self.permutation}",
            disable=verbosity < 1,
        ):
            # prepare data
            if self.permutation:
                joint_p = torch.cat([self.theta_p, self.x_p], dim=1)
                joint_q = torch.cat([self.theta_q, self.x_q], dim=1)
                # permute data (same as permuting the labels)
                joint_p_perm, joint_q_perm = permute_data(joint_p, joint_q, seed=t)
                # extract the permuted P and Q and x
                theta_p_t, x_p_t = (
                    joint_p_perm[:, : self.theta_p.shape[-1]],
                    joint_p_perm[:, self.theta_p.shape[1] :],
                )
                theta_q_t, x_q_t = (
                    joint_q_perm[:, : self.theta_q.shape[-1]],
                    joint_q_perm[:, self.theta_q.shape[1] :],
                )
            else:
                assert (
                    self.null_distribution is not None
                ), "You need to provide a null distribution"
                theta_p_t = self.null_distribution.sample((self.theta_p.shape[0],))
                theta_q_t = self.null_distribution.sample((self.theta_p.shape[0],))
                x_p_t, x_q_t = self.x_p, self.x_q

            if self.z_score:
                theta_p_t = (theta_p_t - self.theta_p_mean) / self.theta_p_std
                theta_q_t = (theta_q_t - self.theta_p_mean) / self.theta_p_std
                x_p_t = (x_p_t - self.x_p_mean) / self.x_p_std
                x_q_t = (x_q_t - self.x_p_mean) / self.x_p_std

            # train
            clf_t = self._train(theta_p_t, theta_q_t, x_p_t, x_q_t, verbosity=0)
            trained_clfs_null[t] = clf_t

        self.trained_clfs_null = trained_clfs_null

    def get_statistics_under_null_hypothesis(
        self,
        theta_o: Tensor,
        x_o: Tensor,
        return_probs: bool = False,
        verbosity: int = 0,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Computes the L-C2ST scores under the null hypothesis.

        Args:
            theta_o: Samples from the posterior conditioned on the observation `x_o`,
                of shape (sample_size, dim).
            x_o: The observation, of shape (, dim_x).
            return_probs: Whether to return the predicted probabilities of being in P,
                defaults to False.
            verbosity: Verbosity level, defaults to 1.

        Returns: one of
            scores: L-C2ST scores under (H0).
            (probs, scores): Predicted probabilities and L-C2ST scores under (H0).
        """

        if self.trained_clfs_null is None:
            raise ValueError(
                "You need to train the classifiers under (H0). \
                    Run `train_under_null_hypothesis`."
            )
        else:
            assert (
                len(self.trained_clfs_null) == self.num_trials_null
            ), "You need one classifier per trial."

        probs_null, stats_null = [], []
        for t in tqdm(
            range(self.num_trials_null),
            desc=f"Computing T under (H0) - permutation = {self.permutation}",
            disable=verbosity < 1,
        ):
            # prepare data
            if self.permutation:
                theta_o_t = theta_o
            else:
                assert (
                    self.null_distribution is not None
                ), "You need to provide a null distribution"

                theta_o_t = self.null_distribution.sample((theta_o.shape[0],))

            if self.z_score:
                theta_o_t = (theta_o_t - self.theta_p_mean) / self.theta_p_std
                x_o = (x_o - self.x_p_mean) / self.x_p_std

            # evaluate
            clf_t = self.trained_clfs_null[t]
            probs, scores = self.get_scores(
                theta_o=theta_o_t, x_o=x_o, trained_clfs=clf_t, return_probs=True
            )
            probs_null.append(probs)
            stats_null.append(scores.mean())

        probs_null, stats_null = np.array(probs_null), np.array(stats_null)

        if return_probs:
            return probs_null, stats_null
        else:
            return stats_null


class LC2ST_NF(LC2ST):
    def __init__(
        self,
        thetas: Tensor,
        xs: Tensor,
        posterior_samples: Tensor,
        flow_inverse_transform: Callable[[Tensor, Tensor], Tensor],
        flow_base_dist: torch.distributions.Distribution,
        num_eval: int = 10_000,
        trained_clfs_null: Optional[Dict[str, List[Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        L-C2ST for Normalizing Flows.

        LC2ST_NF is a subclass of LC2ST that performs the test in the space of the
        base distribution of a normalizing flow. It uses the inverse transform of the
        normalizing flow $T_\\phi^{-1}$ to map the samples from the prior and the
        posterior to the base distribution space. Following Theorem 4, Eq. 17 from [1],
        the new null hypothesis for a Gaussian base distribution is:

            $H_0(x_o) := p(T_\\phi^{-1}(\theta ; x_o) | x_o) = N(0, I_m)$.

        This is because a sample from the NF is a sample from the base distribution
        pushed through the flow:

            $z = T_\\phi^{-1}(\\theta) \\sim N(0, I_m) \\iff theta = T_\\phi(z)$.

        This defines the two classes P and Q for the L-C2ST test, one of which is
        the Gaussion distribution that can be easily be sampled from and is independent
        of the observation `x_o` and the estimator q.

        Important features are:
        - no `theta_o` is passed to the evaluation functions (e.g. `get_scores`),
            as the base distribution is known, samples are drawn at initialization.
        - no permutation method is used, as the null distribution is known,
            samples are drawn during `train_under_null_hypothesis`.
        - the classifiers can be pre-trained under the null and `trained_clfs_null`
            passed as an argument at initialization. They do not depend on the
            observed data (i.e. `posterior_samples` and `xs`).

        Args:
            thetas: Samples from the prior, of shape (sample_size, dim).
            xs: Corresponding simulated data, of shape (sample_size, dim_x).
            posterior_samples: Samples from the estiamted posterior,
                of shape (sample_size, dim).
            flow_inverse_transform: Inverse transform of the normalizing flow.
                Takes thetas and xs as input and returns noise.
            flow_base_dist: Base distribution of the normalizing flow.
            num_eval: Number of samples to evaluate the L-C2ST.
            trained_clfs_null: Pre-trained classifiers under the null.
            kwargs: Additional arguments for the LC2ST class.

        References:
        [1] : https://arxiv.org/abs/2306.03580, https://github.com/JuliaLinhart/lc2st
        """
        # Aplly the inverse transform to the thetas and the posterior samples
        self.flow_inverse_transform = flow_inverse_transform
        inverse_thetas = flow_inverse_transform(thetas, xs).detach()
        inverse_posterior_samples = flow_inverse_transform(
            posterior_samples, xs
        ).detach()

        # Initialize the LC2ST class with the inverse transformed samples
        super().__init__(inverse_thetas, xs, inverse_posterior_samples, **kwargs)

        # Set the parameters for the null hypothesis testing
        self.null_distribution = flow_base_dist
        self.permutation = False
        self.trained_clfs_null = trained_clfs_null

        # Draw samples from the base distribution for evaluation
        self.theta_o = flow_base_dist.sample(torch.Size([num_eval]))

    def get_scores(
        self,
        x_o: Tensor,
        trained_clfs: List[Any],
        return_probs: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Computes the L-C2ST scores given the trained classifiers.

        Args:
            x_o: The observation, of shape (,dim_x).
            trained_clfs: Trained classifiers.
            return_probs: Whether to return the predicted probabilities of being in P,
                defaults to False.
            kwargs: Additional arguments used in the parent class.

        Returns: one of
            scores: L-C2ST scores at `x_o`.
            (probs, scores): Predicted probabilities and L-C2ST scores at `x_o`.
        """
        return super().get_scores(
            theta_o=self.theta_o,
            x_o=x_o,
            trained_clfs=trained_clfs,
            return_probs=return_probs,
        )

    def get_statistic_on_observed_data(
        self,
        x_o: Tensor,
        **kwargs: Any,
    ) -> float:
        """Computes the L-C2ST statistics for the observed data:
        Mean over all cv-scores.

        Args:
            x_o: The observation, of shape (, dim_x).
            kwargs: Additional arguments used in the parent class.

        Returns:
            L-C2ST statistic at `x_o`.
        """
        return super().get_statistic_on_observed_data(theta_o=self.theta_o, x_o=x_o)

    def p_value(
        self,
        x_o: Tensor,
        **kwargs: Any,
    ) -> float:
        r"""Computes the p-value for L-C2ST.

        The p-value is the proportion of times the L-C2ST statistic under the null
        hypothesis is greater than the L-C2ST statistic at the observation `x_o`.
        It is computed by taking the empirical mean over statistics computed on
        several trials under the null hypothesis: $1/H \sum_{h=1}^{H} I(T_h < T_o)$.

        Args:
            x_o: The observation, of shape (, dim_x).
            kwargs: Additional arguments used in the parent class.

        Returns:
            p-value for L-C2ST at `x_o`.
        """
        return super().p_value(theta_o=self.theta_o, x_o=x_o)

    def reject_test(
        self,
        x_o: Tensor,
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> bool:
        """Computes the test result for L-C2ST at a given significance level.

        Args:
            x_o: The observation, of shape (, dim_x).
            alpha: Significance level, defaults to 0.05.
            kwargs: Additional arguments used in the parent class.

        Returns:
            L-C2ST result: True if rejected, False otherwise.
        """
        return super().reject_test(theta_o=self.theta_o, x_o=x_o, alpha=alpha)

    def train_under_null_hypothesis(
        self,
        verbosity: int = 1,
    ) -> None:
        """Computes the L-C2ST scores under the null hypothesis.
        Saves the trained classifiers for the null distribution.

        Args:
            verbosity: Verbosity level, defaults to 1.
        """
        if self.trained_clfs_null is not None:
            raise ValueError(
                "Classifiers have already been trained under the null \
                    and can be used to evaluate any new estimator."
            )
        return super().train_under_null_hypothesis(verbosity=verbosity)

    def get_statistics_under_null_hypothesis(
        self,
        x_o: Tensor,
        return_probs: bool = False,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Computes the L-C2ST scores under the null hypothesis.

        Args:
            x_o: The observation.
                Shape (, dim_x)
            return_probs: Whether to return the predicted probabilities of being in P.
                Defaults to False.
            verbosity: Verbosity level, defaults to 1.
            kwargs: Additional arguments used in the parent class.
        """
        return super().get_statistics_under_null_hypothesis(
            theta_o=self.theta_o,
            x_o=x_o,
            return_probs=return_probs,
            verbosity=verbosity,
        )


def train_lc2st(
    theta_p: Tensor, theta_q: Tensor, x_p: Tensor, x_q: Tensor, clf: BaseEstimator
) -> Any:
    """Trains the classifier on the joint data for the L-C2ST.

    Args:
        theta_p: Samples from P, of shape (sample_size, dim).
        theta_q: Samples from Q, of shape (sample_size, dim).
        x_p: Observations corresponding to P, of shape (sample_size, dim_x).
        x_q: Observations corresponding to Q, of shape (sample_size, dim_x).
        clf: Classifier to train.

    Returns:
        Trained classifier.
    """
    # concatenate to get joint data
    joint_p = np.concatenate([theta_p.cpu().numpy(), x_p.cpu().numpy()], axis=1)
    joint_q = np.concatenate([theta_q.cpu().numpy(), x_q.cpu().numpy()], axis=1)

    # prepare data
    data = np.concatenate((joint_p, joint_q))
    # labels
    target = np.concatenate((
        np.zeros((joint_p.shape[0],)),
        np.ones((joint_q.shape[0],)),
    ))

    # train classifier
    clf_ = clone(clf)
    clf_.fit(data, target)  # type: ignore

    return clf_


def eval_lc2st(
    theta_p: Tensor, x_o: Tensor, clf: BaseEstimator, return_proba: bool = False
) -> Union[float, Tuple[np.ndarray, float]]:
    """Evaluates the classifier returned by `train_lc2st` for one observation
    `x_o` and over the samples `P`.

    Args:
        theta_p: Samples from p (class 0), of shape (sample_size, dim).
        x_o: The observation, of shape (, dim_x).
        clf: Trained classifier.
        return_proba: Whether to return the predicted probabilities of being in P,
            defaults to False.

    Returns:
        L-C2ST score at `x_o`: MSE between 0.5 and the predicted classifier
        probability for class 0 on `theta_p`.
    """
    # concatenate to get joint data
    joint_p = np.concatenate(
        [theta_p.cpu().numpy(), x_o.repeat(len(theta_p), 1).cpu().numpy()], axis=1
    )

    # evaluate classifier
    # probability of being in P (class 0)
    proba = clf.predict_proba(joint_p)[:, 0]  # type: ignore
    # mean squared error between proba and dirac at 0.5
    score = ((proba - [0.5] * len(proba)) ** 2).mean()

    if return_proba:
        return proba, score
    else:
        return score


def permute_data(
    theta_p: Tensor, theta_q: Tensor, seed: int = 1
) -> Tuple[Tensor, Tensor]:
    """Permutes the concatenated data [P,Q] to create null samples.

    Args:
        theta_p: samples from P, of shape (sample_size, dim).
        theta_q: samples from Q, of shape (sample_size, dim).
        seed: random seed, defaults to 1.

    Returns:
        Permuted data [theta_p,theta_qss]
    """
    # set seed
    torch.manual_seed(seed)
    # check inputs
    assert theta_p.shape[0] == theta_q.shape[0]

    sample_size = theta_p.shape[0]
    X = torch.cat([theta_p, theta_q], dim=0)
    x_perm = X[torch.randperm(sample_size * 2)]
    return x_perm[:sample_size], x_perm[sample_size:]


class EnsembleClassifier(BaseEstimator):
    def __init__(self, clf, num_ensemble=1, verbosity=1):
        self.clf = clf
        self.num_ensemble = num_ensemble
        self.trained_clfs = []
        self.verbosity = verbosity

    def fit(self, X, y):
        for n in tqdm(
            range(self.num_ensemble),
            desc="Ensemble training",
            disable=self.verbosity < 1,
        ):
            clf = clone(self.clf)
            if clf.random_state is not None:  # type: ignore
                clf.random_state += n  # type: ignore
            else:
                clf.random_state = n + 1  # type: ignore
            clf.fit(X, y)  # type: ignore
            self.trained_clfs.append(clf)

    def predict_proba(self, X):
        probas = [clf.predict_proba(X) for clf in self.trained_clfs]
        return np.mean(probas, axis=0)
