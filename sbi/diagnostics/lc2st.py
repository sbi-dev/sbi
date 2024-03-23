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
        n_folds: int = 1,
        classifier: str = "mlp",
        z_score: bool = False,
        clf_class: Optional[Any] = None,
        clf_kwargs: Optional[Dict[str, Any]] = None,
        n_trials_null: int = 100,
        permutation: bool = True,
    ) -> None:
        """
        L-C2ST: Local Classifier Two-Sample Test
        -----------------------------------------
        Implementation based on the official code from [1] and the exisiting C2ST
        metric [2], using scikit-learn classifiers.

        Args:
            thetas: Samples from the prior, of shape (sample_size, dim).
            xs: Corresponding simulated data, of shape (sample_size, dim_x).
            posterior_samples: Samples from the estiamted posterior,
                of shape (sample_size, dim)
            seed: Seed for the sklearn classifier and the KFold cross validation,
                defaults to 1.
            n_folds: Number of folds for the cross-validation,
                defaults to 1 (no cross-validation).
            z_score: Whether to z-score to normalize the data, defaults to False.
            classifier: Classification architecture to use,
                possible values: "rf" or "mlp", defaults to "mlp".
            clf_class: Custom sklearn classifier class, defaults to None.
            clf_kwargs: Custom kwargs for the sklearn classifier, defaults to None.
            n_trials_null: Number of trials to estimate the null distribution,
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
        self.P = posterior_samples
        self.x_P = xs
        self.Q = thetas
        self.x_Q = xs

        # z-score normalization parameters
        self.z_score = z_score
        self.P_mean = torch.mean(self.P, dim=0)
        self.P_std = torch.std(self.P, dim=0)
        self.x_P_mean = torch.mean(self.x_P, dim=0)
        self.x_P_std = torch.std(self.x_P, dim=0)

        # set parameters for classifier training
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
        elif "custom":
            if clf_class is None:
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
        self.n_trials_null = n_trials_null
        self.permutation = permutation
        # can be specified if known and independent of x (see `LC2ST-NF`)
        self.null_distribution = None

    def _train(
        self, P: Tensor, Q: Tensor, x_P: Tensor, x_Q: Tensor, verbosity: int = 0
    ) -> List[Any]:
        """Returns the classifiers trained on observed data.

        Args:
            P: Samples from P, of shape (sample_size, dim).
            Q: Samples from Q, of shape (sample_size, dim).
            x_P: Observations corresponding to P, of shape (sample_size, dim_x).
            x_Q: Observations corresponding to Q, of shape (sample_size, dim_x).
            verbosity: Verbosity level, defaults to 0.

        Returns:
            List of trained classifiers for each cv fold.
        """

        # prepare data

        if self.z_score:
            P = (P - self.P_mean) / self.P_std
            Q = (Q - self.P_mean) / self.P_std
            x_P = (x_P - self.x_P_mean) / self.x_P_std
            x_Q = (x_Q - self.x_P_mean) / self.x_P_std

        # initialize classifier
        clf = self.clf_class(**self.clf_kwargs or {})

        # cross-validation
        if self.n_folds > 1:
            trained_clfs = []
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            cv_splits = kf.split(self.P)
            for train_idx, _ in tqdm(
                cv_splits, desc="Cross-validation", disable=verbosity < 1
            ):
                # get train split
                P_train, Q_train = P[train_idx], Q[train_idx]
                x_P_train, x_Q_train = x_P[train_idx], x_Q[train_idx]

                # train classifier
                clf_n = train_lc2st(P_train, Q_train, x_P_train, x_Q_train, clf)

                trained_clfs.append(clf_n)
        else:
            # train single classifier
            clf = train_lc2st(P, Q, x_P, x_Q, clf)
            trained_clfs = [clf]

        return trained_clfs

    def _scores(
        self,
        P_eval: Tensor,
        x_eval: Tensor,
        trained_clfs: List[Any],
        return_probas: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute the L-C2ST scores given the trained classifiers.

        Args:
            P_eval: Samples from P conditioned on the observation `x_eval`,
                of shape (sample_size, dim).
            x_eval: Observation, of shape (,dim_x).
            trained_clfs: Trained classifiers.
            return_probas: Whether to return the predicted probabilities of being in P,
                defaults to False.

        Returns: one of
            scores: L-C2ST scores at `x_eval`.
            (probas, scores): Predicted probabilities and L-C2ST scores at `x_eval`.
        """
        # prepare data
        if self.z_score:
            P_eval = (P_eval - self.P_mean) / self.P_std
            x_eval = (x_eval - self.x_P_mean) / self.x_P_std

        probas, scores = [], []

        # evaluate classifiers
        for clf in trained_clfs:
            proba, score = eval_lc2st(P_eval, x_eval, clf, return_proba=True)
            probas.append(proba)
            scores.append(score)
        probas, scores = np.array(probas), np.array(scores)

        if return_probas:
            return probas, scores
        else:
            return scores

    def train_data(self) -> None:
        """Train the classifiers on the observed data.
        Saves the trained classifiers.
        """
        trained_clfs = self._train(self.P, self.Q, self.x_P, self.x_Q)
        self.trained_clfs = trained_clfs

    def scores_data(
        self, P_eval: Tensor, x_eval: Tensor, return_probas: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute the L-C2ST scores for the observed data.

        Args:
            P_eval: Samples from P conditioned on the observation `x_eval`,
                of shape (sample_size, dim).
            x_eval: Observation, of shape (,dim_x).
            return_probas: Whether to return the predicted probabilities of being in P,
                defaults to False.

        Returns: one of
            scores: L-C2ST scores at `x_eval`.
            (probas, scores): Predicted probabilities and L-C2ST scores at `x_eval`.
        """
        assert (
            self.trained_clfs is not None
        ), "No trained classifiers found. Run `train_data` first."
        return self._scores(
            P_eval=P_eval,
            x_eval=x_eval,
            trained_clfs=self.trained_clfs,
            return_probas=return_probas,
        )

    def statistic_data(
        self,
        P_eval: Tensor,
        x_eval: Tensor,
        return_probas: bool = False,
    ) -> Union[float, Tuple[np.ndarray, float]]:
        """Computes the L-C2ST statistic for the observed data.

        Args:
            P_eval: Samples from P conditioned on the observation `x_eval`,
                of shape (sample_size, dim).
            x_eval: Observation, of shape (, dim_x)
            return_probas: Whether to return the predicted probabilities of being in P,
                defaults to False.

        Returns:
            L-C2ST statistic at `x_eval`.
        """
        probas, scores = self.scores_data(
            P_eval=P_eval, x_eval=x_eval, return_probas=True
        )
        if return_probas:
            return probas, scores.mean()
        else:
            return scores.mean()

    def p_value(
        self,
        P_eval: Tensor,
        x_eval: Tensor,
    ) -> float:
        """Computes the p-value for L-C2ST.

        Args:
            P_eval: Samples from P conditioned on the observation `x_eval`,
                of dhape (sample_size, dim).
            x_eval: Observation, of shape (, dim_x).

        Returns:
            p-value for L-C2ST at `x_eval`.
        """
        _, stat_data = self.statistic_data(
            P_eval=P_eval, x_eval=x_eval, return_probas=True
        )
        _, stats_null = self.statistics_null(
            P_eval=P_eval, x_eval=x_eval, return_probas=True, verbosity=0
        )
        return (stat_data < stats_null).mean()

    def reject(
        self,
        P_eval: Tensor,
        x_eval: Tensor,
        alpha: float = 0.05,
    ) -> bool:
        """Computes the test result for L-C2ST at a given significance level.

        Args:
            P_eval: Samples from P conditioned on the observation `x_eval`,
                of shape (sample_size, dim).
            x_eval: Observation, of shape (, dim_x).
            alpha: Significance level, defaults to 0.05.

        Returns:
            The L-C2ST result: True if rejected, False otherwise.
        """
        return self.p_value(P_eval=P_eval, x_eval=x_eval) < alpha

    def train_null(
        self,
        verbosity: int = 1,
    ) -> None:
        """Compute the L-C2ST scores under the null hypothesis (H0).
        Saves the trained classifiers for each null trial.

        Args:
            verbosity: Verbosity level, defaults to 1.
        """

        trained_clfs_null = {}
        for t in tqdm(
            range(self.n_trials_null),
            desc=f"Training the classifiers under H0, permutation = {self.permutation}",
            disable=verbosity < 1,
        ):
            # prepare data
            if self.permutation:
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
            else:
                assert (
                    self.null_distribution is not None
                ), "You need to provide a null distribution"
                P_t = self.null_distribution.sample((self.P.shape[0],))
                Q_t = self.null_distribution.sample((self.P.shape[0],))
                x_P_t, x_Q_t = self.x_P, self.x_Q

            if self.z_score:
                P_t = (P_t - self.P_mean) / self.P_std
                Q_t = (Q_t - self.P_mean) / self.P_std
                x_P_t = (x_P_t - self.x_P_mean) / self.x_P_std
                x_Q_t = (x_Q_t - self.x_P_mean) / self.x_P_std

            # train
            clf_t = self._train(P_t, Q_t, x_P_t, x_Q_t, verbosity=0)
            trained_clfs_null[t] = clf_t

        self.trained_clfs_null = trained_clfs_null

    def statistics_null(
        self,
        P_eval: Tensor,
        x_eval: Tensor,
        return_probas: bool = False,
        verbosity: int = 0,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute the L-C2ST scores under the null hypothesis.

        Args:
            P_eval: Samples from P conditioned on the observation `x_eval`,
                of shape (sample_size, dim).
            x_eval: Observation, of shape (, dim_x).
            return_probas: Whether to return the predicted probabilities of being in P,
                defaults to False.
            verbosity: Verbosity level, defaults to 1.

        Returns: one of
            scores: L-C2ST scores under (H0).
            (probas, scores): Predicted probabilities and L-C2ST scores under (H0).
        """

        if self.trained_clfs_null is None:
            raise ValueError(
                "You need to train the classifiers under (H0). Run `train_null`."
            )
        else:
            assert (
                len(self.trained_clfs_null) == self.n_trials_null
            ), " You need one classifier per trial."

        probas_null, stats_null = [], []
        for t in tqdm(
            range(self.n_trials_null),
            desc=f"Computing T under (H0) - permutation = {self.permutation}",
            disable=verbosity < 1,
        ):
            # prepare data
            if self.permutation:
                P_eval_t = P_eval
            else:
                assert (
                    self.null_distribution is not None
                ), "You need to provide a null distribution"

                P_eval_t = self.null_distribution.sample((P_eval.shape[0],))

            if self.z_score:
                P_eval_t = (P_eval_t - self.P_mean) / self.P_std
                x_eval = (x_eval - self.x_P_mean) / self.x_P_std

            # evaluate
            clf_t = self.trained_clfs_null[t]
            proba, score = self._scores(
                P_eval=P_eval_t, x_eval=x_eval, trained_clfs=clf_t, return_probas=True
            )
            probas_null.append(proba)
            stats_null.append(score.mean())

        probas_null, stats_null = np.array(probas_null), np.array(stats_null)

        if return_probas:
            return probas_null, stats_null
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
        n_eval: int = 10_000,
        trained_clfs_null: Optional[Dict[str, List[Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        L-C2ST for normalizing flows.

        LC2ST_NF is a subclass of LC2ST that performs the test in the space of the
        base distribution of a normalizing flow. It uses the inverse transform of the
        normalizing flow to map the samples from the prior and the posterior to the
        base distribution space. Important features are:

            - the null distribution is the base distribution of the normalizing flow.
            - no `P_eval` is passed to the evaluation functions (e.g. `_scores`),
                as the base distribution is known, samples are drawn at initialization.
            - no permutation method is used, as the null distribution is known.
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
            n_eval: Number of samples to evaluate the L-C2ST.
            trained_clfs_null: Pre-trained classifiers under the null.
            kwargs: Additional arguments for the LC2ST class.
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
        self.P_eval = flow_base_dist.sample(torch.Size([n_eval]))

    def _scores(
        self,
        x_eval: Tensor,
        trained_clfs: List[Any],
        return_probas: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute the L-C2ST scores given the trained classifiers.

        Args:
            x_eval: Observation, of shape (,dim_x).
            trained_clfs: Trained classifiers.
            return_probas: Whether to return the predicted probabilities of being in P,
                defaults to False.
            kwargs: Additional arguments used in the parent class.

        Returns: one of
            scores: L-C2ST scores at `x_eval`.
            (probas, scores): Predicted probabilities and L-C2ST scores at `x_eval`.
        """
        return super()._scores(
            P_eval=self.P_eval,
            x_eval=x_eval,
            trained_clfs=trained_clfs,
            return_probas=return_probas,
        )

    def scores_data(
        self,
        x_eval: Tensor,
        return_probas: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute the L-C2ST scores for the observed data.

        Args:
            x_eval: Observation, of shape (,dim_x).
            return_probas: Whether to return the predicted probabilities of being in P,
                defaults to False.
            kwargs: Additional arguments used in the parent class.

        Returns: one of
            scores: L-C2ST scores at `x_eval`.
            (probas, scores): Predicted probabilities and L-C2ST scores at `x_eval`.
        """
        return super().scores_data(
            P_eval=self.P_eval, x_eval=x_eval, return_probas=return_probas
        )

    def statistic_data(
        self,
        x_eval: Tensor,
        return_probas: bool = False,
        **kwargs: Any,
    ) -> Union[float, Tuple[np.ndarray, float]]:
        """Computes the L-C2ST statistic for the observed data.

        Args:
            x_eval: Observation, of shape (, dim_x).
            kwargs: Additional arguments used in the parent class.
            return_probas: Whether to return the predicted probabilities of being in P,

        Returns:
            L-C2ST statistic at `x_eval`.
        """
        return super().statistic_data(
            P_eval=self.P_eval, x_eval=x_eval, return_probas=return_probas
        )

    def p_value(
        self,
        x_eval: Tensor,
        **kwargs: Any,
    ) -> float:
        """Computes the p-value for L-C2ST.

        Args:
            x_eval: Observation, of shape (, dim_x).
            kwargs: Additional arguments used in the parent class.

        Returns:
            p-value for L-C2ST at `x_eval`.
        """
        return super().p_value(P_eval=self.P_eval, x_eval=x_eval)

    def reject(
        self,
        x_eval: Tensor,
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> bool:
        """Computes the test result for L-C2ST at a given significance level.

        Args:
            x_eval: Observation, of shape (, dim_x).
            alpha: Significance level, defaults to 0.05.
            kwargs: Additional arguments used in the parent class.

        Returns:
            L-C2ST result: True if rejected, False otherwise.
        """
        return super().reject(P_eval=self.P_eval, x_eval=x_eval, alpha=alpha)

    def train_null(
        self,
        verbosity: int = 1,
    ) -> None:
        """Compute the L-C2ST scores under the null hypothesis.
        Saves the trained classifiers for the null distribution.

        Args:
            verbosity: Verbosity level, defaults to 1.
        """
        if self.trained_clfs_null is not None:
            raise ValueError(
                "Classifiers have already been trained under the null \
                    and can be used to evaluate any new estimator."
            )
        return super().train_null(verbosity=verbosity)

    def statistics_null(
        self,
        x_eval: Tensor,
        return_probas: bool = False,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute the L-C2ST scores under the null hypothesis.

        Args:
            x_eval: Observation.
                Shape (, dim_x)
            return_probas: Whether to return the predicted probabilities of being in P.
                Defaults to False.
            verbosity: Verbosity level, defaults to 1.
            kwargs: Additional arguments used in the parent class.
        """
        return super().statistics_null(
            P_eval=self.P_eval,
            x_eval=x_eval,
            return_probas=return_probas,
            verbosity=verbosity,
        )


def train_lc2st(
    P: Tensor, Q: Tensor, x_P: Tensor, x_Q: Tensor, clf: BaseEstimator
) -> Any:
    """Trains the classifier on the joint data for the L-C2ST.

    Args:
        P: Samples from P, of shape (sample_size, dim).
        Q: Samples from Q, of shape (sample_size, dim).
        x_P: Observations corresponding to P, of shape (sample_size, dim_x).
        x_Q: Observations corresponding to Q, of shape (sample_size, dim_x).
        clf: Classifier to train.

    Returns:
        Trained classifier.
    """
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
    clf_.fit(data, target)  # type: ignore

    return clf_


def eval_lc2st(
    P: Tensor, x_eval: Tensor, clf: BaseEstimator, return_proba: bool = False
) -> Union[float, Tuple[np.ndarray, float]]:
    """Evaluates the classifier returned by `train_lc2st` for one observation
    `x_eval` and over the samples `P`.

    Args:
        P: Samples from P, of shape (sample_size, dim).
        x_eval: Observation, of shape (, dim_x).
        clf: Trained classifier.
        return_proba: Whether to return the predicted probabilities of being in P,
            defaults to False.

    Returns:
        L-C2ST score at `x_eval`.
    """
    # concatenate to get joint data
    joint_P = np.concatenate(
        [P.cpu().numpy(), x_eval.repeat(len(P), 1).cpu().numpy()], axis=1
    )

    # evaluate classifier
    # probability of being in P (class 0)
    proba = clf.predict_proba(joint_P)[:, 0]  # type: ignore
    # mean squared error between proba and dirac at 0.5
    score = ((proba - [0.5] * len(proba)) ** 2).mean()

    if return_proba:
        return proba, score
    else:
        return score


def permute_data(P: Tensor, Q: Tensor, seed: int = 1) -> Tuple[Tensor, Tensor]:
    """Permutes the concatenated data [P,Q] to create null samples.

    Args:
        P: samples form P, of shape (sample_size, dim).
        Q: samples from Q, of shape (sample_size, dim).
        seed: random seed, defaults to 1.

    Returns:
        Permuted data [P,Q]
    """
    # set seed
    torch.manual_seed(seed)
    # check inputs
    assert P.shape[0] == Q.shape[0]

    sample_size = P.shape[0]
    X = torch.cat([P, Q], dim=0)
    X_perm = X[torch.randperm(sample_size * 2)]
    return X_perm[:sample_size], X_perm[sample_size:]
