# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from torch import Tensor
from tqdm import tqdm

from sbi.utils.sbiutils import handle_invalid_x

# Default MLP classifier hyperparameters
DEFAULT_MLP_ACTIVATION = "relu"
# hidden_layer_sizes = (multiplier * ndim,) * 2
DEFAULT_MLP_HIDDEN_LAYER_MULTIPLIER = 10
DEFAULT_MLP_MAX_ITER = 1000
DEFAULT_MLP_SOLVER = "adam"
DEFAULT_MLP_EARLY_STOPPING = True
DEFAULT_MLP_N_ITER_NO_CHANGE = 50


class LC2STState(Enum):
    """Lifecycle states for LC2ST.

    The LC2ST object progresses through these states as training methods are called:
    - INITIALIZED: Object created but no training performed
    - OBSERVED_TRAINED: Classifiers trained on observed data
    - NULL_TRAINED: Classifiers trained under null hypothesis only
    - READY: Both observed and null classifiers trained, ready for inference
    """

    INITIALIZED = auto()
    OBSERVED_TRAINED = auto()
    NULL_TRAINED = auto()
    READY = auto()


@dataclass
class LC2STScores:
    """Structured return type for LC2ST score computations.

    Attributes:
        scores: Array of LC2ST scores, shape (num_folds,) or (num_folds * num_ensemble,)
        probabilities: Optional array of predicted probabilities from classifiers
    """

    scores: np.ndarray
    probabilities: Optional[np.ndarray] = None


class LC2ST:
    r"""L-C2ST: Local Classifier Two-Sample Test.

    Implementation based on the official code from [1] and the exisiting C2ST
    metric [2], using scikit-learn classifiers.

    L-C2ST tests the local consistency of a posterior estimator :math:`q` with
    respect to the true posterior :math:`p`, at a fixed observation :math:`x_o`,
    i.e., whether the following null hypothesis holds:

    :math:`H_0(x_o) := q(\theta \mid x_o) = p(\theta \mid x_o)`.

    L-C2ST proceeds as follows:

    1. It first trains a classifier to distinguish between samples from two joint
       distributions :math:`[\theta_p, x_p]` and :math:`[\theta_q, x_q]`, and
       evaluates the L-C2ST statistic at a given observation :math:`x_o`.

    2. The L-C2ST statistic is the mean squared error between the predicted
       probabilities of being in p (class 0) and a Dirac at 0.5, which corresponds to
       the chance level of the classifier, unable to distinguish between p and q.

    - If ``num_ensemble>1``, the average prediction over all classifiers is used.
    - If ``num_folds>1`` the average statistic over all cv-folds is used.

    To evaluate the test, steps 1 and 2 are performed over multiple trials under the
    null hypothesis (H0). If the null distribution is not known, it is estimated
    using the permutation method, i.e. by training the classifier on the permuted
    data. The statistics obtained under (H0) is then compared to the one obtained
    on observed data to compute the p-value, used to decide whether to reject (H0)
    or not.
    """

    def __init__(
        self,
        prior_samples: Optional[Tensor] = None,
        xs: Optional[Tensor] = None,
        posterior_samples: Optional[Tensor] = None,
        seed: int = 1,
        num_folds: int = 1,
        num_ensemble: int = 1,
        classifier: Union[str, Type[BaseEstimator]] = MLPClassifier,
        z_score: bool = False,
        classifier_kwargs: Optional[Dict[str, Any]] = None,
        num_trials_null: int = 100,
        permutation: bool = True,
        *,
        thetas: Optional[Tensor] = None,  # Deprecated, use prior_samples
    ) -> None:
        """Initialize L-C2ST.

        Args:
            prior_samples: Samples from the prior (Q distribution),
                of shape (sample_size, dim). These are compared against
                posterior_samples (P distribution) by the classifier.
            xs: Corresponding simulated data, of shape (sample_size, dim_x).
            posterior_samples: Samples from the estimated posterior (P distribution),
                of shape (sample_size, dim).
            seed: Seed for the sklearn classifier and the KFold cross validation,
                defaults to 1.
            num_folds: Number of folds for the cross-validation,
                defaults to 1 (no cross-validation).
                This is useful to reduce variance coming from the data.
            num_ensemble: Number of classifiers for ensembling, defaults to 1.
                This is useful to reduce variance coming from the classifier.
            z_score: Whether to z-score to normalize the data, defaults to False.
            classifier: Classification architecture to use, can be one of the following:
                    - "random_forest" or "mlp", defaults to "mlp" or
                    - A classifier class (e.g., RandomForestClassifier, MLPClassifier)
            classifier_kwargs: Custom kwargs for the sklearn classifier,
                defaults to None.
            num_trials_null: Number of trials to estimate the null distribution,
                defaults to 100.
            permutation: Whether to use the permutation method for the null hypothesis,
                defaults to True.
            thetas: Deprecated. Use prior_samples instead.

        References:
        [1] : https://arxiv.org/abs/2306.03580, https://github.com/JuliaLinhart/lc2st
        [2] : https://github.com/sbi-dev/sbi/blob/main/sbi/utils/metrics.py
        """
        # Handle deprecated 'thetas' parameter
        if thetas is not None:
            warnings.warn(
                "Parameter 'thetas' is deprecated and will be removed in a future "
                "version. Use 'prior_samples' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if prior_samples is not None:
                raise ValueError(
                    "Cannot specify both 'thetas' and 'prior_samples'. "
                    "Use 'prior_samples' only."
                )
            prior_samples = thetas

        # Validate required arguments
        if prior_samples is None:
            raise ValueError("prior_samples is required.")
        if xs is None:
            raise ValueError("xs is required.")
        if posterior_samples is None:
            raise ValueError("posterior_samples is required.")

        # Remove NaN/Inf values from all tensors (using mask from xs)
        is_valid_x, num_nans, num_infs = handle_invalid_x(xs, exclude_invalid_x=True)
        if num_nans > 0 or num_infs > 0:
            warnings.warn(
                f"Found {num_nans} NaNs and {num_infs} Infs in xs. "
                f"These rows will be removed from all input tensors. "
                f"Only {is_valid_x.sum()} / {len(xs)} samples remain.",
                stacklevel=2,
            )
        prior_samples = prior_samples[is_valid_x]
        xs = xs[is_valid_x]
        posterior_samples = posterior_samples[is_valid_x]

        # Validate tensor properties after cleaning
        self._validate_inputs(prior_samples, xs, posterior_samples, num_folds, seed)

        # Set observed data for classification (P = posterior, Q = prior)
        self.theta_p = posterior_samples
        self.x_p = xs
        self.theta_q = prior_samples
        self.x_q = xs

        # z-score normalization parameters
        self.z_score = z_score
        self._setup_normalization()

        # Centralized seed management
        self._base_seed = seed
        self.seed = seed  # For backward compatibility
        self.num_folds = num_folds
        self.num_ensemble = num_ensemble

        # Initialize classifier
        self.clf_class = self._resolve_classifier(classifier)
        self.clf_kwargs = self._get_classifier_kwargs(
            classifier_kwargs, prior_samples.shape[-1]
        )

        # Initialize state machine
        self._state = LC2STState.INITIALIZED
        self.trained_clfs: Optional[List[BaseEstimator]] = None
        self.trained_clfs_null: Optional[Dict[int, List[BaseEstimator]]] = None

        # Parameters for the null hypothesis testing
        self.num_trials_null = num_trials_null
        self.permutation = permutation
        # Can be specified if known and independent of x (see LC2ST_NF)
        self.null_distribution: Optional[torch.distributions.Distribution] = None

    def _validate_inputs(
        self,
        prior_samples: Tensor,
        xs: Tensor,
        posterior_samples: Tensor,
        num_folds: int,
        seed: int,
    ) -> None:
        """Validate input tensors and parameters.

        Args:
            prior_samples: Samples from the prior.
            xs: Simulated data.
            posterior_samples: Samples from the estimated posterior.
            num_folds: Number of cross-validation folds.
            seed: Random seed.

        Raises:
            ValueError: If inputs are invalid.
            TypeError: If inputs have wrong types.
        """
        # Check tensor types
        if not isinstance(prior_samples, Tensor):
            raise TypeError(
                f"prior_samples must be a torch.Tensor, got {type(prior_samples)}."
            )
        if not isinstance(xs, Tensor):
            raise TypeError(f"xs must be a torch.Tensor, got {type(xs)}.")
        if not isinstance(posterior_samples, Tensor):
            raise TypeError(
                f"posterior_samples must be a torch.Tensor, "
                f"got {type(posterior_samples)}."
            )

        # Check for empty tensors
        if prior_samples.shape[0] == 0:
            raise ValueError("prior_samples cannot be empty.")
        if xs.shape[0] == 0:
            raise ValueError("xs cannot be empty.")
        if posterior_samples.shape[0] == 0:
            raise ValueError("posterior_samples cannot be empty.")

        # Check sample size consistency
        if not (prior_samples.shape[0] == xs.shape[0] == posterior_samples.shape[0]):
            raise ValueError(
                f"Sample size mismatch: prior_samples has {prior_samples.shape[0]}, "
                f"xs has {xs.shape[0]}, posterior_samples has "
                f"{posterior_samples.shape[0]}. All must have the same number "
                f"of samples."
            )

        # Check dimension consistency
        if prior_samples.shape[-1] != posterior_samples.shape[-1]:
            raise ValueError(
                f"Dimension mismatch: prior_samples has dimension "
                f"{prior_samples.shape[-1]}, but posterior_samples has dimension "
                f"{posterior_samples.shape[-1]}."
            )

        # Check num_folds
        if num_folds < 1:
            raise ValueError(f"num_folds must be >= 1, got {num_folds}.")
        if num_folds > prior_samples.shape[0]:
            raise ValueError(
                f"num_folds ({num_folds}) cannot exceed sample size "
                f"({prior_samples.shape[0]})."
            )

        # Check seed
        if not isinstance(seed, int):
            raise TypeError(f"seed must be an integer, got {type(seed)}.")

    def _setup_normalization(self) -> None:
        """Calculate and store z-score normalization parameters.

        Computes mean and standard deviation for theta and x data from the
        posterior (P) distribution. These parameters are used by _normalize_theta()
        and _normalize_x() when z_score=True.

        This method should be called after self.theta_p and self.x_p are set.
        """
        self.theta_p_mean = torch.mean(self.theta_p, dim=0)
        self.theta_p_std = torch.std(self.theta_p, dim=0)
        self.x_p_mean = torch.mean(self.x_p, dim=0)
        self.x_p_std = torch.std(self.x_p, dim=0)

    def _resolve_classifier(
        self, classifier: Union[str, Type[BaseEstimator]]
    ) -> Type[BaseEstimator]:
        """Resolve classifier string or class to a classifier class.

        Args:
            classifier: Classifier specification (string or class).

        Returns:
            Resolved classifier class.

        Raises:
            ValueError: If classifier string is invalid.
            TypeError: If classifier is not a BaseEstimator subclass.
        """
        if isinstance(classifier, str):
            if classifier.lower() == "mlp":
                return MLPClassifier
            elif classifier.lower() == "random_forest":
                return RandomForestClassifier
            else:
                raise ValueError(
                    f'Invalid classifier: "{classifier}". '
                    'Expected "mlp", "random_forest", '
                    "or a valid scikit-learn classifier class."
                )
        if not issubclass(classifier, BaseEstimator):
            raise TypeError(
                f"classifier must be a string or a subclass of BaseEstimator, "
                f"got {type(classifier).__name__}."
            )
        return classifier

    def _get_classifier_kwargs(
        self, classifier_kwargs: Optional[Dict[str, Any]], ndim: int
    ) -> Dict[str, Any]:
        """Get classifier kwargs with sensible defaults.

        Args:
            classifier_kwargs: User-provided kwargs (may be None).
            ndim: Dimension of the parameter space.

        Returns:
            Dictionary of classifier kwargs.
        """
        if self.clf_class == MLPClassifier:
            hidden_size = DEFAULT_MLP_HIDDEN_LAYER_MULTIPLIER * ndim
            defaults = {
                "activation": DEFAULT_MLP_ACTIVATION,
                "hidden_layer_sizes": (hidden_size, hidden_size),
                "max_iter": DEFAULT_MLP_MAX_ITER,
                "solver": DEFAULT_MLP_SOLVER,
                "early_stopping": DEFAULT_MLP_EARLY_STOPPING,
                "n_iter_no_change": DEFAULT_MLP_N_ITER_NO_CHANGE,
            }
        else:
            defaults = {}

        if classifier_kwargs is not None:
            # Merge user kwargs with defaults (user kwargs take precedence)
            defaults.update(classifier_kwargs)

        return defaults

    def _normalize_theta(self, theta: Tensor) -> Tensor:
        """Normalize theta samples using stored mean and std.

        Args:
            theta: Parameter samples of shape (sample_size, dim).

        Returns:
            Normalized theta if z_score is enabled, otherwise unchanged theta.
        """
        if self.z_score:
            return (theta - self.theta_p_mean) / self.theta_p_std
        return theta

    def _normalize_x(self, x: Tensor) -> Tensor:
        """Normalize observation data using stored mean and std.

        Args:
            x: Observation data of shape (sample_size, dim_x) or (dim_x,).

        Returns:
            Normalized x if z_score is enabled, otherwise unchanged x.
        """
        if self.z_score:
            return (x - self.x_p_mean) / self.x_p_std
        return x

    @property
    def state(self) -> LC2STState:
        """Return the current state of the LC2ST object."""
        return self._state

    def _train(
        self,
        theta_p: Tensor,
        theta_q: Tensor,
        x_p: Tensor,
        x_q: Tensor,
        verbosity: int = 0,
    ) -> List[BaseEstimator]:
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
        # Normalize data
        theta_p = self._normalize_theta(theta_p)
        theta_q = self._normalize_theta(theta_q)
        x_p = self._normalize_x(x_p)
        x_q = self._normalize_x(x_q)

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
                clf_fold = train_lc2st(
                    theta_p_train, theta_q_train, x_p_train, x_q_train, clf
                )

                trained_clfs.append(clf_fold)
        else:
            # train single classifier
            clf = train_lc2st(theta_p, theta_q, x_p, x_q, clf)
            trained_clfs = [clf]

        return trained_clfs

    def get_scores(
        self,
        theta_o: Tensor,
        x_o: Tensor,
        trained_clfs: List[BaseEstimator],
        return_probs: bool = False,
    ) -> Union[LC2STScores, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Computes the L-C2ST scores given the trained classifiers.

        Mean squared error (MSE) between 0.5 and the predicted probabilities
        of being in class 0 over the dataset (`theta_o`, `x_o`).

        Args:
            theta_o: Samples from the posterior conditioned on the observation `x_o`,
                of shape (sample_size, dim).
            x_o: The observation, of shape (,dim_x).
            trained_clfs: List of trained classifiers, of length `num_folds`.
            return_probs: Whether to return the predicted probabilities of being in P,
                defaults to False. Deprecated: Use the LC2STScores.probabilities
                attribute instead.

        Returns:
            LC2STScores object containing scores and optionally probabilities.
            For backward compatibility, if return_probs=True, returns a tuple
            (probs, scores) instead.
        """
        if x_o.shape == self.x_p_mean.shape:
            x_o = x_o.unsqueeze(0)

        # Normalize data
        theta_o = self._normalize_theta(theta_o)
        x_o = self._normalize_x(x_o)

        probs_list, scores_list = [], []

        # Evaluate classifiers
        for clf in trained_clfs:
            proba, score = eval_lc2st(theta_o, x_o, clf, return_proba=True)
            probs_list.append(proba)
            scores_list.append(score)
        probs_arr, scores_arr = np.array(probs_list), np.array(scores_list)

        # Backward compatibility: return tuple if return_probs=True
        if return_probs:
            warnings.warn(
                "The 'return_probs' parameter is deprecated. "
                "Use LC2STScores.probabilities attribute instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return probs_arr, scores_arr

        # Return structured object
        return LC2STScores(scores=scores_arr, probabilities=probs_arr)

    def train_on_observed_data(
        self, seed: Optional[int] = None, verbosity: int = 1
    ) -> "LC2ST":
        """Trains the classifier on the observed data.

        Saves the trained classifier(s) as a list of length `num_folds`.

        Args:
            seed: Random state of the classifier, defaults to None.
            verbosity: Verbosity level, defaults to 1.

        Returns:
            self, for method chaining.
        """
        # Set random state
        if seed is not None:
            if "random_state" in self.clf_kwargs:
                warnings.warn(
                    "Overwriting 'random_state' in classifier_kwargs because "
                    "a 'seed' was provided to train_on_observed_data().",
                    UserWarning,
                    stacklevel=2,
                )
            self.clf_kwargs["random_state"] = seed

        # Train the classifier
        trained_clfs = self._train(
            self.theta_p, self.theta_q, self.x_p, self.x_q, verbosity=verbosity
        )
        self.trained_clfs = trained_clfs

        # Update state
        if self._state == LC2STState.NULL_TRAINED:
            self._state = LC2STState.READY
        else:
            self._state = LC2STState.OBSERVED_TRAINED

        return self

    def get_statistic_on_observed_data(
        self,
        theta_o: Tensor,
        x_o: Tensor,
    ) -> float:
        """Computes the L-C2ST statistics for the observed data.

        Mean over all cv-scores.

        Args:
            theta_o: Samples from the posterior conditioned on the observation `x_o`,
                of shape (sample_size, dim).
            x_o: The observation, of shape (, dim_x)

        Returns:
            L-C2ST statistic at `x_o`.

        Raises:
            RuntimeError: If classifiers have not been trained on observed data.
        """
        if self._state not in (
            LC2STState.OBSERVED_TRAINED,
            LC2STState.READY,
        ):
            raise RuntimeError(
                "Classifiers have not been trained on observed data. "
                "Call train_on_observed_data() before computing statistics."
            )
        result = self.get_scores(
            theta_o=theta_o,
            x_o=x_o,
            trained_clfs=self.trained_clfs,  # type: ignore[arg-type]
        )
        assert isinstance(result, LC2STScores)  # return_probs=False returns LC2STScores
        return float(result.scores.mean())

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
                of shape (sample_size, dim).
            x_o: The observation, of shape (, dim_x).

        Returns:
            p-value for L-C2ST at `x_o`.

        Raises:
            RuntimeError: If the LC2ST is not in READY state (both training
                methods must be called first).
        """
        if self._state != LC2STState.READY:
            missing = []
            if self._state in (LC2STState.INITIALIZED, LC2STState.NULL_TRAINED):
                missing.append("train_on_observed_data()")
            if self._state in (LC2STState.INITIALIZED, LC2STState.OBSERVED_TRAINED):
                missing.append("train_under_null_hypothesis()")
            raise RuntimeError(
                f"LC2ST is not ready to compute p-values. "
                f"Call {' and '.join(missing)} first."
            )

        stat_data = self.get_statistic_on_observed_data(theta_o=theta_o, x_o=x_o)
        _, stats_null = self.get_statistics_under_null_hypothesis(
            theta_o=theta_o, x_o=x_o, return_probs=True, verbosity=0
        )
        return float((stat_data < stats_null).mean())

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
        return bool(self.p_value(theta_o=theta_o, x_o=x_o) < alpha)

    def train_under_null_hypothesis(
        self,
        verbosity: int = 1,
    ) -> "LC2ST":
        """Computes the L-C2ST scores under the null hypothesis (H0).

        Saves the trained classifiers for each null trial.

        Args:
            verbosity: Verbosity level, defaults to 1.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If permutation is False but no null distribution is set.
        """
        trained_clfs_null: Dict[int, List[BaseEstimator]] = {}
        for t in tqdm(
            range(self.num_trials_null),
            desc=f"Training the classifiers under H0, permutation = {self.permutation}",
            disable=verbosity < 1,
        ):
            # Prepare data
            if self.permutation:
                joint_p = torch.cat([self.theta_p, self.x_p], dim=1)
                joint_q = torch.cat([self.theta_q, self.x_q], dim=1)
                # Permute data (same as permuting the labels)
                joint_p_perm, joint_q_perm = permute_data(joint_p, joint_q, seed=t)
                # Extract the permuted P and Q and x
                theta_p_t, x_p_t = (
                    joint_p_perm[:, : self.theta_p.shape[-1]],
                    joint_p_perm[:, self.theta_p.shape[1] :],
                )
                theta_q_t, x_q_t = (
                    joint_q_perm[:, : self.theta_q.shape[-1]],
                    joint_q_perm[:, self.theta_q.shape[1] :],
                )
            else:
                if self.null_distribution is None:
                    raise ValueError(
                        "A null distribution must be provided when permutation=False. "
                        "Set null_distribution or use permutation=True."
                    )
                theta_p_t = self.null_distribution.sample((self.theta_p.shape[0],))
                theta_q_t = self.null_distribution.sample((self.theta_p.shape[0],))
                x_p_t, x_q_t = self.x_p, self.x_q

            # Train (normalization is handled in _train)
            clf_t = self._train(theta_p_t, theta_q_t, x_p_t, x_q_t, verbosity=0)
            trained_clfs_null[t] = clf_t

        self.trained_clfs_null = trained_clfs_null

        # Update state
        if self._state == LC2STState.OBSERVED_TRAINED:
            self._state = LC2STState.READY
        elif self._state == LC2STState.INITIALIZED:
            self._state = LC2STState.NULL_TRAINED

        return self

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

        Raises:
            RuntimeError: If classifiers have not been trained under null hypothesis.
            ValueError: If null distribution required but not set.
        """
        if self._state not in (LC2STState.NULL_TRAINED, LC2STState.READY):
            raise RuntimeError(
                "Classifiers have not been trained under the null hypothesis. "
                "Call train_under_null_hypothesis() first."
            )

        if (
            self.trained_clfs_null is None
            or len(self.trained_clfs_null) != self.num_trials_null
        ):
            raise RuntimeError(
                f"Expected {self.num_trials_null} null classifiers, "
                f"got {len(self.trained_clfs_null) if self.trained_clfs_null else 0}."
            )

        probs_null, stats_null = [], []
        for t in tqdm(
            range(self.num_trials_null),
            desc=f"Computing T under (H0) - permutation = {self.permutation}",
            disable=verbosity < 1,
        ):
            # Prepare data
            if self.permutation:
                theta_o_t = theta_o
            else:
                if self.null_distribution is None:
                    raise ValueError(
                        "A null distribution must be provided when permutation=False."
                    )
                theta_o_t = self.null_distribution.sample((theta_o.shape[0],))

            # Evaluate using LC2STScores (normalization is handled in get_scores)
            clf_t = self.trained_clfs_null[t]
            result = self.get_scores(theta_o=theta_o_t, x_o=x_o, trained_clfs=clf_t)
            assert isinstance(
                result, LC2STScores
            )  # return_probs=False returns LC2STScores
            probs_null.append(result.probabilities)
            stats_null.append(result.scores.mean())

        probs_null_arr, stats_null_arr = np.array(probs_null), np.array(stats_null)

        if return_probs:
            return probs_null_arr, stats_null_arr
        else:
            return stats_null_arr


class LC2ST_NF(LC2ST):
    def __init__(
        self,
        prior_samples: Optional[Tensor] = None,
        xs: Optional[Tensor] = None,
        posterior_samples: Optional[Tensor] = None,
        flow_inverse_transform: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        flow_base_dist: Optional[torch.distributions.Distribution] = None,
        num_eval: int = 10_000,
        trained_clfs_null: Optional[Dict[int, List[BaseEstimator]]] = None,
        *,
        thetas: Optional[Tensor] = None,  # Deprecated, use prior_samples
        **kwargs: Any,
    ) -> None:
        r"""L-C2ST for Normalizing Flows.

        LC2ST_NF is a subclass of LC2ST that performs the test in the space of the
        base distribution of a normalizing flow. It uses the inverse transform of the
        normalizing flow $T_\phi^{-1}$ to map the samples from the prior and the
        posterior to the base distribution space. Following Theorem 4, Eq. 17 from [1],
        the new null hypothesis for a Gaussian base distribution is:

        :math:`H_0(x_o) := p(T_\phi^{-1}(\theta ; x_o) \mid x_o) = \mathcal{N}(0,`
        :math:`I_m)`.

        This is because a sample from the normalizing flow is a sample from the base
        distribution pushed through the flow:

        :math:`z = T_\phi^{-1}(\theta) \sim \mathcal{N}(0, I_m) \iff`
        :math:`\theta = T_\phi(z)`.

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
            prior_samples: Samples from the prior, of shape (sample_size, dim).
            xs: Corresponding simulated data, of shape (sample_size, dim_x).
            posterior_samples: Samples from the estimated posterior,
                of shape (sample_size, dim).
            flow_inverse_transform: Inverse transform of the normalizing flow.
                Takes prior_samples and xs as input and returns noise.
            flow_base_dist: Base distribution of the normalizing flow.
            num_eval: Number of samples to evaluate the L-C2ST.
            trained_clfs_null: Pre-trained classifiers under the null.
            thetas: Deprecated. Use prior_samples instead.
            kwargs: Additional arguments for the LC2ST class.

        References:
        [1] : https://arxiv.org/abs/2306.03580, https://github.com/JuliaLinhart/lc2st
        """
        # Handle deprecated 'thetas' parameter
        if thetas is not None:
            warnings.warn(
                "Parameter 'thetas' is deprecated and will be removed in a future "
                "version. Use 'prior_samples' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if prior_samples is not None:
                raise ValueError(
                    "Cannot specify both 'thetas' and 'prior_samples'. "
                    "Use 'prior_samples' only."
                )
            prior_samples = thetas

        # Validate required arguments
        if prior_samples is None:
            raise ValueError("prior_samples is required.")
        if xs is None:
            raise ValueError("xs is required.")
        if posterior_samples is None:
            raise ValueError("posterior_samples is required.")
        if flow_inverse_transform is None:
            raise ValueError("flow_inverse_transform is required.")
        if flow_base_dist is None:
            raise ValueError("flow_base_dist is required.")

        # Apply the inverse transform to the prior_samples and the posterior samples
        self.flow_inverse_transform = flow_inverse_transform
        inverse_prior_samples = flow_inverse_transform(prior_samples, xs).detach()
        inverse_posterior_samples = flow_inverse_transform(
            posterior_samples, xs
        ).detach()

        # Initialize the LC2ST class with the inverse transformed samples
        super().__init__(
            prior_samples=inverse_prior_samples,
            xs=xs,
            posterior_samples=inverse_posterior_samples,
            **kwargs,
        )

        # Set the parameters for the null hypothesis testing
        self.null_distribution = flow_base_dist
        self.permutation = False
        self.trained_clfs_null = trained_clfs_null

        # Draw samples from the base distribution for evaluation
        self.theta_o = flow_base_dist.sample(torch.Size([num_eval]))

    def get_scores(
        self,
        x_o: Tensor,
        trained_clfs: List[BaseEstimator],
        return_probs: bool = False,
        **kwargs: Any,
    ) -> Union[LC2STScores, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Computes the L-C2ST scores given the trained classifiers.

        Args:
            x_o: The observation, of shape (,dim_x).
            trained_clfs: Trained classifiers.
            return_probs: Whether to return the predicted probabilities of being in P,
                defaults to False.
            kwargs: Additional arguments used in the parent class.

        Returns:
            LC2STScores object, or for backward compatibility with return_probs=True,
            a tuple (probs, scores).
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
    ) -> "LC2ST_NF":
        """Computes the L-C2ST scores under the null hypothesis.

        Saves the trained classifiers for the null distribution.

        Args:
            verbosity: Verbosity level, defaults to 1.

        Returns:
            self, for method chaining.
        """
        if self.trained_clfs_null is not None:
            raise ValueError(
                "Classifiers have already been trained under the null "
                "and can be used to evaluate any new estimator."
            )
        super().train_under_null_hypothesis(verbosity=verbosity)
        return self

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
        x_o: The observation, of shape (1, dim_x).
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
    score = float(((proba - [0.5] * len(proba)) ** 2).mean())

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
    """Ensemble classifier that aggregates predictions from multiple classifiers.

    This class wraps a base classifier and trains multiple instances with different
    random states, then averages their probability predictions for more robust
    classification. It follows the sklearn estimator interface.

    Attributes:
        clf: Base classifier instance to clone for the ensemble.
        num_ensemble: Number of classifiers in the ensemble.
        trained_clfs: List of trained classifier instances.
        verbosity: Verbosity level for progress display (0=silent, 1+=show progress).
    """

    def __init__(
        self,
        clf: BaseEstimator,
        num_ensemble: int = 1,
        verbosity: int = 1,
    ) -> None:
        """Initialize the ensemble classifier.

        Args:
            clf: Base classifier instance to clone for ensemble members.
            num_ensemble: Number of classifiers to train, defaults to 1.
            verbosity: Verbosity level for progress bar, defaults to 1.
        """
        self.clf = clf
        self.num_ensemble = num_ensemble
        self.trained_clfs: List[Any] = []
        self.verbosity = verbosity

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleClassifier":
        """Train the ensemble of classifiers.

        Each classifier in the ensemble is cloned from the base classifier and
        trained with a different random state for diversity.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            self, for method chaining.
        """
        for n in tqdm(
            range(self.num_ensemble),
            desc="Ensemble training",
            disable=self.verbosity < 1,
        ):
            clf = clone(self.clf)
            if clf.random_state is not None:  # type: ignore[union-attr]
                clf.random_state += n  # type: ignore[union-attr]
            else:
                clf.random_state = n + 1  # type: ignore[union-attr]
            clf.fit(X, y)  # type: ignore[union-attr]
            self.trained_clfs.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities by averaging ensemble predictions.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Averaged probability predictions of shape (n_samples, n_classes).
        """
        probas: List[np.ndarray] = [clf.predict_proba(X) for clf in self.trained_clfs]
        return np.mean(probas, axis=0)
