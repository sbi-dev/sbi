# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import logging
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple, Union
from warnings import warn

import torch
from pyknos.nflows.nn import nets
from torch import Tensor, nn, optim, relu
from torch.nn import MSELoss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from sbi.utils.sbiutils import handle_invalid_x, standardizing_net


class Destandardize(nn.Module):
    def __init__(self, mean: Union[Tensor, float], std: Union[Tensor, float]):
        super(Destandardize, self).__init__()
        mean, std = map(torch.as_tensor, (mean, std))
        self.mean = mean
        self.std = std
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, tensor):
        return tensor * self._std + self._mean


def destandardizing_net(batch_t: Tensor, min_std: float = 1e-7) -> nn.Module:
    """Net that de-standardizes the output so the NN can learn the standardized target.

    Args:
        batch_t: Batched tensor from which mean and std deviation (across
            first dimension) are computed.
        min_std:  Minimum value of the standard deviation to use when z-scoring to
            avoid division by zero.

    Returns:
        Neural network module for z-scoring
    """

    is_valid_t, *_ = handle_invalid_x(batch_t, True)

    t_mean = torch.mean(batch_t[is_valid_t], dim=0)
    if len(batch_t > 1):
        t_std = torch.std(batch_t[is_valid_t], dim=0)
        t_std[t_std < min_std] = min_std
    else:
        t_std = 1
        logging.warning(
            """Using a one-dimensional batch will instantiate a Standardize transform
            with (mean, std) parameters which are not representative of the data. We
            allow this behavior because you might be loading a pre-trained. If this is
            not the case, please be sure to use a larger batch."""
        )

    return Destandardize(t_mean, t_std)


def build_input_output_layer(
    batch_theta: Tensor,
    batch_property: Tensor,
    z_score_theta: bool = True,
    z_score_property: bool = True,
    embedding_net_theta: nn.Module = nn.Identity(),
) -> Tuple[nn.Module, nn.Module]:
    r"""Builds input layer for the `ActiveSubspace` that optionally z-scores.

    The regression network used in the `ActiveSubspace` will receive batches of
    $\theta$s and properties.

    Args:
        batch_theta: Batch of $\theta$s, used to infer dimensionality and (optional)
            z-scoring.
        batch_property: Batch of properties, used for (optional) z-scoring.
        z_score_theta: Whether to z-score $\theta$s passing into the network.
        z_score_property: Whether to z-score properties passing into the network.
        embedding_net_theta: Optional embedding network for $\theta$s.

    Returns:
        Input layer that optionally z-scores.
    """

    if z_score_theta:
        input_layer = nn.Sequential(standardizing_net(batch_theta), embedding_net_theta)
    else:
        input_layer = embedding_net_theta

    if z_score_property:
        output_layer = destandardizing_net(batch_property)
    else:
        output_layer = nn.Identity()

    return input_layer, output_layer


class ActiveSubspace:
    def __init__(self, posterior: Any):
        """Identify the active subspace for sensitivity analyses.

        - Introduction to active subspaces: Constantine et al. 2015.
        - Application to analyse the sensitivity in neuroscience models:
            Deistler et al. 2021, in preparation.

        Args:
            posterior: Posterior distribution obtained with `SNPE`, `SNLE`, or `SNRE`.
                Needs to have a `.sample()` method. If we want to analyse the
                sensitivity of the posterior probability, it also must have a
                `.potential()` method.
        """
        self._posterior = posterior
        self._regression_net = None
        self._theta = None
        self._emergent_property = None
        self._device = posterior._device
        self._validation_log_probs = []

    def add_property(
        self,
        theta: Tensor,
        emergent_property: Tensor,
        model: Union[str, Callable] = "resnet",
        hidden_features: int = 100,
        num_blocks: int = 2,
        dropout_probability: float = 0.5,
        z_score_theta: bool = True,
        z_score_property: bool = True,
        embedding_net: nn.Module = nn.Identity(),
    ) -> "ActiveSubspace":
        r"""Add a property whose sensitivity is to be analysed.

        To analyse the sensitivity of an emergent property, we train a neural network
        to predict the property from the parameter set $\theta$. The hyperparameters of
        this neural network also have to be specified here.

        Args:
            theta: Parameter sets $\theta$ sampled from the posterior.
            emergent_property: Tensor containing the values of the property given each
                parameter set $\theta$.
            model: Neural network used to distinguish valid from bad samples. If it is
                a string, use a pre-configured network of the provided type (either
                mlp or resnet). Alternatively, a function that builds a custom
                neural network can be provided. The function will be called with the
                first batch of parameters (theta,), which can thus be used for shape
                inference and potentially for z-scoring. It needs to return a PyTorch
                `nn.Module` implementing the classifier.
            hidden_features: Number of hidden units of the classifier if `model` is a
                string.
            num_blocks: Number of hidden layers of the classifier if `model` is a
                string.
            dropout_probability: Dropout probability of the classifier if `model` is
                `resnet`.
            z_score_theta: Whether to z-score the parameters $\theta$ used to train the
                classifier.
            z_score_property: Whether to z-score the property used to train the
                classifier.
            embedding_net: Neural network used to encode the parameters before they are
                passed to the classifier.

        Returns:
            `ActiveSubspace` to make the call chainable.
        """
        assert emergent_property.shape == (
            theta.shape[0],
            1,
        ), "The `emergent_property` must have shape (N, 1)."

        self._theta = theta
        self._emergent_property = emergent_property

        def build_resnet(theta):
            classifier = nets.ResidualNet(
                in_features=theta.shape[1],
                out_features=1,
                hidden_features=hidden_features,
                context_features=None,
                num_blocks=num_blocks,
                activation=relu,
                dropout_probability=dropout_probability,
                use_batch_norm=True,
            )
            input_layer, output_layer = build_input_output_layer(
                theta,
                emergent_property,
                z_score_theta,
                z_score_property,
                embedding_net,
            )
            classifier = nn.Sequential(input_layer, classifier, output_layer)
            return classifier

        def build_mlp(theta):
            classifier = nn.Sequential(
                nn.Linear(theta.shape[1], hidden_features),
                nn.BatchNorm1d(hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.BatchNorm1d(hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, 1),
            )
            input_layer, output_layer = build_input_output_layer(
                theta,
                emergent_property,
                z_score_theta,
                z_score_property,
                embedding_net,
            )
            classifier = nn.Sequential(input_layer, classifier, output_layer)
            return classifier

        if isinstance(model, str):
            if model == "resnet":
                self._build_nn = build_resnet
            elif model == "mlp":
                self._build_nn = build_mlp
            else:
                raise NameError
        else:
            self._build_nn = model

        return self

    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
    ) -> nn.Module:
        r"""Train a regression network to predict the specified property from $\theta$.

        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use `None` for no clipping.
        """

        assert (
            self._theta is not None and self._emergent_property is not None
        ), "You must call .add_property() first."

        # Get indices for permutation of the data.
        num_examples = len(self._theta)
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(self._theta, self._emergent_property)

        # Create neural_net and validation loaders using a subset sampler.
        train_loader = data.DataLoader(
            dataset,
            batch_size=training_batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices.tolist()),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(training_batch_size, num_examples - num_training_examples),
            shuffle=False,
            drop_last=True,
            sampler=SubsetRandomSampler(val_indices.tolist()),
        )

        if self._regression_net is None:
            self._regression_net = self._build_nn(self._theta[train_indices]).to(
                self._device
            )

        optimizer = optim.Adam(
            list(self._regression_net.parameters()),
            lr=learning_rate,
        )
        max_num_epochs = 2**31 - 1 if max_num_epochs is None else max_num_epochs

        # criterion / loss
        criterion = MSELoss()

        epoch, self._val_log_prob = 0, float("-Inf")
        while epoch <= max_num_epochs and not self._converged(epoch, stop_after_epochs):
            self._regression_net.train()
            for parameters, observations in train_loader:
                optimizer.zero_grad()
                outputs = self._regression_net(parameters.to(self._device))
                loss = criterion(outputs, observations.to(self._device))
                loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._regression_net.parameters(),
                        max_norm=clip_max_norm,
                    )
                optimizer.step()

            epoch += 1

            # calculate validation performance
            self._regression_net.eval()

            val_loss = 0.0
            with torch.no_grad():
                for parameters, observations in val_loader:
                    outputs = self._regression_net(parameters.to(self._device))
                    loss = criterion(outputs, observations.to(self._device))
                    val_loss += loss.item()
            self._val_log_prob = -val_loss / num_validation_examples
            self._validation_log_probs.append(self._val_log_prob)

            print("\r", "Training neural network. Epochs trained: ", epoch, end="")

        return deepcopy(self._regression_net)

    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        r"""Return whether the training converged yet and save best model state so far.

        Checks for improvement in validation performance over previous epochs.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.
        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        converged = False

        assert self._regression_net is not None
        posterior_nn = self._regression_net

        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or self._val_log_prob > self._best_val_log_prob:
            self._best_val_log_prob = self._val_log_prob
            self._epochs_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(posterior_nn.state_dict())
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1:
            posterior_nn.load_state_dict(self._best_model_state_dict)
            converged = True

        return converged

    def find_directions(
        self,
        posterior_log_prob_as_property: bool = False,
        norm_gradients_to_prior: bool = True,
        num_monte_carlo_samples: int = 1000,
    ) -> Tuple[Tensor, Tensor]:
        r"""Return eigenvectors and values corresponding to directions of sensitivity.

        The directions of sensitivity are the directions along which a specific
        property changes in the fastest way. They will have the largest eigenvalues.

        This computes the matrix:
        $\mathbf{M} = \mathbb{E}_{p(\theta|x_o)}[\nabla_{\theta} f(\theta)^T
        \nabla_{\theta}
        f(\theta)]$
        where $f(\cdot)$ is the trained regression network. The expected value is
        approximated with a Monte-Carlo mean. Next, do an eigenvalue
        decomposition of the matrix $\mathbf{M}$:

        $\mathbf{M} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1}$

        We then return the eigenvectors and eigenvalues found by this decomposition.
        Eigenvectors with large eigenvalues are directions along which the property is
        sensitive to changes in the parameters $\theta$ (`active` directions).
        Increases along these directions will increase the value of the property.

        Args:
            posterior_log_prob_as_property: Whether to use the posterior
                log-probability the key property whose sensitivity is analysed. If
                `False`, one must have specified an emergent property and trained a
                regression network using `.add_property().train()`. If `True`,
                any previously specified property is ignored.
            norm_gradients_to_prior: Whether to normalize each entry of the gradient
                by the standard deviation of the prior in each dimension. If set to
                `False`, the directions with the strongest eigenvalues might correspond
                to directions in which the prior is broad.
            num_monte_carlo_samples: Number of Monte Carlo samples that the average is
                based on. A larger value will make the results more accurate while
                requiring more compute time.

        Returns:
            Eigenvectors and corresponding eigenvalues. They are sorted in ascending
            order. The column `eigenvectors[:, j]` is the eigenvector corresponding to
            the `j`-th eigenvalue.
        """

        self._gradients_are_normed = norm_gradients_to_prior

        if self._emergent_property is None and not posterior_log_prob_as_property:
            raise ValueError(
                "You have not yet passed an emergent property whose "
                "sensitivity you would like to analyse. Please use "
                "`.add_emergent_property().train()` to do so. If you want "
                "to use all features that had also been used to infer the "
                "posterior distribution (i.e. you want to analyse the "
                "sensitivity of the posterior probability), use: "
                "`.find_active(posterior_log_prob_as_property=True)`."
            )
        if self._emergent_property is not None and posterior_log_prob_as_property:
            warn(
                "You specified a property with `.add_property()`, but also set "
                "`posterior_log_prob_as_property=True`. The specified property will "
                "be ignored."
            )

        thetas = self._posterior.sample((num_monte_carlo_samples,))

        thetas.requires_grad = True

        if posterior_log_prob_as_property:
            predictions = self._posterior.potential(thetas, track_gradients=True)
        else:
            assert (
                self._regression_net is not None
            ), "self._regression_net is None, you must call `.train()` first."
            predictions = self._regression_net.forward(thetas)
        loss = predictions.mean()
        loss.backward()
        gradients = torch.squeeze(thetas.grad)
        if norm_gradients_to_prior:
            if hasattr(self._posterior.prior, "stddev") and hasattr(
                self._posterior.prior, "mean"
            ):
                self._prior_mean = self._posterior.prior.mean
                self._prior_scale = self._posterior.prior.stddev
            else:
                prior_samples = self._posterior.prior.sample((10000,))
                self._prior_scale = torch.std(prior_samples, dim=0)
                self._prior_mean = torch.mean(prior_samples, dim=0)
            gradients *= self._prior_scale
        outer_products = torch.einsum("bi,bj->bij", (gradients, gradients))
        average_outer_product = outer_products.mean(dim=0)

        eigen_values, eigen_vectors = torch.linalg.eigh(average_outer_product, UPLO="U")

        # Identify the direction of the eigenvectors. Above, we have computed an outer
        # product m*mT=A. Note that the same matrix A can be constructed with the
        # negative vector (-m)(-mT)=A. Thus, when performing an eigen-decomposition of
        # A, we can not determine if the eigenvector was -m or m. We solve this issue
        # below. We use that the average gradient m should be obtained by a mean over
        # the eigenvectors (weighted by the eigenvalues).
        av_gradient = torch.mean(gradients, dim=0)
        av_gradient = av_gradient / torch.norm(av_gradient)
        av_eigenvec = torch.mean(eigen_vectors * eigen_values, dim=1)
        av_eigenvec = av_eigenvec / torch.norm(av_eigenvec)

        # Invert if the negative eigenvectors are closer to the average gradient.
        if (torch.mean((av_eigenvec - av_gradient) ** 2)) > (
            torch.mean((-av_eigenvec - av_gradient) ** 2)
        ):
            eigen_vectors = -eigen_vectors

        self._eigen_vectors = eigen_vectors

        return eigen_values, eigen_vectors

    def project(self, theta: Tensor, num_dimensions: int) -> Tensor:
        r"""Return $\theta$ that were projected into the subspace.

        To identify the dimensionality of the active subspace `num_dimensions`,
        Constantine et al. 2015 suggest to look at gaps in the eigenvalue spectrum.

        Performs a linear projection. Also takes care of normalizing the data. The mean
        and standard deviation used for normalizing are the same as used to compute the
        eigenvectors and eigenvalues (mean and std of prior).

        Args:
            theta: Parameter sets to be projected.
            num_dimensions: Dimensionality of the subspace into which to project.

        Returns:
            Projected parameters of shape `(theta.shape[0], num_dimensions)`.
        """
        theta = theta.to(self._device)
        if self._gradients_are_normed:
            theta = (theta - self._prior_mean) / self._prior_scale

        projection_mat = self._eigen_vectors[:, -num_dimensions:]
        projected_theta = torch.mm(theta, projection_mat)

        return projected_theta
