from copy import deepcopy
from typing import Callable, Optional, Union
from warnings import warn

import torch
import torch.nn.functional as F
from pyknos.nflows.nn import nets
from torch import Tensor, nn, optim, relu, softmax
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from sbi.utils import RejectionClassifier, handle_invalid_x

# TODO: why not integrate this into the `simulate_for_sbi` function? Because I would
#  not use it in this way due to offline simulations. So, I would have to build a
#  workaround around the **flexible** interface -> An "extra flexible interface". We
#  clearly want to avoid this. Thus, I'll make the loop explicit.


def build_rejection_classifier(theta_dim: int,) -> RejectionClassifier:
    """
    Return a classifier that can be trained to reject invalid data.

    See Deistler, Goncalves, Macke, 2020 (in preparation) for details.

    Args:
        theta_dim: Dimensionality of the parameter space.

    Returns:
        Rejection classifier
    """

    return RejectionClassifier(theta_dim)


class RejectionClassifier:
    def __init__(self, theta_dim):
        """
        Args:
            theta_dim: Dimensionality of the parameter space.
        """
        self.net = nets.ResidualNet(
            in_features=theta_dim,
            out_features=1,
            hidden_features=100,
            context_features=None,
            num_blocks=5,
            activation=relu,
            dropout_probability=0.5,
            use_batch_norm=True,
        )

    def train(
        self,
        theta: Tensor,
        x: Tensor,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        subsample_bad_sims_factor: float = 0.0,
        good_bad_criterion: Union[str, Callable] = "valid",
        reweigh_loss: bool = False,
        reweigh_factor: Optional[float] = None,
    ) -> None:
        """
        theta:
        x:
        training_batch_size: Training batch size.
        learning_rate: Learning rate for Adam optimizer.
        validation_fraction: The fraction of data to use for validation.
        stop_after_epochs: The number of epochs to wait for improvement on the
            validation set before terminating training.
        max_num_epochs: Maximum number of epochs to run. If reached, we stop
            training even when the validation loss is still decreasing. If None, we
            train until validation loss increases (see also `stop_after_epochs`).
        clip_max_norm: Value at which to clip the total gradient norm in order to
            prevent exploding gradients. Use None for no clipping.
        subsample_bad_sims_factor: [0, 1]. What fraction of bad simulations should
            randomly be thrown out?
        good_bad_criterion: function. Should take in summary stats x and output whether
            the stats are counted as good simulation (output 1.0) or as bad simulation
            (output 0.0). Other option is good_bad_criterion='nan', which will treat
            simulations with at least one NaN as bad simulation and all others as good.
        reweigh_loss: one way to deal with imbalanced data (e.g. 99% bad simulations).
            If True, we reweigh the CrossEntropyLoss such that it implicitly assigns
            equal prior weight to being a bad or good simulation.
        reweigh_factor: if reweigh_loss is True, but we want to reweigh to a custom
            prior weight, this can be done here. The value assigned will be the
            reweighing factor for bad simulations, (1-reweigh_factor) will be the
            factor for good simulations.

        """
        if good_bad_criterion == "nan":
            label, _, _ = handle_invalid_x(x)
        else:
            label = good_bad_criterion(x)

        # data can be highly unbalanced, e.g. having 99% bad simulations and only 1%
        # good simulations. One way to deal with this is to subsample the bad
        # simulations, which we do here.
        good_theta = theta[label]
        good_label = label[label]
        bad_theta = theta[~label]
        bad_label = label[~label]
        bad_theta = bad_theta[torch.rand(bad_theta.shape) > subsample_bad_sims_factor]
        bad_label = bad_label[torch.rand(bad_label.shape) > subsample_bad_sims_factor]
        theta = torch.cat((good_theta, bad_theta), dim=0).float()
        label = torch.cat((good_label, bad_label), dim=0).long()

        # squeeze dimension from new_sample_stats such that it is just [batchsize]
        # instead of [batchsize x 1]
        label = torch.squeeze(label)

        # compute the fraction of good simulations in dataset.
        # Will be needed for loss-reweighing.
        good_sim_fraction = torch.sum(label, dtype=torch.float) / label.shape[0]

        # get indices for permutation of the data
        num_examples = len(theta)
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(theta, label)

        # Create neural_net and validation loaders using a subset sampler.
        train_loader = data.DataLoader(
            dataset,
            batch_size=training_batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(training_batch_size, num_examples - num_training_examples),
            shuffle=False,
            drop_last=True,
            sampler=SubsetRandomSampler(val_indices),
        )

        optimizer = optim.Adam(list(self.net.parameters()), lr=learning_rate,)

        # define binary cross entropy loss
        if reweigh_loss:
            if reweigh_factor is None:
                # compute the reweighing factor for bad simulations as the fraction of
                # good simulations
                reweigh_factor = good_sim_fraction
        else:
            reweigh_factor = 0.5

        # factor of two such that the average learning rate remains the same.
        # Needed because the average of reweigh_factor and 1-reweigh_factor will be 0.5
        # only.
        reweighing_weights = 2 * torch.tensor([reweigh_factor, 1 - reweigh_factor])
        print("reweighing_weights", reweighing_weights)
        criterion = nn.CrossEntropyLoss(reweighing_weights)
        # todo device again

        epoch, self._val_log_prob = 0, float("-Inf")
        while epoch <= max_num_epochs and not self._converged(epoch, stop_after_epochs):
            # Train for a single epoch.
            self.net.train()
            total_train_loss = 0
            for parameters, observations in train_loader:
                optimizer.zero_grad()
                outputs = self.net(parameters)
                loss = criterion(outputs, observations)
                total_train_loss += loss.item()
                loss.backward()
                optimizer.step()

            epoch += 1

            # calculate validation performance
            self.net.eval()

            val_loss = 0.0
            with torch.no_grad():
                for parameters, observations in val_loader:
                    outputs = self.net(parameters)
                    loss = criterion(outputs, observations)
                    val_loss += loss.item()
            self._val_log_prob = -val_loss / num_validation_examples

            # check for improvement
            print("Validation loss", self._val_log_prob * 1000)

    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        """Return whether the training converged yet and save best model state so far.

        Checks for improvement in validation performance over previous epochs.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.

        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        converged = False

        posterior_nn = self.net

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

    def predict(self, theta, reweigh_factor=0.5):
        """
        Predict labels with hard threshold.
        
        Args:
            theta: Parameters whose label to predict.
            reweigh_factor: Post-hoc correction factor.
        Returns:
            Binary labels.
        """
        # Apply softmax to output.
        pred = F.softmax(self.net.forward(theta))
        ans = []
        # Pick the class with maximum weight
        for t in pred:
            if t[0] * reweigh_factor > t[1] * (1 - reweigh_factor):
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)
