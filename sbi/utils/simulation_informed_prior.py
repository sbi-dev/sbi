from copy import deepcopy
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from pyknos.nflows.nn import nets
from torch import Tensor, nn, optim, relu
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.auto import tqdm

from sbi.types import Shape
from sbi.utils import handle_invalid_x
from sbi.utils.sbiutils import standardizing_net

# TODO: why not integrate this into the `simulate_for_sbi` function? Because I would
#  not use it in this way due to offline simulations. So, I would have to build a
#  workaround around the **flexible** interface -> An "extra flexible interface". We
#  clearly want to avoid this. Thus, I'll make the loop explicit.


# Syntax:
# rej_classifier = RejectionClassifier()
#
# theta, x = simulate_for_sbi(simulator, prior, 1000)
# rej_classifier.add_data().train(theta, x)
# simulation_informed_prior = rej_classifier.build_sip(prior, rej_classifier)


def build_input_layer(
    batch_x: Tensor = None,
    z_score_x: bool = True,
    embedding_net_x: nn.Module = nn.Identity(),
) -> nn.Module:
    """Builds input layer for classifiers that optionally z-scores.

    In SNRE, the classifier will receive batches of thetas and xs.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        hidden_features: Number of hidden features.
        embedding_net_x: Optional embedding network for x.

    Returns:
        Input layer that optionally z-scores.
    """
    if z_score_x:
        input_layer = nn.Sequential(standardizing_net(batch_x), embedding_net_x)
    else:
        input_layer = embedding_net_x

    return input_layer


class RejectionClassifier:
    def __init__(
        self,
        hidden_features: int = 100,
        num_blocks: int = 5,
        dropout_probability: float = 0.5,
        z_score: bool = True,
        embedding_net: nn.Module = nn.Identity(),
    ):
        """
        Classifier that can be trained to reject invalid data.

        Args:
            hidden_features: Number of hidden units.
            num_blocks: Number of hidden layers.
            dropout_probability: Dropout probability.
            z_score: Whether to z-score the parameters used to train the classifier.
            embedding_net: Neural network used to encode the parameters before they are
                passed to the ResNet.
        """
        self._classifier = None

        def build_nn(theta):
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
            input_layer = build_input_layer(theta, z_score, embedding_net)
            classifier = nn.Sequential(input_layer, classifier)

            return classifier

        self._build_nn = build_nn

        self._theta_roundwise = []
        self._x_roundwise = []

    def add_data(self):
        pass

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
    ) -> torch.nn.Module:
        """
        theta: Parameter sets.
        x: Simulation outputs.
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
        subsample_bad_sims_factor: Factor of bad simulations that are randomly thrown
            out. This can be useful when the fraction of bad simulations is extremely
            high and one wants to train on a larger fraction of valid simulations.
            This factor has to be in [0, 1].
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
        if self._classifier is None:
            self._classifier = self._build_nn

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)

        theta = torch.cat(self._theta_roundwise)
        x = torch.cat(self._x_roundwise)

        if good_bad_criterion == "valid":
            label, _, _ = handle_invalid_x(x)
        else:
            label = good_bad_criterion(x)
        label = label.long()

        theta, label = self._subsample_bad_sims(theta, label, subsample_bad_sims_factor)

        # Squeeze dimension from new_sample_stats such that it is just [batchsize]
        # instead of [batchsize x 1].
        print("Label shape before squeezing:  ", label.shape)
        label = torch.squeeze(label)

        # Get indices for permutation of the data.
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

        optimizer = optim.Adam(list(self._classifier.parameters()), lr=learning_rate,)

        # Compute the fraction of good simulations in dataset.
        if reweigh_loss:
            if reweigh_factor is None:
                good_sim_fraction = torch.sum(label, dtype=torch.float) / label.shape[0]
                reweigh_factor = good_sim_fraction
        else:
            reweigh_factor = 0.5

        # Factor of two such that the average learning rate remains the same.
        # Needed because the average of reweigh_factor and 1-reweigh_factor will be 0.5
        # only.
        reweighing_weights = 2 * torch.tensor([reweigh_factor, 1 - reweigh_factor])
        print("Reweighing_weights", reweighing_weights)

        criterion = nn.CrossEntropyLoss(reweighing_weights)
        # todo device again

        epoch, self._val_log_prob = 0, float("-Inf")
        while epoch <= max_num_epochs and not self._converged(epoch, stop_after_epochs):
            self._classifier.train()
            for parameters, observations in train_loader:
                optimizer.zero_grad()
                outputs = self._classifier(parameters)
                loss = criterion(outputs, observations)
                loss.backward()
                optimizer.step()

            epoch += 1

            # calculate validation performance
            self._classifier.eval()

            val_loss = 0.0
            with torch.no_grad():
                for parameters, observations in val_loader:
                    outputs = self._classifier(parameters)
                    loss = criterion(outputs, observations)
                    val_loss += loss.item()
            self._val_log_prob = -val_loss / num_validation_examples

            print("Validation loss", self._val_log_prob * 1000)

        return deepcopy(self._classifier)

    def build_sim_informed_prior(
        self, prior, classifier: Optional[torch.nn.Module] = None
    ):
        if classifier is None:
            classifier = self._classifier
        return SimulationInformedPrior(prior, classifier)

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
        pred = F.softmax(self._classifier.forward(theta))
        ans = []
        # Pick the class with maximum weight
        for t in pred:
            if t[0] * reweigh_factor > t[1] * (1 - reweigh_factor):
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

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

        posterior_nn = self._classifier

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

    def _subsample_bad_sims(
        self, theta: Tensor, label: Tensor, subsample_factor: float = 0.0
    ) -> Tuple[Tensor, Tensor]:
        """
        Data can be highly unbalanced, e.g. having 99% bad simulations and only 1%
        good simulations. One way to deal with this is to subsample the bad
        simulations, which we do here.

        Args:
            theta: Parameters.
            label: Labels indicating if simulation output was valid or not.
            subsample_factor: Factor at which to re-sample, must be in [0,1]

        Returns:
            Subsampled theta and label.
        """

        good_theta = theta[label]
        good_label = label[label]
        bad_theta = theta[~label]
        bad_label = label[~label]
        bad_theta = bad_theta[torch.rand(bad_theta.shape[0]) > subsample_factor]
        bad_label = bad_label[torch.rand(bad_label.shape[0]) > subsample_factor]
        theta = torch.cat((good_theta, bad_theta), dim=0)
        label = torch.cat((good_label, bad_label), dim=0)
        return theta, label


class SimulationInformedPrior:
    def __init__(self, prior, classifier):
        self._prior = prior
        self._classifier = classifier

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        show_progress_bars: bool = False,
        max_sampling_batch_size: int = 10_000,
    ):

        num_samples = sample_shape.numel()
        num_sampled_total, num_remaining = 0, num_samples
        accepted, acceptance_rate = [], float("Nan")

        # Progress bar can be skipped.
        pbar = tqdm(
            disable=not show_progress_bars,
            total=num_samples,
            desc=f"Drawing {num_samples} posterior samples",
        )

        # To cover cases with few samples without leakage:
        sampling_batch_size = min(num_samples, max_sampling_batch_size)
        while num_remaining > 0:

            # Sample and reject.
            candidates = self._prior.sample(sampling_batch_size).reshape(
                sampling_batch_size, -1
            )
            are_accepted_by_classifier = self._classifier.predict(candidates)
            samples = candidates[are_accepted_by_classifier]
            accepted.append(samples)

            # Update.
            num_sampled_total += sampling_batch_size
            num_remaining -= samples.shape[0]
            pbar.update(samples.shape[0])

            # To avoid endless sampling when leakage is high, we raise a warning if the
            # acceptance rate is too low after the first 1_000 samples.
            acceptance_rate = (num_samples - num_remaining) / num_sampled_total

            # For remaining iterations (leakage or many samples) continue sampling with
            # fixed batch size.
            sampling_batch_size = max_sampling_batch_size

        pbar.close()
        print(
            "The classifier rejected",
            acceptance_rate,
            "% of all samples. You will therefore get a speed-up of",
            (1.0 / acceptance_rate - 1.0) * 100,
            "%. (speedup=(1/a)-1)",
        )

        # When in case of leakage a batch size was used there could be too many samples.
        samples = torch.cat(accepted)[:num_samples]
        assert (
            samples.shape[0] == num_samples
        ), "Number of accepted samples must match required samples."

    def predict(self, theta):
        self._classifier.predict(theta)
