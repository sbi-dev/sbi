from copy import deepcopy

import numpy as np
import sbi.simulators as simulators
import sbi.utils as utils
import torch
from sbi.utils.torchutils import get_default_device
from torch import distributions, nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import warnings
from pyknos.mdn.mdn import MultivariateGaussianMDN
from sbi.inference.posteriors.sbi_posterior import Posterior


class SnpeBase:
    def __init__(
        self,
        simulator,
        prior,
        true_observation,
        num_pilot_samples=100,
        density_estimator=None,
        calibration_kernel=None,
        z_score_obs=True,
        use_combined_loss=False,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
        device=None,
        train_with_mcmc=False,
        mcmc_method="slice-np",
    ):
        """
        Args:
            simulator: Python object with 'simulate' method which takes a torch.Tensor
                of parameter values, and returns a simulation result for each parameter
                as a torch.Tensor.
            prior: torch.distribution
                Distribution object with 'log_prob' and 'sample' methods.
            true_observation: torch.Tensor [observation_dim] or [1, observation_dim]
                True observation x0 for which to perform inference on the posterior p(theta | x0).
            num_pilot_samples: int
                Number of simulations that are run when instantiating an object.
                Used to z-score the observations.
            density_estimator: Neural density estimator
                Density estimator to use.
            z_score_obs: bool
                Whether to z-score (=normalize) the data features x
            use_combined_loss: bool
                Whether to jointly neural_net prior samples using maximum likelihood.
                Useful to prevent density leaking when using box uniform priors.
            retrain_from_scratch_each_round: bool
                Whether to retrain the conditional density estimator for the posterior
                from scratch each round.
            discard_prior_samples: bool
                Whether to discard prior samples from round two onwards.
            summary_writer: SummaryWriter
                Optionally pass summary writer. A way to change the log file location.
                If None, will create one internally, saving logs to cwd/logs.
            device: torch.device
                Optionally pass device
                If None, will infer it
        """

        self._simulator = simulator
        self._prior = prior
        self._true_observation = true_observation
        self._device = get_default_device() if device is None else device
        self.z_score_obs = z_score_obs
        self.num_pilot_samples = num_pilot_samples
        self._use_combined_loss = use_combined_loss
        self._discard_prior_samples = discard_prior_samples
        # Need somewhere to store (parameter, observation) pairs from each round.
        self._parameter_bank, self._observation_bank, self._prior_masks = [], [], []
        self._model_bank = []
        self._retrain_from_scratch_each_round = retrain_from_scratch_each_round

        # run prior samples
        self.pilot_parameters, self.pilot_observations = simulators.simulation_wrapper(
            simulator=self._simulator,
            parameter_sample_fn=lambda num_samples: self._prior.sample((num_samples,)),
            num_samples=num_pilot_samples,
        )

        # create the deep neural density estimator
        if density_estimator is None:
            density_estimator = utils.posterior_nn(
                model="maf", prior=self._prior, context=self._true_observation,
            )

        # create the neural posterior which can sample(), log_prob()
        self._neural_posterior = Posterior(
            algorithm="snpe",
            neural_net=density_estimator,
            prior=prior,
            context=true_observation,
            train_with_mcmc=train_with_mcmc,
            mcmc_method=mcmc_method,
        )

        # obtain z-score for observations and define embedding net
        if self.z_score_obs:
            self.obs_mean = torch.mean(self.pilot_observations, dim=0)
            self.obs_std = torch.std(self.pilot_observations, dim=0)
        else:
            self.obs_mean = torch.zeros(self._true_observation.shape)
            self.obs_std = torch.ones(self._true_observation.shape)

        # new embedding_net contains z-scoring
        if not isinstance(self._neural_posterior.neural_net, MultivariateGaussianMDN):
            embedding = nn.Sequential(
                utils.Normalize(self.obs_mean, self.obs_std),
                self._neural_posterior.neural_net._embedding_net,
            )
            self._neural_posterior.set_embedding_net(embedding)
        elif z_score_obs:
            warnings.warn("z-scoring of observation not implemented for MDNs")

        # calibration kernels proposed in Lueckmann, Goncalves et al 2017
        if calibration_kernel is None:
            self.calibration_kernel = lambda context_input: torch.ones(
                [len(context_input)]
            )
        else:
            self.calibration_kernel = calibration_kernel

        # If we're retraining from scratch each round,
        # keep a copy of the original untrained model for reinitialization.
        self._untrained_neural_posterior = deepcopy(self._neural_posterior)

        # Each run also has a dictionary of summary statistics which are populated
        # over the course of training.
        self._summary = {
            "mmds": [],
            "median-observation-distances": [],
            "negative-log-probs-true-parameters": [],
            "neural-net-fit-times": [],
            "epochs": [],
            "best-validation-log-probs": [],
            "rejection-sampling-acceptance-rates": [],
        }

    def run_inference(self, num_rounds, num_simulations_per_round, **kwargs):
        """
        Runs a round of inference

        Args:
            num_rounds: int
                Number of rounds to run.
            num_simulations_per_round: list or int
                list or int: Number of simulator calls per round.

        Returns: None

        """

        if isinstance(num_simulations_per_round, int):
            num_simulations_per_round = [num_simulations_per_round] * num_rounds

        round_description = ""
        tbar = tqdm(range(num_rounds))
        for round_ in tbar:

            tbar.set_description(round_description)

            # run simulations for the round
            self._run_sims(round_, num_simulations_per_round[round_])

            # Fit posterior using newly aggregated data set.
            self._train(round_=round_, **kwargs)

            # Store models at end of each round.
            self._model_bank.append(deepcopy(self._neural_posterior))
            self._model_bank[-1].neural_net.eval()

            # Update description for progress bar.
            round_description = (
                f"-------------------------\n"
                f"||||| ROUND {round_ + 1} STATS |||||:\n"
                f"-------------------------\n"
                f"Epochs trained: {self._summary['epochs'][-1]}\n"
                f"Best validation performance: {self._summary['best-validation-log-probs'][-1]:.4f}\n\n"
            )

            # Update tensorboard and summary dict.
            self._summary_writer, self._summary = utils.summarize(
                summary_writer=self._summary_writer,
                summary=self._summary,
                round_=round_,
                true_observation=self._true_observation,
                parameter_bank=self._parameter_bank,
                observation_bank=self._observation_bank,
                simulator=self._simulator,
                estimate_acceptance_rate=self._neural_posterior.estimate_acceptance_rate(
                    context=self._true_observation,
                ),
            )
        return self._neural_posterior

    def _get_log_prob_proposal_posterior(self, inputs, context, masks):
        """
        Evaluate the log-probability used for the loss. Depending on
        the algorithm, this evaluates a different term.

        Args:
            inputs: torch.tensor(), parameters theta
            context: torch.tensor(), data x
            masks: torch.tensor(), binary, indicates whether to
                use prior samples

        Returns: log-probability
        """
        raise NotImplementedError

    def _run_sims(
        self, round_, num_simulations_per_round,
    ):
        """
        Runs the simulations at the beginning of each round.

        Args:
            round_: int. Round
            num_simulations_per_round: int. Number of simulations in current round

        Returns:
            self._parameter_bank: torch.tensor. theta used for training
            self._observation_bank: torch.tensor. x used for training
            self._prior_masks: torch.tensor. Masks of 0/1 for each prior sample,
                indicating whether prior sample will be used in current round
        """
        # Generate parameters from prior in first round, and from most recent posterior
        # estimate in subsequent rounds.
        if round_ == 0:
            parameters, observations = simulators.simulation_wrapper(
                simulator=self._simulator,
                parameter_sample_fn=lambda num_samples: self._prior.sample(
                    (num_samples,)
                ),
                num_samples=np.maximum(
                    0, num_simulations_per_round - self.num_pilot_samples
                ),
            )
            parameters = torch.cat(
                (parameters, self.pilot_parameters[:num_simulations_per_round]), dim=0
            )
            observations = torch.cat(
                (observations, self.pilot_observations[:num_simulations_per_round]),
                dim=0,
            )
        else:
            parameters, observations = simulators.simulation_wrapper(
                simulator=self._simulator,
                parameter_sample_fn=lambda num_samples: self._neural_posterior.sample(
                    num_samples, context=self._true_observation,
                ),
                num_samples=num_simulations_per_round,
            )

        # Store (parameter, observation) pairs.
        self._parameter_bank.append(torch.Tensor(parameters))
        self._observation_bank.append(torch.Tensor(observations))
        self._prior_masks.append(
            torch.ones(num_simulations_per_round, 1)
            if round_ == 0
            else torch.zeros(num_simulations_per_round, 1)
        )

    def _train(
        self,
        round_,
        batch_size=100,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=20,
        clip_grad_norm=True,
    ):
        """
        Trains the conditional density estimator for the posterior by maximizing the
        proposal posterior using the most recently aggregated bank of (parameter, observation)
        pairs.
        Uses early stopping on a held-out validation set as a terminating condition.

        Args:
            round_: int
                Which round we're currently in. Needed when sampling procedure is
                not simply sampling from (proposal) marginal.
            batch_size: int
                Size of batch to use for training.
            learning_rate: float
                Learning rate for Adam optimizer.
            validation_fraction: float in [0, 1]
                The fraction of data to use for validation.
            stop_after_epochs: int
                The number of epochs to wait for improvement on the
                validation set before terminating training.
            clip_grad_norm: bool
                Whether to clip norm of gradients or not.

        Returns: None
        """

        # get the start index for what training set to use. Either 0 or 1
        ix = int(self._discard_prior_samples and (round_ > 0))

        # Get total number of training examples.
        num_examples = torch.cat(self._parameter_bank[ix:]).shape[0]

        if round_ > 0:
            assert (
                validation_fraction * num_examples >= batch_size
            ), "There are fewer samples in the validation set than the batchsize."

        # Select random neural_net and validation splits from (parameter, observation) pairs.
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(
            torch.cat(self._parameter_bank[ix:]),
            torch.cat(self._observation_bank[ix:]),
            torch.cat(self._prior_masks[ix:]),
        )

        # Create neural_net and validation loaders using a subset sampler.
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(batch_size, num_validation_examples),
            shuffle=False,
            drop_last=True,
            sampler=SubsetRandomSampler(val_indices),
        )

        optimizer = optim.Adam(
            list(self._neural_posterior.neural_net.parameters()), lr=learning_rate,
        )
        # Keep track of best_validation log_prob seen so far.
        best_validation_log_prob = -1e100
        # Keep track of number of epochs since last improvement.
        epochs_since_last_improvement = 0
        # Keep track of model with best validation performance.
        best_model_state_dict = None

        # If we're retraining from scratch each round, reset the neural posterior
        # to the untrained copy we made at the start.
        if self._retrain_from_scratch_each_round and round_ > 0:
            self._neural_posterior = deepcopy(self._untrained_neural_posterior)
            # self._neural_posterior = deepcopy(self._model_bank[0])

        epochs = 0
        converged = False
        while not converged:

            # Train for a single epoch.
            self._neural_posterior.neural_net.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, context, masks = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                )

                # just do maximum likelihood in the first round
                if round_ == 0:
                    log_prob = self._neural_posterior.log_prob(
                        inputs, context, normalize=False
                    )
                else:  # or call the APT loss
                    log_prob = self._get_log_prob_proposal_posterior(
                        inputs, context, masks
                    )
                loss = -torch.mean(log_prob)
                loss.backward()
                if clip_grad_norm:
                    clip_grad_norm_(
                        self._neural_posterior.neural_net.parameters(), max_norm=5.0
                    )
                optimizer.step()

            epochs += 1

            # Calculate validation performance.
            self._neural_posterior.neural_net.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, context, masks = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                        batch[2].to(self._device),
                    )
                    # just do maximum likelihood in the first round
                    if round_ == 0:
                        log_prob = self._neural_posterior.log_prob(
                            inputs, context, normalize=False
                        )
                    else:
                        log_prob = self._get_log_prob_proposal_posterior(
                            inputs, context, masks
                        )
                    log_prob_sum += log_prob.sum().item()
            validation_log_prob = log_prob_sum / num_validation_examples

            # Check for improvement in validation performance over previous epochs.
            if validation_log_prob > best_validation_log_prob:
                best_validation_log_prob = validation_log_prob
                epochs_since_last_improvement = 0
                best_model_state_dict = deepcopy(
                    self._neural_posterior.neural_net.state_dict()
                )
            else:
                epochs_since_last_improvement += 1

            # If no validation improvement over many epochs, stop training.
            if epochs_since_last_improvement > stop_after_epochs - 1:
                self._neural_posterior.neural_net.load_state_dict(best_model_state_dict)
                converged = True

        # Update summary.
        self._summary["epochs"].append(epochs)
        self._summary["best-validation-log-probs"].append(best_validation_log_prob)


class NeuralPotentialFunction:
    """
    Implementation of a potential function for Pyro MCMC which uses a classifier
    to evaluate a quantity proportional to the likelihood.
    """

    def __init__(self, posterior, prior, true_observation):
        """
        Args:
            posterior: nn
            prior: torch.distribution, Distribution object with 'log_prob' method.
            true_observation:torch.Tensor containing true observation x0.
        """

        self.prior = prior
        self.posterior = posterior
        self.true_observation = true_observation

    def __call__(self, parameters_dict):
        """
        Call method allows the object to be used as a function.
        Evaluates the given parameters using a given neural likelhood, prior,
        and true observation.

        Args:
            parameters_dict: dict of parameter values which need evaluation for MCMC.

        Returns:
            torch.Tensor potential ~ -[log r(x0, theta) + log p(theta)]

        """

        parameters = next(iter(parameters_dict.values()))
        potential = -self.posterior.log_prob(
            inputs=parameters, context=self.true_observation
        )
        if isinstance(self.prior, distributions.Uniform):
            log_prob_prior = self.prior.log_prob(parameters).sum(-1)
        else:
            log_prob_prior = self.prior.log_prob(parameters)
        log_prob_prior[~torch.isinf(log_prob_prior)] = 1
        potential *= log_prob_prior

        return potential

    def evaluate(self, point):
        raise NotImplementedError


class SliceNpNeuralPotentialFunction:
    """
    Implementation of a potential function for Pyro MCMC which uses a classifier
    to evaluate a quantity proportional to the likelihood.
    """

    def __init__(self, posterior, prior, true_observation):
        """
        Args:
            posterior: nn
            prior: torch.distribution, Distribution object with 'log_prob' method.
            true_observation:torch.Tensor containing true observation x0.
        """

        self.prior = prior
        self.posterior = posterior
        self.true_observation = true_observation

    def __call__(self, parameters):
        """
        Call method allows the object to be used as a function.
        Evaluates the given parameters using a given neural likelhood, prior,
        and true observation.

        Args:
            parameters_dict: dict of parameter values which need evaluation for MCMC.

        Returns:
            torch.Tensor potential ~ -[log r(x0, theta) + log p(theta)]

        """

        if not np.isinf(self.prior.log_prob(torch.Tensor(parameters)).sum().item()):
            target_log_prob = self.posterior.log_prob(
                inputs=torch.Tensor(parameters).reshape(1, -1),
                context=self.true_observation.reshape(1, -1),
                normalize=False,
            )
        else:
            target_log_prob = -np.inf

        return target_log_prob

    def evaluate(self, point):
        raise NotImplementedError
