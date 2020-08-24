import pytorch_lightning as pl
from torch import optim, ones, eye
import torch
from sbi.utils import (
    batched_mixture_mv,
    batched_mixture_vmv,
    clamp_and_warn,
    del_entries,
    repeat_rows,
)


class PosteriorNet(pl.LightningModule):
    """
    This is a wrapper class for the neural network defined by pyknos / nflows. It wraps
    the neural network into a pytorch_lightning module.
    """

    def __init__(self, net, prior):
        super().__init__()
        self.net = net
        self.prior = prior

    def forward(self, x):
        # TODO: make sure that it is indeed called forward not log_prob().
        return self.net.forward(x)

    def configure_optimizers(self):
        # TODO: lr is set hard rn. Make configurable.
        optimizer = optim.Adam(list(self.net.parameters()), lr=5e-4,)

    def training_step(self, batch, batch_idx):
        # todo need to get the proposal
        theta, x, masks = batch
        loss = -self.log_prob_proposal_posterior(theta, x, masks)
        result = pl.TrainResult(loss)
        return result

    def validation_step(self, batch, batch_idx):
        theta, x, masks = batch
        loss = -self.log_prob_proposal_posterior(theta, x, masks)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss)
        return result

    def log_prob_proposal_posterior(self, theta, x, masks):
        batch_size = theta.shape[0]

        num_atoms = clamp_and_warn(
            "num_atoms", self._num_atoms, min_val=2, max_val=batch_size
        )

        # Each set of parameter atoms is evaluated using the same x,
        # so we repeat rows of the data x, e.g. [1, 2] -> [1, 1, 2, 2]
        repeated_x = repeat_rows(x, num_atoms)

        # To generate the full set of atoms for a given item in the batch,
        # we sample without replacement num_atoms - 1 times from the rest
        # of the theta in the batch.
        probs = ones(batch_size, batch_size) * (1 - eye(batch_size)) / (batch_size - 1)

        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
        contrasting_theta = theta[choices]

        # We can now create our sets of atoms from the contrasting parameter sets
        # we have generated.
        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_atoms, -1
        )

        # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
        log_prob_posterior = self.net.log_prob(atomic_theta, repeated_x)
        self._assert_all_finite(log_prob_posterior, "posterior eval")
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)

        # Get (batch_size * num_atoms) log prob prior evals.
        log_prob_prior = self._prior.log_prob(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
        self._assert_all_finite(log_prob_prior, "prior eval")

        # Compute unnormalized proposal posterior.
        unnormalized_log_prob = log_prob_posterior - log_prob_prior

        # Normalize proposal posterior across discrete set of atoms.
        log_prob_proposal_posterior = unnormalized_log_prob[:, 0] - torch.logsumexp(
            unnormalized_log_prob, dim=-1
        )
        self._assert_all_finite(log_prob_proposal_posterior, "proposal posterior eval")

        # XXX This evaluates the posterior on _all_ prior samples
        if self._use_combined_loss:
            log_prob_posterior_non_atomic = self._posterior.net.log_prob(theta, x)
            masks = masks.reshape(-1)
            log_prob_proposal_posterior = (
                masks * log_prob_posterior_non_atomic + log_prob_proposal_posterior
            )

        return log_prob_proposal_posterior
