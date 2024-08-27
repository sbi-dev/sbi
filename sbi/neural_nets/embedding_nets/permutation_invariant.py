# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

import torch
from torch import Tensor, nn

from sbi.neural_nets.embedding_nets.fully_connected import FCEmbedding


class PermutationInvariantEmbedding(nn.Module):
    """Permutation invariant embedding network.

    Takes as input a tensor with (batch, permutation_dim, input_dim)
    and outputs (batch, output_dim).

    References:
    Chan et al. (2018): "A likelihood-free inference framework for population genetic
    data using exchangeable neural networks"
    Radev et al. (2020): "BayesFlow: Learning complex stochastic models with invertible
    neural networks"
    """

    def __init__(
        self,
        trial_net: nn.Module,
        trial_net_output_dim: int,
        aggregation_fn: Optional[str] = "sum",
        num_hiddens: int = 100,
        num_layers: int = 2,
        output_dim: int = 20,
        aggregation_dim: int = 1,
    ):
        """Permutation invariant multi-layer NN.

        Applies the trial_net to every trial to obtain trial embeddings.
        It then aggregates the trial embeddings across the aggregation dimension to
        construct a permutation invariant embedding across iid trials.
        The resulting embedding is processed further using an additional fully
        connected net. The input to the final embedding net is the trial_net output
        plus the number of trials N: (batch, trial_net_output_dim + 1)

        If the data x has varying number of trials per batch element, missing trials
        should be encoded as NaNs. In the forward pass, the NaNs are masked.

        Args:
            trial_net: Network to process one trial. The combining_operation is
                applied to its output. Takes as input (batch, input_dim), where
                input_dim is the dimensionality of a single trial. Produces output
                (batch, latent_dim).
                Remark: This network should be large enough as it acts on all (iid)
                inputs seperatley and needs enough capacity to process the information
                of all inputs.
            trial_net_output_dim: Dimensionality of the output of the trial_net.
            aggregation_fn: Function to aggregate the trial embeddings. Defaults to
                taking the sum over the non-nan values.
            num_layers: Number of fully connected layer, minimum of 2.
            num_hiddens: Number of hidden dimensions in fully-connected layers.
            output_dim: Dimensionality of the output.
            aggregation_dim: Dimension along which to aggregate the trial embeddings.
        """
        super().__init__()
        self.trial_net = trial_net
        self.aggregation_dim = aggregation_dim
        assert aggregation_fn in [
            "mean",
            "sum",
        ], "aggregation_fn must be 'mean' or 'sum'."
        self.aggregation_fn = aggregation_fn

        # construct fully connected layers
        self.fc_subnet = FCEmbedding(
            input_dim=trial_net_output_dim + 1,  # +1 to encode number of trials
            output_dim=output_dim,
            num_layers=num_layers,
            num_hiddens=num_hiddens,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Network forward pass.
        Args:
            x: Input tensor (batch_size, permutation_dim, input_dim)
        Returns:
            Network output (batch_size, output_dim).
        """

        # Get number of trials from non-nan entries
        num_batch, max_num_trials = x.shape[0], x.shape[self.aggregation_dim]
        nan_counts = (
            torch.isnan(x)
            .sum(dim=self.aggregation_dim)  # count nans over trial dimension
            .reshape(-1)[:num_batch]  # counts are the same across data dims
            .unsqueeze(-1)  # make it (batch, 1) to match embeddings below
        )
        # number of non-nan trials
        trial_counts = max_num_trials - nan_counts

        # get nan entries
        is_nan = torch.isnan(x)
        # apply trial net with nan entries replaced with 0
        masked_x = torch.nan_to_num(x, nan=0.0)
        trial_embeddings = self.trial_net(masked_x)
        # replace previous nan entries with zeros
        trial_embeddings = trial_embeddings * (~is_nan.all(-1, keepdim=True)).float()

        # Take mean over permutation dimension divide by number of trials
        # (instead of just taking torch.mean) to account for masking.
        if self.aggregation_fn == "mean":
            combined_embedding = (
                trial_embeddings.sum(dim=self.aggregation_dim) / trial_counts
            )
        else:
            combined_embedding = trial_embeddings.sum(dim=self.aggregation_dim)

        assert not torch.isnan(combined_embedding).any(), "NaNs in embedding."

        # add number of trials as additional input
        return self.fc_subnet(torch.cat([combined_embedding, trial_counts], dim=1))
