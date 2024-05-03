import warnings
from typing import Optional, Tuple

import numpy as np
import torch
from nflows.utils import torchutils
from torch import Tensor, nn
from torch.nn import functional as F

from sbi.neural_nets.density_estimators.base import DensityEstimator

"""
Implementation of models based on
C. M. Bishop, "Mixture Density Networks", NCRG Report (1994)

Taken from https://github.com/mackelab/pyknos/blob/main/pyknos/mdn/mdn.py.
See there for copyright.
And adapted by CS.

read this: https://scottroy.github.io/to-precision-or-to-variance.html
"""


class MultivariateGaussianMDNFlex(DensityEstimator):
    """
    Conditional density mixture of multivariate Gaussians, after Bishop [1].

    A multivariate Gaussian mixture with full (rather than diagonal) covariances.

    Adapted to deal with flexibel marginalization.

    [1] Bishop, C.: 'Mixture Density Networks', Neural Computing Research Group Report
    1994 https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
    """

    def __init__(
        self,
        features: int,
        partition: Tensor,
        context_features: int,
        hidden_features: int,
        hidden_net: nn.Module,
        num_components: int,
        custom_initialization=False,
        embedding_net=None,
    ):
        """Mixture of multivariate Gaussians with full diagonal.

        Args:
            features: (Max.) Dimension of output density.
            partition: Tensor of ints, they should sum up to features.
                indicating the partition of the parameter (feature) space
            context_features: Dimension of inputs.
            hidden_features: Dimension of final layer of `hidden_net`.
            hidden_net: A Module which outputs final hidden representation before
                paramterization layers (i.e logits, means, and log precisions).
            num_components: Number of mixture components.
            custom_initialization: XXX
        """

        super().__init__(
            net=hidden_net,
            input_shape=torch.Size([features]),
            condition_shape=torch.Size([context_features]),
        )

        # check if partition is matching the full param space
        assert features == partition.sum()

        self._features = features
        self._partition = partition
        self._len_partition = len(partition)
        self._context_features = context_features
        self._hidden_features = hidden_features
        self._num_components = num_components

        self._num_upper_params = (features * (features - 1)) // 2

        self._row_ix, self._column_ix = np.triu_indices(features, k=1)
        self._diag_ix = range(features)

        # Modules
        self._hidden_net = hidden_net

        self._logits_layer = nn.Linear(hidden_features, num_components)

        self._means_layer = nn.Linear(hidden_features, num_components * features)

        self._unconstrained_diagonal_layer = nn.Linear(
            hidden_features, num_components * features
        )
        self._upper_layer = nn.Linear(
            hidden_features, num_components * self._num_upper_params
        )

        # XXX docstring text
        # embedding_net: NOT IMPLEMENTED
        #         A `nn.Module` which has trainable parameters to encode the
        #         context (conditioning). It is trained jointly with the MDN.
        if embedding_net is not None:
            raise NotImplementedError

        # Constant for numerical stability.
        self._epsilon = 1e-4

        # Initialize mixture coefficients and precision factors sensibly.
        if custom_initialization:
            self._initialize()

    def get_parameter_mask(self, model_mask: Tensor):
        """takes the binary model mask and returns extended mask for model parameters
            dependent on self._partition of the parameter space

        Args:
            model_mask (Tensor): (batch, len(_partition) )

        Returns:
            parameter_mask: (batch, features)
        """
        batchsize = model_mask.size()[0]

        parameter_mask = torch.zeros((batchsize, self._features), dtype=bool)

        count = 0
        for i, item in enumerate(self._partition):
            parameter_mask[:, count : count + item] = model_mask[:, i].repeat(item, 1).T
            count += item

        return parameter_mask

    @staticmethod
    def get_p_masks(parameter_mask: Tensor, n_components: int):
        """returns the masks to marginalize out dims of precision matrix

        Args:
            parameter_mask (Tensor): shape: (batch, n_params)
            n_components (int): number of components of the mdn

        Returns:
            Tensors: boolean masks. shape: 4x (batch,n_components, n_params, n_params)
        """

        # Attention: this could be slow in high dimensions!
        # change to matrix multiplication: New = Old @ Mask (or 1-Mask)

        batch, n_params = parameter_mask.shape

        maskxx = torch.zeros((batch, n_params, n_params), dtype=bool)
        maskyy = torch.zeros((batch, n_params, n_params), dtype=bool)
        maskxy = torch.zeros((batch, n_params, n_params), dtype=bool)
        maskyx = torch.zeros((batch, n_params, n_params), dtype=bool)

        for i in range(batch):
            maskxx[i][parameter_mask[i]] = parameter_mask[i]
            maskyy[i][~parameter_mask[i]] = ~parameter_mask[i]
            maskxy[i][parameter_mask[i]] = ~parameter_mask[i]
            maskyx[i][~parameter_mask[i]] = parameter_mask[i]

        # expand to all components:
        return (
            maskxx.repeat(1, n_components, 1).view(
                batch, n_components, n_params, n_params
            ),
            maskyy.repeat(1, n_components, 1).view(
                batch, n_components, n_params, n_params
            ),
            maskxy.repeat(1, n_components, 1).view(
                batch, n_components, n_params, n_params
            ),
            maskyx.repeat(1, n_components, 1).view(
                batch, n_components, n_params, n_params
            ),
        )

    @staticmethod
    def get_marginalized_precisions(precisions: Tensor, mask: Tensor):
        """_summary_

        Args:
            precisions (Tensor): precisions for full dimensions.
                (batch, n_components, parameter_dim, parameter_dim)
            mask (Tensor): boolean mask for parameters (batch, parameter_dim)
        returns:
            list of marginalizes precisions. shape: (batch,n_components,
                                                    _flexibel_,_flexibel_)
        """

        batch_size, n_components, max_parameter_dim, _ = precisions.size()

        # get number of active parameters per sample
        parameter_dim = mask.sum(-1)

        maskxx, maskyy, maskxy, maskyx = MultivariateGaussianMDNFlex.get_p_masks(
            mask, n_components=n_components
        )
        P = []
        for i in range(batch_size):
            dimyy = max_parameter_dim - parameter_dim[i]
            # P = Pxx - Pxy * Pyy^-1 * Pyx
            P.append(
                precisions[i][maskxx[i]].view(
                    n_components, parameter_dim[i], parameter_dim[i]
                )
                - precisions[i][maskxy[i]].view(n_components, parameter_dim[i], dimyy)
                @ torch.linalg.solve(
                    precisions[i][maskyy[i]].view(n_components, dimyy, dimyy),
                    precisions[i][maskyx[i]].view(
                        n_components, dimyy, parameter_dim[i]
                    ),
                )
            )
        return P

    @staticmethod
    def get_marginalized_precisions_dimension_batched(precisions: Tensor, mask: Tensor):
        """

        Args:
            precisions (Tensor): precisions for full dimensions.
                (batch, n_components, parameter_dim, parameter_dim)
            mask (Tensor): boolean mask for parameters (batch, parameter_dim)
        returns:
            list of marginalizes precisions.
            shape: (_flexibel1_,n_components,_flexibel2_,_flexibel2_)
            sorted by number of active components.
            with length of active_components.unique.
        """

        batch_size, n_components, max_parameter_dim, _ = precisions.size()

        # get number of active parameters per sample
        parameter_dim = mask.sum(-1)

        maskxx, maskyy, maskxy, maskyx = MultivariateGaussianMDNFlex.get_p_masks(
            mask, n_components=n_components
        )

        # occurrent dimensions
        possible_dim = torch.unique(parameter_dim)

        P = []
        for dim in possible_dim:
            dimyy = max_parameter_dim - dim
            # P = Pxx - Pxy * Pyy^-1 * Pyx
            P.append(
                precisions[parameter_dim == dim][maskxx[parameter_dim == dim]].view(
                    torch.sum(parameter_dim == dim), n_components, dim, dim
                )
                - precisions[parameter_dim == dim][maskxy[parameter_dim == dim]].view(
                    torch.sum(parameter_dim == dim), n_components, dim, dimyy
                )
                @ torch.linalg.solve(
                    precisions[parameter_dim == dim][maskyy[parameter_dim == dim]].view(
                        torch.sum(parameter_dim == dim), n_components, dimyy, dimyy
                    ),
                    precisions[parameter_dim == dim][maskyx[parameter_dim == dim]].view(
                        torch.sum(parameter_dim == dim), n_components, dimyy, dim
                    ),
                )
            )
        return P

    def get_mixture_components(
        self, context: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return logits, means, precisions and two additional useful quantities.

        Args:
            context: Input to the MDN, leading dimension is batch dimension.
                    (last components include model information)

        Returns:
            A tuple with logits (num_components),
            means (num_components x output_dim),
            precisions (num_components, output_dim, output_dim),
            diagonal of precision factors (num_components, output_dim),
            precision factors
                (upper triangular precision factorA such that SIGMA^-1 = A^T A.)
            All batched.
        """

        h = self._hidden_net(context)

        # Logits and Means are unconstrained and are obtained directly from the
        # output of a linear layer.
        logits = self._logits_layer(h)
        means = self._means_layer(h).view(-1, self._num_components, self._features)

        # Unconstrained diagonal and upper triangular quantities are unconstrained.
        unconstrained_diagonal = self._unconstrained_diagonal_layer(h).view(
            -1, self._num_components, self._features
        )

        # Elements of diagonal of precision factor must be positive
        # (recall precision factor A such that SIGMA^-1 = A^T A).
        diagonal = F.softplus(unconstrained_diagonal)

        # Create empty precision factor matrix, and fill with appropriate quantities.
        precision_factors = torch.zeros(
            means.shape[0],
            self._num_components,
            self._features,
            self._features,
            device=context.device,
        )
        precision_factors[..., self._diag_ix, self._diag_ix] = diagonal

        # one dimensional feature does not involve upper triangular parameters
        if self._features > 1:
            upper = self._upper_layer(h).view(
                -1, self._num_components, self._num_upper_params
            )
            precision_factors[..., self._row_ix, self._column_ix] = upper

        # Precisions are given by SIGMA^-1 = A^T A.
        precisions = torch.matmul(
            torch.transpose(precision_factors, 2, 3), precision_factors
        )
        # Add epsilon to diagnonal for numerical stability.
        precisions[..., torch.arange(self._features), torch.arange(self._features)] += (
            self._epsilon
        )

        return logits, means, precisions, diagonal, precision_factors

    def log_prob(
        self, inputs: Tensor, context=Optional[Tensor], extended_context=True
    ):  # -> Tensor
        """Return log MoG(inputs|context) where MoG is a mixture of Gaussians density.
            the context should include a binary mask in the last entries.

        The MoG's parameters (mixture coefficients, means, and precisions) are the
        outputs of a neural network.

        Args:
            inputs: Input variable, leading dim interpreted as batch dimension.
                potentially containing nans in the marginalized out dimensions.
            context: Conditioning variable, leading dim interpreted as batch dimension.
            extended_context: if last entries of context contain binary mask
                for the partitioned components
                (which are later extended and the corresponding dimensions
                are marginalized out.)

        Returns:
            Log probability of inputs given context under a MoG model.
        """

        if extended_context:
            parameter_mask = self.get_parameter_mask(context[:, -self._len_partition :])
        else:
            raise NotImplementedError

        logits, means, precisions, _, _ = self.get_mixture_components(context)
        return self.log_prob_mog_flex(inputs, logits, means, precisions, parameter_mask)

    @staticmethod
    def log_prob_mog_flex(
        inputs: Tensor, logits: Tensor, means: Tensor, precisions: Tensor, mask: Tensor
    ):  # -> Tensor
        """
        Return the log-probability of `inputs` under a
        (masked) MoG with specified parameters.

        Unlike the `log_prob()` method, this method is fully detached from the neural
        network and can be used independent of the neural net in case the MoG
        parameters are already known.

        Args:
            inputs: Location at which to evaluate the MoG,
                with nans in marginalized dimensions.
                Shape: (batch_size, parameter_dim)
            logits: Log-weights of each component of the MoG. Shape: (batch_size,
                num_components).
            means: Means of each MoG, shape (batch_size, num_components, parameter_dim).
            precisions: Precision matrices of each MoG. Shape:
                (batch_size, num_components, parameter_dim, parameter_dim).
            mask: boolean mask which dimensions are marginalized out.
                Shape: (batch_size, parameter_dim)

        Returns:
            Log-probabilities of each input.
        """

        # get shapes
        batch_size, n_components, max_parameter_dim = means.size()
        inputs = inputs.view(-1, 1, max_parameter_dim)

        # get number of active parameters per sample
        parameter_dim = mask.sum(-1)

        # extend mask to all components
        # (batch, n_components, max_parameter_dim)
        mask_components = mask.repeat(1, n_components).view(
            -1, n_components, max_parameter_dim
        )

        # We need the precision for the marginalized distribution.
        # P = Pxx - Pxy * Pyy^-1 * Pyx
        # we need to loop and store in a list as we get different shapes per sample
        # but we batch over the number of active dimensions such that we only have to
        # loop over the different active dimensions.
        # P = MultivariateGaussianMDNFlex.get_marginalized_precisions(precisions, mask)
        P = MultivariateGaussianMDNFlex.get_marginalized_precisions_dimension_batched(
            precisions, mask
        )

        # Split up evaluation into parts.
        # 1.:  d = -0.5* (x-mu)^T * P * (x-mu)

        # in full dimension:  d1 = (x-mu)
        d1 = (inputs.expand_as(means) - means).view(
            batch_size, n_components, max_parameter_dim, 1
        )

        # flexible dimensions
        d = torch.zeros(batch_size, n_components)
        logdet_p = torch.zeros(batch_size, n_components)

        # for i in range(batch_size):
        #    # compute dets for normalizing factor
        #    logdet_p[i] = torch.log(torch.det(P[i]))
        #
        #    # P * d1 = P * (x-mu)
        #    d2 =
        #     torch.matmul(P[i], d1[i][mask_components[i]].view(n_components, -1, 1))
        #
        #    # -0.5* d1^T * d2 = -0.5* (x-mu)^T * P * (x-mu)
        #    d[i] = -0.5 * torch.matmul(
        #        torch.transpose(
        #            d1[i][mask_components[i]].view(n_components, -1, 1), 1, 2
        #        ),
        #        d2,
        #    ).view(n_components)

        # occurrent dimensions
        possible_dim = torch.unique(parameter_dim)

        for i, dim in enumerate(possible_dim):
            # compute dets for normalizing factor
            logdet_p[parameter_dim == dim] = torch.log(torch.det(P[i]))

            # P * d1 = P * (x-mu)
            d2 = torch.matmul(
                P[i],
                d1[parameter_dim == dim][mask_components[parameter_dim == dim]].view(
                    torch.sum(parameter_dim == dim), n_components, -1, 1
                ),
            )

            # -0.5* d1^T * d2 = -0.5* (x-mu)^T * P * (x-mu)
            d[parameter_dim == dim] = -0.5 * torch.matmul(
                torch.transpose(
                    d1[parameter_dim == dim][
                        mask_components[parameter_dim == dim]
                    ].view(torch.sum(parameter_dim == dim), n_components, -1, 1),
                    2,
                    3,
                ),
                d2,
            ).view(torch.sum(parameter_dim == dim), n_components)

        # 2. mixing factors
        a = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # 3. normalization constant: b,c
        b = (-(parameter_dim / 2.0) * np.log(2 * np.pi)).repeat(n_components, 1).T
        # shape: (batch, n_components)
        c = 0.5 * logdet_p  # shape: (batch, n_components)

        return torch.logsumexp(a + b + c + d, dim=-1)

    def sample(
        self, num_samples: int, context: Tensor, extended_context=True
    ) -> Tensor:
        """
        Return num_samples independent samples from MoG(inputs | context).

        Generates num_samples samples for EACH item in context batch i.e. returns
        (num_samples * batch_size) samples in total.

        Args:
            num_samples: Number of samples to generate.
            context: Conditioning variable, leading dimension is batch dimension.
            extended_context: if last entries of context contain binary mask
                which dimensions are marginalized out.
        Returns:
            Generated samples: (num_samples, output_dim) with leading batch dimension.
        """

        # Get necessary quantities.
        logits, means, precisions, _, _ = self.get_mixture_components(context)

        # parameter mask
        if extended_context:
            parameter_mask = self.get_parameter_mask(context[:, -self._len_partition :])
        else:
            raise NotImplementedError

        return self.sample_mog_flex(
            num_samples, logits, means, precisions, parameter_mask
        )

    @staticmethod
    def sample_mog_flex(
        num_samples: int,
        logits: Tensor,
        means: Tensor,
        precisions: Tensor,
        mask: Tensor,
    ):  # -> Tensor
        """
        Return samples of a MoG_flex with specified parameters and specific
        parameter_mask.
        Currentyly only parameter-batch of size 1 possible.

        Unlike the `sample()` method, this method is fully detached from the neural
        network and can be used independent of the neural net in case the MoG
        parameters are already known.

        Args:
            num_samples: Number of samples to generate.
            logits: Log-weights of each component of the MoG. Shape: (batch_size,
                num_components).
            means: Means of each MoG. Shape: (batch_size, num_components,
                parameter_dim).
            precisions: precisions of each component of the MoG. Shape:
                (batch_size, num_components, parameter_dim, parameter_dim).
            mask: boolean mask which dimensions are marginalized out.
                Shape: (batch_size, parameter_dim)

        Returns:
            Tensor: Samples from the MoG.
                With zeros added to the last entries, such that we always get
                a full dimensional sample.
        """

        batch_size, n_components, max_parameter_dim = means.shape

        if batch_size > 1:
            raise NotImplementedError
        # at the moment only for one context possible.
        # the batch dimension is still in,
        # but the parameter dimension is reduced to the active dimensions
        # of the one context
        parameter_dim = mask.sum(-1)

        # mask the mean and get marginalized precisions:
        # only works for one context as dims may differ

        # extend mask to all components
        # (batch, n_components, max_parameter_dim)
        mask_components = mask.repeat(1, n_components).view(
            -1, n_components, max_parameter_dim
        )

        means = means[mask_components].reshape(batch_size, n_components, -1)

        precisions = MultivariateGaussianMDNFlex.get_marginalized_precisions(
            precisions, mask
        )[0].unsqueeze(
            0
        )  # returns a list of len = batch. so here only one batch element.

        # We need (batch_size * num_samples) samples in total.
        means, precisions = (
            torchutils.repeat_rows(means, num_samples),
            torchutils.repeat_rows(precisions, num_samples),
        )

        # Normalize the logits for the coefficients.
        coefficients = F.softmax(logits, dim=-1)  # [batch_size, num_components]

        # Choose num_samples mixture components per example in the batch.
        # choices are indeces of sampled compoenent
        choices = torch.multinomial(
            coefficients, num_samples=num_samples, replacement=True
        ).view(-1)  # [batch_size, num_samples]

        # Create dummy index for indexing means and precision factors.
        ix = torchutils.repeat_rows(torch.arange(batch_size), num_samples)

        # Select means and precision factors.
        chosen_means = means[ix, choices, :]  # (batch, num_samples, parameter_dim)

        chosen_precisions = precisions[
            ix, choices, :, :
        ]  # (batch, num_samples, parameter_dim,parameter_dim)
        try:
            samples = (
                torch.distributions.multivariate_normal.MultivariateNormal(
                    chosen_means, precision_matrix=chosen_precisions
                )
                .sample((1,))
                .squeeze(0)
            )
        except ValueError:
            warnings.warn("relaxed precision matrix accuracy.", stacklevel=2)
            # print(chosen_precisions.shape)
            # assure that matrices are symmetric
            dim = chosen_precisions.shape[-1]
            upper_ind = torch.triu_indices(dim, dim, offset=1)
            chosen_precisions[:, upper_ind[0], upper_ind[1]] = chosen_precisions[
                :, upper_ind[1], upper_ind[0]
            ]
            # and add epsilon to diagonals
            samples = (
                torch.distributions.multivariate_normal.MultivariateNormal(
                    chosen_means,
                    precision_matrix=chosen_precisions
                    + torch.eye(chosen_precisions.shape[-1])
                    .repeat(chosen_precisions.shape[0], 1)
                    .reshape(
                        -1, chosen_precisions.shape[-1], chosen_precisions.shape[-1]
                    )
                    * 1e-3,
                )
                .sample((1,))
                .squeeze(0)
            )
        samples = samples.reshape(batch_size, num_samples, parameter_dim)

        # append dummy dimensions for later processing
        samples_appended = torch.zeros(batch_size, num_samples, max_parameter_dim)
        samples_appended[:, :, :parameter_dim] = samples
        return samples_appended

    def _initialize(self) -> None:
        """
        Initialize MDN so that mixture coefficients are approximately uniform,
        and covariances are approximately the identity.
        """

        # Initialize mixture coefficients to near uniform.
        self._logits_layer.weight.data = self._epsilon * torch.randn(
            self._num_components, self._hidden_features
        )
        self._logits_layer.bias.data = self._epsilon * torch.randn(self._num_components)

        # Initialize diagonal of precision factors to inverse of softplus at 1.
        self._unconstrained_diagonal_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._features, self._hidden_features
        )
        self._unconstrained_diagonal_layer.bias.data = torch.log(
            torch.exp(torch.tensor([1 - self._epsilon])) - 1
        ) * torch.ones(
            self._num_components * self._features
        ) + self._epsilon * torch.randn(self._num_components * self._features)

        # Initialize off-diagonal of precision factors to zero.
        self._upper_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params, self._hidden_features
        )
        self._upper_layer.bias.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params
        )
