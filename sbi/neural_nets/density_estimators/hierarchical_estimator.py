import functools

import torch

from sbi.neural_nets.density_estimators import DensityEstimator


def split_hierarchical(theta, dim_local):
    return theta[..., :dim_local], theta[..., dim_local:]


def hierachical_simulator(n_extra, dim_local, p_local, simulator=None):
    """Return a hierachical simulator, which returns extra observations.
    """
    if simulator is None:
        return functools.partial(
            hierachical_simulator, n_extra, dim_local, p_local)


    def h_simulator(theta):
        msg = (
            "Hierarchical simulator only work with vector parameters, with "
            f"of shape (n_batch, theta_dim). Got {theta.shape}."
        )
        assert theta.ndim == 2, msg
        n_batch, theta_dim = theta.shape
        local_theta, global_theta = split_hierarchical(theta, dim_local)
        extra_local = p_local.sample((n_batch, n_extra))
        all_theta_local = torch.concatenate(
            (local_theta[:, None], extra_local), dim=1
        )
        all_theta = torch.concatenate(
            (all_theta_local, global_theta.repeat([n_extra+1, 1])
            .view(n_batch, n_extra+1, -1)), dim=2
        )
        observation = simulator(all_theta.view(n_batch * (n_extra+1), -1))
        return observation.view((n_batch,  n_extra + 1, *observation.shape[1:]))

    return h_simulator


class HierarchicalDensityEstimator(DensityEstimator):

    def __init__(
        self, local_flow, global_flow, dim_local, condition_shape,
        embedding_net: torch.nn.Module = torch.nn.Identity()
    ):

        super().__init__()

        self.dim_local = dim_local
        self.local_flow = local_flow
        self.global_flow = global_flow
        self._embedding_net = embedding_net
        self._condition_shape = condition_shape

    @property
    def embedding_net(self):
        return self._embedding_net


    @staticmethod
    def embed_condition(embedding_net, condition, condition_shape):
        '''Embed the condition for the hierarchical flow

        Parameters
        ----------
        condition: torch.Tensor, shape (n_batch, n_extra + 1, *condition_shape)
            The hierarchical condition.

        Returns
        -------
        global_condition: torch.Tensor, shape (n_batch, 2*n_embed)
        local_condition: torch.Tensor, shape (n_batch, n_embed)
        '''
        if condition.ndim < len(condition_shape):
            raise ValueError(
                "condition should be at least with shape (n_extra, *condition_shape) "
                f"but got {condition.shape}. This is likely because there is no "
                "extra observations."
            )
        elif condition.ndim == len(condition_shape):
            batch_condition_shape, n_extra = (), condition.shape[0]
        else:
            *batch_condition_shape, n_extra = condition.shape[:-len(condition_shape)+1]
        condition_shape = condition_shape[1:]  # remove n_extra
        embedded_condition = embedding_net(
            condition.view(-1, *condition_shape)
        ).reshape(*batch_condition_shape, n_extra, -1)

        batch_slice = tuple(slice(None) for _ in range(len(batch_condition_shape)))
        local_slice = (*batch_slice, slice(1))
        agg_slice = (*batch_slice, slice(1, None))

        local_condition = embedded_condition[local_slice]
        agg_condition = torch.mean(
            embedded_condition[agg_slice],
            dim=len(batch_condition_shape), keepdim=True
        )
        global_condition = torch.concatenate(
            (local_condition, agg_condition), dim=len(batch_condition_shape)
        )
        return (
            local_condition.view(*batch_condition_shape, -1),
            global_condition.view(*batch_condition_shape, -1),
        )

    def log_prob(self, theta, condition):
        local_theta, global_theta = split_hierarchical(theta, self.dim_local)
        local_condition, global_condition = self.embed_condition(
            self.embedding_net, condition, self._condition_shape
        )

        log_p_global = self.global_flow.log_prob(
            global_theta, global_condition
        )

        local_condition = torch.concatenate(
            (local_condition, global_theta), dim=-1
        )
        log_p_local = self.local_flow.log_prob(
            local_theta, local_condition
        )
        return log_p_global + log_p_local

    def loss(self, inputs, condition):
        return -self.log_prob(inputs, condition)

    def sample(self, sample_shape, condition):
        local_condition, global_condition = self.embed_condition(
            self.embedding_net, condition, self._condition_shape
        )

        # shape (n_samples, 1)
        global_samples = self.global_flow.sample(
            sample_shape, global_condition
        )
        local_condition = torch.concatenate(
            (local_condition.repeat((*sample_shape, 1)), global_samples), dim=-1
        )
        local_samples = self.local_flow.sample((1,), local_condition)[:, 0]

        samples = torch.cat([local_samples, global_samples], dim=-1)
        return samples

