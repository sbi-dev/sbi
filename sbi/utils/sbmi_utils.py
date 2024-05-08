from copy import deepcopy
from itertools import product

import scipy as scp
import torch
from torch import Tensor

from sbi.inference.posteriors.direct_posterior import DirectPosterior


def get_parameter_mask(model_mask: Tensor, partition: Tensor):
    """takes the binary model mask and returns an extended mask for the model paramters
        dependent on partition of the parameter space

    Args:
        model_mask (Tensor): (batch, len(partition) )
        partition (Tensor): tensor of ints, the partition on the param space

    Returns:
        parameter_mask: (batch, features)
    """
    batchsize = model_mask.size()[0]
    dim = partition.sum()

    parameter_mask = torch.zeros((batchsize, dim), dtype=bool)

    count = 0
    for i, item in enumerate(partition):
        parameter_mask[:, count : count + item] = model_mask[:, i].repeat(item, 1).T
        count += item

    return parameter_mask


def evaluate_model_performance(x, sampler, dim, n_samples=100, data_dim=1):
    """samples model predictions, and returns the mean distance to true model.
        per context.

    Args:
        x (tensor): contexts with model mask (batch, x_shape)
        sampler (SBMISAmplingObject): sbmi sampler
        dim (int, optional): dimension of model space.
        n_samples (int, optional): number of samples per context. Defaults to 100.
        data_dim (int, optional): shape of the simulator output (2 for DDM).
            Defaults to 1.

    Returns:
        Tensor
    """
    _sampler = deepcopy(sampler)

    batch = x.shape[0]

    y = torch.zeros(batch, n_samples, dim)

    for i, x_i in enumerate(x):
        for j in range(n_samples):
            if data_dim == 1:
                y[i, j] = abs(_sampler.sample_model(1, x_i[:-dim]) - x_i[-dim:])
            elif data_dim == 2:
                y[i, j] = abs(_sampler.sample_model(1, x_i[:-dim]) - x_i[-dim:, 0])

    out = 1 - y
    overall_performance = ((out.sum(2) == dim).sum(1) / n_samples).mean()
    return out.mean(1), overall_performance


class SBMISamplingObject:
    def __init__(
        self,
        inference,
        model_posterior,
        embedding_net,
        partition: Tensor,
        data_dim=1,
        model_posterior_type="grassmann",
    ):
        """wrapper for sbmi posterior
        makes it easier to sample from the sbmi posterior,
        automatically adds/removes parameter masks etc.

        Args:
            inference (_type_): sbi inference object or DirectPosterior object
            model_posterior (_type_): _description_
            embedding_net (_type_): _description_
            partition (list): _description_
            data_dim (int, optional): shape of the simulator output (2 for DDM).
                Defaults to 1.
        """
        self.inference = inference
        self.model_posterior = model_posterior
        self.embedding_net = embedding_net
        self.partition = partition
        self.len_partition = len(partition)
        self.n_params = sum(partition)
        self.data_dim = data_dim
        self.model_posterior_type = model_posterior_type

        model_posterior.eval()
        embedding_net.eval()

    def sample_model(self, n: int, x: Tensor):
        """samples models from the model posterior give x

        Args:
            n (int): number of models to sample
            x (Tensor): (context_shape), context without model mask
        """
        if self.data_dim == 1:
            dummy_x = torch.cat((x, torch.ones(self.len_partition) * torch.nan))
        elif self.data_dim == 2:
            dummy_x = torch.cat((x, torch.ones(self.len_partition, 2) * torch.nan))

        embedded_x = self.embedding_net(dummy_x.unsqueeze(0))[:, : -self.len_partition]

        model_mask = torch.zeros(n, self.len_partition)
        for i in range(n):
            model_mask[i] = self.model_posterior.sample(1, context=embedded_x)

        del embedded_x

        return model_mask.detach()

    def model_sample_mean(self, n: int, x: Tensor):
        samples = self.sample_model(n, x)
        return samples.mean(0)

    def inflate_theta(self, theta_raw, model_mask, mode="nan"):
        """inflates theta

        Args:
            theta_raw (_type_): small theta with only active parameters
            model_mask (_type_): _description_
            partition (_type_): _description_

        Returns:
            tensor: full theta with nan/0 for non active components
        """

        if mode == "nan":
            theta = torch.ones(self.partition.sum()) * torch.nan
        elif mode == "zero":
            theta = torch.zeros(self.partition.sum())
        else:
            theta = torch.ones(self.partition.sum())

        count_init = 0
        count_raw = 0
        for i in range(len(self.partition)):
            if model_mask[i] == 1:
                theta[count_init : count_init + self.partition[i]] = theta_raw[
                    count_raw : count_raw + self.partition[i]
                ]
            count_init += self.partition[i]
            count_raw += int(model_mask[i] * self.partition[i])
        return theta

    def theta_log_prob(self, x_i, model_mask, theta, **kwargs):
        """calculates the log_prob of theta given a specific model mask

        Args:
            x_i (_type_): x without model mask
            model_mask (_type_): the model mask
            theta (_type_): theta with nans in the non active components
        returns: log_prob
        """
        # replace nan with 0 st sbi posterior does not give infs
        theta = deepcopy(theta)
        theta[torch.isnan(theta)] = 0

        if self.data_dim == 1:
            x_m = torch.cat([x_i, model_mask])
        elif self.data_dim == 2:
            temp = torch.ones(model_mask.shape[0], 2)
            temp[:, 0] = model_mask
            x_m = torch.cat([x_i, temp])

        if type(self.inference) == DirectPosterior:
            sbi_posterior = self.inference
        else:
            sbi_posterior = self.inference.build_posterior()
        sbi_posterior.set_default_x(x_m)

        return sbi_posterior.log_prob(theta, **kwargs)

    def map_dirty(
        self, x_i, model_mask, runs=1, return_opt_result=False, verbose=False
    ):
        """computing MAP quick and dirty for thetas.
        using scipy optimizer on restricted = active parameters.

        Args:
            x_i (_type_): context withou model mask
            model_mask (_type_): _description_
            partition (_type_): _description_
            posterior (_type_): _description_
            runs (int, optional): _description_. Defaults to 1.

        Remark: self.set_broad_prior() is called, with default parameters.
        the broad bounds need to be changed for other models.

        Returns:
            tensor of tuple: MAP or (MAP, optimization result)
        """
        if self.data_dim == 1:
            x_m = torch.cat([x_i, model_mask])
        elif self.data_dim == 2:
            temp = torch.ones(model_mask.shape[0], 2)
            temp[:, 0] = model_mask
            x_m = torch.cat([x_i, temp])

        if type(self.inference) == DirectPosterior:
            sbi_posterior = self.inference
        else:
            sbi_posterior = self.inference.build_posterior()

        sbi_posterior.set_default_x(x_m)

        def optim_f(theta_small):
            theta = self.inflate_theta(
                torch.tensor(theta_small), model_mask, mode="zero"
            )
            return -sbi_posterior.log_prob(theta.type(torch.FloatTensor))

        optimum = 1e10

        for _ in range(runs):
            # theta_init = torch.zeros(self.partition.sum(), dtype=torch.float32)
            theta_init_small = self.sample_theta(1, x_i, model_mask, verbose=verbose)[
                0
            ][: self.partition[model_mask.type(torch.BoolTensor)].sum()]

            res1 = scp.optimize.minimize(optim_f, theta_init_small)
            if res1.fun < optimum:
                optimum = res1.fun
                res = res1

        x_opt = torch.tensor(res.x)

        if return_opt_result:
            return x_opt, res
        else:
            return x_opt

    def p_model_mask(self, model_masks, x, n=10):
        """computes the probability of a model mask

        Args:
            model_masks (_type_): (batch,len(partition))
            x (Tensor): (context_shape), context without model mask
            n (int): number of internal MADE masks to sample

        Returns:
            Tensor: posterior probability of model mask
        """

        if self.data_dim == 1:
            dummy_x = torch.cat((x, torch.ones(self.len_partition) * torch.nan)).float()
        elif self.data_dim == 2:
            dummy_x = torch.cat((
                x,
                torch.ones(self.len_partition, 2) * torch.nan,
            )).float()

        embedded_x = self.embedding_net(dummy_x.unsqueeze(0))[:, : -self.len_partition]

        p = torch.zeros(model_masks.shape[0])

        if self.model_posterior_type == "grassmann":
            for j, model_mask in enumerate(model_masks.float()):
                p[j] = self.model_posterior.forward(
                    model_mask.unsqueeze(0), context=embedded_x
                )
        elif self.model_posterior_type == "categorical":
            for j, model_mask in enumerate(model_masks.float()):
                p[j] = torch.exp(
                    self.model_posterior.forward(
                        model_mask.unsqueeze(0), context=embedded_x
                    )
                )
        else:
            raise Warning(
                f"{self.model_posterior_type} model_posterior not yet implemented here."
            )

        return p.detach().clone()

    def map_model_mask(self, x: Tensor, verbose=False, return_all=False):
        """evaluates the model posterior of ALL possible models
        Attention: this is only feasible for small models

        Args:
            x (Tensor): (context_shape), context without model mask
            verbose (bool, optional): print intermediate results. Defaults to False.
        returns: map and p(map) OR map, p(map), masks_all, p(masks)

        """

        all_masks = torch.tensor(
            [item for item in product(range(2), repeat=len(self.partition))],
            dtype=torch.float,
        )

        p_m = torch.zeros(all_masks.shape[0])

        if self.data_dim == 1:
            dummy_x = torch.cat((x, torch.ones(self.len_partition) * torch.nan))
        elif self.data_dim == 2:
            dummy_x = torch.cat((x, torch.ones(self.len_partition, 2) * torch.nan))

        embedded_x = self.embedding_net(dummy_x.unsqueeze(0))[:, : -self.len_partition]

        map_model = {"p": -1}
        loss_all = 0

        for j, model_mask in enumerate(all_masks):
            out = self.model_posterior.forward(
                model_mask.unsqueeze(0), context=embedded_x
            )

            if self.model_posterior_type == "grassmann":
                loss = out
                p_m[j] = loss.detach().clone()
                loss_all += loss
            elif self.model_posterior_type == "categorical":
                loss = torch.exp(out)
                p_m[j] = loss.detach().clone()
                loss_all += loss
            else:
                raise Warning(
                    f"{self.model_posterior_type} \
                        model_posterior not yet implemented here."
                )

            if verbose:
                print(f"mask: {model_mask}, loss: {loss}")

            if loss > map_model["p"]:
                map_model["mask"] = model_mask
                map_model["p"] = loss.detach()

        if verbose:
            print(f"sum loss {loss_all}")

        if return_all:
            return map_model["mask"], map_model["p"], all_masks, p_m
        else:
            return map_model["mask"], map_model["p"]

    def sample_theta(self, n: int, x: Tensor, model_mask: Tensor, verbose=True):
        """samples theta from the sbi_posterior

        Args:
            n (int): number of samples to draw from sbi_posterior
            x (Tensor): (context_shape), context without model mask
            model_mask (Tensor): (model_components)

        Returns:
            _type_: _description_
        """

        if self.data_dim == 1:
            x_o = torch.cat((x, model_mask), 0)
        elif self.data_dim == 2:
            inflated_model_mask = torch.zeros(model_mask.shape[0], 2)
            inflated_model_mask[:, 0] = model_mask
            x_o = torch.cat((x, inflated_model_mask), 0).unsqueeze(0)

        # change the prior to match the model mask
        # (needs to be done for rejection sampling to work properly)

        if type(self.inference) == DirectPosterior:
            sbi_posterior = self.inference
        else:
            sbi_posterior = self.inference.build_posterior()

        samples = sbi_posterior.sample((n,), x=x_o, show_progress_bars=verbose)

        samples = self.postprocess_samples(samples, model_mask.unsqueeze(0))
        return samples.detach()

    def sample(self, n: int, x: Tensor, verbose=False):
        """samples one model and n corresponding theta

        Args:
            n (int): n_samples for theta
            x (Tensor): context w/o model mask
        """

        model_mask = self.sample_model(1, x).squeeze()
        if model_mask.sum() == 0:
            raise ValueError(
                "no model component sampled. so no parameters can be sampled. "
            )

        if verbose:
            print("sampled model_mask:", model_mask)
        thetas = self.sample_theta(n, x, model_mask, verbose=verbose)

        return model_mask.detach(), thetas

    def postprocess_samples(self, thetas: Tensor, model_mask: Tensor):
        """cuts the full theta samples to the appropriate length of model mask

        Args:
            thetas (Tensor): samples with attached dummy dimensions
            model_mask (Tensor): used model mask to generate samples
        """

        parameter_mask = get_parameter_mask(model_mask, self.partition).squeeze(0)
        parameter_dim = parameter_mask.sum(-1)

        return thetas[:, :parameter_dim]
