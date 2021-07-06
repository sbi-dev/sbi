import torch
import numpy as np
from torch import nn

import typing
from first_second_order_helpers import diagonal_hessian


# For uncertainity quantification, REMOVE

class BayesianLinear(nn.Linear):
    """ This layer will learn the mean and variance of the Gaussian random variables which gives the weight. """

    def __init__(self, *args, **kwargs):
        self.kl_loss = torch.zeros(1)
        self.lam = kwargs.pop("lam", 0.5)
        super().__init__(*args, **kwargs)
        self.logvar_weight = nn.Parameter(torch.zeros(self.weight.shape))
        self.logvar_bias = nn.Parameter(torch.zeros(self.bias.shape))
        self.mean = False
        self.init_parameters()

    def init_parameters(self) -> None:
        """ Start with prior mean and variance """
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.logvar_weight, -np.log(self.lam))
        nn.init.constant_(self.logvar_bias, -np.log(self.lam))

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Posterior scales
        sigma_w = torch.exp(0.5 * self.logvar_weight)
        sigma_b = torch.exp(0.5 * self.logvar_bias)

        # Then the posterior mean is used as point estimate
        if self.mean:
            sigma_w = 0
            sigma_b = 0

        # Sample weights and biases
        new_weight = self.weight + sigma_w * torch.randn(self.weight.shape)
        new_bias = self.bias + sigma_b * torch.randn(self.bias.shape)

        # Accumulate KL divergence terms for training
        if self.training:
            l2 = torch.norm(self.weight) ** 2 + torch.norm(self.bias) ** 2
            num_elements = new_weight.numel() + new_bias.numel()
            kl_loss = 0.5 * (
                self.lam
                * (
                    l2
                    + torch.sum(self.logvar_weight.exp())
                    + torch.sum(self.logvar_bias.exp())
                )
                + num_elements * np.log(1 / self.lam)
                - self.logvar_weight.sum()
                - self.logvar_bias.sum()
                - num_elements
            )
            self.kl_loss = kl_loss.flatten()
        return torch.functional.F.linear(input, new_weight, new_bias)


def replace_linear(model):
    """This will replace all linear layers with Bayesian Linear layers
    """
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_linear(module)

        if isinstance(module, torch.nn.Linear):
            ## simple module
            setattr(
                model,
                n,
                BayesianLinear(
                    in_features=module.in_features, out_features=module.out_features
                ),
            )


def collect_kl_loss(module):
    """ This will collect the KL loss if one trains with SVI """
    loss = 0
    for m in module.modules():
        if hasattr(m, "kl_loss"):
            loss += m.kl_loss
    return loss


def mean(module, val: bool):
    """ This will switch to MAP mode """
    for m in module.modules():
        if hasattr(m, "mean"):
            m.mean = val


def transform_to_BNN(net, module_names_to_transform=None):
    """ This will change all linear layers to bayesian linear layers"""
    if module_names_to_transform is not None:
        modules = torch.nn.ModuleList(
            [m for name, m in net.modules if name in module_names_to_transform]
        )
    else:
        modules = net
    replace_linear(modules)
    setattr(modules, "kl_loss", collect_kl_loss)
    setattr(modules, "mean", lambda x: mean(modules, x))


def laplace_approximation(model, data, layers, prior_precision=1, min_var=1e-6):
    """This will perform an diagonal guassian laplace approximation.
    """
    for layer in layers:
        loss = model.log_prob(*data).mean()
        diag_hessian = diagonal_hessian(loss, layer.parameters())
        parameter_distributions = []
        for mean, val in zip(layer.parameters(), diag_hessian):
            variance = torch.clamp(-1 / (val - prior_precision), min=min_var)
            q_w = torch.distributions.Normal(mean.data.clone(), variance)
            parameter_distributions.append(q_w)

        def resample_weights():
            for para, q_w in zip(layer.parameters(), parameter_distributions):
                para.data = q_w.sample()

        setattr(layer, "resample_weights", resample_weights)


def resample_weights(model):
    """ This will resample the network weights if a laplace approximation was perforemed."""
    for module in model.modules():
        if hasattr(module, "resample_weights"):
            module.resample_weights()
