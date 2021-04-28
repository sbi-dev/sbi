import torch
from torch import nn


class BayesianLinear(nn.Linear):
    """ This layer will learn the mean and variance of the Gaussian random variables which gives the weight. """

    def __init__(self, *args, **kwargs):
        self.kl_loss = torch.zeros(1)
        self.lam = kwargs.pop("lam", 0.5)
        super().__init__(*args, **kwargs)
        self.logvar_weight = nn.Parameter(torch.zeros(self.weight.shape))
        self.logvar_bias = nn.Parameter(torch.zeros(self.bias.shape))
        self.init_parameters()

    def init_parameters(self) -> None:
        """ Start with prior mean and variance """
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.logvar_weight, -np.log(self.lam))
        nn.init.constant_(self.logvar_bias, -np.log(self.lam))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        sigma_w = torch.exp(0.5 * self.logvar_weight)
        sigma_b = torch.exp(0.5 * self.logvar_bias)

        new_weight = self.weight + sigma_w * torch.randn(self.weight.shape)
        new_bias = self.bias + sigma_b * torch.randn(self.bias.shape)
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
