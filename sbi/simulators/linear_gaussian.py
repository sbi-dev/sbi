import torch


def linear_gaussian(parameters: torch.Tensor) -> torch.Tensor:

    # fixed variance
    std = 1

    if parameters.ndim == 1:
        parameters = parameters[None, :]

    return parameters + std * torch.randn_like(parameters)


def get_ground_truth_posterior_samples_linear_gaussian(
    observation: torch.Tensor, num_samples: int = 1000, std=1
):
    assert observation.ndim == 2, "needs batch dimension in observation"
    mean = observation
    dim = mean.shape[1]
    std = torch.sqrt(torch.Tensor([std ** 2 / (std ** 2 + 1)]))
    c = torch.Tensor([1 / (std ** 2 + 1)])
    return c * mean + std * torch.randn(num_samples, dim)
