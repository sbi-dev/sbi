
from typing import Callable, Any
from tqdm.auto import tqdm
import torch
from sbi.inference.potentials.score_based_potential import ScoreBasedPotential
from math import sqrt

@torch.no_grad()
def score_based_sampler(
    score_based_potential: ScoreBasedPotential,
    proposal: Any, 
    drift: Callable,
    diffusion: Callable,
    ts: torch.Tensor,
    dim_theta: int,
    num_samples: int = 1,
    show_progress_bars: bool = True,
      
):
    r"""Returns a sampler for score-based methods.

    Args:
        score_based_potential: The score-based potential function.
        proposal: The proposal (noise) distribution .
        drift: The drift function of the SDE.
        diffusion: The diffusion function of the SDE.

    Returns:
        A sampler for score-based methods.
    """
    iid2, batch, condition_shape = score_based_potential.x_o.shape
    sample_shape = (num_samples, batch)
    theta = proposal.sample(sample_shape)
    delta_t = (1/ts.numel())
    delta_t_sqrt = sqrt(delta_t)
    pbar = tqdm(
        ts,
        disable=not show_progress_bars,
        desc=f"Drawing {num_samples} posterior samples",
    )
    shape = theta.shape

    for t in pbar:
        f = drift(theta, t)
        g = diffusion(theta, t)
        score = score_based_potential(theta, t)

        theta = theta - (f-g**2*score) *delta_t + g * torch.randn(sample_shape + (dim_theta,)) * delta_t_sqrt
        
        theta.reshape(shape)

    return theta
