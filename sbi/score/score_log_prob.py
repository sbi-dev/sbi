
import torch
import numpy as np
from scipy import integrate
from score_sampling import get_score_fn

from sbi.vi.first_second_order_helpers import jacobian_in_batch
from utils import *

def get_random_vec(shape, type="gaussian"):
    if type.lower() == "gaussian":
        return torch.randn(shape)
    elif type.lower() == "rademacher":
        return torch.randint(shape, low=0, high=2).float() * 2 - 1.
    else:
        raise NotImplementedError("We only implement gaussian or rademacher as type")


def get_div_fn_hutchinson(fn, slice_type="gaussian", n_particles=1):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, context=None):
    with torch.enable_grad():
        x = x.repeat(n_particles,1)
        t = t.reshape(-1,1).repeat(n_particles,1)
        eps = get_random_vec(x.shape, type=slice_type)
        x.requires_grad_(True)
        fn_eps = torch.sum(fn(x, t, context=context) * eps)
        grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        grad_fn_eps.reshape(n_particles,*x.shape[1:])
    x.requires_grad_(False)
    return torch.mean(torch.sum(grad_fn_eps * eps, dim=tuple(range(1,len(x.shape)))))

  return div_fn

def get_div_fn_exact(fn):
    """ Create the divergence function of 'fn' exactly using autograd"""
    def div_fn(x,t, context=None):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = fn(x,t, context=context)
            jac = jacobian_in_batch(y,x)
            trace = jac.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        x.requires_grad_(False)
        return trace
    return div_fn


def get_likelihood_fn(sde, inverse_scaler=lambda x:x, trace_estimator='exact', trace_estimator_kwargs=dict(),
                      rtol=1e-3, atol=1e-3, method='RK45', eps=1e-5):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.
  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.
  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """
  if trace_estimator == "exact":
      get_div_fn = get_div_fn_exact
  else:
      get_div_fn = lambda fn:get_div_fn_hutchinson(fn, **trace_estimator_kwargs)

  def drift_fn(model, x, t,context=None):
    """The drift function of the reverse-time SDE."""
    score_fn = get_score_fn(model, context)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def div_fn(model, x, t, context=None):
    return get_div_fn(lambda xx, tt, context: drift_fn(model, xx, tt, context=context))(x, t, context=context)

  def likelihood_fn(model, sample, context=None):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.
    Args:
      model: A score model.
      data: A PyTorch tensor.
    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    shape = sample.shape

    def ode_func(t, x):
        x = from_flattened_numpy(x[:-shape[0]], shape).to(sample.device).type(torch.float32)
        vec_t = torch.ones(x.shape[0], device=sample.device) * t
        drift = to_flattened_numpy(drift_fn(model, x, vec_t, context=context))
        logp_grad = to_flattened_numpy(div_fn(model, x, vec_t, context=context))
        return np.concatenate([drift, logp_grad], axis=0)

    init = np.concatenate([to_flattened_numpy(sample), np.zeros((shape[0],))], axis=0)
    solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
    zp = solution.y[:, -1]
    z = from_flattened_numpy(zp[:-shape[0]], shape).to(sample.device).type(torch.float32)
    delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(sample.device).type(torch.float32)
    prior_logp = sde.prior_logp(z)
    logp = prior_logp + delta_logp

    return logp

  return likelihood_fn
