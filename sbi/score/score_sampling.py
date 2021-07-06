import functools

import torch
import numpy as np
import abc

from scipy import integrate
from utils import *
import score_sde

_CORRECTORS = {}
_PREDICTORS = {}





def get_score_fn(model, context):
    model.eval()

    def score_fn(x, t):
        return model(x, t, context=context)

    return score_fn


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]

def get_sampling_fn(sampler_name, sde, **kwargs):
    if sampler_name == "ode_sampler":
        return get_ode_sampler(sde, **kwargs)
    elif sampler_name == "predictor_corrector":
        predictor = get_predictor(kwargs.get("predictor", "euler_maruyama"))
        corrector = get_corrector(kwargs.get("corrector", "none"))
        return get_pc_sampler(sde, predictor, corrector)
    else:
        raise NotImplementedError("We currently only implement ode_sampler and predictor_corrector")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t,h=None, x_prev=None):
        """One update of the predictor.
    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.
    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.
    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.
    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, h=None, x_prev=None):
        dt = -1.0 / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name="adaptive")
class AdaptivePredictor(Predictor):
    def __init__(
        self,
        sde,
        score_fn,
        probability_flow=False,
        eps=1e-7,
        abstol=1e-4,
        reltol=1e-4,
        error_use_prev=True,
        norm="L2_scaled",
        safety=0.9,
        sde_improved_euler=True,
        extrapolation=True,
        exp=0.9,
    ):
        super().__init__(sde, score_fn, probability_flow)
        self.h_min = 1e-10  # min step-size
        self.t = sde.T  # starting t
        self.eps = eps  # end t
        self.abstol = abstol
        self.reltol = reltol
        self.error_use_prev = error_use_prev
        self.norm = norm
        self.safety = safety
        self.sde_improved_euler = sde_improved_euler
        self.extrapolation = extrapolation
        self.exp = exp

        if self.norm == "L2_scaled":

            def norm_fn(x):
                return torch.sqrt(torch.sum((x) ** 2, axis=-1, keepdims=True) / self.n)

        elif self.norm == "L2":

            def norm_fn(x):
                return torch.sqrt(torch.sum((x) ** 2, axis=-1, keepdims=True))

        elif self.norm == "Linf":

            def norm_fn(x):
                return torch.max(torch.abs(x), axis=-1, keepdims=True)

        else:
            raise NotImplementedError(self.norm)
        self.norm_fn = norm_fn

    def update_fn(self, x, t, h, x_prev):
        # Note: both h and t are vectors with batch_size elems (this is because we want adaptive step-sizes for each sample separately)
        my_rsde = self.rsde.sde
        self.n = x.shape[-1]

        # h_ = jnp.expand_dims(h, (1,2,3)) # expand for multiplications
        # t_ = jnp.expand_dims(t, (1,2,3)) # expand for multiplications

        z = torch.randn(x.shape)
        drift, diffusion = my_rsde(x, t)

        if not self.sde_improved_euler:  # Like Lamba's algorithm
            x_mean_new = x - h * drift
            drift_Heun, _ = my_rsde(x_mean_new, t - h)  # Heun's method on the ODE
            if self.extrapolation:  # Extrapolate using the Heun's method result
                x_mean_new = x - (h / 2) * (drift + drift_Heun)
            x_new = x_mean_new + diffusion * torch.sqrt(h) * z
            E = (h / 2) * drift_Heun - drift  # local-error between EM and Heun (ODEs)
            x_check = x_mean_new
        else:
            # Heun's method for SDE (while Lamba method only focuses on the non-stochastic part, this also includes the stochastic part)
            K1_mean = -h * drift
            K1 = K1_mean + diffusion * torch.sqrt(h) * z

            drift_Heun, diffusion_Heun = my_rsde(x + K1, t - h)
            K2_mean = -h * drift_Heun
            K2 = K2_mean + diffusion_Heun * torch.sqrt(h) * z
            E = 1 / 2 * (K2 - K1)  # local-error between EM and Heun (SDEs) (right one)
            # E = 1/2*(K2_mean - K1_mean) # a little bit better with VE, but not that much
            if self.extrapolation:  # Extrapolate using the Heun's method result
                x_new = x + (1 / 2) * (K1 + K2)
                x_check = x + K1
                x_check_other = x_new
            else:
                x_new = x + K1
                x_check = x + (1 / 2) * (K1 + K2)
                x_check_other = x_new

        # Calculating the error-control
        if self.error_use_prev:
            reltol_ctl = (
                torch.maximum(torch.abs(x_prev), torch.abs(x_check)) * self.reltol
            )
        else:
            reltol_ctl = torch.abs(x_check) * self.reltol
        err_ctl = torch.maximum(reltol_ctl, torch.tensor([self.abstol]))

        # Normalizing for each sample separately
        E_scaled_norm = self.norm_fn(E / err_ctl)

        # Accept or reject x_{n+1} and t_{n+1} for each sample separately
        # accept = torch.vmap(lambda a: a <= 1)(E_scaled_norm)
        accept = E_scaled_norm <= 1
        x = torch.where(accept, x_new, x)
        x_prev = torch.where(accept, x_check, x_prev)
        t_ = torch.where(accept, t - h, t)

        # Change the step-size
        h_max = torch.maximum(
            t_ - self.eps, torch.zeros(1)
        )  # max step-size must be the distance to the end (we use maximum between that and zero in case of a tiny but negative value: -1e-10)
        E_pow = torch.where(
            h == 0, h, torch.pow(E_scaled_norm, -self.exp)
        )  # Only applies power when not zero, otherwise, we get nans
        h_new = torch.minimum(h_max, self.safety * h * E_pow)

        return x, x_prev, t_.reshape((-1)), h_new.reshape((-1))


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t,h, x_prev):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G * z
        return x, x_mean


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, score_sde.VPSDE)
            and not isinstance(sde, score_sde.VESDE)
            and not isinstance(sde, score_sde.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, score_sde.VPSDE) or isinstance(sde, score_sde.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size * grad
            x = x_mean + torch.sqrt(step_size * 2) * noise

        return x, x_mean


@register_corrector(name="ald")
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.
  We include this corrector only for completeness. It was not directly used in our paper.
  """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, score_sde.VPSDE)
            and not isinstance(sde, score_sde.VESDE)
            and not isinstance(sde, score_sde.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, score_sde.VPSDE) or isinstance(sde, score_sde.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)

        return x, x_mean


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


def shared_predictor_update_fn(
    x, t, sde, model, predictor, probability_flow, continuous, context,h=None, x_prev=None, **kwargs,
):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = get_score_fn(model, context)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow, **kwargs)
    return predictor_obj.update_fn(x, t,h, x_prev)


def shared_corrector_update_fn(
    x, t, sde, model, corrector, continuous, snr, n_steps, context,
):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = get_score_fn(model, context)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)


def get_adaptive_pc_sampler(sde,predictor, corrector,inverse_scaler=lambda x:x,
    snr=2,
    n_steps=1,
    probability_flow=False,
    denoise=True,
    continuous=True,
    device="cpu",
    h_init=1e-2,
    abstol = 1e-2,
    reltol = 1e-2, 
    error_use_prev=True,
    norm = "L2_scaled",
    safety = .9, 
    extrapolation = True,
    sde_improved_euler=True,
    exp = 0.9):


    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def pc_sampler_adaptive(model,shape, context):
        """ The PC sampler funciton.
        Args:
        model: A score model.
        Returns:
        Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            eps = model.eps
            h = torch.ones(shape[0],1) * h_init
            t = torch.ones(shape[0],1) * sde.T
            N = sde.N -1
            mask = t.flatten() < eps
            for _ in range(N):
                t = t.reshape(-1,1)
                h = h.reshape(-1,1)
                x, x_prev = corrector_update_fn(x, t, context=context, model=model)
                x, x_prev, t,h = predictor_update_fn(x, t, h=h, x_prev=x_prev, context=context, model=model, eps=eps)
                if (t < eps).all():
                    break

        return inverse_scaler(x)

    return pc_sampler_adaptive




def get_pc_sampler(
    sde,
    predictor,
    corrector,
    inverse_scaler=lambda x:x,
    snr=2,
    n_steps=1,
    probability_flow=False,
    denoise=True,
    continuous=True,
    device="cpu",
):
    """Create a Predictor-Corrector (PC) sampler.
  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.
  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
    # Create predictor & corrector update functions

    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def pc_sampler(model,shape, context):
        """ The PC sampler funciton.
    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            eps = model.eps
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, context=context, model=model)
                x, x_mean = predictor_update_fn(x, vec_t, context=context, model=model)

            return inverse_scaler(x_mean if denoise else x)

    return pc_sampler


def get_ode_sampler(
    sde,
    inverse_scaler=lambda x:x,
    denoise=False,
    rtol=1e-3,
    atol=1e-3,
    method="RK45",
    device="cpu",
):
    """Probability flow ODE sampler with the black-box ODE solver.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.
  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
    def denoise_update_fn(model, x, context=None):
        score_fn = get_score_fn(model, context=context)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * model.eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t, context=None):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(model, context=context)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, shape, context, z=None, rtol=rtol, atol=atol, method="RK45"):
        """The probability flow ODE sampler with black-box ODE solver.
    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t, context=context)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func,
                (sde.T, model.eps),
                to_flattened_numpy(x),
                rtol=rtol,
                atol=atol,
                method=method,
            )
            x = (
                torch.tensor(solution.y[:, -1])
                .reshape(shape)
                .to(device)
                .type(torch.float32)
            )

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x, context=context)

            x = inverse_scaler(x)
            return x

    return ode_sampler


