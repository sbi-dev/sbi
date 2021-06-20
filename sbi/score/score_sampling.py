import torch
import numpy as np

from scipy import integrate


def ode_sampler(
    score_model,
    diffusion_coeff,
    input_size,
    batch_size=64,
    atol=1e-5,
    rtol=1e-5,
    device="cpu",
    z=None,
    eps=1e-3,
):
    t = torch.ones(batch_size, device=device)
    if z is None:
        init_x = torch.randn(
            batch_size, input_size, device=device
        ) * score_model.marginal_prob(t)[1].unsqueeze(1)
    else:
        init_x = z

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(
            time_steps, device=device, dtype=torch.float32
        ).reshape((sample.shape[0], 1))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(
        ode_func,
        (1.0, eps),
        init_x.reshape(-1).cpu().numpy(),
        rtol=rtol,
        atol=atol,
        method="RK45",
    )
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x


def Euler_Maruyama_sampler(
    score_model, diffusion_coeff, input_dim, batch_size=64, num_steps=2000, eps=1e-3
):
    t = torch.ones(batch_size)
    init_x = torch.randn(batch_size, input_dim) * score_model.marginal_prob(t)[
        1
    ].unsqueeze(1)
    time_steps = torch.linspace(1.0, eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size) * time_step
            g = diffusion_coeff(batch_time_step).unsqueeze(1)
            mean_x = (
                x + (g ** 2) * score_model(x, batch_time_step.unsqueeze(1)) * step_size
            )
            x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x)
    return mean_x
