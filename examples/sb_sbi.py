import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from HH_helper_functions import HHsimulator, calculate_summary_statistics, syn_current
from torch.distributions.multivariate_normal import MultivariateNormal

import sbi.utils as utils
from sbi.inference import simulate_for_sbi
from sbi.inference.snpe.snpe_a import SNPE_A
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.utils import pairplot, posterior_nn
from sbi.utils.user_input_checks import prepare_for_sbi

I, t_on, t_off, dt, t, A_soma = syn_current()


def run_HH_model(params):
    params = np.asarray(params)

    # input current, time step
    I, t_on, t_off, dt, t, A_soma = syn_current()

    t = np.arange(0, len(I), 1) * dt

    # initial voltage
    V0 = -70

    states = HHsimulator(V0, params.reshape(1, -1), dt, t, I)

    return dict(data=states.reshape(-1), time=t, dt=dt, I=I.reshape(-1))


def simulation_wrapper(params):
    """
    Returns summary statistics from conductance values in `params`.

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    obs = run_HH_model(params)
    summstats = torch.as_tensor(calculate_summary_statistics(obs))
    return summstats


if __name__ == "__main__":
    # Configure.
    torch.manual_seed(0)
    num_sim = 300
    true_params = np.array([50.0, 5.0])
    labels_params = [r"$g_{Na}$", r"$g_{K}$"]
    observation_trace = run_HH_model(true_params)
    observation_summary_statistics = calculate_summary_statistics(observation_trace)
    method = "SNPE_A"
    num_rounds = 3
    num_components = 4
    # TODO test MVN prior
    prior_min = [0.5, 1e-4]
    prior_max = [80.0, 15.0]
    prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
    )

    # mean = torch.tensor([45, 6.5])
    # cov = torch.tensor([[3 * math.sqrt(45), 0], [0, 3 * math.sqrt(6.5)]])
    # prior = MultivariateNormal(loc=mean, covariance_matrix=cov)

    if method == "SNPE_A":
        density_estimator = "mdn_snpe_a"
        density_estimator = posterior_nn(
            model=density_estimator, num_components=num_components
        )
        snpe = SNPE_A(num_components, num_rounds, prior, density_estimator)

    else:
        density_estimator = "maf"
        density_estimator = posterior_nn(
            model=density_estimator, num_components=num_components
        )
        snpe = SNPE_C(prior, density_estimator)

    simulator, prior = prepare_for_sbi(simulation_wrapper, prior)
    proposal = prior

    fig_th, ax_th = plt.subplots(1)

    # Start multi-round training.
    for r in range(num_rounds + 1):
        # Simulate and append.
        thetas, data_sim = simulate_for_sbi(
            simulator=simulator,
            proposal=proposal,
            num_simulations=num_sim,
            num_workers=20,
        )
        snpe.append_simulations(thetas, data_sim, proposal)

        # Plot the sampled thetas.
        ax_th.scatter(
            x=thetas[:, 0].numpy(), y=thetas[:, 1].numpy(), label=f"round {r}", s=10
        )
        if r == num_rounds:
            break

        # Train.
        density_estimator = snpe.train(retrain_from_scratch_each_round=False)

        if method == "SNPE_A":
            posterior = snpe.build_posterior(
                proposal=proposal,
                density_estimator=density_estimator,
                sample_with_mcmc=False,
            )
        else:
            posterior = snpe.build_posterior(
                density_estimator=density_estimator, sample_with_mcmc=False
            )

        # Pretend we obtained the perfect
        posterior.set_default_x(observation_summary_statistics)
        proposal = posterior

    fig = plt.figure(figsize=(7, 5))
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax = plt.subplot(gs[0])
    plt.plot(observation_trace["time"], observation_trace["data"])
    plt.ylabel("voltage (mV)")
    plt.title("observed data")
    plt.setp(ax, xticks=[], yticks=[-80, -20, 40])

    ax = plt.subplot(gs[1])
    plt.plot(observation_trace["time"], I * A_soma * 1e3, "k", lw=2)
    plt.xlabel("time (ms)")
    plt.ylabel("input (nA)")

    ax.set_xticks(
        [0, max(observation_trace["time"]) / 2, max(observation_trace["time"])]
    )
    ax.set_yticks([0, 1.1 * np.max(I * A_soma * 1e3)])
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))

    # Analysis of the posterior given the observed data
    samples = posterior.sample((10000,), x=observation_summary_statistics)

    fig, axes = pairplot(
        samples,
        limits=[[0.5, 80], [1e-4, 15.0]],
        ticks=[[0.5, 80], [1e-4, 15.0]],
        figsize=(5, 5),
        points=true_params,
        points_offdiag={"markersize": 6},
        points_colors="r",
    )

    # Draw a sample from the posterior and convert to numpy for plotting.
    posterior_sample = posterior.sample((1,), x=observation_summary_statistics).numpy()

    fig = plt.figure(figsize=(7, 5))

    # plot observation
    t = observation_trace["time"]
    y_obs = observation_trace["data"]
    plt.plot(t, y_obs, lw=2, label="observation")

    # simulate and plot samples
    x = run_HH_model(posterior_sample)
    plt.plot(t, x["data"], "--", lw=2, label="posterior sample")

    plt.xlabel("time (ms)")
    plt.ylabel("voltage (mV)")

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.3, 1), loc="upper right")

    ax.set_xticks([0, 60, 120])
    ax.set_yticks([-80, -20, 40])

    plt.show()
