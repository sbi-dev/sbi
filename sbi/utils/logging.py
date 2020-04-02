import torch

import sbi.simulators as simulators
import sbi.utils as utils


def summarize(
    summary_writer,
    summary,
    round_,
    true_observation,
    parameter_bank,
    observation_bank,
    simulator,
    posterior_samples_acceptance_rate=None,
):
    # get ground truth if available
    try:
        (
            _,
            prior,
            ground_truth_parameters,
            ground_truth_observation,
        ) = simulators.get_simulator_prior_and_groundtruth(simulator.name)
    # Update summaries.
    except:
        pass

    try:
        mmd = utils.unbiased_mmd_squared(
            parameter_bank[-1],
            simulator.get_ground_truth_posterior_samples(num_samples=1000),
        )
        summary["mmds"].append(mmd.item())
    except:
        pass

    try:
        # Median |x - x0| for most recent round.
        median_observation_distance = torch.median(
            torch.sqrt(
                torch.sum(
                    (observation_bank[-1] - true_observation.reshape(1, -1)) ** 2,
                    dim=-1,
                )
            )
        )
        summary["median_observation_distances"].append(
            median_observation_distance.item()
        )

        summary_writer.add_scalar(
            tag="median_observation_distance",
            scalar_value=summary["median_observation_distances"][-1],
            global_step=round_ + 1,
        )

    except:
        pass

    try:
        # KDE estimate of negative log prob true parameters using
        # parameters from most recent round.

        negative_log_prob_true_parameters = -utils.gaussian_kde_log_eval(
            samples=parameter_bank[-1], query=ground_truth_parameters.reshape(1, -1),
        )
        summary["negative_log_probs_true_parameters"].append(
            negative_log_prob_true_parameters.item()
        )

        summary_writer.add_scalar(
            tag="negative_log_prob_true_parameters",
            scalar_value=summary["negative_log_probs_true_parameters"][-1],
            global_step=round_ + 1,
        )
    except:
        pass

    try:
        # Rejection sampling acceptance rate
        summary["rejection_sampling_acceptance-rates"].append(
            posterior_samples_acceptance_rate
        )

        summary_writer.add_scalar(
            tag="rejection_sampling_acceptance_rate",
            scalar_value=summary["rejection_sampling_acceptance_rates"][-1],
            global_step=round_ + 1,
        )
    except:
        pass

    try:
        # Plot most recently sampled parameters.
        parameters = utils.tensor2numpy(parameter_bank[-1])
        figure = utils.plot_hist_marginals(
            data=parameters,
            ground_truth=utils.tensor2numpy(ground_truth_parameters).reshape(-1),
            lims=simulator.parameter_plotting_limits,
        )
        summary_writer.add_figure(
            tag="posterior_samples", figure=figure, global_step=round_ + 1
        )
    except:
        pass

    # Write quantities using SummaryWriter.
    summary_writer.add_scalar(
        tag="epochs_trained",
        scalar_value=summary["epochs"][-1],
        global_step=round_ + 1,
    )

    summary_writer.add_scalar(
        tag="best_validation_log_prob",
        scalar_value=summary["best_validation_log_probs"][-1],
        global_step=round_ + 1,
    )

    if summary["mmds"]:
        summary_writer.add_scalar(
            tag="mmd", scalar_value=summary["mmds"][-1], global_step=round_ + 1,
        )

    summary_writer.flush()

    return summary_writer, summary
