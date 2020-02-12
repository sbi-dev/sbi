import sbi.utils as utils
import torch


def summarize(
    summary_writer,
    summary,
    round_,
    true_observation,
    parameter_bank,
    observation_bank,
    simulator,
    estimate_acceptance_rate=None,
):
    # Update summaries.
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
        summary["median-observation-distances"].append(
            median_observation_distance.item()
        )

        summary_writer.add_scalar(
            tag="median-observation-distance",
            scalar_value=summary["median-observation-distances"][-1],
            global_step=round_ + 1,
        )

    except:
        pass

    try:
        # KDE estimate of negative log prob true parameters using
        # parameters from most recent round.
        negative_log_prob_true_parameters = -utils.gaussian_kde_log_eval(
            samples=parameter_bank[-1],
            query=simulator.get_ground_truth_parameters().reshape(1, -1),
        )
        summary["negative-log-probs-true-parameters"].append(
            negative_log_prob_true_parameters.item()
        )

        summary_writer.add_scalar(
            tag="negative-log-prob-true-parameters",
            scalar_value=summary["negative-log-probs-true-parameters"][-1],
            global_step=round_ + 1,
        )
    except:
        pass

    try:
        # Rejection sampling acceptance rate
        rejection_sampling_acceptance_rate = estimate_acceptance_rate()
        summary["rejection-sampling-acceptance-rates"].append(
            rejection_sampling_acceptance_rate
        )

        summary_writer.add_scalar(
            tag="rejection-sampling-acceptance-rate",
            scalar_value=summary["rejection-sampling-acceptance-rates"][-1],
            global_step=round_ + 1,
        )
    except:
        pass

    try:
        # Plot most recently sampled parameters.
        parameters = utils.tensor2numpy(parameter_bank[-1])
        figure = utils.plot_hist_marginals(
            data=parameters,
            ground_truth=utils.tensor2numpy(
                simulator.get_ground_truth_parameters()
            ).reshape(-1),
            lims=_simulator.parameter_plotting_limits,
        )
        summary_writer.add_figure(
            tag="posterior-samples", figure=figure, global_step=round_ + 1
        )
    except:
        pass

    # Write quantities using SummaryWriter.
    summary_writer.add_scalar(
        tag="epochs-trained",
        scalar_value=summary["epochs"][-1],
        global_step=round_ + 1,
    )

    summary_writer.add_scalar(
        tag="best-validation-log-prob",
        scalar_value=summary["best-validation-log-probs"][-1],
        global_step=round_ + 1,
    )

    if summary["mmds"]:
        summary_writer.add_scalar(
            tag="mmd", scalar_value=summary["mmds"][-1], global_step=round_ + 1,
        )

    summary_writer.flush()

    return summary_writer, summary
