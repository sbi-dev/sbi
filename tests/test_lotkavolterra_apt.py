import torch
from matplotlib import pyplot as plt
from sbi import inference, simulators, utils

# use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")


def test_lotkavolterra_based_on_gtparams():

    # get simulator and prior
    (
        simulator,
        prior,
        ground_truth_parameters,
        ground_truth_observation,
    ) = simulators.get_simulator_prior_and_groundtruth("lotka-volterra")

    parameter_dim = ground_truth_parameters.shape[0]
    observation_dim = ground_truth_observation.shape[0]

    # create inference method
    inference_method = inference.APT(
        simulator=simulator,
        prior=prior,
        true_observation=ground_truth_observation,
        density_estimator="maf",
        num_atoms=-1,
    )

    # run inference
    inference_method.run_inference(num_rounds=2, num_simulations_per_round=1000)

    # sample posterior
    samples = inference_method.sample_posterior(num_samples=10000)

    posterior_sample_mean = samples.mean(axis=0)
    posterior_sample_std = samples.std(axis=0)

    for th0, m, std in zip(
        ground_truth_parameters, posterior_sample_mean, posterior_sample_std
    ):
        print(th0, m, std)
        assert (
            m - std < th0 and m + std > th0
        ), f"gt parameter outside of posterior mean +- std"
