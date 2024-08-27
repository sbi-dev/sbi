
# Tutorials for using the `sbi` toolbox

Before running the notebooks, follow our instructions to [install
sbi](../install.md). Alternatively, you can also open a [codespace on
GitHub](https://codespaces.new/sbi-dev/sbi) and work through the tutorials in
the browser. The numbers of the notebooks are not informative of the order,
please follow this structure depending on which group you identify with.

Once you have familiarised yourself with the methods and identified how to apply
SBI to your use case, ensure you work through the **Diagnostics** tutorials
linked below, to identify failure cases and assess the quality of your
inference.

## Introduction

<div class="grid cards" markdown>
- [Getting started](00_getting_started.md)
- [Amortized inference](01_gaussian_amortized.md)
- [More flexibility for training and sampling](18_training_interface.md)
- [Implemented algorithms](16_implemented_methods.md)
</div>

## Advanced

<div class="grid cards" markdown>
- [Multi-round inference](02_multiround_inference.md)
- [Sampling algorithms in sbi](09_sampler_interface.md)
- [Custom density estimators](03_density_estimators.md)
- [Embedding nets for observations](04_embedding_networks.md)
- [SBI with trial-based data](12_iid_data_and_permutation_invariant_embeddings.md)
- [Handling invalid simulations](06_restriction_estimator.md)
- [Crafting summary statistics](08_crafting_summary_statistics.md)
- [Importance sampling posteriors](15_importance_sampled_posteriors.md)
</div>

## Diagnostics

<div class="grid cards" markdown>
- [Posterior predictive checks](10_diagnostics_posterior_predictive_checks.md)
- [Simulation-based calibration](11_diagnostics_simulation_based_calibration.md)
- [Local-C2ST coverage checks](13_diagnostics_lc2st.md)
- [Density plots and MCMC diagnostics with ArviZ](14_mcmc_diagnostics_with_arviz.md)
</div>

## Analysis

<div class="grid cards" markdown>
- [Conditional distributions](05_conditional_distributions.md)
- [Posterior sensitivity analysis](07_sensitivity_analysis.md)
- [Plotting functionality](17_plotting_functionality.md)
</div>

## Examples

<div class="grid cards" markdown>
- [Hodgkin-Huxley model](Example_00_HodgkinHuxleyModel.md)
- [Decision-making model](Example_01_DecisionMakingModel.md)
</div>
