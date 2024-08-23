
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
- [Getting started](00_getting_started_flexible.md)
- [Amortized inference](01_gaussian_amortized.md)
- [Implemented algorithms](16_implemented_methods.md)
</div>

## Advanced

<div class="grid cards" markdown>
- [Multi-round inference](03_multiround_inference.md)
- [Sampling algorithms in sbi](11_sampler_interface.md)
- [Custom density estimators](04_density_estimators.md)
- [Embedding nets for observations](05_embedding_net.md)
- [SBI with trial-based data](14_iid_data_and_permutation_invariant_embeddings.md)
- [Handling invalid simulations](08_restriction_estimator.md)
- [Crafting summary statistics](10_crafting_summary_statistics.md)
- [Importance sampling posteriors](17_importance_sampled_posteriors.md)
</div>

## Diagnostics

<div class="grid cards" markdown>
- [Posterior predictive checks](12_diagnostics_posterior_predictive_check.md)
- [Simulation-based calibration](13_diagnostics_simulation_based_calibration.md)
- [Density plots and MCMC diagnostics with ArviZ](15_mcmc_diagnostics_with_arviz.md)
- [Local-C2ST coverage checks](18_diagnostics_lc2st.md)
</div>


## Analysis

<div class="grid cards" markdown>
- [Conditional distributions](07_conditional_distributions.md)
- [Posterior sensitivity analysis](09_sensitivity_analysis.md)
- [Plotting functionality](19_plotting_functionality.md)
</div>

## Examples

<div class="grid cards" markdown>
- [Hodgkin-Huxley model](Example_00_HodgkinHuxleyModel.md)
- [Decision-making model](Example_01_DecisionMakingModel.md)
</div>
