# API Reference

## Inference - helpers

::: sbi.inference.base.infer

::: sbi.utils.user_input_checks.prepare_for_sbi

::: sbi.inference.base.simulate_for_sbi


## Inference - methods

::: sbi.inference.snpe.snpe_a.SNPE_A
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.snpe.snpe_c.SNPE_C
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.snle.snle_a.SNLE_A
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.snre.snre_a.SNRE_A
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.snre.snre_b.SNRE_B
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.snre.snre_c.SNRE_C
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.snre.bnre.BNRE
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.abc.mcabc.MCABC
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.abc.smcabc.SMCABC
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

## Posteriors

::: sbi.inference.posteriors.direct_posterior.DirectPosterior
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.posteriors.importance_posterior.ImportanceSamplingPosterior
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.posteriors.mcmc_posterior.MCMCPosterior
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.posteriors.rejection_posterior.RejectionPosterior
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.posteriors.vi_posterior.VIPosterior
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

## Models

::: sbi.utils.get_nn_models.posterior_nn
      show_object_full_path: true

::: sbi.utils.get_nn_models.likelihood_nn
      show_object_full_path: true

::: sbi.utils.get_nn_models.classifier_nn
      show_object_full_path: true

::: sbi.neural_nets.density_estimators.DensityEstimator

## Potentials

::: sbi.inference.potentials.posterior_based_potential.posterior_estimator_based_potential
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.potentials.likelihood_based_potential.likelihood_estimator_based_potential
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.potentials.ratio_based_potential.ratio_estimator_based_potential
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

## Analysis

::: sbi.analysis.plot.pairplot
      show_object_full_path: true

::: sbi.analysis.plot.marginal_plot
      show_object_full_path: true

::: sbi.analysis.plot.conditional_pairplot
      show_object_full_path: true

::: sbi.analysis.conditional_density.conditional_corrcoeff
      show_object_full_path: true
