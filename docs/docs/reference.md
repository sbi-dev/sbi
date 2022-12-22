# API Reference

## Inference

::: sbi.inference.base.infer
    rendering:
      show_root_heading: true

::: sbi.utils.user_input_checks.prepare_for_sbi
    rendering:
      show_root_heading: true

::: sbi.inference.base.simulate_for_sbi
    rendering:
      show_root_heading: true

::: sbi.inference.snpe.snpe_a.SNPE_A
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.snpe.snpe_c.SNPE_C
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.snle.snle_a.SNLE_A
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.snre.snre_a.SNRE_A
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.snre.snre_b.SNRE_B
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.snre.snre_c.SNRE_C
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.snre.bnre.BNRE
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.abc.mcabc.MCABC
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.abc.smcabc.SMCABC
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

## Posteriors

::: sbi.inference.posteriors.direct_posterior.DirectPosterior
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.posteriors.importance_posterior.ImportanceSamplingPosterior
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.posteriors.mcmc_posterior.MCMCPosterior
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.posteriors.rejection_posterior.RejectionPosterior
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.posteriors.vi_posterior.VIPosterior
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

## Models

::: sbi.utils.get_nn_models.posterior_nn
    rendering:
      show_root_heading: true
      show_object_full_path: true

::: sbi.utils.get_nn_models.likelihood_nn
    rendering:
      show_root_heading: true
      show_object_full_path: true

::: sbi.utils.get_nn_models.classifier_nn
    rendering:
      show_root_heading: true
      show_object_full_path: true

## Potentials

::: sbi.inference.potentials.posterior_based_potential.posterior_estimator_based_potential
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.potentials.likelihood_based_potential.likelihood_estimator_based_potential
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true

::: sbi.inference.potentials.ratio_based_potential.ratio_estimator_based_potential
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true
  
## Analysis

::: sbi.analysis.plot.pairplot
    rendering:
      show_root_heading: true
      show_object_full_path: true

::: sbi.analysis.plot.marginal_plot
    rendering:
      show_root_heading: true
      show_object_full_path: true

::: sbi.analysis.plot.conditional_pairplot
    rendering:
      show_root_heading: true
      show_object_full_path: true

::: sbi.analysis.conditional_density.conditional_corrcoeff
    rendering:
      show_root_heading: true
      show_object_full_path: true
