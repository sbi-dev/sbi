# API Reference

## Inference

::: sbi.inference.base.infer
    rendering:
      show_root_heading: true

::: sbi.user_input.user_input_checks.prepare_for_sbi
    rendering:
      show_root_heading: true

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
      
::: sbi.inference.posteriors.likelihood_based_posterior.LikelihoodBasedPosterior
    rendering:
      show_root_heading: true
    selection:
      filters: [ "!^_", "^__", "!^__class__" ]
      inherited_members: true
      
::: sbi.inference.posteriors.ratio_based_posterior.RatioBasedPosterior
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

## Utils

::: sbi.utils.plot.pairplot
    rendering:
      show_root_heading: true
      show_object_full_path: true
