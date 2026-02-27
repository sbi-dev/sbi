.. This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
.. under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

.. _how_to_guide:


How-to guide
============

The `how-to guide` provides brief answers to specific questions. For each of the steps
in the ``sbi`` workflow, it also provides a guide on how to choose among the options
in the ``sbi`` toolbox (e.g., how to choose the inference method?).

.. toctree::
   :maxdepth: 1
   :hidden:

   how_to_guide/prior_and_simulator
   how_to_guide/neural_nets
   how_to_guide/training
   how_to_guide/sampling
   how_to_guide/diagnostics
   how_to_guide/visualization
   how_to_guide/posterior_parameters


Prior and simulator
-------------------

- :doc:`how_to_guide/00_custom_prior`
- :doc:`how_to_guide/01_crafting_summary_statistics`


Neural nets
-----------

- :doc:`how_to_guide/03_choose_neural_net`
- :doc:`how_to_guide/04_embedding_networks`
- :doc:`how_to_guide/20_time_series_embedding`
- :doc:`how_to_guide/08_permutation_invariant_embeddings`
- :doc:`how_to_guide/03_density_estimators`


Training
--------

- :doc:`how_to_guide/06_choosing_inference_method`
- :doc:`how_to_guide/02_multiround_inference`
- :doc:`how_to_guide/07_gpu_training`
- :doc:`how_to_guide/07_save_and_load`
- :doc:`how_to_guide/07_resume_training`
- :doc:`how_to_guide/21_hyperparameter_tuning`


Sampling
--------

- :doc:`how_to_guide/09_sampler_interface`
- :doc:`how_to_guide/10_refine_posterior_with_importance_sampling`
- :doc:`how_to_guide/11_iid_sampling_with_nle_or_nre`
- :doc:`how_to_guide/12_mcmc_diagnostics_with_arviz`
- :doc:`how_to_guide/23_using_pyro_with_sbi`


Diagnostics
-----------

- :doc:`how_to_guide/14_choose_diagnostic_tool`
- :doc:`how_to_guide/15_expected_coverage`
- :doc:`how_to_guide/16_sbc`
- :doc:`how_to_guide/13_diagnostics_lc2st`
- :doc:`how_to_guide/17_tarp`
- :doc:`how_to_guide/18_model_misspecification`


Visualization
-------------

- :doc:`how_to_guide/05_conditional_distributions`


Posterior Parameters
--------------------

- :doc:`how_to_guide/19_posterior_parameters`
