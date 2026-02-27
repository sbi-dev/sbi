.. This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
.. under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

API reference
=============

.. toctree::
   :maxdepth: 1
   :hidden:

   api_reference/prior_and_simulator
   api_reference/neural_nets
   api_reference/embedding_nets
   api_reference/training
   api_reference/potentials
   api_reference/posteriors
   api_reference/posterior_parameters
   api_reference/diagnostics
   api_reference/visualization
   api_reference/other_utilities


Prior and simulator
-------------------

.. autosummary::
   :nosignatures:

   sbi.utils.process_prior
   sbi.utils.process_simulator
   sbi.utils.BoxUniform
   sbi.utils.MultipleIndependent
   sbi.utils.RestrictedPrior
   sbi.utils.RestrictionEstimator
   sbi.inference.simulate_for_sbi


Neural nets
-----------

.. autosummary::
   :nosignatures:

   sbi.neural_nets.posterior_nn
   sbi.neural_nets.likelihood_nn
   sbi.neural_nets.classifier_nn
   sbi.neural_nets.posterior_score_nn
   sbi.neural_nets.posterior_flow_nn
   sbi.neural_nets.marginal_nn


Embedding nets
--------------

.. autosummary::
   :nosignatures:

   sbi.neural_nets.embedding_nets.CausalCNNEmbedding
   sbi.neural_nets.embedding_nets.CNNEmbedding
   sbi.neural_nets.embedding_nets.FCEmbedding
   sbi.neural_nets.embedding_nets.LRUEmbedding
   sbi.neural_nets.embedding_nets.PermutationInvariantEmbedding
   sbi.neural_nets.embedding_nets.ResNetEmbedding1D
   sbi.neural_nets.embedding_nets.ResNetEmbedding2D
   sbi.neural_nets.embedding_nets.TransformerEmbedding


Training
--------

.. autosummary::
   :nosignatures:

   sbi.inference.NPE_C
   sbi.inference.NPE_A
   sbi.inference.MNPE
   sbi.inference.FMPE
   sbi.inference.NPSE
   sbi.inference.NLE_A
   sbi.inference.NRE_A
   sbi.inference.NRE_B
   sbi.inference.NRE_C
   sbi.inference.BNRE
   sbi.inference.MCABC
   sbi.inference.SMCABC


Potentials
----------

.. autosummary::
   :nosignatures:

   sbi.inference.posterior_estimator_based_potential
   sbi.inference.likelihood_estimator_based_potential
   sbi.inference.ratio_estimator_based_potential
   sbi.inference.vector_field_estimator_based_potential


Posteriors
----------

.. autosummary::
   :nosignatures:

   sbi.inference.DirectPosterior
   sbi.inference.ImportanceSamplingPosterior
   sbi.inference.MCMCPosterior
   sbi.inference.RejectionPosterior
   sbi.inference.VectorFieldPosterior
   sbi.inference.VIPosterior
   sbi.inference.EnsemblePosterior


Posterior Parameters
--------------------

.. autosummary::
   :nosignatures:

   sbi.inference.posteriors.DirectPosteriorParameters
   sbi.inference.posteriors.ImportanceSamplingPosteriorParameters
   sbi.inference.posteriors.MCMCPosteriorParameters
   sbi.inference.posteriors.RejectionPosteriorParameters
   sbi.inference.posteriors.VectorFieldPosteriorParameters
   sbi.inference.posteriors.VIPosteriorParameters


Diagnostics
-----------

.. autosummary::
   :nosignatures:

   sbi.diagnostics.run_sbc
   sbi.diagnostics.check_sbc
   sbi.analysis.sbc_rank_plot
   sbi.diagnostics.run_tarp
   sbi.diagnostics.check_tarp
   sbi.analysis.plot_tarp
   sbi.diagnostics.LC2ST
   sbi.analysis.pp_plot_lc2st
   sbi.diagnostics.get_nltp
   sbi.analysis.pp_plot
   sbi.inference.MarginalTrainer
   sbi.diagnostics.calc_misspecification_logprob
   sbi.diagnostics.calc_misspecification_mmd


Visualization
-------------

.. autosummary::
   :nosignatures:

   sbi.analysis.pairplot
   sbi.analysis.marginal_plot
   sbi.analysis.conditional_pairplot


Other utilities
---------------

.. autosummary::
   :nosignatures:

   sbi.utils.get_density_thresholder
   sbi.utils.transformed_potential
   sbi.utils.mcmc_transform
   sbi.analysis.conditional_corrcoeff
   sbi.analysis.conditional_potential
   sbi.analysis.ActiveSubspace
