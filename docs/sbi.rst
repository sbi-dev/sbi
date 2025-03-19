API reference
=============

Prior and simulator
-------------------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   sbi.utils.user_input_checks.process_prior
   sbi.utils.user_input_checks.process_simulator
   sbi.inference.trainers.base.simulate_for_sbi


Neural nets
-----------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   sbi.neural_nets.factory.posterior_nn
   sbi.neural_nets.factory.likelihood_nn
   sbi.neural_nets.factory.classifier_nn
   sbi.neural_nets.factory.flowmatching_nn
   sbi.neural_nets.factory.posterior_score_nn
   sbi.neural_nets.estimators.ConditionalDensityEstimator


Training
--------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   sbi.inference.trainers.npe.npe_c.NPE_C
   sbi.inference.trainers.npe.npe_a.NPE_A
   sbi.inference.trainers.fmpe.fmpe.FMPE
   sbi.inference.trainers.npse.npse.NPSE
   sbi.inference.trainers.nle.nle_a.NLE_A
   sbi.inference.trainers.nre.nre_a.NRE_A
   sbi.inference.trainers.nre.nre_b.NRE_B
   sbi.inference.trainers.nre.nre_c.NRE_C
   sbi.inference.trainers.nre.bnre.BNRE
   sbi.inference.abc.mcabc.MCABC
   sbi.inference.abc.smcabc.SMCABC


Potentials
----------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   sbi.inference.potentials.posterior_based_potential.posterior_estimator_based_potential
   sbi.inference.potentials.likelihood_based_potential.likelihood_estimator_based_potential
   sbi.inference.potentials.ratio_based_potential.ratio_estimator_based_potential
   sbi.inference.potentials.score_based_potential.score_estimator_based_potential


Posteriors
----------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   sbi.inference.posteriors.direct_posterior.DirectPosterior
   sbi.inference.posteriors.importance_posterior.ImportanceSamplingPosterior
   sbi.inference.posteriors.mcmc_posterior.MCMCPosterior
   sbi.inference.posteriors.rejection_posterior.RejectionPosterior
   sbi.inference.posteriors.score_posterior.ScorePosterior
   sbi.inference.posteriors.vi_posterior.VIPosterior


Visualization
-------------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   sbi.analysis.plot.pairplot
   sbi.analysis.plot.marginal_plot
   sbi.analysis.plot.conditional_pairplot
   sbi.analysis.conditional_density.conditional_corrcoeff
