.. This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
.. under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

Neural nets
===========

The neural network of an inference method is described by a *config* object: one
class per model, holding only the settings that model accepts. Pass it to the
trainer as ``density_estimator``, ``classifier``, or ``vf_estimator``. See
:doc:`../how_to_guide/27_estimator_configs` for how to use them.


Density estimator configs
-------------------------

For ``NPE`` and ``NLE``, passed as ``density_estimator``.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.DensityConfigBase
   sbi.neural_nets.MADEConfig
   sbi.neural_nets.MAFConfig
   sbi.neural_nets.MAFRQSConfig
   sbi.neural_nets.MDNConfig
   sbi.neural_nets.NSFConfig
   sbi.neural_nets.ZukoBPFConfig
   sbi.neural_nets.ZukoGFConfig
   sbi.neural_nets.ZukoMAFConfig
   sbi.neural_nets.ZukoNAFConfig
   sbi.neural_nets.ZukoNCSFConfig
   sbi.neural_nets.ZukoNICEConfig
   sbi.neural_nets.ZukoNSFConfig
   sbi.neural_nets.ZukoSOSPFConfig
   sbi.neural_nets.ZukoUNAFConfig


Pretrained estimator configs
----------------------------

For ``NPE_PFN``, passed as ``density_estimator``. ``TabPFNConfig`` is not a
trainable density estimator for ``NPE`` or ``NLE``.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.TabPFNConfig


Mixed density estimator configs
-------------------------------

For ``MNPE`` and ``MNLE``, where part of the data is discrete.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.MixedConfig


Classifier configs
------------------

For the ``NRE`` variants, passed as ``classifier``.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.ClassifierConfigBase
   sbi.neural_nets.LinearClassifierConfig
   sbi.neural_nets.MLPClassifierConfig
   sbi.neural_nets.ResNetClassifierConfig


Marginal density estimator configs
----------------------------------

For ``MarginalTrainer``, which fits an unconditional density.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.MarginalConfigBase
   sbi.neural_nets.MarginalBPFConfig
   sbi.neural_nets.MarginalGFConfig
   sbi.neural_nets.MarginalMAFConfig
   sbi.neural_nets.MarginalNAFConfig
   sbi.neural_nets.MarginalNCSFConfig
   sbi.neural_nets.MarginalNICEConfig
   sbi.neural_nets.MarginalNSFConfig
   sbi.neural_nets.MarginalSOSPFConfig
   sbi.neural_nets.MarginalUNAFConfig


Vector field estimator configs
------------------------------

For ``FMPE`` and ``NPSE``, passed as ``vf_estimator``. The estimator config
selects flow matching or a score estimator; its ``net`` config selects the
network architecture.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.VectorFieldConfigBase
   sbi.neural_nets.FlowMatchingConfig
   sbi.neural_nets.ScoreConfigBase
   sbi.neural_nets.VEScoreConfig
   sbi.neural_nets.VPScoreConfig
   sbi.neural_nets.SubVPScoreConfig
   sbi.neural_nets.MLPConfig
   sbi.neural_nets.AdaMLPConfig
   sbi.neural_nets.TransformerConfig


Factory functions
-----------------

The factory functions predate the config classes and are kept for backwards
compatibility. They take the model name as a string and return a build
function.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.classifier_nn
   sbi.neural_nets.likelihood_nn
   sbi.neural_nets.marginal_nn
   sbi.neural_nets.posterior_flow_nn
   sbi.neural_nets.posterior_nn
   sbi.neural_nets.posterior_score_nn
