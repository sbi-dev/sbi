.. This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
.. under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

Neural nets
===========

Conditional density configs
---------------------------

Used by ``NPE`` and ``NLE``.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.MADEConfig
   sbi.neural_nets.MAFConfig
   sbi.neural_nets.MAFRQSConfig
   sbi.neural_nets.MDNConfig
   sbi.neural_nets.NSFConfig
   sbi.neural_nets.TabPFNConfig
   sbi.neural_nets.ZukoBPFConfig
   sbi.neural_nets.ZukoGFConfig
   sbi.neural_nets.ZukoMAFConfig
   sbi.neural_nets.ZukoNAFConfig
   sbi.neural_nets.ZukoNCSFConfig
   sbi.neural_nets.ZukoNICEConfig
   sbi.neural_nets.ZukoNSFConfig
   sbi.neural_nets.ZukoSOSPFConfig
   sbi.neural_nets.ZukoUNAFConfig


Mixed config
------------

For data with continuous and categorical parts, used by ``MNPE`` and ``MNLE``.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.MixedConfig


Classifier configs
------------------

Used by ``NRE``.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.LinearClassifierConfig
   sbi.neural_nets.MLPClassifierConfig
   sbi.neural_nets.ResNetClassifierConfig


Marginal density configs
-------------------------

Used by ``MarginalTrainer``.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.MarginalBPFConfig
   sbi.neural_nets.MarginalGFConfig
   sbi.neural_nets.MarginalMAFConfig
   sbi.neural_nets.MarginalNAFConfig
   sbi.neural_nets.MarginalNCSFConfig
   sbi.neural_nets.MarginalNICEConfig
   sbi.neural_nets.MarginalNSFConfig
   sbi.neural_nets.MarginalSOSPFConfig
   sbi.neural_nets.MarginalUNAFConfig


Vector field builder
--------------------

Used by ``FMPE`` and ``NPSE``.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.VectorFieldEstimatorBuilder


Deprecated factory functions
----------------------------

Deprecated since v0.28.0, to be removed in v0.29.0. Pass the matching config above
instead, e.g. ``posterior_nn(model="nsf")`` becomes ``NSFConfig()``.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   sbi.neural_nets.classifier_nn
   sbi.neural_nets.likelihood_nn
   sbi.neural_nets.marginal_nn
   sbi.neural_nets.posterior_flow_nn
   sbi.neural_nets.posterior_nn
   sbi.neural_nets.posterior_score_nn
