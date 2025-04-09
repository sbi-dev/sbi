---
title: 'sbi reloaded: a toolkit for simulation-based inference workflows'
tags:
  - Python
  - PyTorch
  - Bayesian Inference
  - Simulation-Based Inference
  - Scientific Discovery
  - Conditional Density Estimation

authors:
  - name: Jan Boelts
    affiliation: "1, 2, 3"
    corresponding: true
    equal-contrib: true
    role: "Maintainer, core contributor"

  - name: Michael Deistler
    affiliation: "1, 2"
    corresponding: true
    equal-contrib: true
    role: "Maintainer, core contributor"

  - name: Manuel Gloeckler
    affiliation: "1, 2"
    role: "Core contributor"

  - name: Álvaro Tejero-Cantero
    affiliation: "3, 4"
    role: "Core contributor"

  - name: Jan-Matthis Lueckmann
    affiliation: "5"
    role: "Core contributor"

  - name: Guy Moss
    affiliation: "1, 2"
    role: "Core contributor"

  - name: Peter Steinbach
    affiliation: "6"
    role: "Major contributor"

  - name: Thomas Moreau
    affiliation: "7"
    role: "Major contributor"

  - name: Fabio Muratore
    affiliation: "8"
    role: "Major contributor"

  - name: Julia Linhart
    affiliation: "7"
    role: "Major contributor"

  - name: Conor Durkan
    affiliation: "9"
    role: "Major contributor"

  - name: Julius Vetter
    affiliation: "1, 2"

  - name: Benjamin Kurt Miller
    affiliation: "10"

  - name: Maternus Herold
    affiliation: "3, 11, 12"

  - name: Abolfazl Ziaeemehr
    affiliation: "13"

  - name: Matthijs Pals
    affiliation: "1, 2"

  - name: Theo Gruner
    affiliation: "14"

  - name: Sebastian Bischoff
    affiliation: "1, 2, 15"

  - name: Nastya Krouglova
    affiliation: "16, 17"

  - name: Richard Gao
    affiliation: "1, 2"

  - name: Janne K Lappalainen
    affiliation: "1, 2"

  - name: Bálint Mucsányi
    affiliation: "1, 2, 18"

  - name: Felix Pei
    affiliation: "19"

  - name: Auguste Schulz
    affiliation: "1, 2"

  - name: Zinovia Stefanidi
    affiliation: "1, 2"

  - name: Pedro Rodrigues
    affiliation: "20"

  - name: Cornelius Schröder
    affiliation: "1, 2"

  - name: Faried Abu Zaid
    affiliation: "3"

  - name: Jonas Beck
    affiliation: "2, 21"

  - name: Jaivardhan Kapoor
    affiliation: "1, 2"

  - name: David S. Greenberg
    affiliation: "22, 23"

  - name: Pedro J. Gonçalves
    affiliation: "17, 24"

  - name: Jakob H. Macke
    affiliation: "1, 2, 25"
    corresponding: true

affiliations:
  - index: 1
    name: Machine Learning in Science, University of Tübingen
  - index: 2
    name: Tübingen AI Center
  - index: 3
    name: TransferLab, appliedAI Institute for Europe
  - index: 4
    name: ML Colab, Cluster ML in Science, University of Tübingen
  - index: 5
    name: Google Research
  - index: 6
    name: Helmholtz-Zentrum Dresden-Rossendorf
  - index: 7
    name: Université Paris-Saclay, INRIA, CEA, Palaiseau, France
  - index: 8
    name: Robert Bosch GmbH
  - index: 9
    name: School of Informatics, University of Edinburgh
  - index: 10
    name: University of Amsterdam
  - index: 11
    name: Research and Innovation Center, BMW Group
  - index: 12
    name: Institute for Applied Mathematics and Scientific Computing, University of the Bundeswehr Munich, Germany
  - index: 13
    name: Aix Marseille, INSERM, INS, France
  - index: 14
    name: TU Darmstadt, hessian.AI, Germany
  - index: 15
    name: University Hospital Tübingen and M3 Research Center
  - index: 16
    name: Faculty of Science, B-3000, KU Leuven, Belgium
  - index: 17
    name: VIB-Neuroelectronics Research Flanders (NERF) and imec, Belgium
  - index: 18
    name: Methods of Machine Learning, University of Tübingen
  - index: 19
    name: Neuroscience Institute, Carnegie Mellon University
  - index: 20
    name: Université Grenoble Alpes, INRIA, CNRS, Grenoble INP, LJK, France
  - index: 21
    name: Hertie Institute for AI in Brain Health, University of Tübingen
  - index: 22
    name: Institute of Coastal Systems - Analysis and Modeling
  - index: 23
    name: Helmholtz AI
  - index: 24
    name: Departments of Computer Science Electrical Engineering, KU Leuven, Belgium
  - index: 25
    name: Department Empirical Inference, Max Planck Institute for Intelligent Systems, Tübingen

date: 16 October 2024
bibliography: paper.bib

---

# Abstract

Scientists and engineers use simulators to model empirically observed phenomena. However, tuning the parameters of a simulator to ensure its outputs match observed data presents a significant challenge. Simulation-based inference (SBI) addresses this by enabling Bayesian inference for simulators, identifying parameters that match observed data and align with prior knowledge. Unlike traditional Bayesian inference, SBI only needs access to simulations from the model and does not require evaluations of the likelihood-function. In addition, SBI algorithms do not require gradients through the simulator, allow for massive parallelization of simulations, and can perform inference for different observations without further simulations or training, thereby amortizing inference.
Over the past years, we have developed, maintained, and extended `sbi`, a PyTorch-based package that implements Bayesian SBI algorithms based on neural networks. The `sbi` toolkit implements a wide range of inference methods, neural network architectures, sampling methods, and diagnostic tools. In addition, it provides well-tested default settings but also offers flexibility to fully customize every step of the simulation-based inference workflow. Taken together, the `sbi` toolkit enables scientists and engineers to apply state-of-the-art SBI methods to black-box simulators, opening up new possibilities for aligning simulations with empirically observed data.

# Statement of need

Bayesian inference is a principled approach for determining parameters consistent with empirical observations: Given a prior over parameters, a forward-model (defining the likelihood), and observations, it returns a posterior distribution.
The posterior distribution captures the entire space of parameters that are compatible with the observations and the prior and it quantifies parameter uncertainty.
When the forward-model is given by a stochastic simulator, Bayesian inference can be challenging: (1) the forward-model can be slow to evaluate, making algorithms that rely on sequential evaluations of the likelihood (such as Markov-Chain Monte-Carlo, MCMC) impractical, (2) the simulator can be non-differentiable, prohibiting the use of gradient-based MCMC or variational inference (VI) methods, and (3) likelihood-evaluations can be intractable, meaning that we can only generate samples from the model, but not evaluate their likelihoods.

Recently, simulation-based inference (SBI) algorithms based on neural networks have been
developed to overcome these limitations [@papamakarios2016fast;
@papamakarios2019sequential; @hermans2020likelihood]. Unlike classical methods from
Approximate Bayesian Computation (ABC, @sisson2018_chapter1), these methods use neural
networks to learn the relationship between parameters and simulation outputs. Neural SBI
algorithms (1) allow for massive parallelization of simulations (in contrast to
sequential evaluations in MCMC methods), (2) do not require gradients through the
simulator, and (3) do not require evaluations of the likelihood, but only samples from
the simulator. Finally, many of these algorithms allow for \emph{amortized} inference,
that is, after a large upfront cost of simulating data for the training phase, they can
return the posterior distribution for any observation without requiring further
simulations or retraining.

To aid in the effective application of these algorithms to a wide range of problems, we developed the `sbi` toolkit. `sbi` implements a variety of state-of-the-art SBI algorithms, offering both high-level interfaces, extensive documentation and tutorials for practitioners, as well as low-level interfaces for experienced users and SBI researchers (giving full control over simulations, the training loop, and the sampling procedure). Since the original release of the `sbi` package [@tejerocantero2020sbi], the community of contributors has expanded significantly, resulting in a large number of improvements that have made `sbi` more flexible, performant, and reliable. `sbi` now supports a wider range of amortized and sequential inference methods, neural network architectures (including normalizing flows, flow- and score-matching, and various embedding network architectures), samplers (including MCMC, variational inference, importance sampling, and rejection sampling), diagnostic tools, visualization tools, and a comprehensive set of tutorials on how to use these features.

The `sbi` package is already used extensively by the machine learning research community
[@deistler2022truncated; @gloecklervariational; @muratore2022neural;
@gloeckler2023adversarial; @dyer2022calibrating; @wiqvist2021sequential;
@spurio2023bayesian; @dirmeier2023simulation;@gloeckler2024allinone; @hermans2022crisis; @linhart2024c2st; @boelts2022flexible]
but has also fostered the application of SBI in various fields of research
[@groschner2022biophysical;@bondarenko2023embryo; @confavreux2023meta;
@myers2024disinhibition; @avecilla2022neural; @lowet2023theta; @bernaerts2023combined;
@mishra2022neural; @dyer2022black; @hashemi2023amortized; @hahn2022accelerated;
@lemos2024field; @deistler2022energy; @rossler2023skewed; @dingeldein2023simulation;
@jin2023bayesian; @boelts2023simulation; @gao2024deep; @wang2024comprehensive].

# Description

`sbi` is a flexible and extensive toolkit for running simulation-based Bayesian inference workflows. `sbi` supports any kind of (offline) simulator and prior, a wide range of inference methods, neural networks, and samplers, as well as diagnostic methods and analysis tools (\autoref{fig:fig1}).

![**Features of the `sbi` package.** Components that were added since the initial release described in @tejerocantero2020sbi are marked in red.\label{fig:fig1}](sbi_toolbox.png)

A significant challenge in making SBI algorithms accessible to a broader community lies
in accommodating diverse and complex simulators, as well as varying degrees of
flexibility in each step of the inference process. To address this, `sbi`
provides pre-configured defaults for all inference methods, but also allows full
customization of every step in the process (including simulation, training, sampling,
diagnostics and analysis).

**Simulator \& prior:** The `sbi` toolkit requires only simulation parameters and
simulated data as input, without needing direct access to the simulator itself. However,
if the simulator can be provided as a Python callable, `sbi` can optionally parallelize
running the simulations from a given prior using `joblib` [@joblib]. Additionally, `sbi` can automatically handle failed
simulations or missing values, it supports both discrete and continuous parameters and
observations (or mixtures thereof) and it provides utilities to flexibly define priors.

**Methods:** `sbi` implements a wide range of neural network-based SBI algorithms, among
them Neural Posterior Estimation (NPE) with various conditional estimators, Neural
Likelihood Estimation (NLE), and Neural Ratio Estimation (NRE). Each of these methods
can be run either in an *amortized* mode, where the neural network is trained once on a
set of pre-existing simulation results and then performs inference on *any* observation
without further simulations or retraining, or in a *sequential* mode where inference is
focused on one observation to improve simulation efficiency with active learning,
running simulations with parameters likely to have resulted in the observation.

**Neural networks and training:** `sbi` implements a wide variety of state-of-the-art
conditional density estimators for NPE and NLE, including normalizing flows
[@papamakarios2021normalizing; @greenberg2019automatic] (via `nflows` [@nflows-repo] and
`zuko` [@zuko-repo]), diffusion models [@song2021scorebased; @geffner2023compositional;
@sharrock2022sequential], mixture density networks [@Bishop_94], and flow matching
[@lipman2023flow; @dax2023flow] (via `zuko`), as well as ensembles of any of these
networks. `sbi` also implements a large set of embedding networks that can automatically
learn summary statistics of (potentially) high-dimensional simulation outputs (including
multilayer perceptrons, convolutional networks, and permutation-invariant networks).
The neural networks can be trained with a pre-configured training loop with established
default values, but `sbi` also allows full access over the training loop when desired.

**Sampling:** For NLE and NRE, `sbi` implements a large range of samplers, including
MCMC (with chains vectorized across observations), variational inference, rejection
sampling, or importance sampling, as well as wrappers to use MCMC samplers from Pyro and
PyMC [@bingham2019pyro; @abril2023pymc]. `sbi` can perform inference for single
observations or for multiple *i.i.d.* observations, and can use importance sampling to
correct for potential inaccuracies in the posterior if the likelihood is available.

**Diagnostics and analysis:** The `sbi` toolkit also implements a large set of
diagnostic tools, such as simulation-based calibration (SBC) [@talts2018validating],
expected coverage [@hermans2022crisis; @deistler2022truncated], local C2ST
[@linhart2024c2st], and TARP [@lemos2023sampling]. Additionally, `sbi` offers
visualization tools for the posterior, including marginal and conditional corner
plots to visualize high-dimensional distributions, calibration plots, and wrappers for
Arviz [@arviz_2019] diagnostic plots.

With `sbi`, our goal is to advance scientific discovery and computational engineering by
making Bayesian inference accessible to a broad range of models, including those with
inaccessible likelihoods, and to a broader range of users, including both machine
learning researchers and domain practitioners. We have created an open architecture and
embraced community-driven development practices to encourage collaboration with other
machine learning researchers and applied scientists to join us in this long-term vision.

# Related software

Simulation-based inference methods implemented in the `sbi` package require only access
to simulated data, which can also be generated offline in other programming languages or
frameworks. This sets `sbi` apart from toolboxes for traditional Bayesian inference,
such as MCMC-based methods [@abril2023pymc; @bingham2019pyro; @gelman2015stan], which
rely on likelihood evaluations, and from probabilistic programming languages (e.g., Pyro
[@bingham2019pyro], NumPyro [@phan2019composable], Stan [@gelman2015stan], or Turing.jl
[@ge2018t]), which typically require the simulator to be differentiable and implemented
within their respective frameworks [@quera-bofarull2023].

Since the original release of the `sbi` package, several other packages that implement
neural network-based SBI algorithms have emerged.
The `lampe` [@rozet_2021_lampe] package offers neural posterior and neural ratio estimation,
primarily targeting SBI researchers with a low-level API and full flexibility over the
training loop. Its development has stopped in favor of the `sbi` project in July 2024.
The `BayesFlow` package [@bayesflow_2023_software] focuses on a set of amortized SBI algorithms
based on posterior and likelihood estimation that have been developed in the respective
research labs [@radev2020bayesflow].
The `swyft` package [@swyft] specializes in algorithms based on neural ratio estimation.
The `sbijax` package [@dirmeier2024simulationbasedinferencepythonpackage] implements a set
of inference methods in JAX.

# Author contributions

This work represents a collaborative effort with contributions from a large and diverse
team.  Author contributions are categorized as follows: Jan Boelts and Michael Deistler
are the current maintainers and lead developers of the sbi package and contributed
equally to this work.  Manuel Gloeckler, Álvaro Tejero-Cantero, Jan-Matthis Lueckmann,
and Guy Moss have made substantial and sustained core contributions to the codebase and
project direction. Peter Steinbach, Thomas Moreau, Fabio Muratore, Julia Linhart, and
Conor Durkan have made major contributions to specific features or aspects of the
package.  All other authors listed have contributed to the sbi package through code,
documentation, or discussions.

# Acknowledgements

This work has been supported by the German Federal Ministry of Education and Research
(BMBF, projects "Simalesam", FKZ 01IS21055 A-B and "DeepHumanVision", FKZ: 031L0197B,
and the Tübingen AI Center FKZ: 01IS18039A), the German Research Foundation (DFG)
through Germany’s Excellence Strategy (EXC-Number 2064/1, PN 390727645) and SFB1233 (PN
276693517), SFB 1089 (PN 227953431), SPP 2041 (PN 34721065), SPP 2041 "Computational
Connectomics", SPP 2298-2 (PN 543917411), SFB 1233 "Robust Vision", and Germany's
Excellence Strategy EXC-Number 2064/1/Project number 390727645, the "Certification and
Foundations of Safe Machine Learning Systems in Healthcare" project funded by the Carl
Zeiss Foundation, the Else Kröner Fresenius Stiftung (Project "ClinbrAIn"), and the
European Union (ERC, "DeepCoMechTome", ref. 101089288). CD was supported by the EPSRC
Centre for Doctoral Training in Data Science, funded by the UK Engineering and Physical
Sciences Research Council (grant EP/L016427/1) and the University of Edinburgh. BKM is
part of the ELLIS PhD program, receiving travel support from the ELISE mobility program
which has received funding from the European Union's Horizon 2020 research and
innovation programme under ELISE grant agreement No 951847. DSG is supported by
Helmholtz AI. JL is a recipient of the Pierre-Aguilar Scholarship and thankful for the
funding of the Capital Fund Management (CFM). ANK is supported by an FWO grant
(G097022N). TG was supported by "Third Wave of AI”, funded by the Excellence Program of
the Hessian Ministry of Higher Education, Science, Research and Art. TM and PLCR were
supported from a national grant managed by the French National Research Agency (Agence
Nationale de la Recherche) attributed to the ExaDoST project of the NumPEx PEPR program,
under the reference ANR-22-EXNU-0004. PS is supported by the Helmholtz Association
Initiative and Networking Fund through the Helmholtz AI platform grant. MD, MG, GM, JV,
MP, SB, JKL, AS, ZS, JB are members of the International Max Planck Research School for
Intelligent Systems (IMPRS-IS).

# References
