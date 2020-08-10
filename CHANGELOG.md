# v0.11.3

- Added an FAQ (#293)


# v0.11.2

- Fixed pickling of SNRE by moving StandardizeInputs (#291)
- Added check to ensure correct round number when presimulated data is provided
- Subclassed Posterior depending on inference algorithm (#282, #285)
- Pinned pyro to v1.3.1 as a temporary workaround (see #288) 
- Detaching weights for MCMC SIR init immediately to save memory (#292)


# v0.11.1

- Bug fix for log_prob() in SNRE (#280)


# v0.11.0

- Changed the API to do multi-round inference (#273)
- Allow to continue inference (#273)


# v0.10.2

- Added missing type imports (#275)
- Made compatible for Python 3.6 (#275)


# v0.10.1

- Added `mcmc_parameters` to init methods of inference methods (#270)
- Fixed detaching of `log_weights` when using `sir` MCMC init (#270)
- Fixed logging for SMC-ABC


# v0.10.0

- Added option to pass external data (#264)
- Added setters for MCMC parameters (#267)
- Added check for `density_estimator` argument (#263)
- Fixed `NeuralPosterior` pickling error (#265)
- Added code coverage reporting (#269)


# v0.9.0

- Added ABC methods (#250)
- Added multiple chains for MCMC and new init strategy (#247)
- Added options for z-scoring for all inference methods (#256)
- Simplified swapping out neural networks (#256)
- Improved tutorials
- Fixed device keyword argument (#253)
- Removed need for passing x-shapes (#259)


# v0.8.0

- First public version
