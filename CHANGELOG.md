# v0.13.0
- Conditional distributions and correlations for analysing the posterior (#321)
- Moved rarely used arguments from pairplot into kwargs (#321)
- Sampling from conditional posterior (#327)


# v0.12.2

- Bug fix for zero simulations in later rounds (#318)
- Bug fix for sbi.utils.sbiutils.Standardize; mean and std are now registered in state dict (thanks @plcrodrigues, #325)
- Tutorials on embedding_net and presimulated data (thanks @plcrodrigues, #314, #318)
- FAQ entry for pickling error


# v0.12.1

- Bug fix for broken NSF (#310, thanks @tvwenger).


# v0.12.0

- Add FAQ (#293)
- Fix bug in embedding_net when output dimension does not equal input dimension (#299)
- Expose arguments of functions used to build custom networks (#299)
- Implement non-atomic APT (#301)
- Depend on pyknos 0.12 and nflows 0.12
- Improve documentation (#302, #305, thanks to @agramfort)
- Fix bug for 1D uniform priors (#307).

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
