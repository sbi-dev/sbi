# v0.15.0

## Major changes
- Active subspaces for sensitivity analysis (#394, [tutorial](https://www.mackelab.org/sbi/tutorial/09_sensitivity_analysis/))
- Method to compute the maximum-a-posteriori estimate from the posterior (#412)

## API changes
- `pairplot()`, `conditional_pairplot()`, and `conditional_corrcoeff()` should now be imported from `sbi.analysis` instead of `sbi.utils` (#394).
- Changed `fig_size` to `figsize` in pairplot (#394).
- moved `user_input_checks` to `sbi.utils` (#430).

## Minor changes
- Depend on new `joblib=1.0.0` and fix progress bar updates for multiprocessing (#421).
- Fix for embedding nets with `SNRE` (thanks @adittmann, #425)
- Is it now optional to pass a prior distribution when using SNPE (#426)
- Support loading of posteriors saved after `sbi v0.15.0` (#427, thanks @psteinb)
- Neural network training can be resumed (#431)
- Allow using NSF to estimate 1D distributions (#438)


# v0.14.3

- Fixup for conditional correlation matrix (thanks @JBeckUniTb, #404)
- z-score data using only the training data (#411)


# v0.14.2

- Small fix for SMC-ABC with semi-automatic summary statistics (#402)


# v0.14.1

- Support for training and sampling on GPU including fixes from `nflows` (#331)
- Bug fix for SNPE with neural spline flow and MCMC (#398)
- Small fix for SMC-ABC particles covariance
- Small fix for rejection-classifier (#396)


# v0.14.0

- New flexible interface API (#378). This is going to be a breaking change for users of 
the flexible interface and you will have to change your code. Old syntax:
```python
from sbi.inference import SNPE, prepare_for_sbi

simulator, prior = prepare_for_sbi(simulator, prior)
inference = SNPE(simulator, prior)

# Simulate, train, and build posterior.
posterior = inference(num_simulation=1000)
```
New syntax:
```python
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

simulator, prior = prepare_for_sbi(simulator, prior)
inference = SNPE(prior)

theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=1000)
density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)  # MCMC kwargs go here.
```
More information can be found here [here](https://www.mackelab.org/sbi/tutorial/02_flexible_interface/).
- Fixed typo in docs for `infer` (thanks @glouppe, #370)
- New `RestrictionEstimator` to learn regions of bad simulation outputs (#390)
- Improvements for and new ABC methods (#395)
    - Linear regression adjustment as in Beaumont et al. 2002 for both MCABC and SMCABC
    - Semi-automatic summary statistics as in Fearnhead & Prangle 2012 for both MCABC and SMCABC
    - Small fixes to perturbation kernel covariance estimation in SMCABC.


# v0.13.2

- Fix bug in SNRE (#363)
- Fix warnings for multi-D x (#361)
- Small improvements to MCMC, verbosity and continuing of chains (#347, #348)


# v0.13.1

- Make logging of vectorized numpy slice sampler slightly less verbose and address NumPy future warning (#347)
- Allow continuation of MCMC chains (#348)


# v0.13.0

- Conditional distributions and correlations for analysing the posterior (#321)
- Moved rarely used arguments from pairplot into kwargs (#321)
- Sampling from conditional posterior (#327)
- Allow inference with multi-dimensional x when appropriate embedding is passed (#335)
- Fixes a bug with clamp_and_warn not overriding num_atoms for SNRE and the warning message itself (#338)
- Compatibility with Pyro 1.4.0 (#339)
- Speed up posterior rejection sampling by introducing batch size (#340, #343)
- Allow vectorized evaluation of numpy potentials (#341)
- Adds vectorized version of numpy slice sampler which allows parallel log prob evaluations across all chains (#344)


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
