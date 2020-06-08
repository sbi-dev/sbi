### Easy/beginner mode
- automatically runs single round
- allows for almost no additional arguments (maybe `num_workers` and a string for the nn)
```
infer = SNPE(prior=prior, simulator=simulator)
posterior = infer(num_simulations=1000)
```




### Advanced mode
- unlocks all features :)
```
sbi.inference.snpe.set_mode('advanced')

simulator, prior, x_shape = prepare_sbi_problem(simulator, prior)
my_density_estimator = get_nn_models('nsf', prior, x_shape, num_hiddens=50, num_layers=10)

infer = SNPE(prior, simulator, density_estimator=my_density_estimator)
posterior = infer(num_rounds=2, x_o=observation, num_simulations_per_round=1000)
```


### Coding details

- new class `SNPE_easy_interface`, which inherits from `SNPE`.
- at `SNPE_easy_interface.__init__()`, we set the advanced parameters of `SNPE` to fixed values.
- in addition `SNPE_easy_interface.__init__()` also runs `prepare_sbi_problem()`
- by default, we have an alias `SNPE_easy_interface=SNPE`, so we will automatically run the easy interface.
- calling `sbi.inference.snpe.set_mode('advanced')` changes the alias.

### Single function call
```
# single-round
amortized_post = parameters(simulator, prior, num_simulations=1000, method='SNPE')
# multi-round (open to having defaults for rounds and method)
focused_post = parameters(simulator, prior, focus_on=x_o, rounds=3, num_simulations = 1000, method='SNPE')
```
