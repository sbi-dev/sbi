# What should I do when my 'posterior samples are outside of the prior support' in SNPE?

When working with **multi-round** SNPE, you might have experienced the following
warning: 
```
Only x% posterior samples are within the prior support. It may take a long time to collect the remaining 10000 samples. Consider interrupting (Ctrl-C) and switching to 'sample_with_mcmc=True'.
```

This reason for this issue is described in more detail 
[here](https://arxiv.org/abs/2002.03712) and [here](https://arxiv.org/abs/1905.07488). 
The following fixes are possible:  

- sample with MCMC: `samples = posterior((num_samples,), x=x_o, sample_with_mcmc=True)`.
This will make sampling slower, but samples will not 'leak'.  

- resort to single-round SNPE and (if necessary) increase your simulation budget.  

- if your prior is either Gaussian (torch.distributions.multivariateNormal) or Uniform 
(sbi.utils.BoxUniform), you can avoid leakage by using a mixture density network as 
density estimator. I.e., using the 
[flexible interface](https://www.mackelab.org/sbi/tutorial/03_flexible_interface/), set 
`density_estimator='mdn'`. When running inference, there should be a print statement 
"Using SNPE-C with non-atomic loss"

- use a different algorithm, e.g. SNRE and SNLE. Note, however, that these algorithms
can have different issues and potential pitfalls.  
