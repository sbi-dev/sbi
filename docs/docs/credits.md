# Credits

`sbi` is licensed under the [Affero General Public License version 3 (AGPLv3)](https://www.gnu.org/licenses/agpl-3.0.html) and 
Copyright (C) 2020 Michael Deistler, Jan F. Bölts, Jan-Matthis Lückmann, Álvaro Tejero-Cantero.
Copyright (C) 2020 Conor M. Durkan


##  Code

`sbi` uses density estimators from [bayesiains/nflows](https://github.com/bayesiains/nsf) by [Conor M.Durkan](https://conormdurkan.github.io/), [George Papamakarios](https://gpapamak.github.io/) and [Artur Bekasov](https://arturbekasov.github.io/).

`sbi` started as a fork of [conormdurkan/lfi](https://github.com/conormdurkan/lfi), by [Conor M.Durkan](https://conormdurkan.github.io/.

## Inference methods

`sbi` implements inference methods reported in the following contributions:

- **Fast ε-free Inference of Simulation Models with Bayesian Conditional Density
  Estimation**<br> by Papamakarios G. and Murray I. (NeurIPS 2016)
  <br>[[PDF]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation.pdf)
  [[BibTeX]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation/bibtex).

- **Flexible statistical inference for mechanistic models of neural dynamics** <br> by
  Lueckmann J-M., Gonçalves P., Bassetto G., Öcal K., Nonnenmacher M. and Macke J. (NeurIPS 2017)
  <br>[[PDF]](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics.pdf)
  [[BibTeX]](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics/bibtex).

- **Automatic posterior transformation for likelihood-free inference**<br>by Greenberg, Nonnenmacher M. and Macke J. (ICML 2019) <br>[[PDF]](http://proceedings.mlr.press/v97/greenberg19a/greenberg19a.pdf) <br> [[BibTeX]](http://proceedings.mlr.press/v97/greenberg19a.html).

- **On Contrastive Learning for Likelihood-free Inference**<br>Durkan C.,
  Murray I., and Papamakarios G.(ICML 2020) <br>[[PDF]](https://arxiv.org/abs/2002.03712).

We refer to these methods as SNPE-A, SNPE-B, and SNPE-C/APT, respectively.
