# Credits

##  Code

This code builds heavily on previous work by [Conor Durkan](https://conormdurkan.github.io/), [George Papamakarios](https://gpapamak.github.io/) and [Artur Bekasov](https://arturbekasov.github.io/).
Relevant repositories include [bayesiains/nsf](https://github.com/bayesiains/nsf) and [conormdurkan/lfi](https://github.com/conormdurkan/lfi).


## Algorithms

`sbi` implements algorithms from the following papers:

- **Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation**<br> by Papamakarios & Murray (NeurIPS 2016) <br>[[PDF]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation.pdf) [[BibTeX]](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation/bibtex)

- **Flexible statistical inference for mechanistic models of neural dynamics** <br> by Lueckmann, Goncalves, Bassetto, Öcal, Nonnenmacher & Macke (NeurIPS 2017) <br>[[PDF]](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics.pdf) [[BibTeX]](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics/bibtex)

- **Automatic posterior transformation for likelihood-free inference**<br>by Greenberg, Nonnenmacher & Macke (ICML 2019) <br>[[PDF]](http://proceedings.mlr.press/v97/greenberg19a/greenberg19a.pdf) [[BibTeX]](data:text/plain;charset=utf-8,%0A%0A%0A%0A%0A%0A%40InProceedings%7Bpmlr-v97-greenberg19a%2C%0A%20%20title%20%3D%20%09%20%7BAutomatic%20Posterior%20Transformation%20for%20Likelihood-Free%20Inference%7D%2C%0A%20%20author%20%3D%20%09%20%7BGreenberg%2C%20David%20and%20Nonnenmacher%2C%20Marcel%20and%20Macke%2C%20Jakob%7D%2C%0A%20%20booktitle%20%3D%20%09%20%7BProceedings%20of%20the%2036th%20International%20Conference%20on%20Machine%20Learning%7D%2C%0A%20%20pages%20%3D%20%09%20%7B2404--2414%7D%2C%0A%20%20year%20%3D%20%09%20%7B2019%7D%2C%0A%20%20editor%20%3D%20%09%20%7BChaudhuri%2C%20Kamalika%20and%20Salakhutdinov%2C%20Ruslan%7D%2C%0A%20%20volume%20%3D%20%09%20%7B97%7D%2C%0A%20%20series%20%3D%20%09%20%7BProceedings%20of%20Machine%20Learning%20Research%7D%2C%0A%20%20address%20%3D%20%09%20%7BLong%20Beach%2C%20California%2C%20USA%7D%2C%0A%20%20month%20%3D%20%09%20%7B09--15%20Jun%7D%2C%0A%20%20publisher%20%3D%20%09%20%7BPMLR%7D%2C%0A%20%20pdf%20%3D%20%09%20%7Bhttp%3A%2F%2Fproceedings.mlr.press%2Fv97%2Fgreenberg19a%2Fgreenberg19a.pdf%7D%2C%0A%20%20url%20%3D%20%09%20%7Bhttp%3A%2F%2Fproceedings.mlr.press%2Fv97%2Fgreenberg19a.html%7D%2C%0A%20%20abstract%20%3D%20%09%20%7BHow%20can%20one%20perform%20Bayesian%20inference%20on%20stochastic%20simulators%20with%20intractable%20likelihoods%3F%20A%20recent%20approach%20is%20to%20learn%20the%20posterior%20from%20adaptively%20proposed%20simulations%20using%20neural%20network-based%20conditional%20density%20estimators.%20However%2C%20existing%20methods%20are%20limited%20to%20a%20narrow%20range%20of%20proposal%20distributions%20or%20require%20importance%20weighting%20that%20can%20limit%20performance%20in%20practice.%20Here%20we%20present%20automatic%20posterior%20transformation%20(APT)%2C%20a%20new%20sequential%20neural%20posterior%20estimation%20method%20for%20simulation-based%20inference.%20APT%20can%20modify%20the%20posterior%20estimate%20using%20arbitrary%2C%20dynamically%20updated%20proposals%2C%20and%20is%20compatible%20with%20powerful%20flow-based%20density%20estimators.%20It%20is%20more%20flexible%2C%20scalable%20and%20efficient%20than%20previous%20simulation-based%20inference%20techniques.%20APT%20can%20operate%20directly%20on%20high-dimensional%20time%20series%20and%20image%20data%2C%20opening%20up%20new%20applications%20for%20likelihood-free%20inference.%7D%0A%7D%0A)

- **On Contrastive Learning for Likelihood-free Inference**<br>by Durkan,
  Murray, Papamakarios (arXiv 2020) <br>[[PDF]](https://arxiv.org/abs/2002.03712)

We refer to these algorithms as SNPE-A, SNPE-B, and SNPE-C/APT, respectively.
