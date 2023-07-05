# Credits

## License

`sbi` is licensed under the [Affero General Public License version 3 (AGPLv3)](https://www.gnu.org/licenses/agpl-3.0.html) and

> Copyright (C) 2020 Álvaro Tejero-Cantero, Jakob H. Macke, Jan-Matthis Lückmann,
> Michael Deistler, Jan F. Bölts.

> Copyright (C) 2020 Conor M. Durkan.

## Support

`sbi` has been supported by the German Federal Ministry of Education and Research (BMBF) through the project ADIMEM, FKZ 01IS18052 A-D). [ADIMEM](https://fit.uni-tuebingen.de/Project/Details?id=9199) is a collaborative project between the groups of Jakob Macke (Uni Tübingen), Philipp Berens (Uni Tübingen), Philipp Hennig (Uni Tübingen) and Marcel Oberlaender (caesar Bonn) which aims to develop inference methods for mechanistic models.

![](static/logo_bmbf.svg)

## Important dependencies and prior art

* `sbi` is the successor to [`delfi`](https://github.com/mackelab/delfi), a Theano-based
  toolbox for sequential neural posterior estimation developed at [mackelab](https://uni-tuebingen.de/en/research/core-research/cluster-of-excellence-machine-learning/research/research/cluster-research-groups/professorships/machine-learning-in-science/). If you were
  using `delfi`, we strongly recommend to move your inference over to `sbi`. Please open
  issues if you find unexpected behaviour or missing features. We will consider these
  bugs and give them priority.

* `sbi` as a PyTorch-based toolbox started as a fork of
  [conormdurkan/lfi](https://github.com/conormdurkan/lfi), by [Conor
  M.Durkan](https://conormdurkan.github.io/).

* `sbi` uses density estimators from
[bayesiains/nflows](https://github.com/bayesiains/nsf) by [Conor
M.Durkan](https://conormdurkan.github.io/), [George
Papamakarios](https://gpapamak.github.io/) and [Artur
Bekasov](https://arturbekasov.github.io/). These are proxied through
[`pyknos`](https://github.com/mackelab/pyknos), a package focused on density estimation.

* `sbi` uses `PyTorch` and tries to align with the interfaces (e.g. for probability
  distributions) adopted by `PyTorch`.

* See [README.md](https://github.com/mackelab/sbi/blob/master/README.md) for a list of
  publications describing the methods implemented in `sbi`.
