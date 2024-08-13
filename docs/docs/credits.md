# Credits

## Community and Contributions

`sbi` is a community-driven package. We are grateful to all our contributors who have
played a significant role in shaping `sbi`. Their valuable input, suggestions, and
direct contributions to the codebase have been instrumental in the development of `sbi`.

## License

`sbi` is licensed under the [Apache License
(Apache-2.0)](https://www.apache.org/licenses/LICENSE-2.0) and

> Copyright (C) 2020 Álvaro Tejero-Cantero, Jakob H. Macke, Jan-Matthis Lückmann,
> Michael Deistler, Jan F. Bölts.

> Copyright (C) 2020 Conor M. Durkan.

> All contributors hold the copyright of their specific contributions.

## Support

`sbi` has been supported by the German Federal Ministry of Education and Research (BMBF)
through project ADIMEM (FKZ 01IS18052 A-D), project SiMaLeSAM (FKZ 01IS21055A) and the
Tübingen AI Center (FKZ 01IS18039A). Since 2024, `sbi` has been supported by the
appliedAI Institute for Europe gGmbH.

![](static/logo_bmbf.svg)

## Important dependencies and prior art

- `sbi` is the successor to [`delfi`](https://github.com/mackelab/delfi), a Theano-based
  toolbox for sequential neural posterior estimation developed at
  [mackelab](https://www.mackelab.org).If you were using `delfi`, we strongly recommend
  moving your inference over to `sbi`. Please open issues if you find unexpected
  behavior or missing features. We will consider these bugs and give them priority.
- `sbi` as a PyTorch-based toolbox started as a fork of
  [conormdurkan/lfi](https://github.com/conormdurkan/lfi), by [Conor
  M.Durkan](https://conormdurkan.github.io/).
- `sbi` uses `PyTorch` and tries to align with the interfaces (e.g. for probability
  distributions) adopted by `PyTorch`.
- See [README.md](https://github.com/mackelab/sbi/blob/master/README.md) for a
  list of publications describing the methods implemented in `sbi`.
