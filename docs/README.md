# Documentation

The documentation is available at: <http://mackelab.org/sbi>


## Building the Documentation

To build the docs you will need to:
```bash
pip install mkdocs-material markdown-include mknotebooks mkdocs-redirects mkdocstrings
```

If you have made changes to the notebooks, you have to convert them to markdown:
```bash
./docs/tutorial/convert_nb_to_md.sh
```

You can build the docs locally by running in this subfolder:
```bash
mkdocs serve
```

## Updating documentation on GitHub

```bash
mkdocs gh-deploy
```
