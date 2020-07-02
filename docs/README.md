# Documentation

The documentation is available at: <http://mackelab.org/sbi>


## Building the Documentation

You can build the docs locally by running the following command from this subfolder:
```bash
jupyter nbconvert --to markdown ../tutorial/*.ipynb --output-dir docs/tutorial/ && mkdocs serve
```

The docs can be updated to GitHub using:
```bash
jupyter nbconvert --to markdown ../tutorial/*.ipynb --output-dir docs/tutorial/ && mkdocs gh-deploy
```
