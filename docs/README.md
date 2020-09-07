# Documentation

The documentation is available at: <http://mackelab.org/sbi>


## Building the Documentation

You can build the docs locally by running the following command from this subfolder:
```bash
jupyter nbconvert --to markdown ../tutorials/*.ipynb --output-dir docs/tutorial/
jupyter nbconvert --to markdown ../examples/*.ipynb --output-dir docs/examples/
mkdocs serve
```

The docs can be updated to GitHub using:
```bash
jupyter nbconvert --to markdown ../tutorials/*.ipynb --output-dir docs/tutorial/
jupyter nbconvert --to markdown ../examples/*.ipynb --output-dir docs/examples/
mkdocs gh-deploy
```

## Contributing FAQ

Create a new markdown file named `question_XX.md` in the `docs/faq` folder, where `XX` 
is a running index for the questions. The file should start with the question as title 
(i.e. starting with a `#`) and then have the answer below. Additionally, you need to 
add a link to your question in the markdown file `docs/faq.md`.