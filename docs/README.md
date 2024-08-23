# Documentation

The documentation is available at: [sbi-dev.github.io/sbi](http://sbi-dev.github.io/sbi)

## Building the Documentation

We use [`mike`](https://github.com/jimporter/mike) to manage, build, and deploy our
documentation with [`mkdocs`](https://www.mkdocs.org/). To build the documentation
locally, follow these steps:

1. Install the documentation dependencies:

    ```bash
    python -m pip install .[doc]
    ```

2. Convert the current version of the documentation notebooks to markdown and build the
   website locally using `mkdocs`:

    ```bash
    jupyter nbconvert --to markdown ../tutorials/*.ipynb --output-dir docs/tutorials/
    mkdocs serve
    ```

### Deployment

Website deployment is managed with `mike` and happens automatically:

- With every push to `main`, a `dev` version of the most recent documentation is built.
- With every new published **release**, the current documentation is deployed on the
  website.

Thus, the documentation on the website always refers to the latest release, and not
necessarily to the version on `main`.

## Contributing FAQ

We welcome contributions to our list of frequently asked questions. To contribute:

1. Create a new markdown file named `question_XX.md` in the `docs/faq` folder, where
   `XX` is a running index for the questions.
2. The file should start with the question as the title (i.e. starting with a `#`) and
   then have the answer below.
3. Add a link to your question in the [`docs/faq.md`] file using the same index.
