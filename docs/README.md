# Documentation

## Building and serving the documentation locally

To build the Sphinx documentation with autoreload (recommended for development):

```bash
# Install documentation dependencies
uv sync --extra doc

# Start development server with autoreload
sphinx-autobuild . _build/html
```

This will automatically rebuild the documentation when you make changes and serve it locally (usually at `http://127.0.0.1:8000`).

Alternatively, for a one-time build and simple HTTP server:

```bash
make html && python -m http.server --directory _build/html
```

The autoreload method is recommended as it will find all Jupyter notebooks, run them, collect the output, and incorporate them into the documentation, while automatically rebuilding when you make changes.
