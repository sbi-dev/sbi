# Documentation

To build the sphinx documentation, run
```
make html && python -m http.server --directory _build/html
```
This will find all jupyter notebooks, run them, collect the output, and incorporate them into the documentation.
