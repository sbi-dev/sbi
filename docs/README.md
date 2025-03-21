# Documentation

To build the sphinx documentation, run
```
make html
cd _build/html
python -m http.server
```
This will find all jupyter notebooks, run them, collect the output, and incorporate them into the documentation.
