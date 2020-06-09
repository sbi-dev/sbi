#!/usr/bin/env bash

jupyter nbconvert --to markdown notebooks/00_getting_started.ipynb --output-dir markdown_files
jupyter nbconvert --to markdown notebooks/01_gaussian_amortized.ipynb --output-dir markdown_files
jupyter nbconvert --to markdown notebooks/02_HH_simulator.ipynb --output-dir markdown_files
jupyter nbconvert --to markdown notebooks/03_flexible_interface.ipynb --output-dir markdown_files