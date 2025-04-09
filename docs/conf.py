# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'sbi'
copyright = '2020, sbi team'
author = 'sbi team'


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_math_dollar",
    "sphinx_design",
    "sphinx.ext.mathjax",
    "myst_nb",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

source_suffix = {'.rst': 'restructuredtext', '.myst': 'myst-nb', '.ipynb': 'myst-nb'}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Myst-NB
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]
nb_execution_timeout = 600
nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
html_title = ""
html_logo = "logo.png"
html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/sbi-dev/sbi',
    "use_repository_button": True,
    "use_download_button": False,
    'repository_branch': 'main',
    "path_to_docs": 'docs',
    'launch_buttons': {
        'colab_url': 'https://colab.research.google.com',
        'binderhub_url': 'https://mybinder.org',
    },
    "toc_title": "Navigation",
    "show_navbar_depth": 1,
    "show_toc_level": 3,
    "pygment_light_style": "default",
    "pygment_dark_style": "github-dark",  # All styles: https://pygments.org/styles/
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']

autosummary_generate = True
autodoc_typehints = "description"
add_module_names = False
autodoc_member_order = "bysource"
