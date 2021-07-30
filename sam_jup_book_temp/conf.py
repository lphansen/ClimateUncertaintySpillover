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
import os
import sys
# sys.path.insert(0, os.path.abspath('../src/'))
sys.path.append(os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'UncertaintySpillover'
copyright = '2021, SuriChen'
author = 'SuriChen'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
import sphinx_material
html_context = sphinx_material.get_html_context()
html_theme_path = sphinx_material.html_theme_path()
html_theme = 'sphinx_material'
html_theme_options = {
    'navigation_with_keys': True,
    'nav_title': 'UncertaintySpillover',
    'color_primary': 'blue-grey',
    'color_accent': 'indigo',
    'repo_url': 'https://github.com/lphansen/WrestlingClimate',
    'repo_name': 'Uncertainty Spillover',
    'repo_type': 'github',
    'globaltoc_depth': 3,
    'globaltoc_collapse': True,
    'master_doc': True,
    'logo_icon': '&#xe55d',
}
html_show_sourcelink = False
html_sidebars = {
    '**': [
        'globaltoc.html',
        'localtoc.html',
        'searchbox.html',
        'logo-text.html',
    ]
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# biblipgraphy file
bibtex_bibfiles = ['climate.bib']

# nbsphinx
nbsphinx_allow_errors = True
nbsphinx_kernel_name = "python"
