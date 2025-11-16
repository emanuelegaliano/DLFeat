# conf.py

import os
import sys
# Point to the project root so Sphinx can find your DLFeat.py module
# Assuming DLFeat.py is in the root of the repository.
sys.path.insert(0, os.path.abspath('../..')) 

project = 'DLFeat'
copyright = '2025, Your Antonino Furnari' # UPDATE THIS
author = 'Antonino Furnari/Gemini'         # UPDATE THIS
release = '0.3.1' # UPDATE THIS to your current DLFeat version

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

templates_path = ['_templates'] # sphinx-quickstart creates this, can be empty
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '__init__',
    'show-inheritance': True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True 
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'PIL': ('https://pillow.readthedocs.io/en/stable/', None),
}
