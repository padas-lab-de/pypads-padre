# PadrePads
An extension of pypads that implements and tracks other concepts from machine learning experiments.   

[![Documentation Status](https://readthedocs.org/projects/pypads-onto/badge/?version=latest)](https://pypads.readthedocs.io/projects/pypads-onto/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pypads-padre.svg)](https://badge.fury.io/py/pypads-padre)  

<!--- ![Build status](https://gitlab.padim.fim.uni-passau.de/RP-17-PaDReP/padre-pads/badges/master/pipeline.svg) --->

# Intalling
This tool requires those libraries to work:

    Python (>= 3.6),
    pypads (>= 0.1.8)
    
PadrePads only support python 3.6 and higher. To install pypads_padre run this in you terminal

**Using source code**

First, you have to install **poetry** if not installed

    pip install poetry
    poetry build (in the root folder of the repository padre-pads/)

This would create two files under dist/ that can be used to install,

    pip install dist/pypads-padre-X.X.X.tar.gz
    OR
    pip install dist/pypads-padre-X.X.X-py3-none-any.whl
    
 
**Using pip ([PyPi release](https://pypi.org/project/pypads-padre/))**

The package can be found on PyPi in following [project](https://pypi.org/project/pypads-padre/).

    pip install pypads_padre


### Tests
The unit tests can be found under 'test/' and can be executed using

    poetry run pytest test/

# Documentation

For more information, look into the [official documentation of PadrePads](https://pypads.readthedocs.io/en/latest/projects/pypads-padre.html).

# Scientific work disclaimer
This was created in scope of scientific work of the Data Science Chair at the University of Passau. If you want to use this tool or any of its resources in your scientific work include a citation.

# Acknowledgement
This work has been partially funded by the **Bavarian Ministry of Economic Affairs, Regional Development and Energy** by means of the funding programm **"Internetkompetenzzentrum Ostbayern"** as well as by the **German Federal Ministry of Education and Research** in the project **"Provenance Analytics"** with grant agreement number *03PSIPT5C*.
