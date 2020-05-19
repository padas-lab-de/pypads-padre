.. padrepads documentation master file, created by
sphinx-quickstart on Tue May 19 15:15:40 2020.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

PadrePaDS: Documentation!
=====================================

Building on the PyPaDS toolset, `PadrePaDS`_ aims to add additional semantic information to tracked experiments.

.. _PyPaDS: https://github.com/padre-lab-eu/pypads

Install PyPads
--------------

Logging your experiments manually can be overwhelming and exhaustive? PyPads is a tool to help automate logging as much information as possible by
tracking the libraries of your choice.

* **Installing PyPads**:
   :ref:`With pip <install_official_release>` |
   :ref:`From source <install_from_source>`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Install PyPads:

   install

Related Projects
----------------
- **PaDRe-Pads** is a tool that builds on PyPads and add some semantics to the tracked data of Machine learning experiments. See the `padre-pads documentation <https://github.com/padre-lab-eu/padre-pads>`_.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Related Projects:

   related_projects

Concepts
========

PadrePads builds upon pypads when it comes to tracking, but it also adds a layer of loggers that tracks semantic information from experiments executions.

Dataset tracking
----------------

PadrePads have a dataset logger that tries to identify the object returned by the tracked function hooked with 'pypads_dataset'.
After collecting as mush metadata on this object, padrepads then dumps it on disk along with the metadata and link to the current run ID.

The currently supported dataset providers by padrepads::

    - Scikit-learn (sklearn.datasets).
    - Keras datasets.
    - torchvision datasets.

Split tracking
--------------

Decisions tracking
------------------

Grid Search
-----------


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
