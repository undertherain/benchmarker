.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python

.. image:: https://api.travis-ci.org/undertherain/benchmarker.svg?branch=master
    :target: https://travis-ci.org/undertherain/benchmarker
    :alt: build status from Travis CI

.. image:: https://coveralls.io/repos/github/undertherain/benchmarker/badge.svg?branch=master
    :target: https://coveralls.io/github/undertherain/benchmarker?branch=master
    :alt: coveralls badge

.. image:: https://api.codacy.com/project/badge/Grade/a6094d03cc8446a9b851ce9dd298d3c8    
    :target: https://www.codacy.com/project/undertherain/benchmarker/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=undertherain/benchmarker&amp;utm_campaign=Badge_Grade_Dashboard

========
Synopsis
========

Benchmarker is a modular framework to automate a set of performance
benchmarks, mostly for deep learning.

============
Installation
============

Clone Benchmarker, install required packages:

.. code-block:: bash

   git clone --recursive https://github.com/undertherain/benchmarker.git
   cd benchmarker
   pip3 install [--user] -r requirements.txt


===
Run
===

Run Benchmarker as a python module from the root directory of the
repo:

.. code-block:: bash

   python3 -m benchmarker \
       --mode=training \
       --framework=pytorch \
       --problem=resnet50 \
       --problem_size=32 \
       --batch_size=4


==========
Motivation
==========

Various devices, frameworks und underlying software stacks, network
architectures make benchmarking a considerable challege.  To tackle
this problem we introduce Benchmarker.  It is a framework for
*performance benchmarking* with emphasis on deep learning.

It is modular, which means it is easy to extend with any new benchmark
or with any new framework, see these TODO tutorials:


============
Contributors
============

The original version was developed in 2017 by `Aleksandr Drozd
<https://blackbird.pw/>`_.  Since then the following people (in
alphabetical order) contributed the project:

- Zhengyang Bai
- `Kevin Brown <https://kevinabrown.github.io/>`_
- `Mateusz Bysiek <https://mbdevpl.github.io/>`_
- `Aleksandr Drozd <https://blackbird.pw/>`_
- `Artur Podobas <http://podobas.net/>`_
- Shweta Salaria
- `Emil Vatai <http://vatai.github.io/>`_
