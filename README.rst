.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python

.. image:: https://api.travis-ci.org/undertherain/benchmarker.svg?branch=master
    :target: https://travis-ci.org/undertherain/benchmarker
    :alt: build status from Travis CI

.. image:: https://api.codacy.com/project/badge/Grade/a6094d03cc8446a9b851ce9dd298d3c8    
    :target: https://www.codacy.com/project/undertherain/benchmarker/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=undertherain/benchmarker&amp;utm_campaign=Badge_Grade_Dashboard

========
Synopsis
========

Benchmarker is a modular framework to automate a set of performance benchmarks, mostly for deep learning. 

===
Run
===

python3 -m benchmarker  --mode=training --framework=chainer --problem=resnet50 --problem_size=32 --batch_size=4


==========
Motivation
==========

various devices, frameworks und underlying software stacks, network architectures etc.

============
Installation
============

Clone, install required packages
for example by running

``git clone --recursive https://github.com/undertherain/benchmarker.git``

``pip3 install [--user] -r requirements.txt``


=============
API Reference
=============

under development 


============
Contributors
============

Aleksandr Drozd

Kevin Brown

Artur Podobas

Mateusz Bysiek

=======
License
=======

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


