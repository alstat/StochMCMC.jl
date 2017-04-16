****
StochMCMC.jl
****

:Author: Al-Ahmadgaid B. Asaad (alasaadstat@gmail.com | https://alstatr.blogspot.com/)
:Requires: julia releases 0.4.1 or later
:Date: |today|
:License: `MIT <https://github.com/alstat/StochMCMC.jl/blob/master/LICENSE.md>`_
:Website: https://github.com/alstat/StochMCMC.jl


A julia package for Stochastic Gradient Markov Chain Monte Carlo. The package is part of my master's thesis entitled
**Bayesian Autoregressive Distributed Lag** *via* **Stochastic Gradient Hamiltonian Monte Carlo** or **BADL-SGHMC**,
under the supervision of **Dr. Joselito C. Magadia** of School of Statistics, University of the Philippines Diliman.
This work aims to accommodate other Stochastic Gradient MCMCs in the near future.

Installation
============
To install the package, run the following

.. code-block:: julia

    Pkg.clone("https://github.com/alstat/StochMCMC.jl")

And to load the package, run

.. code-block:: julia

    using StochMCMC

Contents
~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    mh.rst
    hmc.rst
    sghmc.rst


Indices
~~~~~~~~~~

* :ref:`genindex`
