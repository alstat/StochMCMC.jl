****
StochMCMC.jl
****

:Description: *A Julia package for Stochastic Gradient Markov Chain Monte Carlo*
:Maintainer: Al-Ahmadgaid B. Asaad (alasaadstat@gmail.com)
:Requires: julia releases 0.4.1 or later
:Date: |today|
:Web site: https://github.com/alstat/StochMCMC.jl
:License: `MIT <https://github.com/brian-j-smith/Mamba.jl/blob/master/LICENSE.md>`_

This package is part of my master's thesis entitled **Bayesian Autoregressive Distributed Lag** *via* **Stochastic Gradient Hamiltonian Monte Carlo** or BADL-SGHMC,
under the supervision of Dr. Joselito C. Magadia of School of Statistics, University of the Philippines Diliman. This package aims to accommodate other
Stochastic Gradient MCMCs in the near future. At the latest, the following are the MCMC algorithms available:

1. Metropolis-Hasting
2. Hamiltonian Monte Carlo
3. Stochastic Gradient Hamiltonian Monte Carlo

Installation
============
To install the package, simply run the following codes

.. code-block:: julia

    Pkg.clone("https://github.com/alstat/StochMCMC.jl")

And to load the package:

.. code-block:: julia

    using StochMCMC

.. toctree::
    :maxdepth: 2

    tutorial.rst
    mh.rst



Indices
~~~~~~~~~~

* :ref:`genindex`
