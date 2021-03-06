# StochMCMC.jl

[![Documentation Status](https://readthedocs.org/projects/stochmcmcjl/badge/?version=latest)](http://stochmcmcjl.readthedocs.io/en/latest/?badge=latest)

*A Julia package for Stochastic Gradient Markov Chain Monte Carlo*

This package is part of my master's thesis entitled **Bayesian Autoregressive Distributed Lag** *via* **Stochastic Gradient Hamiltonian Monte Carlo** or BADL-SGHMC. However as the title says, this package aims to accomodate other Stochastic Gradient MCMCs in the near future. At the latest, the following are the MCMC algorithms available:

1. Metropolis-Hasting
2. Hamiltonian Monte Carlo
3. Stochastic Gradient Hamiltonian Monte Carlo

## Installation
To install the package, simply run the following codes
```julia
Pkg.clone("https://github.com/alstat/StochMCMC.jl")
```
And to load the package:
```julia
using StochMCMC
```

## Documentation
The complete documentation is available at: http://stochmcmcjl.readthedocs.io/

---
<table width=100%>
<tr><td>author:</td><td><b>Al-Ahmadgaid B. Asaad</b></td><td>thesis advisor:</td><td><b>Joselito C. Magadia, Ph.D.</b></td></tr>
<tr><td>email:</td><td>alasaadstat@gmail.com</td><td>website:</td><td>http://stat.upd.edu.ph/</td></tr>
<tr><td>blog:</td><td>http://alstatr.blogspot.com/</td></tr>
</table>
