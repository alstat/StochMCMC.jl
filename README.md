# StochMCMC.jl
*A Julia package for Stochastic Gradient Markov Chain Monte Carlo*

This package is part of my master's thesis entitled **Bayesian Autoregressive Distributed Lag** *via* **Stochastic Gradient Hamiltonian Monte Carlo** or BADL-SGHMC. However, as the title of this says, this package aims to accomodate other Stochastic Gradient MCMCs in the near future. At the latest, the following are the MCMC algorithms available:

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

## Tutorial 1: Bayesian Linear Regression
In order to illustrate the modeling, the data is simulated from a simple linear regression expectation function. That is the model is given by

\begin{equation}
y_i= w_0 + w_1 x_i + \varepsilon_i,\quad\varepsilon_i\sim\mathcal{N}\left(0,\alpha^{-1}\right)
\end{equation}

To do so, let $\mathbf{B}\triangleq[w_0\;w_1]^{\text{T}}=[.2\;\;-.9]^{\text{T}}, \alpha = 1 / 5.$. Generate 200 hypothetical data:

```julia
using DataFrames
using Distributions
using Gadfly
Gadfly.push_theme(:dark)

srand(123);

# Define data parameters
w0 = .2; w1 = -.9; stdev = 5.;
alpha = 1 / stdev;

# Generate Hypothetical Data
n = 200;
x = rand(Uniform(-1, 1), n);
A = [ones(length(x)) x];
B = [w0; w1];
f = A * B;
y = f + rand(Normal(0, alpha), n);

my_df = DataFrame(Independent = round(x, 4), Dependent = round(y, 4));
my_df |> head # View the first six observations
```
