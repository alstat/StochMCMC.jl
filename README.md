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

## Tutorial: Bayesian Linear Regression
In order to illustrate the modeling, the data is simulated from a simple linear regression expectation function. That is the model is given by

```
y_i= w_0 + w_1 x_i + e_i,   e_i ~ N(0, 1 / a)
```

To do so, let `B = [w_0 ;w_1]'=[.2  -.9]', a = 1 / 5.`. Generate 200 hypothetical data:

```julia
using DataFrames
using Distributions
using Gadfly
using StochMCMC
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
```

To view the head of the data, run the following:
```julia
head(my_df)

# 6×2 DataFrames.DataFrame
# │ Row │ Independent │ Dependent │
# ├─────┼─────────────┼───────────┤
# │ 1   │  0.5369     │ -0.3164   │
# │ 2   │  0.8810     │ -0.5236   │
# │ 3   │  0.3479     │  0.2077   │
# │ 4   │ -0.2091     │  0.3833   │
# │ 5   │ -0.3735     │  0.5150   │
# │ 6   │  0.3251     │ -0.3508   │
```
Next is to plot this data which can be done as follows:
```julia
plot(my_df, x = :Independent, y = :Dependent)
```

![(Right) Triangular Membership Function](https://github.com/alstat/StochMCMC.jl/blob/master/figures/plot1.svg)
