# StochMCMC.jl
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

## Tutorial: Bayesian Linear Regression
In order to illustrate the modeling, the data is simulated from a simple linear regression expectation function. That is the model is given by

```
y_i = w_0 + w_1 x_i + e_i,   e_i ~ N(0, 1 / a)
```
### Data Simulation
To do so, let `B = [w_0, w_1]' = [.2, -.9]', a = 1 / 5`. Generate 200 hypothetical data:

```julia
using DataFrames
using Distributions
using Gadfly
using StochMCMC
Gadfly.push_theme(:dark)

srand(123);

# Define data parameters
w0 = -.3; w1 = -.5; stdev = 5.;
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
# │ 1   │  0.5369     │ -0.6016   │
# │ 2   │  0.8810     │ -0.6712   │
# │ 3   │  0.3479     │ -0.1531   │
# │ 4   │ -0.2091     │ -0.2004   │
# │ 5   │ -0.3735     │ -0.1345   │
# │ 6   │  0.3251     │ -0.7208   │
```
Next is to plot this data which can be done as follows:
```julia
plot(my_df, x = :Independent, y = :Dependent)
```

![(Right) Triangular Membership Function](https://github.com/alstat/StochMCMC.jl/blob/master/figures/plot1.png)

### Setup Probabilities
In order to proceed with the Bayesian inference, the parameters of the model is considered to be random modeled by a standard Gaussian distribution. That is, `B ~ N(0, I)`, where `0` is the zero vector. The likelihood of the data is given by,

```
L(w|[x, y], b) = ∏_{i=1}^n N([x_i, y_i]|w, b)
```
Thus the posterior is given by,
```
P(w|[x, y]) ∝ P(w)L(w|[x, y], b)
```

To start programming, define the probabilities
```julia
"""
The log prior function is given by the following codes:
"""
function logprior(theta::Array{Float64}; mu::Array{Float64} = zero_vec, s::Array{Float64} = s)
  w0_prior = log(pdf(Normal(mu[1, 1], s[1, 1]), theta[1]))
  w1_prior = log(pdf(Normal(mu[2, 1], s[2, 2]), theta[2]))
   w_prior = [w0_prior w1_prior]

  return w_prior |> sum
end

"""
The log likelihood function is given by the following codes:
"""
function loglike(theta::Array{Float64}; alpha::Float64 = alpha, x::Array{Float64} = x, y::Array{Float64} = y)
  yhat = theta[1] * exp(x / theta[2])

  likhood = Float64[]
  for i in 1:length(yhat)
    push!(likhood, log(pdf(Normal(yhat[i], alpha), y[i])))
  end

  return likhood |> sum
end

"""
The log posterior function is given by the following codes:
"""
function logpost(theta::Array{Float64})
  loglike(theta, alpha = alpha, x = x, y = y) + logprior(theta, mu = zero_vec, s = eye_mat)
end
```
### Estimation: Metropolis-Hasting
To start the estimation, define the necessary parameters for the Metropolis-Hasting algorithm
```julia
# Scale parameter for the likelihood
alpha =  1 / 5.

# Hyperparameters
zero_vec = zeros(2)
eye_mat = eye(2)
```
Run the MCMC:
```julia
mh_object = MH(logpost; init_est = [.1; .1]);
chain1 = mcmc(mh_object, r = 10000);
```
Extract the estimate
```julia
burn_in = 100
thinning = 10

# Expetation of the Posterior
est = mapslices(mean, chain1[(burn_in + 1):thinning:end, :], [1]);
est
# 1×2 Array{Float64,2}:
#  -0.250422  0.759721
```
### Estimation: Hamiltonian Monte Carlo
Setup the necessary paramters including the gradients. The potential energy is the negative logposterior given by `U`, the gradient is `dU`; the kinetic energy is the standard Gaussian function given by `K`, with gradient `dK`. Thus,

```julia
U(theta::Array{Float64}) = - logpost(theta)
K(p::Array{Float64}; Σ = eye(length(p))) = (p' * inv(Σ) * p) / 2
function dU(theta::Array{Float64}; alpha::Float64 = alpha, b::Float64 = eye_mat[1, 1])
  [-alpha * sum(y - (theta[1] + theta[2] * x));
   -alpha * sum((y - (theta[1] + theta[2] * x)) .* x)] + b * theta
end
dK(p::AbstractArray{Float64}; Σ::Array{Float64} = eye(length(p))) = inv(Σ) * p;
```
---
* author: **AL-AHMADGAID B. ASAAD**
* email: alasaadstat@gmail.com
* blog: http://alstatr.blogspot.com/
