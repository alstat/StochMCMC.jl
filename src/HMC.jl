"""
HAMILTONIAN MONTE CARLO TYPE INSTANTIATION

Approximate Bayesian Inference via Hamiltonian Monte Carlo.

**Examples:**
```julia
using DataFrames
using Distributions
using Gadfly
using StochMCMC
Gadfly.push_theme(:dark)

srand(123);

# Define data parameters
w0 = -.3; w1 = -.5; stdev = 5.; a =  1 / stdev

# Generate Hypothetical Data
n = 200;
x = rand(Uniform(-1, 1), n);
A = [ones(length(x)) x];
B = [w0; w1];
f = A * B;
y = f + rand(Normal(0, a), n);

my_df = DataFrame(Independent = round(x, 4), Dependent = round(y, 4));

"""
The log prior function is given by the following codes:
"""
function logprior(theta::Array{Float64}; mu::Array{Float64} = zero_vec, s::Array{Float64} = eye_mat)
  w0_prior = log(pdf(Normal(mu[1, 1], s[1, 1]), theta[1]))
  w1_prior = log(pdf(Normal(mu[2, 1], s[2, 2]), theta[2]))
   w_prior = [w0_prior w1_prior]

  return w_prior |> sum
end

"""
The log likelihood function is given by the following codes:
"""
function loglike(theta::Array{Float64}; alpha::Float64 = a, x::Array{Float64} = x, y::Array{Float64} = y)
  yhat = theta[1] + theta[2] * x

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
  loglike(theta, alpha = a, x = x, y = y) + logprior(theta, mu = zero_vec, s = eye_mat)
end

U(theta::Array{Float64}) = - logpost(theta);
K(p::Array{Float64}; Σ = eye(length(p))) = (p' * inv(Σ) * p) / 2;
function dU(theta::Array{Float64}; alpha::Float64 = a, b::Float64 = eye_mat[1, 1])
  [-alpha * sum(y - (theta[1] + theta[2] * x));
   -alpha * sum((y - (theta[1] + theta[2] * x)) .* x)] + b * theta
end
dK(p::AbstractArray{Float64}; Σ::Array{Float64} = eye(length(p))) = inv(Σ) * p;

srand(123);
HMC_object = HMC(U, K, dU, dK, zeros(2), 2);
chain2 = mcmc(HMC_object, leapfrog_params = Dict([:ɛ => .09, :τ => 20]), r = 10000);
```


"""
immutable HMC
  U       ::Function
  K       ::Function
  dU      ::Function
  dK      ::Function
  init_est::Array{Float64}
  d       ::Int64
end

function mcmc(parameters::HMC;
  leapfrog_params::Dict{Symbol, Real} = Dict([:ɛ => 0.05, :τ => 20]),
  set_seed::Int64 = 123,
  r::Int64 = 1000)

  U, K, dU, dK, w, d = parameters.U, parameters.K, parameters.dU, parameters.dK, parameters.init_est, parameters.d
  ɛ, τ = leapfrog_params[:ɛ], leapfrog_params[:τ]
  H(x::AbstractArray{Float64}, p::AbstractArray{Float64}) = U(x) + K(p)

  if typeof(set_seed) == Int64
    srand(set_seed)
  end

  chain = zeros(r, d);
  chain[1, :] = w
  for i in 1:(r - 1)
    w = chain[i, :]
    p = randn(length(w))
    oldE = H(w, p)

    for j in 1:τ
      p = p - (ɛ / 2) * dU(w)
      w = w + ɛ * dK(p)
      p = p - (ɛ / 2) * dU(w)
    end

    newE = H(w, p)
    dE = newE - oldE

    if dE[1] < 0
      chain[i + 1, :] = w
    elseif rand(Uniform()) < exp(-dE)[1]
      chain[i + 1, :] = w
    else
      chain[i + 1, :] = chain[i, :]
    end
  end

  return chain
end
