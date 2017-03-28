"""
HAMILTONIAN MONTE CARLO

The following codes explore the use of HMC for Bayesian Inference.
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
