"""
STOCHASTIC GRADIENT HAMILTONIAN MONTE CARLO

The following codes explore the use of SGHMC for Bayesian Inference
"""
immutable SGHMC
  dU      ::Function
  dK      ::Function
  dKΣ     ::Array{Float64}
  C       ::Array{Float64}
  V       ::Array{Float64}
  init_est::Array{Float64}
  d       ::Int64
end

function mcmc(parameters::SGHMC;
  leapfrog_params::Dict{Symbol, Real} = Dict([:ɛ => .05, :τ => 20]),
  set_seed::Int64 = 123,
  r::Int64 = 1000)

  dU, dK, dKΣ, C, V, w, d = parameters.dU, parameters.dK, parameters.dKΣ, parameters.C, parameters.V, parameters.init_est, parameters.d
  ɛ, τ = leapfrog_params[:ɛ], leapfrog_params[:τ]

  if typeof(set_seed) == Int64
    srand(set_seed)
  end

  chain = zeros(r, d);
  B = .5 * V * ɛ
  D = sqrt(2 * (C - B) * ɛ)
  if size(B) != size(C)
    error("C and V should have the same dimension.")
  else
    if sum(size(B)) > 1
      if det(B) > det(C)
        error("ɛ is too big. Consider decreasing it.")
      end
    else
      if det(B[1]) > det(C[1])
        error("ɛ is too big. Consider decreasing it.")
      end
    end
  end

  for i in 1:r
    p = randn(d, 1)

    for j in 1:τ
      p = p - dU(w) * ɛ - C * inv(dKΣ) * p + D * randn(d, 1);
      w = w + dK(p) * ɛ;
    end

    chain[i, :] = w
  end

  return chain
end
