"""
METROPOLIS-HASTING


The following codes explore the use of MH for Bayesian Inference.
"""
immutable MH
  logposterior::Function
  proposal    ::Function
  init_est    ::Array{Float64}
  d           ::Int64

  function default_proposal(theta::Array{Float64}, d::Int64)
    sigmas = ones(d);
    random = Float64[]

    for i in 1:length(theta)
      push!(random, rand(Normal(theta[i], sigmas[i])))
    end

    return random
  end

  MH(logposterior = Union{}; proposal::Function = default_proposal, init_est::Array{Float64} = [0.; 0.], d::Int64 = 2) = new(logposterior, proposal, init_est, d)
end

function mcmc(parameters::MH; r = 1000)
  chain = zeros(r, parameters.d)
  chain[1, :] = parameters.init_est

  for i in 1:(r - 1)
    propose = parameters.proposal(chain[i, :], parameters.d)
    probab = exp(parameters.logposterior(propose) - parameters.logposterior(chain[i, :]))

    if rand(Uniform()) < probab
      chain[i + 1, :] = propose
    else
      chain[i + 1, :] = chain[i, :]
    end
  end

  return chain

end
