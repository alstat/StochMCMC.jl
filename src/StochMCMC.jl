module StochMCMC

using Distributions

include("MH.jl")
include("HMC.jl")
include("SGHMC.jl")

export mcmc, MH, HMC, SGHMC

end
