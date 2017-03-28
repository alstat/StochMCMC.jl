module StochMCMC

using Distributions

include(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/MH.jl"));
include(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/HMC.jl"));
include(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/SG HMC.jl"));

export mcmc, MH, HMC, SGHMC

end

Pkg.clone(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/StochMCMC.jl"))
