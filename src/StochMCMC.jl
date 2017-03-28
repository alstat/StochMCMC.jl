module StochMCMC

using Distributions

include("MH.jl");
include("HMC.jl");
include("SGHMC.jl");

export mcmc, MH, HMC, SGHMC

end

Pkg.clone(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/StochMCMC.jl"))
