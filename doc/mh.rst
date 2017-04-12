Metropolis-Hasting
===================

A markov chain monte carlo sampler based on the method of Metropolis and Hasting.

.. function:: MH(logposterior::Function, proposal::Function, init_est::Array{Float64}, d::Int64)

    Construct a ``Sampler`` object for Metropolis-Hasting sampling.

    **Arguments**

        * ``logposterior`` : the logposterior of the parameters of interest.
        * ``proposal`` : the proposal distribution for random steps of the MCMC.
        * ``init_est`` : the initial/starting value for the markov chain.
        * ``d`` : the dimension of the posterior distribution.

    **Value**

        Returns a ``MH`` type object.

    **Example**
        See <section-tutorial>.
