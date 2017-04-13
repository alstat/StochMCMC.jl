Metropolis-Hasting
===================

Implementation of the Metropolis-Hasting sampler for Bayesian inference.

.. function:: MH(logposterior::Function, proposal::Function, init_est::Array{Float64}, d::Int64)

    Construct a ``Sampler`` object for Metropolis-Hasting sampling.

    **Arguments**

        * ``logposterior`` : the logposterior of the parameter of interest.
        * ``proposal`` : the proposal distribution for random steps of the MCMC.
        * ``init_est`` : the initial/starting value for the markov chain.
        * ``d`` : the dimension of the posterior distribution.

    **Value**

        Returns a ``MH`` type object.

    **Example**

    In order to illustrate the modeling, the data is simulated from a simple linear regression expectation function. That is the model is given by

    .. code-block:: txt

        y_i = w_0 + w_1 x_i + e_i,   e_i ~ N(0, 1 / a)

    To do so, let :code:`B = [w_0, w_1]' = [.2, -.9]', a = 1 / 5`. Generate 200 hypothetical data:

    .. code-block:: julia

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

    Next is to plot this data which can be done as follows:

    .. code-block:: julia

        plot(my_df, x = :Independent, y = :Dependent)

    .. image:: figures/plot1.png
        :width: 80%
        :align: center
        :alt: alternate text

    In order to proceed with the Bayesian inference, the parameters of the model is considered to be random modeled by a standard Gaussian distribution. That is, :code:`B ~ N(0, I)`, where :code:`0` is the zero vector. The likelihood of the data is given by,

    .. code-block:: txt

        L(w|[x, y], b) = ∏_{i=1}^n N([x_i, y_i]|w, b)

    Thus the posterior is given by,

    .. code-block:: txt

        P(w|[x, y]) ∝ P(w)L(w|[x, y], b)

    To start programming, define the probabilities

    .. code-block:: julia

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

    To start the estimation, define the necessary parameters for the Metropolis-Hasting algorithm

    .. code-block:: julia

        # Hyperparameters
        zero_vec = zeros(2)
        eye_mat = eye(2)

    Run the MCMC:

    .. code-block:: julia

        srand(123);
        mh_object = MH(logpost; init_est = zeros(2));
        chain1 = mcmc(mh_object, r = 10000);

    Extract the estimate

    .. code-block:: julia

        burn_in = 100;
        thinning = 10;

        # Expetation of the Posterior
        est1 = mapslices(mean, chain1[(burn_in + 1):thinning:end, :], [1]);
        est1
        # 1×2 Array{Float64,2}:
        #  -0.313208  -0.46376
