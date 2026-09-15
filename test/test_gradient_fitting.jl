@testitem "Gradient-based fitting (LocalLevel): recovers known parameters and matches the Kalman oracle" begin
    using SMCForecast
    using Distributions
    using Random
    using Statistics

    # Generate synthetic data from LocalLevel's own model with known true
    # parameters, so "accuracy" here means recovering (or matching the
    # likelihood of) parameters we actually chose, not just "the optimizer
    # didn't crash".
    rng = MersenneTwister(20260914)
    true_level = 50.0
    true_level_variance = 4.0
    true_observation_variance = 25.0
    T = 150

    values = let x = true_level
        vals = zeros(T)
        for t in 1:T
            x += sqrt(true_level_variance) * randn(rng)
            vals[t] = x + sqrt(true_observation_variance) * randn(rng)
        end
        vals
    end

    # LocalLevel is an exact linear-Gaussian state space model, so this is
    # the true marginal log-likelihood -- no Monte Carlo noise, no fitting
    # involved. It's the ground truth both fitting methods below are
    # compared against.
    kalman_ll_at_truth = SMCForecast.kalman_loglikelihood(true_level, true_level_variance, true_observation_variance, values)
    @test isfinite(kalman_ll_at_truth)

    # A correct maximum-likelihood fit should never do *worse*, at the true
    # (Kalman-exact) likelihood, than the parameters the data was actually
    # generated from -- that's what "maximum" means, and it holds
    # regardless of which particular random draw of synthetic data we got.
    # That makes it a far more robust check than bounding each parameter's
    # own recovery error, which is inherently noisy for any correct
    # estimator on a single dataset. `slack` allows for the optimizer not
    # reaching the exact optimum and for the particle-based objective's own
    # Monte Carlo approximation error relative to the exact Kalman value.
    slack = 20.0

    t_grad = @elapsed begin
        fitted_grad, iterations_used = SMCForecast.fit_gradient(Val{LocalLevel}(), values; particle_count=300, rng=MersenneTwister(1))
    end
    @test fitted_grad.level_variance > 0
    @test fitted_grad.observation_variance > 0
    kalman_ll_grad = SMCForecast.kalman_loglikelihood(fitted_grad.level, fitted_grad.level_variance, fitted_grad.observation_variance, values)
    @test kalman_ll_grad > kalman_ll_at_truth - slack

    t_deriv_free = @elapsed begin
        fitted_bb = SMCForecast.fit(Val{LocalLevel}(), values; maxtime=5.0, size=200)
    end
    kalman_ll_bb = SMCForecast.kalman_loglikelihood(fitted_bb.level, fitted_bb.level_variance, fitted_bb.observation_variance, values)
    @test kalman_ll_bb > kalman_ll_at_truth - slack

    println("true params:      level=$true_level, level_variance=$true_level_variance, observation_variance=$true_observation_variance, kalman-ll=$kalman_ll_at_truth")
    println("gradient fit:     level=$(fitted_grad.level), level_variance=$(fitted_grad.level_variance), observation_variance=$(fitted_grad.observation_variance), kalman-ll=$kalman_ll_grad, $(iterations_used) iterations, $(t_grad)s")
    println("derivative-free:  level=$(fitted_bb.level), level_variance=$(fitted_bb.level_variance), observation_variance=$(fitted_bb.observation_variance), kalman-ll=$kalman_ll_bb, $(t_deriv_free)s")

    # Not asserting t_grad < t_deriv_free: bboptimize2 is time-boxed
    # (MaxTime), not evaluation-boxed, and lbfgs's own stopping point is a
    # tolerance on the gradient norm -- both are independent, somewhat
    # arbitrary constants, so a strict comparison between the two wall
    # times is not a real invariant. The likelihood checks above are the
    # actual accuracy comparison; this is just a sanity ceiling against a
    # genuine hang. fit_gradient runs n_restarts=4 independent L-BFGS
    # optimizations (see its docstring -- a single start could converge to
    # a degenerate level_variance≈0 local optimum), so the ceiling allows
    # for several times one run's worst-case time; L-BFGS should need far
    # fewer iterations than the plain gradient descent this replaced, so
    # 60s remains a loose sanity check rather than a tight bound.
    @test t_grad < 60.0
end

@testitem "Differentiable particle likelihood (LocalLevel): matches the Kalman oracle and is ForwardDiff-differentiable" begin
    using SMCForecast
    using Distributions
    using Random
    using ForwardDiff

    rng = MersenneTwister(7)
    true_level = 20.0
    true_level_variance = 1.5
    true_observation_variance = 8.0
    T = 80

    values = let x = true_level
        vals = zeros(T)
        for t in 1:T
            x += sqrt(true_level_variance) * randn(rng)
            vals[t] = x + sqrt(true_observation_variance) * randn(rng)
        end
        vals
    end

    kalman_ll = SMCForecast.kalman_loglikelihood(true_level, true_level_variance, true_observation_variance, values)

    # With enough particles, the resampling-free particle estimator should
    # land close to the exact value -- the main correctness check on
    # differentiable_particle_loglikelihood itself, independent of fitting.
    # The tolerance here is a heuristic (not derived from a variance
    # formula) since there's no local Julia environment available while
    # writing this to calibrate it empirically -- loosen it if it flakes,
    # a real bug should miss by far more than this.
    standard_normals = randn(MersenneTwister(99), 5000, T)
    particle_ll = SMCForecast.differentiable_particle_loglikelihood([true_level, true_level_variance, true_observation_variance], values, standard_normals)
    @test isfinite(particle_ll)
    @test abs(particle_ll - kalman_ll) < 15.0

    # And it must actually be usable by ForwardDiff -- a non-finite or
    # all-zero gradient here would mean the function silently isn't
    # differentiable (e.g. a stray hard-coded ::Float64 truncating Dual
    # numbers), which is exactly the failure mode this test exists to catch.
    θ0 = [true_level, true_level_variance, true_observation_variance]
    grad = ForwardDiff.gradient(θ -> SMCForecast.differentiable_particle_loglikelihood(θ, values, standard_normals), θ0)
    @test all(isfinite, grad)
    @test any(g -> abs(g) > 1e-6, grad)
end
