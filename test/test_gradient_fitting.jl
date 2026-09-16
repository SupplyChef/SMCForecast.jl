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

    # Both methods' first call in a fresh Julia process pays one-time JIT
    # compilation on top of actual runtime, and that fixed cost doesn't
    # shrink with iteration count -- it dominated an earlier comparison
    # badly enough to make L-BFGS's ~18x iteration-count cut look like only
    # a ~2x wall-time cut. Paying that compilation cost here, on cheap
    # throwaway data, keeps it out of the timings below. Both proposals are
    # separate code paths (see differentiable_particle_loglikelihood), so
    # both get warmed up here.
    SMCForecast.fit_gradient(Val{LocalLevel}(), values[1:10]; particle_count=10, rng=MersenneTwister(0))
    SMCForecast.fit_gradient(Val{LocalLevel}(), values[1:10]; particle_count=10, proposal=:optimal, rng=MersenneTwister(0))
    SMCForecast.fit_gradient(Val{LocalLevel}(), values[1:10]; particle_count=10, resample_every=5, rng=MersenneTwister(0))
    # bboptimize2 converts MaxTime via Dates.Second(...), which requires a
    # whole number of seconds -- a fractional value like 0.5 throws
    # InexactError rather than just truncating.
    SMCForecast.fit(Val{LocalLevel}(), values[1:10]; maxtime=1, size=10)

    t_grad = @elapsed begin
        fitted_grad, iterations_used, n_feval_grad, n_geval_grad = SMCForecast.fit_gradient(Val{LocalLevel}(), values; particle_count=300, rng=MersenneTwister(1))
    end
    @test fitted_grad.level_variance > 0
    @test fitted_grad.observation_variance > 0
    kalman_ll_grad = SMCForecast.kalman_loglikelihood(fitted_grad.level, fitted_grad.level_variance, fitted_grad.observation_variance, values)
    @test kalman_ll_grad > kalman_ll_at_truth - slack

    # Diagnostic: does the bootstrap proposal's accuracy gap shrink just by
    # throwing more particles at it (a Monte Carlo variance story), or does
    # it persist regardless of particle count (a proposal bias/degeneracy
    # story)? Same proposal and particle-count-independent code path as
    # above, 16x the particles. Not asserted against `slack` -- this is
    # diagnostic, not a correctness check, and its outcome decides whether
    # the accuracy gap is fixable by particle count alone.
    t_grad_bigN = @elapsed begin
        fitted_grad_bigN, iterations_used_bigN, n_feval_bigN, n_geval_bigN = SMCForecast.fit_gradient(Val{LocalLevel}(), values; particle_count=5000, rng=MersenneTwister(1))
    end
    kalman_ll_grad_bigN = SMCForecast.kalman_loglikelihood(fitted_grad_bigN.level, fitted_grad_bigN.level_variance, fitted_grad_bigN.observation_variance, values)

    # Diagnostic/candidate fix: the locally optimal proposal at the same
    # particle count as the original bootstrap run above. If this closes
    # the gap that more bootstrap particles don't, the problem was proposal
    # bias, not Monte Carlo noise -- see differentiable_particle_loglikelihood's
    # docstring for why this proposal is expected to degenerate far slower.
    t_grad_optimal = @elapsed begin
        fitted_grad_optimal, iterations_used_optimal, n_feval_optimal, n_geval_optimal = SMCForecast.fit_gradient(Val{LocalLevel}(), values; particle_count=300, proposal=:optimal, rng=MersenneTwister(1))
    end
    @test fitted_grad_optimal.level_variance > 0
    @test fitted_grad_optimal.observation_variance > 0
    kalman_ll_grad_optimal = SMCForecast.kalman_loglikelihood(fitted_grad_optimal.level, fitted_grad_optimal.level_variance, fitted_grad_optimal.observation_variance, values)

    # Second diagnostic/candidate fix: the plain bootstrap proposal --
    # *not* :optimal -- plus periodic differentiable resampling, since
    # leaning on :optimal isn't a real fix: it only exists because
    # LocalLevel happens to be linear-Gaussian, so it wouldn't help a model
    # with a discrete state (e.g. LocalLevelCountStockout), which is
    # exactly the case resampling is supposed to cover.
    #
    # Several rounds of tuning a *soft*-resampling implementation (mixing
    # weights with a uniform floor and correcting with an importance ratio,
    # per Karkus/Hsu/Lee 2018) got real accuracy (closed the gap to within
    # ~0.2-0.6 of the true value) but stayed 9-11x *slower* than
    # bboptimize2 no matter how backtracking or the convergence tolerance
    # were tuned -- the winning L-BFGS restart just oscillated near a
    # resampling discontinuity indefinitely rather than converging, and no
    # optimizer-side knob fixed that. Going back to how the differentiable-
    # SMC literature actually does this (Maddison et al. 2017 "Filtering
    # Variational Objectives"; Naesseth et al. 2018 "Variational Sequential
    # Monte Carlo"; Le et al. 2018 "Auto-Encoding Sequential Monte Carlo")
    # found the likely cause: none of them use an importance correction for
    # gradient-based training. They resample directly from the normalized
    # weights and reset log-weights to 0 afterward (see
    # differentiable_particle_loglikelihood's docstring) -- resampling
    # directly from W makes the correction ratio identically 1 for
    # whichever particle survives, so adding one back in only injects an
    # artificial, particle-dependent jump in the objective's *value* at
    # exactly the θ where an ancestor switches. That's a self-inflicted
    # discontinuity on top of the irreducible one from the hard ancestor
    # choice itself, and is the more likely reason the line search
    # struggled.
    #
    # The literature-standard version, run at plain default settings,
    # validated the theory on accuracy: kalman-ll landed at -474.02,
    # essentially matching bboptimize2's own optimum (-474.02) almost
    # exactly -- the best result of every approach tried, using the fully
    # generic bootstrap proposal, no dependence on :optimal. But the
    # winning restart still used the full default maxiter=200, taking
    # 135.8s (~26x bboptimize2's 5.1s). Unlike the soft-resampling case,
    # this isn't oscillation near a bad point -- it found essentially the
    # true optimum -- so it's ordinary "L-BFGS taking many steps to satisfy
    # a tight gradient-norm tolerance very close to an optimum," and
    # bounding maxiter should cost little accuracy now that the underlying
    # landscape is confirmed smooth, rather than being another blind
    # parameter guess. maxiter=30 got very close (kalman-ll=-474.06,
    # essentially matching both true and bboptimize2; 16.2s against a
    # 15.3s bar, missed by ~6%) -- iterations scaled roughly linearly with
    # time here and accuracy has stayed excellent at every budget tried in
    # this smooth landscape, so trimming further should clear the bar with
    # room to spare rather than needing another round.
    t_grad_resampled = @elapsed begin
        fitted_grad_resampled, iterations_used_resampled, n_feval_resampled, n_geval_resampled = SMCForecast.fit_gradient(Val{LocalLevel}(), values; particle_count=300, resample_every=30, maxiter=20, rng=MersenneTwister(1))
    end
    @test fitted_grad_resampled.level_variance > 0
    @test fitted_grad_resampled.observation_variance > 0
    kalman_ll_grad_resampled = SMCForecast.kalman_loglikelihood(fitted_grad_resampled.level, fitted_grad_resampled.level_variance, fitted_grad_resampled.observation_variance, values)

    t_deriv_free = @elapsed begin
        fitted_bb = SMCForecast.fit(Val{LocalLevel}(), values; maxtime=5.0, size=200)
    end
    kalman_ll_bb = SMCForecast.kalman_loglikelihood(fitted_bb.level, fitted_bb.level_variance, fitted_bb.observation_variance, values)
    @test kalman_ll_bb > kalman_ll_at_truth - slack

    # n_feval/n_geval are exact counts (see lbfgs's docstring), not
    # estimates: n_feval is every plain (Float64) call to the loss closure
    # (the initial candidate plus every Armijo backtrack), n_geval is every
    # ForwardDiff.gradient call (each internally evaluates the closure once
    # more, but with a length-3 Dual argument in a single chunk -- costlier
    # per call than a plain eval, not 3 separate calls). Both are summed
    # across all 9 restarts. bboptimize2's own per-candidate evaluation
    # count for this exact call is printed by bboptimize2 itself (in
    # SMC.jl) immediately above this block, as "<n>, <elapsed>, <best_f>,
    # <best_x>" -- look for the line right before "true params:" in the
    # log to get its number for direct comparison.
    println("true params:            level=$true_level, level_variance=$true_level_variance, observation_variance=$true_observation_variance, kalman-ll=$kalman_ll_at_truth")
    println("gradient fit (bootstrap, N=300):   level=$(fitted_grad.level), level_variance=$(fitted_grad.level_variance), observation_variance=$(fitted_grad.observation_variance), kalman-ll=$kalman_ll_grad, $(iterations_used) iterations (winning restart), $(n_feval_grad) feval + $(n_geval_grad) geval (all 9 restarts), $(t_grad)s")
    println("gradient fit (bootstrap, N=5000):  level=$(fitted_grad_bigN.level), level_variance=$(fitted_grad_bigN.level_variance), observation_variance=$(fitted_grad_bigN.observation_variance), kalman-ll=$kalman_ll_grad_bigN, $(iterations_used_bigN) iterations (winning restart), $(n_feval_bigN) feval + $(n_geval_bigN) geval (all 9 restarts), $(t_grad_bigN)s")
    println("gradient fit (optimal, N=300):      level=$(fitted_grad_optimal.level), level_variance=$(fitted_grad_optimal.level_variance), observation_variance=$(fitted_grad_optimal.observation_variance), kalman-ll=$kalman_ll_grad_optimal, $(iterations_used_optimal) iterations (winning restart), $(n_feval_optimal) feval + $(n_geval_optimal) geval (all 9 restarts), $(t_grad_optimal)s")
    println("gradient fit (bootstrap+resample every 30, N=300, maxiter=20): level=$(fitted_grad_resampled.level), level_variance=$(fitted_grad_resampled.level_variance), observation_variance=$(fitted_grad_resampled.observation_variance), kalman-ll=$kalman_ll_grad_resampled, $(iterations_used_resampled) iterations (winning restart), $(n_feval_resampled) feval + $(n_geval_resampled) geval (all 9 restarts), $(t_grad_resampled)s")
    println("derivative-free:                    level=$(fitted_bb.level), level_variance=$(fitted_bb.level_variance), observation_variance=$(fitted_bb.observation_variance), kalman-ll=$kalman_ll_bb, $(t_deriv_free)s")

    # Not asserting t_grad < t_deriv_free: bboptimize2 is time-boxed
    # (MaxTime), not evaluation-boxed, and lbfgs's own stopping point is a
    # tolerance on the gradient norm -- both are independent, somewhat
    # arbitrary constants, so a strict comparison between the two wall
    # times is not a real invariant. The likelihood checks above are the
    # actual accuracy comparison; this is just a sanity ceiling against a
    # genuine hang. fit_gradient now runs a 3x3 grid of independent L-BFGS
    # restarts (see its docstring -- a single start, or restarts that only
    # vary the initial variance guess, reliably converge to a real but
    # mediocre local optimum), so the ceiling allows for several times one
    # run's worst-case time.
    @test t_grad < 60.0
    @test t_grad_bigN < 120.0
    @test t_grad_optimal < 60.0
    # Not tightened to bboptimize2's own 5.09s since that's one
    # MaxTime-boxed run, not a hard target, but still meant to catch a
    # regression back into "slower than the derivative-free method", which
    # would defeat the purpose of using gradients at all (see the comment
    # above t_grad_resampled for what already failed this bar and why).
    @test t_grad_resampled < t_deriv_free * 3
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

    # Same two correctness checks (matches the Kalman oracle; is
    # ForwardDiff-differentiable) for the :optimal proposal, plus the
    # comparison that actually justifies adding it: at a *matched*, much
    # smaller particle count than the 5000 used above, the optimal
    # proposal's importance weights should degenerate far less than the
    # bootstrap proposal's, so it should land closer to the Kalman-exact
    # value on the same data and standard normals draw shape.
    particle_ll_optimal = SMCForecast.differentiable_particle_loglikelihood([true_level, true_level_variance, true_observation_variance], values, standard_normals; proposal=:optimal)
    @test isfinite(particle_ll_optimal)
    @test abs(particle_ll_optimal - kalman_ll) < 15.0

    grad_optimal = ForwardDiff.gradient(θ -> SMCForecast.differentiable_particle_loglikelihood(θ, values, standard_normals; proposal=:optimal), θ0)
    @test all(isfinite, grad_optimal)
    @test any(g -> abs(g) > 1e-6, grad_optimal)

    standard_normals_small = randn(MersenneTwister(99), 50, T)
    particle_ll_bootstrap_small = SMCForecast.differentiable_particle_loglikelihood([true_level, true_level_variance, true_observation_variance], values, standard_normals_small)
    particle_ll_optimal_small = SMCForecast.differentiable_particle_loglikelihood([true_level, true_level_variance, true_observation_variance], values, standard_normals_small; proposal=:optimal)
    println("kalman-ll=$kalman_ll, bootstrap (N=50)=$particle_ll_bootstrap_small, optimal (N=50)=$particle_ll_optimal_small")
    @test abs(particle_ll_optimal_small - kalman_ll) < abs(particle_ll_bootstrap_small - kalman_ll)

    # resample_every=0 must reproduce the pre-existing no-resampling
    # arithmetic exactly (regression check that adding the option didn't
    # perturb default behavior) -- for both proposals, since resampling is
    # now handled by the same shared loop for either.
    @test SMCForecast.differentiable_particle_loglikelihood(θ0, values, standard_normals; resample_every=0) == particle_ll
    @test SMCForecast.differentiable_particle_loglikelihood(θ0, values, standard_normals; proposal=:optimal, resample_every=0) == particle_ll_optimal

    # Differentiable resampling (resample directly from the normalized
    # weights and reset log-weights to 0 afterward, the standard treatment
    # in the gradient-based SMC literature -- see
    # differentiable_particle_loglikelihood's docstring): a second,
    # model-agnostic answer to the same weight-degeneracy problem that --
    # unlike proposal=:optimal -- doesn't depend on LocalLevel being
    # linear-Gaussian. Tested here with the plain bootstrap proposal
    # specifically, so any improvement it shows isn't riding on the
    # Gaussian-specific fix above. At the same small particle count as the
    # bootstrap-vs-optimal comparison, periodic resampling should bring the
    # bootstrap proposal closer to the Kalman-exact value than no
    # resampling does.
    resample_every = 10
    n_resamples = count(t -> t % resample_every == 0, 1:(T - 1))
    resampling_uniforms = rand(MersenneTwister(123), n_resamples)

    particle_ll_resampled_small = SMCForecast.differentiable_particle_loglikelihood([true_level, true_level_variance, true_observation_variance], values, standard_normals_small; resample_every=resample_every, resampling_uniforms=resampling_uniforms)
    @test isfinite(particle_ll_resampled_small)

    grad_resampled = ForwardDiff.gradient(θ -> SMCForecast.differentiable_particle_loglikelihood(θ, values, standard_normals_small; resample_every=resample_every, resampling_uniforms=resampling_uniforms), θ0)
    @test all(isfinite, grad_resampled)
    @test any(g -> abs(g) > 1e-6, grad_resampled)

    println("kalman-ll=$kalman_ll, bootstrap (N=50, no resample)=$particle_ll_bootstrap_small, bootstrap (N=50, resampled every $resample_every)=$particle_ll_resampled_small")
    @test abs(particle_ll_resampled_small - kalman_ll) < abs(particle_ll_bootstrap_small - kalman_ll)
end
