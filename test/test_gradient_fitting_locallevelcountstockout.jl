@testitem "Gradient-based fitting (LocalLevelCountStockout): recovers known parameters and matches a large-N particle-filter oracle" begin
    using SMCForecast
    using Distributions
    using Random
    using StaticArrays
    using Statistics

    # LocalLevelCountStockout has no closed-form marginal likelihood (a
    # genuinely discrete in-stock/stockout regime plus a non-Gaussian
    # zero-inflated generalized-Poisson/Poisson observation model) --
    # unlike LocalLevel and LocalLevelChange there is no Kalman-equivalent
    # ground truth here. In its place, get_loss_function(::Val{LocalLevelCountStockout},
    # ...) at a large particle count -- the same, already-existing,
    # already-tested bootstrap filter bboptimize2 itself optimizes against
    # (see LocalLevelCountStockout.jl) -- is used as the reference
    # likelihood: not exact, but independently implemented (it performs
    # real resampling every step via filter!/resample!, not the
    # resampling-optional surrogate this file's own
    # differentiable_particle_loglikelihood uses) and already exercised by
    # this package's own test suite, so it's the most trustworthy
    # substitute available.
    true_system = LocalLevelCountStockout(; level1=40.0, level2=2.0, level_variance=9.0,
                                           zero_inflation=0.1, overdispersion=0.15,
                                           level_matrix=[0.95 0.05; 0.2 0.8])
    T = 150

    rng = MersenneTwister(20260916)
    smc_gen = SMC{MVector{3, Float64}, LocalLevelCountStockout}(true_system, 1)
    initialize!(smc_gen; rng=rng)
    # happy_only=false is required to actually generate stockout periods --
    # predict_states/predict_observations default to happy_only=true (used
    # for forecasting from an already-fitted model, where you don't want a
    # single simulated path stuck in a stockout state dominating a
    # forecast), which would silently reject every transition into state 2
    # and produce data indistinguishable from a plain LocalLevel series.
    obs, _ = predict_observations(smc_gen, T; happy_only=false, rng=rng)
    values = map(o -> Float64(round(o[1])), obs)

    true_xs = [true_system.level1, true_system.level2, true_system.level_variance,
               true_system.zero_inflation, true_system.overdispersion,
               true_system.level_matrix[1, 2], true_system.level_matrix[2, 2]]

    reference_loss = SMCForecast.get_loss_function(Val{LocalLevelCountStockout}(), values; size=3000)
    true_nll = reference_loss(true_xs)
    @test isfinite(true_nll)

    # Generous slack relative to LocalLevel's 20.0: this model has 7 active
    # parameters (vs 3), a Monte Carlo reference likelihood itself (not an
    # exact Kalman value), and a differentiable surrogate that -- unlike
    # the reference filter -- uses a plain bootstrap proposal for the
    # continuous level (no locally-optimal proposal exists for this model;
    # see DifferentiableLocalLevelCountStockout.jl's module docstring).
    slack = 40.0

    # JIT warm-up on cheap throwaway data, same rationale as LocalLevel's
    # own test -- both proposal-free code paths (plain and resampled) are
    # exercised here since they're separate branches internally.
    SMCForecast.fit_gradient(Val{LocalLevelCountStockout}(), values[1:10]; particle_count=10, maxiter=2, rng=MersenneTwister(0))
    SMCForecast.fit_gradient(Val{LocalLevelCountStockout}(), values[1:10]; particle_count=10, resample_every=5, maxiter=2, rng=MersenneTwister(0))
    SMCForecast.fit(Val{LocalLevelCountStockout}(), values[1:10]; maxtime=1, size=10)

    t_grad = @elapsed begin
        fitted_grad, iterations_used, n_feval, n_geval = SMCForecast.fit_gradient(Val{LocalLevelCountStockout}(), values;
                                                                                   particle_count=300, resample_every=30, maxiter=50, rng=MersenneTwister(1))
    end
    grad_xs = [fitted_grad.level1, fitted_grad.level2, fitted_grad.level_variance,
               fitted_grad.zero_inflation, fitted_grad.overdispersion,
               fitted_grad.level_matrix[1, 2], fitted_grad.level_matrix[2, 2]]
    grad_nll = reference_loss(grad_xs)
    @test isfinite(grad_nll)
    @test grad_nll < true_nll + slack

    t_deriv_free = @elapsed begin
        fitted_bb = SMCForecast.fit(Val{LocalLevelCountStockout}(), values; maxtime=5.0, size=200)
    end
    bb_xs = [fitted_bb.level1, fitted_bb.level2, fitted_bb.level_variance,
             fitted_bb.zero_inflation, fitted_bb.overdispersion,
             fitted_bb.level_matrix[1, 2], fitted_bb.level_matrix[2, 2]]
    bb_nll = reference_loss(bb_xs)
    @test isfinite(bb_nll)

    # n_feval/n_geval are exact counts (see lbfgs's docstring), summed
    # across all 4 restarts (see fit_gradient's docstring for why this
    # model uses a narrower 2x2 grid than LocalLevel's 3x3).
    println("true params:      level1=$(true_system.level1), level2=$(true_system.level2), level_variance=$(true_system.level_variance), zero_inflation=$(true_system.zero_inflation), overdispersion=$(true_system.overdispersion), p12=$(true_system.level_matrix[1,2]), p22=$(true_system.level_matrix[2,2]), reference-nll=$true_nll")
    println("gradient fit:     level1=$(fitted_grad.level1), level2=$(fitted_grad.level2), level_variance=$(fitted_grad.level_variance), zero_inflation=$(fitted_grad.zero_inflation), overdispersion=$(fitted_grad.overdispersion), p12=$(fitted_grad.level_matrix[1,2]), p22=$(fitted_grad.level_matrix[2,2]), reference-nll=$grad_nll, $(iterations_used) iterations (winning restart), $(n_feval) feval + $(n_geval) geval (all 4 restarts), $(t_grad)s")
    println("derivative-free:  level1=$(fitted_bb.level1), level2=$(fitted_bb.level2), level_variance=$(fitted_bb.level_variance), zero_inflation=$(fitted_bb.zero_inflation), overdispersion=$(fitted_bb.overdispersion), p12=$(fitted_bb.level_matrix[1,2]), p22=$(fitted_bb.level_matrix[2,2]), reference-nll=$bb_nll, $(t_deriv_free)s")

    # Sanity ceiling against a genuine hang, not a strict comparison against
    # t_deriv_free -- see LocalLevel's own test for why that comparison
    # isn't a real invariant (bboptimize2 is time-boxed, not
    # evaluation-boxed).
    @test t_grad < 120.0
end

@testitem "Differentiable particle likelihood (LocalLevelCountStockout): finite, ForwardDiff-differentiable, and resampling helps" begin
    using SMCForecast
    using Distributions
    using Random
    using ForwardDiff
    using StaticArrays

    true_system = LocalLevelCountStockout(; level1=30.0, level2=1.0, level_variance=6.0,
                                           zero_inflation=0.1, overdispersion=0.15,
                                           level_matrix=[0.93 0.07; 0.25 0.75])
    T = 60
    rng = MersenneTwister(55)
    smc_gen = SMC{MVector{3, Float64}, LocalLevelCountStockout}(true_system, 1)
    initialize!(smc_gen; rng=rng)
    obs, _ = predict_observations(smc_gen, T; happy_only=false, rng=rng)
    values = map(o -> Float64(round(o[1])), obs)

    θ0 = [true_system.level1, true_system.level2, true_system.level_variance,
          true_system.zero_inflation, true_system.overdispersion,
          true_system.level_matrix[1, 2], true_system.level_matrix[2, 2]]

    # No Kalman-exact oracle exists here (see the other testitem), so this
    # file's own function at a large particle count and no resampling
    # stands in as the reference value for the differentiability/resampling
    # checks below -- the same role kalman_ll plays in LocalLevel's
    # equivalent test, just a Monte Carlo one rather than an exact one.
    standard_normals = randn(MersenneTwister(99), 5000, T)
    reference_ll = SMCForecast.differentiable_particle_loglikelihood(Val{LocalLevelCountStockout}(), θ0, values, standard_normals)
    @test isfinite(reference_ll)

    # Must actually be usable by ForwardDiff -- a non-finite or all-zero
    # gradient would mean the function silently isn't differentiable (e.g.
    # a stray hard-coded ::Float64/::Int guard truncating Dual numbers, the
    # exact failure mode log_zigp_pmf's docstring calls out). Checking
    # *every* component, not just "any", specifically guards against the
    # real bug this function's Rao-Blackwellized regime treatment replaced:
    # an earlier, regime-sampling version of this function had an exactly
    # zero gradient w.r.t. p12/p22 (grad[6]/grad[7]) for any theta, which
    # "any(... > 1e-6)" alone would not have caught since the other 5
    # components were always nonzero.
    grad = ForwardDiff.gradient(θ -> SMCForecast.differentiable_particle_loglikelihood(Val{LocalLevelCountStockout}(), θ, values, standard_normals), θ0)
    @test all(isfinite, grad)
    @test all(g -> abs(g) > 1e-8, grad)

    # resample_every=0 must reproduce the pre-existing no-resampling
    # arithmetic exactly (regression check that adding the option didn't
    # perturb default behavior).
    @test SMCForecast.differentiable_particle_loglikelihood(Val{LocalLevelCountStockout}(), θ0, values, standard_normals; resample_every=0) == reference_ll

    standard_normals_small = randn(MersenneTwister(99), 50, T)
    particle_ll_small = SMCForecast.differentiable_particle_loglikelihood(Val{LocalLevelCountStockout}(), θ0, values, standard_normals_small)

    resample_every = 10
    n_resamples = count(t -> t % resample_every == 0, 1:(T - 1))
    resampling_uniforms = rand(MersenneTwister(123), n_resamples)

    particle_ll_resampled_small = SMCForecast.differentiable_particle_loglikelihood(Val{LocalLevelCountStockout}(), θ0, values, standard_normals_small;
                                                                                     resample_every=resample_every, resampling_uniforms=resampling_uniforms)
    @test isfinite(particle_ll_resampled_small)

    grad_resampled = ForwardDiff.gradient(θ -> SMCForecast.differentiable_particle_loglikelihood(Val{LocalLevelCountStockout}(), θ, values, standard_normals_small;
                                                                                                   resample_every=resample_every, resampling_uniforms=resampling_uniforms), θ0)
    @test all(isfinite, grad_resampled)
    @test all(g -> abs(g) > 1e-8, grad_resampled)

    println("reference (N=5000, no resample)=$reference_ll, small (N=50, no resample)=$particle_ll_small, small (N=50, resampled every $resample_every)=$particle_ll_resampled_small")
    @test abs(particle_ll_resampled_small - reference_ll) < abs(particle_ll_small - reference_ll)
end
