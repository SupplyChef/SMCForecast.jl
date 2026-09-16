# --------------------------------------------------------------------------
# Differentiable (gradient-based) fitting for LocalLevelCountStockout.
#
# Unlike LocalLevel/LocalLevelChange, this model has no closed-form
# marginal likelihood to use as a ground-truth oracle in tests: it has a
# genuinely discrete hidden state (a 2-state in-stock/stockout Markov
# chain, not just a continuous one with a near-point-mass prior) and a
# non-Gaussian, non-linear observation model (a zero-inflated generalized
# Poisson in-stock, a plain Poisson out-of-stock). This is exactly the case
# differentiable resampling was built to reach and :optimal never could
# (see DifferentiableLocalLevel.jl): there is no analog of "the locally
# optimal proposal" here, so this file only ever uses the bootstrap
# proposal, made fast the same way it was for the other two models
# (resample directly from the normalized weights, reset log-weights to 0,
# no importance correction -- see DifferentiableLocalLevel.jl's docstring
# for the full history of why).
#
# Tests use the existing, already-validated get_loss_function(::Val{LocalLevelCountStockout}, ...)
# (the same properly-resampling bootstrap filter bboptimize2 itself
# optimizes against) at a large particle count as the reference likelihood
# in place of a Kalman oracle -- not exact, but independently implemented
# and already exercised by this package's own test suite, which is the
# best available substitute here.
#
# Two more simplifications relative to the full LocalLevelCountStockoutModel,
# both restricted to this differentiable path (the non-differentiable
# fit/get_loss_function path is untouched):
#   - the discrete Markov regime transition is proposed from a fixed
#     Uniform(0,1) draw per particle per timestep compared against the
#     (θ-dependent) transition probability -- the same reparameterization
#     trick as resampling's ancestor selection, and for the same reason:
#     the *choice* of new regime is a genuinely discrete, non-differentiable
#     function of θ, but the *value* an in-stock particle's continuous
#     level carries forward, and the observation weight computed from
#     whichever regime was chosen, are still smooth functions of θ almost
#     everywhere;
#   - the initial regime is fixed at "in stock" for every particle, rather
#     than sampled from the θ-dependent stationary-ish distribution
#     LocalLevelCountStockoutModel's own sample_initial_state uses. This
#     only biases the first few timesteps' worth of likelihood before the
#     chain mixes away from it, and keeps the initial state a clean,
#     differentiable point mass instead of requiring a differentiable
#     categorical draw with no fixed particles to reparameterize against.
# --------------------------------------------------------------------------

"""
    log_poisson_pmf(k, lambda)

log pdf of Poisson(lambda) at nonnegative integer `k`, `lambda` a smooth
(possibly Dual) value. Hand-written rather than calling
`Distributions.logpdf(Poisson(lambda), k)` so it's guaranteed to work
uniformly for any real-valued `lambda` (including ForwardDiff Duals)
without depending on that method's own internals supporting them.
"""
function log_poisson_pmf(k::Int, lambda)
    return k * log(lambda) - lambda - logfactorial(k)
end

"""
    log_zigp_pmf(k, lambda, theta, pi)

log pdf of the zero-inflated generalized Poisson distribution at
nonnegative integer `k`, matching `zigp_pmf`/`log_generalized_poisson_pmf`
in CountStockoutCore.jl exactly in its math, but written to stay in log
space throughout and to avoid that function's `if lambda < 0; return 0.0;
end` guard -- returning a bare `Float64` there would truncate a Dual
argument and silently zero out its derivative, which is fine for
`zigp_pmf`'s existing (non-differentiable) callers but not for this one.
`lambda`, `theta`, `pi` are expected to be valid (`lambda > 0`, `theta`
and `pi` in the model's usual ranges) by construction of the caller's
parameterization (log/logit-transformed in `fit_gradient`), so that guard
is not reproduced here.
"""
function log_zigp_pmf(k::Int, lambda, theta, pi)
    if k == 0
        return log(pi + (1 - pi) * exp(-lambda))
    end
    log_one_minus_pi = log(1 - pi)
    if k == 1
        return log_one_minus_pi + log(lambda) - (lambda + theta)
    end
    v = lambda + k * theta
    return log_one_minus_pi + log(lambda) + (k - 1) * log(v) - v - logfactorial(k)
end

"""
    differentiable_particle_loglikelihood(θ, values, standard_normals, regime_uniforms; resample_every=0, resampling_uniforms=nothing)

`θ` is `[level1, level2, level_variance, zero_inflation, overdispersion,
p12, p22]` (matching `get_loss_function(::Val{LocalLevelCountStockout},
...)`'s parameter order exactly): `level1` the initial/in-stock level,
`level2` the out-of-stock Poisson mean and the level's floor, `p12` =
P(in-stock -> stockout), `p22` = P(stockout -> stockout). `values` are
the observed counts. `standard_normals` (n_particles x length(values))
are fixed N(0,1) draws for the level's transition noise; `regime_uniforms`
(same shape) are fixed Uniform(0,1) draws used to propose each particle's
new regime, the reparameterization trick applied to the discrete Markov
transition -- see the module-level note above. Bootstrap proposal only
(there is no locally-optimal proposal for this model); `resample_every`/
`resampling_uniforms` behave exactly as on the other two models' versions
of this function, reusing `systematic_resample_indices` unchanged.
"""
function differentiable_particle_loglikelihood(θ, values, standard_normals::AbstractMatrix, regime_uniforms::AbstractMatrix;
                                                resample_every::Int=0,
                                                resampling_uniforms::Union{Nothing,AbstractVector}=nothing)
    level1, level2, level_variance, zero_inflation, overdispersion, p12, p22 = θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7]
    n_particles = size(standard_normals, 1)
    T = length(values)

    RT = promote_type(typeof(level1), typeof(level2), typeof(level_variance), typeof(zero_inflation), typeof(overdispersion), typeof(p12), typeof(p22))
    value = fill(convert(RT, level1), n_particles)
    value_buf = similar(value)
    regime = ones(Int, n_particles) # fixed "in stock" initial regime -- see module note
    regime_buf = similar(regime)
    log_weights = zeros(RT, n_particles)
    total_loglik = zero(RT)
    resample_count = 0

    level_sd = sqrt(level_variance)
    one_minus_od_over_one_minus_zi = (1 - overdispersion) / (1 - zero_inflation)

    for t in 1:T
        y = Int(round(values[t]))
        @inbounds for i in 1:n_particles
            new_regime = regime[i] == 1 ? (regime_uniforms[i, t] < p12 ? 2 : 1) : (regime_uniforms[i, t] < p22 ? 2 : 1)
            new_value = max(value[i] + level_sd * standard_normals[i, t], level2)

            if new_regime == 2
                log_weights[i] += log_poisson_pmf(y, level2)
            else
                lambda = new_value * one_minus_od_over_one_minus_zi
                log_weights[i] += log_zigp_pmf(y, lambda, overdispersion, zero_inflation)
            end

            value[i] = new_value
            regime[i] = new_regime
        end

        if resample_every > 0 && t < T && t % resample_every == 0
            resample_count += 1
            u0 = resampling_uniforms[resample_count]

            m = maximum(log_weights)
            w_unnorm = exp.(log_weights .- m)
            w_sum = sum(w_unnorm)
            total_loglik += m + log(w_sum) - log(n_particles)

            W = w_unnorm ./ w_sum
            ancestors = systematic_resample_indices(W, u0)

            @inbounds for i in 1:n_particles
                value_buf[i] = value[ancestors[i]]
                regime_buf[i] = regime[ancestors[i]]
            end
            value, value_buf = value_buf, value
            regime, regime_buf = regime_buf, regime
            fill!(log_weights, zero(RT))
        end
    end

    m = maximum(log_weights)
    total_loglik += m + log(sum(exp(lw - m) for lw in log_weights)) - log(n_particles)
    return total_loglik
end

"""
    get_loss_function_gradient(::Val{LocalLevelCountStockout}, values; particle_count=200, resample_every=0, rng=Random.default_rng())

Gradient-friendly counterpart to get_loss_function(::Val{LocalLevelCountStockout}, ...):
returns `φ -> -loglik` where `φ` is `[log(level1), log(level2),
log(level_variance), logit(zero_inflation), logit(overdispersion),
logit(p12), logit(p22)]` -- log/logit-transformed so every one of the 7
parameters is unconstrained for the optimizer while `θ` itself always
stays in its valid range (positive levels/variance, probabilities in
(0,1)), the same role LocalLevel/LocalLevelChange's log-transformed
variances play.
"""
function get_loss_function_gradient(::Val{LocalLevelCountStockout}, values; particle_count=200, resample_every::Int=0, rng=Random.default_rng())
    standard_normals = randn(rng, particle_count, length(values))
    regime_uniforms = rand(rng, particle_count, length(values))
    n_resamples = resample_every > 0 ? count(t -> t % resample_every == 0, 1:(length(values) - 1)) : 0
    resampling_uniforms = resample_every > 0 ? rand(rng, n_resamples) : nothing
    sigmoid(x) = 1 / (1 + exp(-x))
    return φ -> begin
        level1, level2, level_variance = exp(φ[1]), exp(φ[2]), exp(φ[3])
        zero_inflation, overdispersion, p12, p22 = sigmoid(φ[4]), sigmoid(φ[5]), sigmoid(φ[6]), sigmoid(φ[7])
        -differentiable_particle_loglikelihood([level1, level2, level_variance, zero_inflation, overdispersion, p12, p22], values,
                                                standard_normals, regime_uniforms;
                                                resample_every=resample_every, resampling_uniforms=resampling_uniforms)
    end
end

"""
    fit_gradient(::Val{LocalLevelCountStockout}, values; particle_count=200, maxiter=200, tol=1e-6, resample_every=0, max_backtracks=20, rng=Random.default_rng())

Gradient-based counterpart to fit(::Val{LocalLevelCountStockout}, ...).
Restarts over `level1` (mean/max of `values`, spanning the plausible
in-stock range) x variance `scale` (1/10, deliberately narrower than the
other two models' 0.1/1/10 -- level_variance's own scale is already tiny
relative to the data here, see `base_variance_guess`, and a 0.1x-smaller
guess on top of that produced a degenerate near-zero starting variance in
early testing), 2x2=4 restarts rather than LocalLevel's 3x3=9: each
restart is markedly more expensive here (7 active parameters and a
per-particle ZIGP evaluation, not a handful of Gaussian arithmetic ops),
and unlike LocalLevel this model's likelihood surface hasn't been
observed to need a wide multi-start grid to avoid a bad local optimum --
this restart grid exists mainly to hedge against one unlucky starting
`level1`, not to escape a known degenerate basin. `level2`,
`zero_inflation`, `overdispersion`, `p12`, `p22` are not gridded, fixed at
single reasonable starting guesses instead, mirroring bboptimize2's own
`fit`'s single incumbent starting point for the same 4 parameters.

Returns (fitted LocalLevelCountStockout, iterations_used, total_n_feval,
total_n_geval) -- see LocalLevel's fit_gradient docstring for what these
mean.
"""
function fit_gradient(::Val{LocalLevelCountStockout}, values; particle_count=200, maxiter=200, tol::Real=1e-6, resample_every::Int=0, max_backtracks::Int=20, rng=Random.default_rng())
    loss = get_loss_function_gradient(Val{LocalLevelCountStockout}(), values; particle_count=particle_count, resample_every=resample_every, rng=rng)

    logit(p) = log(p / (1 - p))

    level1_guesses = (sum(values) / length(values), maximum(values))
    base_variance_guess = var(values) / length(values)
    scale_guesses = (1.0, 10.0)
    level2_guess = max(minimum(values), 0.05)
    zero_inflation_guess = 0.05
    overdispersion_guess = 0.1
    p12_guess = 0.05
    p22_guess = 0.8

    best_φ = nothing
    best_f = Inf
    best_iterations = 0
    total_n_feval = 0
    total_n_geval = 0
    for level1_0 in level1_guesses, scale in scale_guesses
        φ0 = [log(level1_0), log(level2_guess), log(base_variance_guess * scale),
              logit(zero_inflation_guess), logit(overdispersion_guess), logit(p12_guess), logit(p22_guess)]

        φ_opt, f_opt, iterations_used, n_feval, n_geval = lbfgs(loss, φ0; maxiter=maxiter, tol=tol, max_backtracks=max_backtracks)
        total_n_feval += n_feval
        total_n_geval += n_geval
        if f_opt < best_f
            best_f = f_opt
            best_φ = φ_opt
            best_iterations = iterations_used
        end
    end

    sigmoid(x) = 1 / (1 + exp(-x))
    level1, level2, level_variance = exp(best_φ[1]), exp(best_φ[2]), exp(best_φ[3])
    zero_inflation, overdispersion, p12, p22 = sigmoid(best_φ[4]), sigmoid(best_φ[5]), sigmoid(best_φ[6]), sigmoid(best_φ[7])

    fitted = LocalLevelCountStockout(; level1=level1, level2=level2, level_variance=level_variance,
                                      zero_inflation=zero_inflation, overdispersion=overdispersion,
                                      level_matrix=[1-p12 p12; 1-p22 p22])
    return fitted, best_iterations, total_n_feval, total_n_geval
end
