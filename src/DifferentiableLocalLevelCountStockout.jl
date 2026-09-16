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
# The discrete in-stock/stockout regime is NOT reparameterized the way
# resampling's ancestor selection is (a fixed uniform draw compared
# against a threshold). An earlier version of this file did exactly that
# (regime_uniforms[i, t] < p12/p22 choosing a hard new_regime), and it
# compiled and ran, but it has a real, structural bug: p12/p22 then only
# ever decide *which branch* of the log-weight update executes, and
# neither branch's own value (log_zigp_pmf/log_poisson_pmf) contains p12
# or p22 at all -- so for any *fixed* sequence of regime_uniforms draws,
# the whole log-likelihood is a piecewise-*constant* function of p12/p22
# (not just piecewise-smooth), with a derivative of exactly zero
# everywhere except at the measure-zero threshold crossings themselves.
# ForwardDiff never sees those crossings, so ForwardDiff.gradient's p12/p22
# components come back as exactly 0.0 for any θ, and L-BFGS can never move
# them off their starting guess -- confirmed in CI: the winning restart's
# fitted p12/p22 matched their initial guesses to full floating point
# precision. This is a different situation from resampling's ancestor
# choice, where the *value* an ancestor carries forward is itself a smooth
# function of θ, so a real gradient signal still flows through every later
# timestep even though *which* ancestor was picked doesn't.
#
# The fix used here instead is a Rao-Blackwellised particle filter (Doucet,
# de Freitas, Murphy & Russell, 2000, "Rao-Blackwellised particle filtering
# for dynamic Bayesian networks"): since the regime's transition doesn't
# depend on the continuous level and the level's transition doesn't depend
# on the regime (only the *observation* density couples them), the 2-state
# regime can be marginalized out analytically per particle instead of
# sampled. Each particle carries a belief vector (P(in stock), P(stockout))
# that is exactly propagated through the level_matrix transition (a plain
# 2x2 matrix-vector product, smooth in p12/p22) and exactly updated by
# Bayes' rule against the observation mixture -- the standard HMM forward
# algorithm, run independently per particle since each particle's own
# level path enters the observation densities differently. This removes
# regime sampling (and its regime_uniforms input) entirely: p12/p22 now
# enter the log-likelihood continuously (as belief-mixing weights), giving
# a real, non-zero gradient, and it's a strict accuracy improvement too --
# an exact treatment of the discrete component has no Monte Carlo error of
# its own, unlike sampling it would.
#
# The initial belief is computed the same way LocalLevelCountStockoutModel's
# own sample_initial_state does (level_matrix^10 applied to a point mass at
# "in stock"), just as a smooth, differentiable 2-vector recursion instead
# of a stationary-ish sampling distribution -- an exact match to the real
# model's own initial-state distribution rather than a simplification of
# it.
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
    differentiable_particle_loglikelihood(::Val{LocalLevelCountStockout}, θ, values, standard_normals; resample_every=0, resampling_uniforms=nothing)

`θ` is `[level1, level2, level_variance, zero_inflation, overdispersion,
p12, p22]` (matching `get_loss_function(::Val{LocalLevelCountStockout},
...)`'s parameter order exactly): `level1` the initial/in-stock level,
`level2` the out-of-stock Poisson mean and the level's floor, `p12` =
P(in-stock -> stockout), `p22` = P(stockout -> stockout). `values` are
the observed counts. `standard_normals` (n_particles x length(values)) are
fixed N(0,1) draws for the level's transition noise -- the same
reparameterization trick as the other two models. There is no
`regime_uniforms` input: the in-stock/stockout regime is not sampled at
all, but exactly marginalized per particle (Rao-Blackwellization -- see
the module-level note above for why the sampled version this replaced had
an exactly-zero gradient w.r.t. p12/p22, a real bug, not just a variance
problem). Bootstrap proposal only for the continuous level (there is no
locally-optimal proposal for this model); `resample_every`/
`resampling_uniforms` behave exactly as on the other two models' versions
of this function, reusing `systematic_resample_indices` unchanged --
resampling now carries each particle's (value, belief) pair forward
together.

The leading `::Val{LocalLevelCountStockout}` argument disambiguates this
method from LocalLevel's and LocalLevelChange's own
`differentiable_particle_loglikelihood` methods, which take different
numbers of plain `AbstractMatrix` positional arguments and would otherwise
risk exactly this kind of dispatch collision as this file's argument
shapes evolve (see fit_gradient's docstring/git history for a version of
this function that collided with LocalLevelChange's for exactly this
reason).
"""
function differentiable_particle_loglikelihood(::Val{LocalLevelCountStockout}, θ, values, standard_normals::AbstractMatrix;
                                                resample_every::Int=0,
                                                resampling_uniforms::Union{Nothing,AbstractVector}=nothing)
    level1, level2, level_variance, zero_inflation, overdispersion, p12, p22 = θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7]
    n_particles = size(standard_normals, 1)
    T = length(values)

    RT = promote_type(typeof(level1), typeof(level2), typeof(level_variance), typeof(zero_inflation), typeof(overdispersion), typeof(p12), typeof(p22))
    value = fill(convert(RT, level1), n_particles)
    value_buf = similar(value)

    # Per-particle belief over regime (P(in stock), P(stockout)), exactly
    # marginalized rather than sampled -- see the module-level note. The
    # initial belief (before any observation) is identical across
    # particles: 10 applications of the transition matrix to a point mass
    # on "in stock", matching LocalLevelCountStockoutModel's own
    # level_weights10 = (level_matrix^10)[1, :] used by sample_initial_state.
    b1_0, b2_0 = one(RT), zero(RT)
    for _ in 1:10
        b1_0, b2_0 = b1_0 * (1 - p12) + b2_0 * (1 - p22), b1_0 * p12 + b2_0 * p22
    end
    belief1 = fill(b1_0, n_particles)
    belief2 = fill(b2_0, n_particles)
    belief1_buf = similar(belief1)
    belief2_buf = similar(belief2)

    log_weights = zeros(RT, n_particles)
    total_loglik = zero(RT)
    resample_count = 0

    level_sd = sqrt(level_variance)
    one_minus_od_over_one_minus_zi = (1 - overdispersion) / (1 - zero_inflation)

    for t in 1:T
        y = Int(round(values[t]))
        @inbounds for i in 1:n_particles
            new_value = max(value[i] + level_sd * standard_normals[i, t], level2)

            # Predict step: propagate this particle's own regime belief
            # through the transition matrix (independent of the level).
            predicted_b1 = belief1[i] * (1 - p12) + belief2[i] * (1 - p22)
            predicted_b2 = belief1[i] * p12 + belief2[i] * p22

            # Update step: weight each regime hypothesis by its
            # observation density at this particle's own (just-transitioned)
            # level, in log space to avoid underflow in either tail before
            # combining them (the same log-sum-exp treatment used below for
            # resampling/final normalization).
            lambda = new_value * one_minus_od_over_one_minus_zi
            log_lik_in_stock = log_zigp_pmf(y, lambda, overdispersion, zero_inflation)
            log_lik_stockout = log_poisson_pmf(y, level2)

            log_predicted_b1 = log(predicted_b1)
            log_predicted_b2 = log(predicted_b2)
            joint1 = log_predicted_b1 + log_lik_in_stock
            joint2 = log_predicted_b2 + log_lik_stockout
            m = max(joint1, joint2)
            log_mixture_lik = m + log(exp(joint1 - m) + exp(joint2 - m))

            log_weights[i] += log_mixture_lik
            belief1[i] = exp(joint1 - log_mixture_lik)
            belief2[i] = exp(joint2 - log_mixture_lik)
            value[i] = new_value
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
                belief1_buf[i] = belief1[ancestors[i]]
                belief2_buf[i] = belief2[ancestors[i]]
            end
            value, value_buf = value_buf, value
            belief1, belief1_buf = belief1_buf, belief1
            belief2, belief2_buf = belief2_buf, belief2
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
    n_resamples = resample_every > 0 ? count(t -> t % resample_every == 0, 1:(length(values) - 1)) : 0
    resampling_uniforms = resample_every > 0 ? rand(rng, n_resamples) : nothing
    sigmoid(x) = 1 / (1 + exp(-x))
    return φ -> begin
        level1, level2, level_variance = exp(φ[1]), exp(φ[2]), exp(φ[3])
        zero_inflation, overdispersion, p12, p22 = sigmoid(φ[4]), sigmoid(φ[5]), sigmoid(φ[6]), sigmoid(φ[7])
        -differentiable_particle_loglikelihood(Val{LocalLevelCountStockout}(), [level1, level2, level_variance, zero_inflation, overdispersion, p12, p22], values,
                                                standard_normals;
                                                resample_every=resample_every, resampling_uniforms=resampling_uniforms)
    end
end

"""
    fit_gradient(::Val{LocalLevelCountStockout}, values; particle_count=200, maxiter=200, tol=1e-6, resample_every=0, max_backtracks=20, rng=Random.default_rng())

Gradient-based counterpart to fit(::Val{LocalLevelCountStockout}, ...).
Restarts over `level1` (mean/max of `values`, spanning the plausible
in-stock range) x variance `scale` (1/10), 2x2=4 restarts rather than
LocalLevel's 3x3=9: each restart is markedly more expensive here (7 active
parameters and a per-particle ZIGP evaluation, not a handful of Gaussian
arithmetic ops). This grid is a starting point, not yet stress-tested the
way LocalLevel's was -- LocalLevel's own 3x3 grid over level x
variance-scale exists specifically because a narrower grid reliably found
a real but mediocre local optimum (level_variance collapsing towards 0,
another parameter absorbing the noise instead) that only a grid spanning
the model's actual data range escaped; if the same symptom shows up here
(check level_variance and the reference-nll gap in
test_gradient_fitting_locallevelcountstockout.jl), widen this grid the
same way rather than tuning something else. `level2`, `zero_inflation`,
`overdispersion`, `p12`, `p22` are not gridded, fixed at single reasonable
starting guesses instead, mirroring bboptimize2's own `fit`'s single
incumbent starting point for the same 4 parameters.

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
