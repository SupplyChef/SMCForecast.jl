# --------------------------------------------------------------------------
# Differentiable (gradient-based) fitting for LocalLevel, kept alongside the
# existing derivative-free fit(::Val{LocalLevel}, ...)/bboptimize2 path so
# both can be compared for speed and accuracy on the same data.
#
# LocalLevel is the only model in this package with no discrete hidden state
# (no in-stock/stockout regime, no zero-inflation mixture indicator), so it's
# the one place a gradient-based fit doesn't first have to solve the harder
# problem of differentiating through a discrete branch. It still shares the
# other obstacle every model here has: filter!'s resampling step is a hard,
# non-differentiable categorical choice. Two pieces work around that for
# LocalLevel specifically, without touching filter!/resample! at all:
#
#   - kalman_loglikelihood: LocalLevel is an exact linear-Gaussian state
#     space model (a random walk observed with Gaussian noise), so its
#     marginal log-likelihood has a closed form via the Kalman filter --
#     no particles, no resampling, no Monte Carlo noise. This is a
#     ground-truth oracle for tests, not a fitting method itself.
#
#   - differentiable_particle_loglikelihood: a particle log-likelihood
#     using the same transition and observation densities as LocalLevel's
#     sample_states/observation_probability, but with resampling dropped
#     and the particles' underlying standard-normal draws fixed once per
#     fit (the reparameterization trick) so the result is a smooth,
#     ForwardDiff-differentiable function of the parameters. Skipping
#     resampling means this degrades for long series exactly the way
#     un-resampled SIS always does (weight variance grows with T) -- it's
#     a fitting-time surrogate for short-to-moderate series, not a
#     filter!-with-gradients replacement.
# --------------------------------------------------------------------------

"""
    kalman_loglikelihood(level, level_variance, observation_variance, values)

Exact marginal log-likelihood of `values` under LocalLevel's model
(x_t = x_{t-1} + N(0, level_variance), y_t = x_t + N(0, observation_variance),
x_0 == level), computed via the Kalman filter -- no particles, no
randomness. Used in tests as a ground-truth check for both fitting methods.
"""
function kalman_loglikelihood(level, level_variance, observation_variance, values)
    # LocalLevel's own prior is Uniform(level-0.0001, level+0.0001); its
    # variance is (0.0002)^2/12, negligible but included for fidelity to
    # the particle model this is meant to check against.
    a = level
    P = (0.0002)^2 / 12
    loglik = zero(promote_type(typeof(level), typeof(level_variance), typeof(observation_variance), eltype(values)))
    for y in values
        F = P + observation_variance
        v = y - a
        loglik += -0.5 * (log(2 * pi) + log(F) + v^2 / F)
        K = P / F
        a = a + K * v
        P = P * (1 - K) + level_variance
    end
    return loglik
end

"""
    differentiable_particle_loglikelihood(θ, values, standard_normals)

`θ` is `[level, level_variance, observation_variance]`. `standard_normals` is an
(n_particles x length(values)) matrix of fixed N(0,1) draws -- fixing them,
rather than drawing fresh ones inside this function, is what makes the
result a smooth function of θ (the reparameterization trick) so ForwardDiff
can differentiate through it. Implements sequential importance sampling
with the bootstrap (transition) proposal and no resampling; see the
module-level note above for what that does and doesn't make valid.
"""
function differentiable_particle_loglikelihood(θ, values, standard_normals::AbstractMatrix)
    level, level_variance, observation_variance = θ[1], θ[2], θ[3]
    n_particles = size(standard_normals, 1)
    T = length(values)

    RT = promote_type(typeof(level), typeof(level_variance), typeof(observation_variance))
    level_sd = sqrt(level_variance)
    log_norm_const = -0.5 * log(2 * pi * observation_variance)

    x = fill(convert(RT, level), n_particles)
    log_weights = zeros(RT, n_particles)

    for t in 1:T
        @inbounds for i in 1:n_particles
            x[i] = x[i] + level_sd * standard_normals[i, t]
            log_weights[i] += log_norm_const - 0.5 * (values[t] - x[i])^2 / observation_variance
        end
    end

    m = maximum(log_weights)
    return m + log(sum(exp(lw - m) for lw in log_weights)) - log(n_particles)
end

"""
    get_loss_function_gradient(::Val{LocalLevel}, values; particle_count=200, rng=Random.default_rng())

Gradient-friendly counterpart to get_loss_function(::Val{LocalLevel}, ...):
returns `φ -> -loglik` where `φ` is `[level, log(level_variance),
log(observation_variance)]` (fitting in log-space keeps the variances
positive without box constraints in the optimizer). Draws the particles'
standard normals once and closes over them, so repeated calls to the
returned function -- as an optimizer makes -- evaluate a fixed,
deterministic, differentiable surface rather than a fresh Monte Carlo draw
each time.
"""
function get_loss_function_gradient(::Val{LocalLevel}, values; particle_count=200, rng=Random.default_rng())
    standard_normals = randn(rng, particle_count, length(values))
    return φ -> begin
        level, level_variance, observation_variance = φ[1], exp(φ[2]), exp(φ[3])
        -differentiable_particle_loglikelihood([level, level_variance, observation_variance], values, standard_normals)
    end
end

"""
    gradient_descent(g, φ0; maxiter=500, tol=1e-6, initial_step=1.0, armijo_c=1e-4, backtrack_factor=0.5)

Minimal backtracking-line-search gradient descent using ForwardDiff for the
gradient. Deliberately simple and dependency-free (no Optim.jl) -- the point
here is comparing against derivative-free search, not fielding a tuned
L-BFGS. Plain (non-Newton) gradient descent converges slowly very close to
an optimum (the gradient norm shrinks only linearly step to step there), so
`tol` is a practically-tight-enough stopping point rather than machine
precision -- a much stricter tol mostly buys extra iterations circling the
optimum, not a meaningfully better fit. Returns (φ_opt, f_opt, iterations_used).
"""
function gradient_descent(g, φ0::AbstractVector{<:Real}; maxiter=500, tol=1e-6, initial_step=1.0, armijo_c=1e-4, backtrack_factor=0.5)
    φ = copy(φ0)
    f_val = g(φ)
    iterations_used = 0

    for iter in 1:maxiter
        iterations_used = iter
        grad = ForwardDiff.gradient(g, φ)
        if !all(isfinite, grad)
            break
        end
        grad_norm_sq = sum(abs2, grad)
        if sqrt(grad_norm_sq) < tol
            break
        end

        step = initial_step
        φ_candidate = φ .- step .* grad
        f_candidate = g(φ_candidate)
        # `f_candidate > ...` is false whenever f_candidate is NaN (any IEEE 754
        # comparison against NaN is false), so an overshoot that blows up the
        # objective (e.g. exp(φ) overflowing) would otherwise read as "Armijo
        # satisfied" and get accepted, permanently poisoning φ with NaN for
        # every later iteration. Reject non-finite candidates explicitly.
        while (!isfinite(f_candidate) || f_candidate > f_val - armijo_c * step * grad_norm_sq) && step > 1e-14
            step *= backtrack_factor
            φ_candidate = φ .- step .* grad
            f_candidate = g(φ_candidate)
        end

        # Backtracking exhausted the step all the way to the floor without
        # finding a finite, improving point: no further progress is possible
        # along this direction, so stop rather than accept a broken step.
        if !isfinite(f_candidate)
            break
        end

        φ = φ_candidate
        f_val = f_candidate
    end

    return φ, f_val, iterations_used
end

"""
    fit_gradient(::Val{LocalLevel}, values; particle_count=200, maxiter=500, rng=Random.default_rng())

Gradient-based counterpart to fit(::Val{LocalLevel}, ...): uses ForwardDiff
through a resampling-free particle likelihood (see
get_loss_function_gradient) instead of bboptimize2's derivative-free
search. Returns (fitted LocalLevel, iterations_used) -- the iteration count
is exposed because one gradient-descent iteration and one bboptimize2
function evaluation aren't the same unit of work, so wall-clock time and
iteration count both matter for comparing the two.
"""
function fit_gradient(::Val{LocalLevel}, values; particle_count=200, maxiter=500, rng=Random.default_rng())
    loss = get_loss_function_gradient(Val{LocalLevel}(), values; particle_count=particle_count, rng=rng)

    initial_variance_guess = var(values) / length(values)
    φ0 = [values[1], log(initial_variance_guess), log(initial_variance_guess)]

    φ_opt, _, iterations_used = gradient_descent(loss, φ0; maxiter=maxiter)

    level, level_variance, observation_variance = φ_opt[1], exp(φ_opt[2]), exp(φ_opt[3])
    return LocalLevel(level, level_variance, observation_variance), iterations_used
end
