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
    lbfgs_direction(grad, s_history, y_history, rho_history)

The standard L-BFGS two-loop recursion (Nocedal & Wright, Algorithm 7.4):
turns the current gradient and a short history of position/gradient
differences into an approximate Newton descent direction, without ever
forming the (d x d) Hessian approximation explicitly. Returns the search
direction (already negated, i.e. a descent direction, not just -Hg).
"""
function lbfgs_direction(grad, s_history, y_history, rho_history)
    q = copy(grad)
    m = length(s_history)
    alpha = zeros(eltype(grad), m)
    for i in m:-1:1
        alpha[i] = rho_history[i] * dot(s_history[i], q)
        q .-= alpha[i] .* y_history[i]
    end

    gamma = m > 0 ? dot(s_history[m], y_history[m]) / dot(y_history[m], y_history[m]) : one(eltype(grad))
    r = gamma .* q

    for i in 1:m
        beta = rho_history[i] * dot(y_history[i], r)
        r .+= s_history[i] .* (alpha[i] - beta)
    end

    return -r
end

"""
    lbfgs(g, φ0; maxiter=200, tol=1e-6, memory=10, initial_step=1.0, armijo_c=1e-4, backtrack_factor=0.5)

Limited-memory BFGS with a backtracking (Armijo) line search, using
ForwardDiff for gradients. Dependency-free (no Optim.jl) -- Optim's latest
release moved autodiff selection to an ADTypes-based API that isn't
verifiable to resolve compatibly across this repo's Julia 1.8/latest CI
matrix without a local Julia environment (see fit_gradient's history for
why that risk wasn't worth taking), so this reuses the same
ForwardDiff + hand-rolled-line-search approach as the plain gradient
descent it replaces, just with a curvature-aware search direction instead
of steepest descent. Plain gradient descent needs far more iterations to
converge close to an optimum (the gradient norm there shrinks only
linearly step to step); L-BFGS approximates the inverse Hessian from the
last `memory` (position, gradient) changes and gets superlinear
convergence instead, which is the actual "a gradient is cheap, dimension
doesn't matter" argument for preferring gradients over derivative-free
search -- plain steepest descent doesn't realize that argument by itself.
Returns (φ_opt, f_opt, iterations_used).
"""
function lbfgs(g, φ0::AbstractVector{<:Real}; maxiter=200, tol=1e-6, memory=10, initial_step=1.0, armijo_c=1e-4, backtrack_factor=0.5)
    φ = copy(φ0)
    f_val = g(φ)
    grad = ForwardDiff.gradient(g, φ)

    s_history = typeof(φ)[]
    y_history = typeof(φ)[]
    rho_history = eltype(φ)[]

    iterations_used = 0
    for iter in 1:maxiter
        iterations_used = iter

        if !all(isfinite, grad)
            break
        end
        if sqrt(sum(abs2, grad)) < tol
            break
        end

        direction = lbfgs_direction(grad, s_history, y_history, rho_history)
        directional_derivative = dot(grad, direction)
        # The two-loop recursion is only guaranteed to produce a descent
        # direction when the accumulated curvature pairs keep the implicit
        # Hessian approximation positive definite; numerically that can
        # slip (or the history can be empty on iteration 1, giving
        # direction = -grad, which is always fine). Fall back to steepest
        # descent whenever it doesn't.
        if !isfinite(directional_derivative) || directional_derivative >= 0
            direction = -grad
            directional_derivative = -sum(abs2, grad)
        end

        step = initial_step
        φ_candidate = φ .+ step .* direction
        f_candidate = g(φ_candidate)
        # The non-finite check is required, not optional: any IEEE 754
        # comparison against NaN is false, so an overshot candidate that
        # blows up the objective would otherwise read as "Armijo satisfied"
        # and get accepted, poisoning every later iteration with NaN.
        while (!isfinite(f_candidate) || f_candidate > f_val + armijo_c * step * directional_derivative) && step > 1e-14
            step *= backtrack_factor
            φ_candidate = φ .+ step .* direction
            f_candidate = g(φ_candidate)
        end

        if !isfinite(f_candidate)
            break
        end

        grad_candidate = ForwardDiff.gradient(g, φ_candidate)
        s = φ_candidate .- φ
        y = grad_candidate .- grad
        sy = dot(s, y)
        # Skip the curvature update (rather than push a degenerate pair)
        # when the curvature condition sy > 0 fails to hold with enough
        # margin -- pushing it anyway can make later two-loop recursions
        # produce an ascent direction.
        if isfinite(sy) && sy > 1e-10
            push!(s_history, s)
            push!(y_history, y)
            push!(rho_history, 1 / sy)
            if length(s_history) > memory
                popfirst!(s_history)
                popfirst!(y_history)
                popfirst!(rho_history)
            end
        end

        φ = φ_candidate
        f_val = f_candidate
        grad = grad_candidate
    end

    return φ, f_val, iterations_used
end

"""
    fit_gradient(::Val{LocalLevel}, values; particle_count=200, maxiter=200, rng=Random.default_rng())

Gradient-based counterpart to fit(::Val{LocalLevel}, ...): uses ForwardDiff
through a resampling-free particle likelihood (see
get_loss_function_gradient) instead of bboptimize2's derivative-free
search, optimizing with L-BFGS (see lbfgs) rather than plain gradient
descent. Returns (fitted LocalLevel, iterations_used) -- the iteration
count from whichever restart won is exposed because one L-BFGS iteration
and one bboptimize2 function evaluation aren't the same unit of work, so
wall-clock time and iteration count both matter for comparing the two.

Unlike bboptimize2, which explores many candidates at once via its
population, a single L-BFGS run has no global search of its own -- it
just follows the local (curvature-corrected) gradient from wherever it
starts, and converges to whichever stationary point is reachable from
there, good or bad. This is restarted from a small grid of initial
guesses to compensate, but the grid has to actually cover the same
ground bboptimize2's population does to help: an earlier version of this
function only varied the initial *variance* guess across restarts,
always starting `level` at `values[1]` -- every restart shared the same
basin of attraction in the level direction, and L-BFGS reliably
converged to a real but mediocre local optimum (level_variance collapsing
towards 0, observation_variance absorbing the noise instead) that
bboptimize2's search over the full `(minimum(values), maximum(values))`
level range does not get stuck in. Restarting over a level x variance-scale
grid (spanning the same level range bboptimize2 searches) is what
actually lets L-BFGS reach a comparable optimum, not the choice of
optimizer -- L-BFGS was already finding that same mediocre point in far
fewer iterations than plain gradient descent, just as reliably.
"""
function fit_gradient(::Val{LocalLevel}, values; particle_count=200, maxiter=200, rng=Random.default_rng())
    loss = get_loss_function_gradient(Val{LocalLevel}(), values; particle_count=particle_count, rng=rng)

    level_guesses = (minimum(values), sum(values) / length(values), maximum(values))
    base_variance_guess = var(values) / length(values)
    scale_guesses = (0.1, 1.0, 10.0)

    best_φ = nothing
    best_f = Inf
    best_iterations = 0
    for level0 in level_guesses, scale in scale_guesses
        φ0 = [level0, log(base_variance_guess * scale), log(base_variance_guess * scale)]

        φ_opt, f_opt, iterations_used = lbfgs(loss, φ0; maxiter=maxiter)
        if f_opt < best_f
            best_f = f_opt
            best_φ = φ_opt
            best_iterations = iterations_used
        end
    end

    level, level_variance, observation_variance = best_φ[1], exp(best_φ[2]), exp(best_φ[3])
    return LocalLevel(level, level_variance, observation_variance), best_iterations
end
