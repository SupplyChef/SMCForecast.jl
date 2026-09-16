# --------------------------------------------------------------------------
# Differentiable (gradient-based) fitting for LocalLevelChange, following the
# exact same pattern validated for LocalLevel in DifferentiableLocalLevel.jl:
# a closed-form Kalman oracle for tests, a resampling-free-by-default
# particle log-likelihood differentiable via ForwardDiff, and periodic
# differentiable resampling (resample directly from the normalized weights,
# reset log-weights to 0 -- see DifferentiableLocalLevel.jl's docstring for
# why, and why an importance-correction "soft resampling" variant was tried
# first and rejected) as the model-agnostic fix for the bootstrap proposal's
# weight degeneracy. lbfgs and systematic_resample_indices are reused as-is
# from DifferentiableLocalLevel.jl -- neither is LocalLevel-specific.
#
# LocalLevelChange is a second, independent linear-Gaussian model (a local
# linear trend: a level with its own random-walk "change"/slope component),
# not just a re-tuning of the LocalLevel fix -- it exists to check that the
# resampling mechanism actually generalizes rather than having been shaped
# to fit one model's particular likelihood surface. Unlike LocalLevel, no
# locally-optimal proposal is implemented here: the point of this file is
# specifically to validate the generic (proposal-agnostic) resampling fix,
# not to re-derive a Gaussian-specific shortcut for a second model.
# --------------------------------------------------------------------------

"""
    kalman_loglikelihood(level, change, level_variance, change_variance, observation_variance, values)

Exact marginal log-likelihood of `values` under LocalLevelChange's model
(a local linear trend: level_t = level_{t-1} + change_{t-1} + N(0,
level_variance), change_t = change_{t-1} + N(0, change_variance), y_t =
level_t + N(0, observation_variance)), computed via the Kalman filter for
a 2-dimensional state -- no particles, no randomness. Used in tests as a
ground-truth check for both fitting methods, the same role
`kalman_loglikelihood(level, level_variance, observation_variance,
values)` plays for LocalLevel (this is a separate method of that same
name, dispatched on argument count).

The covariance recursion is hand-unrolled into named scalars (P11, P12,
P22 for the symmetric 2x2 state covariance) rather than using matrix
types, mirroring LocalLevel's own hand-scalar Kalman filter -- this keeps
it simple to differentiate (if ever needed) and avoids any ambiguity in
how StaticArrays/LinearAlgebra matrix operations interact with Dual
numbers.
"""
function kalman_loglikelihood(level, change, level_variance, change_variance, observation_variance, values)
    a1, a2 = level, change
    # LocalLevelChange's own prior is a near-point-mass Uniform on level
    # (variance (0.0002)^2/12, matching LocalLevel's convention) and an
    # exact point mass on change (sample_initial_state always sets it to
    # system.change with no randomness), hence P22 starts at exactly 0.
    P11 = (0.0002)^2 / 12
    P12 = zero(P11)
    P22 = zero(P11)

    RT = promote_type(typeof(level), typeof(change), typeof(level_variance), typeof(change_variance), typeof(observation_variance), eltype(values))
    loglik = zero(RT)
    for y in values
        a1p = a1 + a2
        a2p = a2
        P11p = P11 + 2 * P12 + P22 + level_variance
        P12p = P12 + P22
        P22p = P22 + change_variance

        v = y - a1p
        Fv = P11p + observation_variance
        loglik += -0.5 * (log(2 * pi) + log(Fv) + v^2 / Fv)

        K1 = P11p / Fv
        K2 = P12p / Fv
        a1 = a1p + K1 * v
        a2 = a2p + K2 * v
        P11 = P11p * (1 - K1)
        P12 = P12p * (1 - K1)
        P22 = P22p - K2 * P12p
    end
    return loglik
end

"""
    differentiable_particle_loglikelihood(θ, values, standard_normals_level, standard_normals_change; resample_every=0, resampling_uniforms=nothing)

`θ` is `[level, change, level_variance, change_variance,
observation_variance]`. `standard_normals_level`/`standard_normals_change`
are (n_particles x length(values)) matrices of fixed N(0,1) draws for the
level and change transition noise respectively -- fixing them is the same
reparameterization trick `differentiable_particle_loglikelihood` for
LocalLevel uses, and for the same reason (a smooth function of θ that
ForwardDiff can differentiate through). Bootstrap proposal only (see the
module-level note above for why no `:optimal` proposal is implemented
here); `resample_every`/`resampling_uniforms` behave exactly as documented
on LocalLevel's version of this function, including reusing
`systematic_resample_indices` unchanged -- it only ever operates on a
weight vector and a scalar offset, nothing LocalLevel-specific.
"""
function differentiable_particle_loglikelihood(θ, values, standard_normals_level::AbstractMatrix, standard_normals_change::AbstractMatrix;
                                                resample_every::Int=0,
                                                resampling_uniforms::Union{Nothing,AbstractVector}=nothing)
    level0, change0, level_variance, change_variance, observation_variance = θ[1], θ[2], θ[3], θ[4], θ[5]
    n_particles = size(standard_normals_level, 1)
    T = length(values)

    RT = promote_type(typeof(level0), typeof(change0), typeof(level_variance), typeof(change_variance), typeof(observation_variance))
    level = fill(convert(RT, level0), n_particles)
    change = fill(convert(RT, change0), n_particles)
    level_buf = similar(level)
    change_buf = similar(change)
    log_weights = zeros(RT, n_particles)
    total_loglik = zero(RT)
    resample_count = 0

    level_sd = sqrt(level_variance)
    change_sd = sqrt(change_variance)
    log_norm_const = -0.5 * log(2 * pi * observation_variance)

    for t in 1:T
        y = values[t]
        @inbounds for i in 1:n_particles
            level[i] = level[i] + change[i] + level_sd * standard_normals_level[i, t]
            change[i] = change[i] + change_sd * standard_normals_change[i, t]
            log_weights[i] += log_norm_const - 0.5 * (y - level[i])^2 / observation_variance
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
                level_buf[i] = level[ancestors[i]]
                change_buf[i] = change[ancestors[i]]
            end
            level, level_buf = level_buf, level
            change, change_buf = change_buf, change
            fill!(log_weights, zero(RT))
        end
    end

    m = maximum(log_weights)
    total_loglik += m + log(sum(exp(lw - m) for lw in log_weights)) - log(n_particles)
    return total_loglik
end

"""
    get_loss_function_gradient(::Val{LocalLevelChange}, values; particle_count=200, resample_every=0, rng=Random.default_rng())

Gradient-friendly counterpart to get_loss_function(::Val{LocalLevelChange}, ...):
returns `φ -> -loglik` where `φ` is `[level, change,
log(level_variance), log(change_variance), log(observation_variance)]`.
Same role as LocalLevel's version of this function; see its docstring.
"""
function get_loss_function_gradient(::Val{LocalLevelChange}, values; particle_count=200, resample_every::Int=0, rng=Random.default_rng())
    standard_normals_level = randn(rng, particle_count, length(values))
    standard_normals_change = randn(rng, particle_count, length(values))
    n_resamples = resample_every > 0 ? count(t -> t % resample_every == 0, 1:(length(values) - 1)) : 0
    resampling_uniforms = resample_every > 0 ? rand(rng, n_resamples) : nothing
    return φ -> begin
        level0, change0 = φ[1], φ[2]
        level_variance, change_variance, observation_variance = exp(φ[3]), exp(φ[4]), exp(φ[5])
        -differentiable_particle_loglikelihood([level0, change0, level_variance, change_variance, observation_variance], values,
                                                standard_normals_level, standard_normals_change;
                                                resample_every=resample_every, resampling_uniforms=resampling_uniforms)
    end
end

"""
    fit_gradient(::Val{LocalLevelChange}, values; particle_count=200, maxiter=200, tol=1e-6, resample_every=0, max_backtracks=20, rng=Random.default_rng())

Gradient-based counterpart to fit(::Val{LocalLevelChange}, ...); same role
and same restart-grid rationale as LocalLevel's fit_gradient (see its
docstring for why a single L-BFGS run needs restarting at all). Grids over
`level` (min/mean/max of `values`, spanning bboptimize2's own search
range) and a variance `scale` (0.1/1/10) applied to all three variances at
once, the same 3x3=9 restarts LocalLevel uses; `change` is not gridded
(fixed at the mean successive difference of `values`) to avoid a
combinatorial blow-up in restart count, since gridding every one of 5
parameters independently was never needed to make LocalLevel's fit work.
`change_variance`'s base guess is scaled down by 1000x from
`level`/`observation`'s, mirroring bboptimize2's own `fit`'s SearchRange
for it (LocalLevelChange's constructor also enforces change_variance <=
level_variance).

Returns (fitted LocalLevelChange, iterations_used, total_n_feval,
total_n_geval) -- see LocalLevel's fit_gradient docstring for what these
mean.
"""
function fit_gradient(::Val{LocalLevelChange}, values; particle_count=200, maxiter=200, tol::Real=1e-6, resample_every::Int=0, max_backtracks::Int=20, rng=Random.default_rng())
    loss = get_loss_function_gradient(Val{LocalLevelChange}(), values; particle_count=particle_count, resample_every=resample_every, rng=rng)

    level_guesses = (minimum(values), sum(values) / length(values), maximum(values))
    change0_guess = (values[end] - values[1]) / (length(values) - 1)
    base_variance_guess = var(values) / length(values)
    scale_guesses = (0.1, 1.0, 10.0)

    best_φ = nothing
    best_f = Inf
    best_iterations = 0
    total_n_feval = 0
    total_n_geval = 0
    for level0 in level_guesses, scale in scale_guesses
        φ0 = [level0, change0_guess,
              log(base_variance_guess * scale), log(base_variance_guess * scale / 1000), log(base_variance_guess * scale)]

        φ_opt, f_opt, iterations_used, n_feval, n_geval = lbfgs(loss, φ0; maxiter=maxiter, tol=tol, max_backtracks=max_backtracks)
        total_n_feval += n_feval
        total_n_geval += n_geval
        if f_opt < best_f
            best_f = f_opt
            best_φ = φ_opt
            best_iterations = iterations_used
        end
    end

    level, change = best_φ[1], best_φ[2]
    level_variance, change_variance, observation_variance = exp(best_φ[3]), exp(best_φ[4]), exp(best_φ[5])
    return LocalLevelChange(level, change, level_variance, change_variance, observation_variance), best_iterations, total_n_feval, total_n_geval
end
