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
#     filter!-with-gradients replacement. It supports two orthogonal ways
#     to fight that degeneracy without touching filter!/resample!: the
#     locally optimal proposal (:optimal, exact only because LocalLevel is
#     linear-Gaussian) and periodic differentiable resampling
#     (resample_every > 0, model-agnostic -- the one of the two that would
#     still apply to a model with a discrete state). See the function's
#     docstring for both.
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
    differentiable_particle_loglikelihood(θ, values, standard_normals; proposal=:bootstrap, resample_every=0, resampling_uniforms=nothing)

`θ` is `[level, level_variance, observation_variance]`. `standard_normals` is an
(n_particles x length(values)) matrix of fixed N(0,1) draws -- fixing them,
rather than drawing fresh ones inside this function, is what makes the
result a smooth function of θ (the reparameterization trick) so ForwardDiff
can differentiate through it. Implements sequential importance sampling
with no resampling; see the module-level note above for what that does and
doesn't make valid.

`proposal=:bootstrap` samples each `x_t` from the transition density alone
(`x_{t-1} + N(0, level_variance)`), ignoring `values[t]`, then weights by
the observation density. That's the textbook bootstrap filter, but
proposing without looking at the observation is exactly what makes its
importance weights degenerate quickly as `T` grows -- most of the particles
end up carrying negligible weight, so the effective number of particles
actually informing the likelihood estimate can be far smaller than
`n_particles`, and that shortfall doesn't average out by drawing more
standard normals with the same proposal.

`proposal=:optimal` instead uses the proposal that minimizes importance
weight variance for a fixed observation, which is tractable here because
LocalLevel is linear-Gaussian: it samples `x_t` from
`p(x_t | x_{t-1}, values[t])` (a Gaussian combining the transition and
observation precisions, i.e. one step of a Kalman update) and weights by
`p(values[t] | x_{t-1}) = N(values[t]; x_{t-1}, level_variance +
observation_variance)` -- the same one-step-ahead predictive density
`kalman_loglikelihood` accumulates. This is the standard "optimal
importance function" for SIS (Doucet, Godsill & Andrieu, 2000): still
exact importance sampling for any `θ` (unbiased regardless of how far `θ`
is from the data-generating parameters, not just at the optimum), it just
uses `values[t]` when proposing instead of only when weighting, so weight
variance grows far more slowly with `T`. It requires no resampling and
therefore raises none of resampling's differentiability problems -- but it
only exists because LocalLevel's transition and observation are both
linear-Gaussian; a model with a discrete state (e.g. a stockout regime)
has no such closed form.

# Resampling

`resample_every` (`0` by default, meaning "never") is a second, more
broadly applicable answer to the same degeneracy problem, that doesn't
depend on any of that: periodic resampling, following the same treatment
used to train SMC-based models with gradients in the literature (Maddison
et al. 2017, "Filtering Variational Objectives"; Naesseth et al. 2018,
"Variational Sequential Monte Carlo"; Le et al. 2018, "Auto-Encoding
Sequential Monte Carlo") -- resample ancestors directly from the
normalized weights `W` (via systematic resampling) and reset log-weights
to exactly 0 afterward, with *no* importance-sampling correction:

  - *which* particle survives is decided by comparing `W`'s cumulative sum
    against one pre-drawn offset per resampling event
    (`resampling_uniforms`, the same fixed-randomness trick as
    `standard_normals`) -- that choice itself carries no gradient (there's
    no way around that for a genuinely discrete selection), but the
    *value* carried forward by the surviving particle is a smooth function
    of θ, so gradient information still flows through it via every later
    timestep;
  - resetting log-weights to 0 (rather than a correction like
    `W[ancestor] / q[ancestor]`, tried first and found to perform badly --
    see fit_gradient's history) is what the cited papers do, and it isn't
    just simpler: resampling directly from `W` means that ratio would be
    identically 1 for whichever particle survives, so a correction term
    would only ever inject an artificial, particle-dependent jump in the
    objective's *value* (not just its derivative) at exactly the θ where
    an ancestor switches -- exactly the kind of discontinuity a
    line-search-based optimizer struggles with, and precisely the effect
    observed. Dropping the correction removes that self-inflicted jump.
    The tradeoff is a technically biased gradient of this Monte Carlo
    estimator (a well-known property of this approach), but the
    estimator's *value* stays an unbiased likelihood estimate regardless,
    and this is the standard, empirically-validated choice for gradient-
    based SMC parameter learning, not a shortcut invented for this case.

This works regardless of whether `proposal` is `:bootstrap` or `:optimal`,
and regardless of whether the model's likelihood has any closed form at
all, which is the point: it's the option that would still apply to a model
`:optimal`-style proposals can't reach. `resample_every=0` performs no
resampling and reproduces the exact same arithmetic (down to floating
point) as before this option existed.
"""
function differentiable_particle_loglikelihood(θ, values, standard_normals::AbstractMatrix;
                                                proposal::Symbol=:bootstrap,
                                                resample_every::Int=0,
                                                resampling_uniforms::Union{Nothing,AbstractVector}=nothing)
    if proposal !== :bootstrap && proposal !== :optimal
        throw(ArgumentError("proposal must be :bootstrap or :optimal, got $(repr(proposal))"))
    end

    level, level_variance, observation_variance = θ[1], θ[2], θ[3]
    n_particles = size(standard_normals, 1)
    T = length(values)

    RT = promote_type(typeof(level), typeof(level_variance), typeof(observation_variance))
    x = fill(convert(RT, level), n_particles)
    x_buf = similar(x)
    log_weights = zeros(RT, n_particles)
    total_loglik = zero(RT)
    resample_count = 0

    level_sd = sqrt(level_variance)
    log_norm_const_bootstrap = -0.5 * log(2 * pi * observation_variance)
    marginal_variance = level_variance + observation_variance
    post_variance = level_variance * observation_variance / marginal_variance
    post_sd = sqrt(post_variance)
    log_norm_const_optimal = -0.5 * log(2 * pi * marginal_variance)

    for t in 1:T
        y = values[t]
        if proposal === :bootstrap
            @inbounds for i in 1:n_particles
                x[i] = x[i] + level_sd * standard_normals[i, t]
                log_weights[i] += log_norm_const_bootstrap - 0.5 * (y - x[i])^2 / observation_variance
            end
        else # :optimal
            @inbounds for i in 1:n_particles
                pred_mean = x[i]
                log_weights[i] += log_norm_const_optimal - 0.5 * (y - pred_mean)^2 / marginal_variance
                post_mean = post_variance * (pred_mean / level_variance + y / observation_variance)
                x[i] = post_mean + post_sd * standard_normals[i, t]
            end
        end

        if resample_every > 0 && t < T && t % resample_every == 0
            resample_count += 1
            u0 = resampling_uniforms[resample_count]

            m = maximum(log_weights)
            w_unnorm = exp.(log_weights .- m)
            w_sum = sum(w_unnorm)
            # The marginal likelihood contribution of this block has to be
            # banked now, before the particle set gets replaced by
            # resampling -- see the docstring's "Resampling" section: with
            # log-weights reset to 0 after resampling (no correction term),
            # the next block's own contribution starts fresh rather than
            # picking up a carried-forward correction.
            total_loglik += m + log(w_sum) - log(n_particles)

            W = w_unnorm ./ w_sum
            ancestors = systematic_resample_indices(W, u0)

            # Gather into the preallocated buffer and swap, rather than
            # x = x[ancestors]: that allocates a fresh array every
            # resampling event (5 per call here), and profiling this
            # (via feval/geval counts against wall time -- see
            # fit_gradient's docstring) showed each resampling event
            # costing far more than a 300-element gather should, which
            # points at allocation/GC pressure across the thousands of
            # calls a 9-restart L-BFGS grid makes. fill! avoids the same
            # problem for log_weights, which only ever needs to become
            # all-zero here, not a new array.
            @inbounds for i in 1:n_particles
                x_buf[i] = x[ancestors[i]]
            end
            x, x_buf = x_buf, x
            fill!(log_weights, zero(RT))
        end
    end

    m = maximum(log_weights)
    total_loglik += m + log(sum(exp(lw - m) for lw in log_weights)) - log(n_particles)
    return total_loglik
end

"""
    systematic_resample_indices(q, u0)

Systematic resampling: given a probability vector `q` (summing to 1) and a
single fixed offset `u0 ~ Uniform(0,1)`, returns `length(q)` ancestor
indices via one sorted sweep over `q`'s cumulative sum, rather than
`length(q)` independent draws -- the standard low-variance resampling
scheme (the same one `resample!` in SMC.jl uses; see its docstring),
reimplemented here so it also works when `q` carries ForwardDiff Dual
numbers. Index selection compares `cumsum(q)` against plain `Float64`
positions, which only ever inspects `q`'s primal value -- exactly why the
choice of *which* ancestor gets picked carries no gradient of its own; see
`differentiable_particle_loglikelihood`'s docstring for why that's fine.
"""
function systematic_resample_indices(q::AbstractVector, u0::Real)
    n = length(q)
    cumq = cumsum(q)
    ancestors = Vector{Int}(undef, n)
    j = 1
    for i in 1:n
        position = (i - 1 + u0) / n
        while j < n && cumq[j] < position
            j += 1
        end
        ancestors[i] = j
    end
    return ancestors
end

"""
    get_loss_function_gradient(::Val{LocalLevel}, values; particle_count=200, proposal=:bootstrap, resample_every=0, rng=Random.default_rng())

Gradient-friendly counterpart to get_loss_function(::Val{LocalLevel}, ...):
returns `φ -> -loglik` where `φ` is `[level, log(level_variance),
log(observation_variance)]` (fitting in log-space keeps the variances
positive without box constraints in the optimizer). Draws the particles'
standard normals (and, if `resample_every > 0`, the resampling offsets)
once and closes over them, so repeated calls to the returned function --
as an optimizer makes -- evaluate a fixed, deterministic, differentiable
surface rather than a fresh Monte Carlo draw each time. `proposal` and
`resample_every` are passed through to
`differentiable_particle_loglikelihood` (see its docstring).
"""
function get_loss_function_gradient(::Val{LocalLevel}, values; particle_count=200, proposal::Symbol=:bootstrap, resample_every::Int=0, rng=Random.default_rng())
    standard_normals = randn(rng, particle_count, length(values))
    n_resamples = resample_every > 0 ? count(t -> t % resample_every == 0, 1:(length(values) - 1)) : 0
    resampling_uniforms = resample_every > 0 ? rand(rng, n_resamples) : nothing
    return φ -> begin
        level, level_variance, observation_variance = φ[1], exp(φ[2]), exp(φ[3])
        -differentiable_particle_loglikelihood([level, level_variance, observation_variance], values, standard_normals;
                                                proposal=proposal, resample_every=resample_every,
                                                resampling_uniforms=resampling_uniforms)
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
    lbfgs(g, φ0; maxiter=200, tol=1e-6, memory=10, initial_step=1.0, armijo_c=1e-4, backtrack_factor=0.5, max_backtracks=20, gradient_function=ForwardDiff.gradient)

Limited-memory BFGS with a backtracking (Armijo) line search, using
ForwardDiff for gradients by default. Dependency-free (no Optim.jl) -- Optim's latest
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
Returns (φ_opt, f_opt, iterations_used, n_feval, n_geval): `n_feval`
counts every plain (`Float64`-in, `Float64`-out) call to `g`, and
`n_geval` counts every `ForwardDiff.gradient(g, ...)` call -- each of
those internally evaluates `g` exactly once too, but with a length-3
`Dual` argument (this function's `φ` always has 3 components), which
ForwardDiff computes in a single chunk covering all 3 partials at once,
not 3 separate evaluations. These counts exist to answer, with real
numbers instead of an estimate, "how many evaluations does this actually
make, and how does that compare to bboptimize2's?" -- see fit_gradient's
docstring for how they're used.

`max_backtracks` bounds the Armijo backtracking loop below to at most
that many halvings (in addition to the implicit ~46 from `step > 1e-14`,
whichever binds first) before giving up and accepting whatever candidate
it has -- which the loop falls through to unconditionally already once
backtracking bottoms out, so this only changes how *many* evaluations
that costs, not whether an unsatisfying step can be accepted. This matters
because `g` isn't always smooth: `differentiable_particle_loglikelihood`
with `resample_every > 0` has genuine, if measure-zero in θ-space,
discontinuities where a resampling ancestor selection flips, and a
discontinuous function can fail the Armijo condition unpredictably rather
than smoothly approaching it, which was observed to make backtracking
repeatedly nearly bottom out -- each such iteration costs up to ~46 extra
evaluations of `g` (and its ForwardDiff gradient afterward), which is
exactly what made an earlier resampling experiment take 516s. Capping it
here bounds the worst case for every caller, resampling or not, rather
than working around it only where it was first noticed.

`gradient_function` takes `(g, φ)` and returns `∇g(φ)`, defaulting to
`ForwardDiff.gradient` -- every existing caller (LocalLevel,
LocalLevelChange) keeps using exactly that, unchanged. It exists so a
caller whose loss has many parameters can swap in a reverse-mode AD tool
instead: forward-mode's cost scales with the *number of parameters*
(ForwardDiff differentiates by propagating one Dual partial derivative
slot per parameter through every operation), while reverse-mode's cost is
roughly a constant multiple of a single forward pass regardless of
parameter count -- see DifferentiableLocalLevelCountStockout.jl's
`fit_gradient` docstring for where this mattered enough to use (7
parameters, a per-particle inner loop with real transcendental-function
cost) and which reverse-mode tool was actually compatible with this
repo's Julia 1.8 CI leg.
"""
function lbfgs(g, φ0::AbstractVector{<:Real}; maxiter=200, tol=1e-6, memory=10, initial_step=1.0, armijo_c=1e-4, backtrack_factor=0.5, max_backtracks=20, gradient_function=ForwardDiff.gradient)
    φ = copy(φ0)
    f_val = g(φ)
    n_feval = 1
    grad = gradient_function(g, φ)
    n_geval = 1

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
        n_feval += 1
        # The non-finite check is required, not optional: any IEEE 754
        # comparison against NaN is false, so an overshot candidate that
        # blows up the objective would otherwise read as "Armijo satisfied"
        # and get accepted, poisoning every later iteration with NaN.
        backtracks = 0
        while (!isfinite(f_candidate) || f_candidate > f_val + armijo_c * step * directional_derivative) && step > 1e-14 && backtracks < max_backtracks
            step *= backtrack_factor
            φ_candidate = φ .+ step .* direction
            f_candidate = g(φ_candidate)
            n_feval += 1
            backtracks += 1
        end

        if !isfinite(f_candidate)
            break
        end

        grad_candidate = gradient_function(g, φ_candidate)
        n_geval += 1
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

    return φ, f_val, iterations_used, n_feval, n_geval
end

"""
    fit_gradient(::Val{LocalLevel}, values; particle_count=200, maxiter=200, tol=1e-6, proposal=:bootstrap, resample_every=0, max_backtracks=20, rng=Random.default_rng())

Gradient-based counterpart to fit(::Val{LocalLevel}, ...): uses ForwardDiff
through a resampling-free particle likelihood (see
get_loss_function_gradient) instead of bboptimize2's derivative-free
search, optimizing with L-BFGS (see lbfgs) rather than plain gradient
descent. Returns (fitted LocalLevel, iterations_used, total_n_feval,
total_n_geval) -- the iteration count from whichever restart won is
exposed because one L-BFGS iteration and one bboptimize2 function
evaluation aren't the same unit of work, so wall-clock time and iteration
count both matter for comparing the two; `total_n_feval`/`total_n_geval`
sum lbfgs's own per-restart counts (see its docstring) across all 9
restarts below, to answer "how many evaluations does the whole fit
actually make" with a real number instead of an estimate -- each restart
runs independently and none of them short-circuit the others, so the
total is a straight sum, not something bounded by whichever restart won.

`proposal` and `resample_every` are passed through to
`get_loss_function_gradient`/`differentiable_particle_loglikelihood` (see
their docstrings) and both default to their no-op values (`:bootstrap`,
`0`) so this function's behavior is unchanged from earlier callers unless
explicitly requested. `proposal=:optimal` reduces importance weight
degeneracy using LocalLevel's closed-form optimal proposal; independently,
`resample_every > 0` reduces it via periodic differentiable resampling, a
mechanism that isn't specific to linear-Gaussian models.

`tol` and `max_backtracks` are passed through to `lbfgs`. With
`resample_every > 0`, `tol`'s default (1e-6) can make every restart burn
its full `maxiter` without ever terminating early: near one of
resampling's discontinuities (see `differentiable_particle_loglikelihood`)
the gradient norm can legitimately never settle below a tight tolerance,
since the function isn't smooth there, so `lbfgs` keeps iterating for a
convergence criterion the objective's own shape can't satisfy. A looser
`tol` lets it stop once practically converged instead of paying for
iterations that aren't buying more accuracy.

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
function fit_gradient(::Val{LocalLevel}, values; particle_count=200, maxiter=200, tol::Real=1e-6, proposal::Symbol=:bootstrap, resample_every::Int=0, max_backtracks::Int=20, rng=Random.default_rng())
    loss = get_loss_function_gradient(Val{LocalLevel}(), values; particle_count=particle_count, proposal=proposal,
                                       resample_every=resample_every, rng=rng)

    level_guesses = (minimum(values), sum(values) / length(values), maximum(values))
    base_variance_guess = var(values) / length(values)
    scale_guesses = (0.1, 1.0, 10.0)

    best_φ = nothing
    best_f = Inf
    best_iterations = 0
    total_n_feval = 0
    total_n_geval = 0
    for level0 in level_guesses, scale in scale_guesses
        φ0 = [level0, log(base_variance_guess * scale), log(base_variance_guess * scale)]

        φ_opt, f_opt, iterations_used, n_feval, n_geval = lbfgs(loss, φ0; maxiter=maxiter, tol=tol, max_backtracks=max_backtracks)
        total_n_feval += n_feval
        total_n_geval += n_geval
        if f_opt < best_f
            best_f = f_opt
            best_φ = φ_opt
            best_iterations = iterations_used
        end
    end

    level, level_variance, observation_variance = best_φ[1], exp(best_φ[2]), exp(best_φ[3])
    return LocalLevel(level, level_variance, observation_variance), best_iterations, total_n_feval, total_n_geval
end
