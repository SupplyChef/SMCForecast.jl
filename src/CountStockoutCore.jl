using DecisionTree: Root, Leaf, Node, LeafOrNode, print_tree

# --------------------------------------------------------------------------
# Mutable regression-tree helpers.
#
# DecisionTree.jl's fitted trees are immutable, but LocalLevelCountStockoutExplanatoryML
# needs to refine leaf values with a derivative-free optimizer after fitting the tree
# structure. These wrappers copy a fitted DecisionTree.jl tree into a structurally
# identical, mutable-leaf-value shape so leaf values can be swapped in during that
# optimization without refitting the tree.
# --------------------------------------------------------------------------

mutable struct MutableLeaf
    majority::Float64
end

struct MutableNode
    featid::Int64
    featval::Float64
    left::Union{Nothing, MutableLeaf, MutableNode}
    right::Union{Nothing, MutableLeaf, MutableNode}
end

struct MutableRoot
    node::Union{MutableLeaf, MutableNode}
end

copy_tree1(leaf::Leaf, overrides::IdDict{Leaf{Float64}, Float64})::MutableLeaf = MutableLeaf(get(overrides, leaf, leaf.majority))
function copy_tree1(tree::Root{Float64,Float64}, overrides::IdDict{Leaf{Float64}, Float64})::MutableRoot
    return MutableRoot(copy_tree1(tree.node, overrides))
end
function copy_tree1(tree::Node{Float64,Float64}, overrides::IdDict{Leaf{Float64}, Float64})::MutableNode
    if tree.featid == 0
        return MutableNode(tree.featid, 0.0, copy_tree1(tree.left, overrides), nothing)
    else
        return MutableNode(tree.featid, tree.featval, copy_tree1(tree.left, overrides), copy_tree1(tree.right, overrides))
    end
end

apply_tree1(leaf::MutableLeaf, feature::AbstractVector{Float64})::Float64 = leaf.majority
function apply_tree1(tree::MutableRoot, features::AbstractVector{Float64})::Float64
    apply_tree1(tree.node, features)
end
function apply_tree1(tree::MutableNode, features::AbstractVector{Float64})::Float64
    if tree.featid == 0
        return apply_tree1(tree.left, features)
    elseif features[tree.featid] < tree.featval
        return apply_tree1(tree.left, features)
    else
        return apply_tree1(tree.right, features)
    end
end

get_leaves(leaf::Leaf)=[leaf]
function get_leaves(tree::Root{S,T}) where {S,T}
    get_leaves(tree.node)
end
function get_leaves(tree::Node{S,T}) where {S,T}
    if tree.featid == 0
        return get_leaves(tree.left)
    else
        return vcat(get_leaves(tree.left), get_leaves(tree.right))
    end
end

function print_tree1(root::MutableRoot, featnames=[])
    print_tree1(root.node, featnames)
end
function print_tree1(leaf::MutableLeaf, featnames=[], indent=0)
    return " " ^ (indent - 3) * "$(leaf.majority)\n"
end
function print_tree1(tree::MutableNode, featnames=[], indent=0)
    if tree.featid == 0
        return print_tree1(tree.left, featnames, indent)
    else
        lead1 = " " ^ (indent) * "├─ "
        lead2 = " " ^ (indent) * "└─ "
        return "$(featnames[tree.featid]) < $(tree.featval) ?\n" *
                lead1 * print_tree1(tree.left, featnames, indent+3) *
                lead2 * print_tree1(tree.right, featnames, indent+3)
    end
end

# --------------------------------------------------------------------------
# Zero-inflated generalized-Poisson observation distribution helpers.
# --------------------------------------------------------------------------

function negative_binomial_pmf(r, p, logp, logonep, current_observation::Int64)
    return binomial_coefficient(current_observation + r - 1, current_observation) * exp(logp*r + logonep*current_observation)
end

function binomial_coefficient(n::Float64, k::Int64)::Float64
    if k == 0
        return 1.0
    end
    binomial_coefficient = n
    @inbounds for i in 2:k
        binomial_coefficient *= (n + 1 - i) / i
    end
    return binomial_coefficient
end

# Compute the log of the PMF of the Generalized Poisson Distribution
function log_generalized_poisson_pmf(k::Int, lambda::Float64, theta::Float64)::Float64
    if k == 0
        return -lambda
    elseif k == 1
        log_lambda = log(lambda)
        log_term3 = -(lambda + theta)

        log_pmf = log_lambda + log_term3
        return log_pmf
    else
        log_lambda = log(lambda)
        log_term2 = (k - 1) * log(lambda + k * theta)
        log_term3 = -(lambda + k * theta)

        #TODO: precompute the factorial
        log_k_factorial = logfactorial(k)

        log_pmf = log_lambda + log_term2 + log_term3 - log_k_factorial
        return log_pmf
    end
end

# Compute the PMF of the Zero-Inflated Generalized Poisson Distribution
function zigp_pmf(k::Int, lambda::Float64, theta::Float64, pi::Float64)
    if lambda < 0
        return 0.0
    end
    if k == 0
        p_zero = pi + (1 - pi) * exp(-lambda)
        return p_zero
    else
        p_k = (1 - pi) * exp(log_generalized_poisson_pmf(k, lambda, theta))
        return p_k
    end
end

function sample_zigp(lambda::Float64, theta::Float64, pi::Float64; rng=Random.default_rng())
    if lambda <= 0
        return 0
    end
    u = rand(rng)

    k = 0
    cum_pmf = zigp_pmf(k, lambda, theta, pi)
    while cum_pmf <= u && k <= max(1000, 3*lambda)
        k = k+1
        cum_pmf += zigp_pmf(k, lambda, theta, pi)
    end
    return k
end

# --------------------------------------------------------------------------
# Mean-adjustment components.
#
# LocalLevelCountStockout, LocalLevelCountStockoutExplanatory, and
# LocalLevelCountStockoutExplanatoryML share an identical hidden 2-state
# (in-stock/stockout) Markov chain and zero-inflated generalized-Poisson
# observation model; they differ only in how the latent level is adjusted
# for exogenous effects before/after the random-walk transition. That
# adjustment is factored out here as a small trait hierarchy so the three
# variants can share one SMCSystem implementation (see LocalLevelCountStockoutModel
# below) instead of duplicating it three times.
# --------------------------------------------------------------------------

abstract type MeanAdjustment end

"No exogenous adjustment: the latent level is used as-is."
struct IdentityAdjustment <: MeanAdjustment end

"Multiplicative adjustment: prod_i (coefficients[i] + 1) over active (> 0) regressors."
struct LinearAdjustment <: MeanAdjustment
    exogenous::Matrix{Float64}
    coefficients::Vector{Float64}
end

"Adjustment given by a (possibly leaf-value-overridden) regression tree over the regressors."
struct TreeAdjustment <: MeanAdjustment
    exogenous::Matrix{Float64}
    machine::MutableRoot
end

exogenous_row(::IdentityAdjustment, time::Int) = nothing
exogenous_row(adjustment::LinearAdjustment, time::Int) = @view adjustment.exogenous[:, time]
exogenous_row(adjustment::TreeAdjustment, time::Int) = @view adjustment.exogenous[:, time]

deadjust(::IdentityAdjustment, value::Float64, row)::Float64 = value
readjust(::IdentityAdjustment, value::Float64, row)::Float64 = value

function deadjust(adjustment::LinearAdjustment, value::Float64, row)::Float64
    prod = 1.0
    @inbounds for (i, e) in enumerate(row)
        if e > 0
            prod *= (adjustment.coefficients[i] + 1)
        end
    end
    return value / prod
end

function readjust(adjustment::LinearAdjustment, value::Float64, row)::Float64
    prod = 1.0
    @inbounds for (i, e) in enumerate(row)
        if e > 0
            prod *= (adjustment.coefficients[i] + 1)
        end
    end
    return value * prod
end

deadjust(adjustment::TreeAdjustment, value::Float64, row)::Float64 = value - apply_tree1(adjustment.machine, row)
readjust(adjustment::TreeAdjustment, value::Float64, row)::Float64 = value + apply_tree1(adjustment.machine, row)

# --------------------------------------------------------------------------
# The shared count-stockout system.
# --------------------------------------------------------------------------

"""
    LocalLevelCountStockoutModel{A<:MeanAdjustment}

A local-level, zero-inflated generalized-Poisson count model with a hidden
2-state (in-stock/stockout) Markov chain, parameterized by a `MeanAdjustment`
that describes how the latent level is adjusted for exogenous effects.

`LocalLevelCountStockout`, `LocalLevelCountStockoutExplanatory`, and
`LocalLevelCountStockoutExplanatoryML` are aliases of this type with
`IdentityAdjustment`, `LinearAdjustment`, and `TreeAdjustment` respectively;
use their keyword constructors rather than constructing this type directly.

`initial_state_weights` and `adjust_initial_value` preserve two pre-existing
differences between those three variants' initial-state sampling (the
Explanatory variant historically did not re-derive its initial-state weights
from `level_matrix`, and did not apply the exogenous adjustment to its
initial value): they are not modeled as arbitrary knobs, just as the minimal
surface needed to keep each variant's existing behavior identical after
sharing this implementation.
"""
struct LocalLevelCountStockoutModel{A<:MeanAdjustment} <: SMCSystem{SizedVector{3, Float64, Vector{Float64}}}
    adjustment::A

    level1::Float64
    level2::Float64

    level_matrix::Array{Float64, 2}
    levels::Array{Int64, 1}

    level_variance::Float64

    zero_inflation::Float64
    overdispersion::Float64

    # the members below are used to speed up computation
    level2_exp::Float64

    level_weights::Array{ProbabilityWeights, 1}
    level_weights10::ProbabilityWeights
    level_equal_weights::ProbabilityWeights

    adjust_sampling::Bool

    initial_state_weights::ProbabilityWeights
    adjust_initial_value::Bool

    function LocalLevelCountStockoutModel(adjustment::A;
                                           level1,
                                           level2,
                                           level_matrix,
                                           level_variance,
                                           zero_inflation,
                                           overdispersion,
                                           adjust_sampling::Bool,
                                           initial_state_weights::Union{Nothing, ProbabilityWeights}=nothing,
                                           adjust_initial_value::Bool=false) where {A <: MeanAdjustment}
        levels = [1, 2]
        level_weights = [pweights(level_matrix[i,:]) for i in 1:size(level_matrix, 1)]
        level_weights10 = pweights((level_matrix^10)[1,:])
        level_equal_weights = pweights([0.5, 0.5])

        new{A}(adjustment,
               level1,
               level2,
               level_matrix,
               levels,
               level_variance,
               zero_inflation,
               overdispersion,
               exp(-level2), level_weights, level_weights10, level_equal_weights,
               adjust_sampling,
               isnothing(initial_state_weights) ? level_weights10 : initial_state_weights,
               adjust_initial_value)
    end
end

function sample_initial_state(system::LocalLevelCountStockoutModel, count; rng=Random.default_rng())::Array{SizedVector{3, Float64, Vector{Float64}}, 1}
    initial_value = system.level1
    if system.adjust_initial_value
        row = exogenous_row(system.adjustment, 1)
        initial_value = readjust(system.adjustment, system.level1, row)
    end

    states = sample(rng, [1, 2], system.initial_state_weights, count)
    return [SizedVector{3, Float64, Vector{Float64}}(1.0, initial_value, states[i]) for i in eachindex(states)]
end

function sample_states(system::LocalLevelCountStockoutModel,
                       current_states::Vector{SizedVector{3, Float64, Vector{Float64}}},
                       next_observation::Union{Missing, Float64},
                       new_states, sampling_probabilities; happy_only=false, rng=Random.default_rng())
    time = Int(current_states[1][1])

    current_row = exogenous_row(system.adjustment, time)
    next_row = exogenous_row(system.adjustment, time + 1)

    n = Normal(0, sqrt(system.level_variance))

    for (i, current_state) in enumerate(current_states)
        value = deadjust(system.adjustment, current_state[2], current_row)
        state = Int(current_state[3])

        sampling_probabilities[i] = 1

        @inbounds new_state = (rand(rng) > system.level_matrix[state, 1]) + 1
        if happy_only
            while new_state == 2
                new_state = sample(rng, system.levels, system.level_weights[state])
            end
        else
            if !ismissing(next_observation) && next_observation == 0 && system.adjust_sampling
                new_state = sample(rng, system.levels, system.level_equal_weights)
                sampling_probabilities[i] = system.level_matrix[state, new_state] / 0.5
            end
        end
        ϵ = rand(rng, n)
        new_value = max(value + ϵ, system.level2)

        new_states[i][1] = time + 1
        new_states[i][2] = readjust(system.adjustment, new_value, next_row)
        new_states[i][3] = new_state
    end
end

function sample_observation(system::LocalLevelCountStockoutModel, current_state::SizedVector{3}; rng=Random.default_rng())
    value::Float64 = current_state[2]
    state = Int(current_state[3])

    if state == 2
        return rand(rng, Poisson(system.level2))
    end

    value = value * (1 - system.overdispersion) / (1 - system.zero_inflation)
    return sample_zigp(value, system.overdispersion, system.zero_inflation)
end

function transition_probability(system::LocalLevelCountStockoutModel, state1::SizedVector{3, Float64, Vector{Float64}}, new_observation, state2::SizedVector{3, Float64, Vector{Float64}})::Float64
    time = Int(state1[1])
    current_row = exogenous_row(system.adjustment, time)
    value = deadjust(system.adjustment, state1[2], current_row)
    state = Int(state1[3])

    new_time = Int(state2[1])
    new_row = exogenous_row(system.adjustment, new_time)
    new_value = deadjust(system.adjustment, state2[2], new_row)
    new_state = Int(state2[3])

    n = Normal(0, sqrt(system.level_variance))
    if new_value > system.level2
        p = pdf(n, new_value - value)
    else
        p = cdf(n, new_value - value)
    end
    probability = system.level_matrix[state, new_state] * p

    return probability
end

function observation_probability(system::LocalLevelCountStockoutModel, current_state::SizedVector{3, Float64, Vector{Float64}}, current_observation)::Float64
    value = current_state[2]
    state = Int(current_state[3])

    if state == 2
        if current_observation == 0
            return system.level2_exp
        end
        return pdf(Poisson(system.level2), current_observation)
    end

    value = value * (1 - system.overdispersion) / (1 - system.zero_inflation)
    return zigp_pmf(Int(current_observation), value, system.overdispersion, system.zero_inflation)
end

function average_state(system::LocalLevelCountStockoutModel, states, weights)
    return SizedVector{3, Float64, Vector{Float64}}([states[1][1],
                           sum(states[i][2] * weights[i] for i in eachindex(weights)),
                           sum(states[i][3] * weights[i] for i in eachindex(weights))])
end
