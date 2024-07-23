using DecisionTree: Root, Leaf, Node, LeafOrNode#, apply_tree

#import DecisionTree: apply_tree

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

copy_tree2(leaf::Leaf, overrides::IdDict{Leaf{Float64}, Float64})::MutableLeaf = MutableLeaf(get(overrides, leaf, leaf.majority))
function copy_tree2(tree::Root{Float64,Float64}, overrides::IdDict{Leaf{Float64}, Float64})::MutableRoot
    return MutableRoot(copy_tree2(tree.node, overrides))
end
function copy_tree2(tree::Node{Float64,Float64}, overrides::IdDict{Leaf{Float64}, Float64})::MutableNode
    if tree.featid == 0
        return MutableNode(tree.featid, 0.0, copy_tree2(tree.left, overrides), nothing)
    else
        return MutableNode(tree.featid, tree.featval, copy_tree2(tree.left, overrides), copy_tree2(tree.right, overrides))
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

apply_tree2(leaf::Leaf, feature::AbstractVector{Float64}, overrides::IdDict{Leaf, Float64})::Float64 = get(overrides, leaf, leaf.majority)
function apply_tree2(tree::Root{Float64,Float64}, features::AbstractVector{Float64}, overrides::IdDict{Leaf, Float64})::Float64
    apply_tree2(tree.node, features, overrides)
end
function apply_tree2(tree::Node{Float64,Float64}, features::AbstractVector{Float64}, overrides::IdDict{Leaf, Float64})::Float64
    if tree.featid == 0
        return apply_tree2(tree.left, features, overrides)
    elseif features[tree.featid] < tree.featval
        return apply_tree2(tree.left, features, overrides)
    else
        return apply_tree2(tree.right, features, overrides)
    end
end
        
# function apply_tree(tree::Root{S,T}, features::AbstractMatrix{S}, overrides::Dict{Leaf, Float64}) where {S,T}
#     apply_tree(tree.node, features, overrides)
# end
# function apply_tree(tree::LeafOrNode{S,T}, features::AbstractMatrix{S}, overrides::Dict{Leaf, Float64}) where {S,T}
#     N = size(features, 1)
#     predictions = Array{T}(undef, N)
#     for i in 1:N
#         predictions[i] = apply_tree(tree, features[i, :], overrides)
#     end
#     return predictions
# end

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

struct LocalLevelCountJumpExplanatoryML <: SMCSystem{SizedVector{3, Float64, Vector{Float64}}}
    exogenous::Matrix{Float64}

    level1::Float64
    level2::Float64
    level_matrix::Array{Float64, 2}
    levels::Array{Int64, 1}
    
    machine::MutableRoot

    level_variance::Float64
    
    zero_inflation::Float64
    overdispersion::Float64

    level2_exp::Float64

    level_weights::Array{ProbabilityWeights, 1}
    level_weights10::ProbabilityWeights
    level_equal_weights::ProbabilityWeights

    adjust_sampling::Bool

    function LocalLevelCountJumpExplanatoryML(;exogenous, level1, level2, level_matrix, machine, level_variance, zero_inflation, overdispersion, adjust_sampling=false)
        levels = [1, 2]
        level_weights = [pweights(level_matrix[i,:]) for i in 1:size(level_matrix, 1)]
        level_weights10 = pweights((level_matrix^10)[1,:])
        level_equal_weights = pweights([0.5, 0.5])

        new(exogenous, 
            level1, 
            level2, 
            level_matrix, 
            levels, 
            machine, 
            level_variance, 
            zero_inflation,
            overdispersion,
            exp(-level2), level_weights, level_weights10, level_equal_weights, 
            adjust_sampling)
    end
end

function forecast(::Val{LocalLevelCountJumpExplanatoryML}, exogenous, values, horizon; maxtime=10.0, size=500, forecast_percentiles=0.5)
    fcs = fit(Val{LocalLevelCountJumpExplanatoryML}(), exogenous, values; maxtime=maxtime, size=size)
    smc = SMC{SizedVector{3, Float64}, LocalLevelCountJumpExplanatoryML}(fcs, 1_000)
    filter!(smc, values; record=false)
    obs, weights = predict_observations(smc, horizon)
    if isa(forecast_percentiles, Real)
        return percentiles(forecast_percentiles, obs, weights)
    else
        return [percentiles(p, obs, weights) for p in forecast_percentiles]
    end
end

function fit(::Val{LocalLevelCountJumpExplanatoryML}, exogenous, values; maxtime=10, regularization=0.0, size=100,
                                                                        min_stay_outofstock_probability=0.0001, 
                                                                        rng=Random.default_rng())

    DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
    model = DecisionTreeRegressor(max_depth=5, min_samples_split=30)
    mach = MLJ.machine(model, table(exogenous[:, 1:length(values)]'), values) |> MLJ.fit!

    leaves = get_leaves(mach.fitresult[1])

    loss_function = get_loss_function(Val{LocalLevelCountJumpExplanatoryML}(), exogenous, values, mach; regularization=regularization, size=size)
    
    dim = 7

    xs = bboptimize2(loss_function,
                       vcat([values[1], 
                        0.00001, 
                        max((var(values) - (length(values) * mean(values))) / length(values),  0.001), 
                        0.0, 
                        0.0, 
                        0.1, 
                        max(0.00001, 0.9)],
                        [leaf.majority for leaf in leaves]
                       ),
                        Dict(:SearchRange => vcat([(0, maximum(values)), 
                                                (0.00001, mean(values) / 5), 
                                                (0.00001, var(values) / length(values)),
                                                (0.00001, .9999),
                                                (0.00001, .9999), 
                                                (0.00001, .9999), 
                                                (min_stay_outofstock_probability, .9999)],
                                                [(leaf.majority - 1, leaf.majority + 1) for leaf in leaves]),
                        :NumDimensions => dim, 
                        :MaxTime => maxtime),
                        rng=rng
                    )

    machine = copy_tree2(mach.fitresult[1], IdDict(l => xs[7 + i] for (i, l) in enumerate(leaves)))
    fcs2 = LocalLevelCountJumpExplanatoryML(;exogenous=exogenous, 
                                            machine = machine,
                                            level1=xs[1], 
                                            level2=xs[2],
                                            level_variance=abs(xs[3]), 
                                            zero_inflation=abs(xs[4]),
                                            overdispersion=abs(xs[5]),
                                            level_matrix=[1-xs[6] xs[6]; 
                                                        1-xs[7] xs[7]])
    return fcs2
end

function get_loss_function(::Val{LocalLevelCountJumpExplanatoryML}, exogenous, values, mach; regularization=0.0, size=1000)
    leaves = get_leaves(mach.fitresult[1])
    machine = copy_tree2(mach.fitresult[1], IdDict(l => values[7 + i] for (i, l) in enumerate(leaves)))
    
    return xs -> begin
        fcs2 = LocalLevelCountJumpExplanatoryML(;exogenous=exogenous, 
                                                machine = machine,
                                                level1=xs[1], 
                                                level2=xs[2],
                                                level_variance=abs(xs[3]), 
                                                zero_inflation=abs(xs[4]),
                                                overdispersion=abs(xs[5]),
                                                level_matrix=[1-xs[6] xs[6]; 
                                                            1-xs[7] xs[7]])
        smc = SMC{SizedVector{3, Float64, Vector{Float64}}, LocalLevelCountJumpExplanatoryML}(fcs2, size)
        rng = MersenneTwister(1)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false, rng=rng)
        return -likelihood 
    end
end

function sample_initial_state(system::LocalLevelCountJumpExplanatoryML, count; rng=Random.default_rng())::Array{SizedVector{3, Float64, Vector{Float64}}, 1}
    states = sample(rng, [1, 2], pweights([0.9, 0.1]), count)
    return [SizedVector{3, Float64, Vector{Float64}}(1.0, system.level1, states[i]) for i in eachindex(states)]
end

function sample_states(system::LocalLevelCountJumpExplanatoryML, 
                      current_states::Vector{SizedVector{3, Float64, Vector{Float64}}},
                      next_observation::Union{Missing, Float64}, 
                      new_states::Vector{SizedVector{3, Float64, Vector{Float64}}}, 
                      sampling_probabilities::Vector{Float64}; 
                      happy_only=false, rng=Random.default_rng())
    time = Int(current_states[1][1])

    @views exogenous_time::AbstractVector{Float64} = system.exogenous[:, time]
    @views exogenous_time_plus_1::AbstractVector{Float64} = system.exogenous[:, time + 1]
    result::MutableRoot = system.machine
    for (i, current_state) in enumerate(current_states)
        value = de_exogenous_ml(current_state, result, exogenous_time)
        state = Int(current_state[3])
        sampling_probabilities[i] = 1

        n = Normal(0, sqrt(system.level_variance))

        new_state = sample(rng, system.levels, system.level_weights[state])
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
        new_states[i][2] = re_exogenous_ml(new_value, result, exogenous_time_plus_1)
        new_states[i][3] = new_state
    end
end

function sample_observation(system::LocalLevelCountJumpExplanatoryML, current_state::SizedVector{3}; rng=Random.default_rng())
    value::Float64 = current_state[2]
    state = Int(current_state[3])

    if state == 2
        return rand(rng, Poisson(system.level2))
    end

    value = value * (1 - system.overdispersion) / (1 - system.zero_inflation)
    return sample_zigp(value, system.overdispersion, system.zero_inflation)
end

function transition_probability(system::LocalLevelCountJumpExplanatoryML, 
                                state1::SizedVector{3, Float64, Vector{Float64}}, 
                                new_observation, 
                                state2::SizedVector{3, Float64, Vector{Float64}})::Float64
    result::MutableRoot = system.machine
    
    time = Int(state1[1])
    @views exogenous_time::AbstractVector{Float64} = system.exogenous[:, time]
    value = de_exogenous_ml(state1, result, exogenous_time)
    state = Int(state1[3])

    new_time = Int(state2[1])
    @views exogenous_newtime::AbstractVector{Float64} = system.exogenous[:, new_time]
    new_value = de_exogenous_ml(state2, result, exogenous_newtime)
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

function observation_probability(system::LocalLevelCountJumpExplanatoryML, current_state::SizedVector{3, Float64, Vector{Float64}}, current_observation)::Float64
    time = current_state[1]
    value = current_state[2]
    state = current_state[3]

    if state == 2
        if current_observation == 0
            return system.level2_exp
        end
        return pdf(Poisson(system.level2), current_observation)
    end

    value = value * (1 - system.overdispersion) / (1 - system.zero_inflation)
    return zigp_pmf(Int(current_observation), value, system.overdispersion, system.zero_inflation)
end

function average_state(system::LocalLevelCountJumpExplanatoryML, states, weights)
    return SizedVector{3, Float64, Vector{Float64}}([states[1][1], 
                           sum(states[i][2] * weights[i] for i in eachindex(weights)), 
                           sum(states[i][3] * weights[i] for i in eachindex(weights))])
end
function de_exogenous_ml(state::SizedVector{3, Float64, Vector{Float64}}, 
                        result::MutableRoot, 
                        exogenous::AbstractVector{Float64})::Float64
    time::Int64 = Int(state[1])
    value::Float64 = state[2]
    if time == 0
        return value
    end
    
    #machine::Root{Float64, Float64} = system.machine.fitresult[1]
    estimate::Float64 = apply_tree1(result, exogenous)
    
    return value - estimate
end

function re_exogenous_ml(value::Float64, 
                         result::MutableRoot,
                         exogenous::AbstractVector{Float64})::Float64
    if time == 0
        return value
    end
    
    #machine::Root{Float64, Float64} = system.machine.fitresult[1]
    estimate::Float64 = apply_tree1(result, exogenous)
    
    return value + estimate
end
