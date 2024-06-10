struct LocalLevelCountJumpExplanatory <: SMCSystem{SizedVector{3, Float64}}
    exogenous::Matrix{Float64}

    level1::Float64
    level2::Float64
    level_matrix::Array{Float64, 2}
    levels::Array{Int64, 1}
    level_weights::Array{ProbabilityWeights, 1}

    intercept::Float64
    coefficients::Array{Float64, 1}

    level_variance::Float64
    observation_variance::Float64

    function LocalLevelCountJumpExplanatory(;exogenous, level1, level2, level_matrix, intercept, coefficients, level_variance, observation_variance)
        levels = [1,2]
        level_weights = [pweights(level_matrix[i,:]) for i in 1:size(level_matrix, 1)]
        new(exogenous, level1, level2, level_matrix, levels, level_weights, intercept, coefficients, level_variance, observation_variance)
    end
end

function fit2(::Val{LocalLevelCountJumpExplanatory}, exogenous, values; regularization=0.0, maxtime=10)
    dim = 4 + size(exogenous, 1)
    res = bboptimize(get_loss_function(Val{LocalLevelCountJumpExplanatory}(), exogenous, values; regularization=regularization); 
                    SearchRange = (-50.0, 50.0), 
                    NumDimensions = dim, 
                    MaxTime = maxtime)

    println(best_fitness(res))
    println(best_candidate(res))

    xs = best_candidate(res)
    fcs2 = LocalLevelJumpExplanatory(;exogenous=exogenous, 
                              level1=xs[1], 
                              level2=0.0, 
                              level_matrix=[0.9 0.1; 0.9 0.1],
                              intercept=xs[2], 
                              coefficients=xs[3:(2 + size(exogenous, 1))],
                              level_variance=abs(xs[2 + size(exogenous, 1) + 1]),
                              observation_variance=abs(xs[2 + size(exogenous, 1) + 2]))
    return fcs2
end

function fit(::Val{LocalLevelCountJumpExplanatory}, exogenous, values; regularization=0.0, maxtime=10)
    loss_function = get_loss_function(Val{LocalLevelCountJumpExplanatory}(), exogenous, values; regularization=regularization)
    
    dim = 4 + size(exogenous, 1)

    xs = bboptimize2(loss_function,
                    vcat([values[1]], [0.0], zeros(size(exogenous, 1)), [var(values), var(values)]),
                    Dict(:SearchRange => vcat([(-maximum(values), maximum(values))],
                                              [(-maximum(values), maximum(values))],
                                              repeat([(minimum(values)-maximum(values), maximum(values)-minimum(values))], size(exogenous, 1)),
                                              [(0.00001, var(values))],
                                              [(0.00001, var(values))]
                                        ),
                         :NumDimensions => dim, 
                         :MaxTime => maxtime)
                         )

    fcs2 = LocalLevelCountJumpExplanatory(;exogenous=exogenous, 
                              level1=xs[1], 
                              level2=0.0, 
                              level_matrix=[0.9 0.1; 0.9 0.1],
                              intercept=xs[2], 
                              coefficients=xs[3:(2 + size(exogenous, 1))],
                              level_variance=abs(xs[2 + size(exogenous, 1) + 1]),
                              observation_variance=abs(xs[2 + size(exogenous, 1) + 2]))
    return fcs2
end

function get_loss_function(::Val{LocalLevelCountJumpExplanatory}, exogenous, values; regularization=0.0)
    return xs -> begin
        fcs2 = LocalLevelCountJumpExplanatory(;exogenous=exogenous, 
                              level1=xs[1], 
                              level2=0.0, 
                              level_matrix=[0.9 0.1; 0.9 0.1],
                              intercept=xs[2], 
                              coefficients=xs[3:(2 + size(exogenous, 1))],
                              level_variance=abs(xs[2 + size(exogenous, 1) + 1]),
                              observation_variance=abs(xs[2 + size(exogenous, 1) + 2]))
        smc = SMC{SizedVector{3, Float64}, LocalLevelCountJumpExplanatory}(fcs2, 30)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false)
        return -likelihood + regularization * sum(x^2 for x in xs)
    end
end

function sample_initial_state(system::LocalLevelCountJumpExplanatory, count; rng=Random.default_rng())::Array{SizedVector{3, Float64}, 1}
    states = sample(rng, [1,2], pweights([0.9, 0.1]), count)
    return [SizedVector{3, Float64}([0, system.level1, states[i]]) for i in eachindex(states)]
end

function sample_states(system::LocalLevelCountJumpExplanatory, 
                      current_states::Vector{SizedVector{3, Float64}}, next_observation::Union{Missing, Float64}, 
                      new_states, sampling_probabilities)
    time = Int(current_states[1][1])

    for (i, current_state) in enumerate(current_states)
        value = de_exogenous(current_state, system.exogenous, system.intercept, system.coefficients)
        state = Int(current_state[3])
        
        n = Normal(0, sqrt(system.level_variance))

        new_state = sample(system.levels, system.level_weights[state])
        ϵ = rand(n)
        new_value = value + ϵ

        p = pdf(n, ϵ)
        
        new_states[i][1] = time + 1
        new_states[i][2] = re_exogenous(new_value, time + 1, system.exogenous, system.intercept, system.coefficients)
        new_states[i][3] = new_state

        sampling_probabilities[i] = 1
    end
end

function sample_observation(system::LocalLevelCountJumpExplanatory, current_state::SizedVector{3})
    time = Int(current_state[1])
    value = current_state[2]
    state = Int(current_state[3])

    if state == 2
        value = system.level2
    end

    return rand(Normal(value, sqrt(system.observation_variance)))
end

function transition_probability(system::LocalLevelCountJumpExplanatory, state1::SizedVector{3, Float64}, observation, state2::SizedVector{3, Float64})::Float64
    time = Int(state1[1])
    value = de_exogenous(state1, system.exogenous, system.intercept, system.coefficients)
    state = Int(state1[3])

    new_value = de_exogenous(state2, system.exogenous, system.intercept, system.coefficients)
    new_state = Int(state2[3])
    
    return system.level_matrix[state, new_state] * Distributions.normpdf(value, sqrt(system.level_variance), new_value)
end

function observation_probability(system::LocalLevelCountJumpExplanatory, current_state::SizedVector{3, Float64}, current_observation)::Float64
    time = Int(current_state[1])
    value = current_state[2]
    state = Int(current_state[3])

    if state == 2
        value = system.level2
    end

    return Distributions.normpdf(value, sqrt(system.observation_variance), current_observation)
end

function average_state(system::LocalLevelCountJumpExplanatory, states, weights)
    return SizedVector{3}([states[1][1], 
                           sum(states[i][2] * weights[i] for i in eachindex(weights)), 
                           sum(states[i][3] * weights[i] for i in eachindex(weights))])
end