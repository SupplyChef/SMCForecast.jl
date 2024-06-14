
struct LocalLevelJump <: SMCSystem{SizedVector{3, Float64, Vector{Float64}}}
    level1::Float64
    level2::Float64
    
    level_matrix::Array{Float64, 2}
    levels::Array{Int64, 1}
    level_weights::Array{ProbabilityWeights, 1}
    level_weights10::ProbabilityWeights

    level_variance::Float64
    observation_variance::Float64

    function LocalLevelJump(level1, level2, level_matrix, level_variance, observation_variance)
        levels = [1, 2]
        level_weights = [pweights(level_matrix[i,:]) for i in 1:size(level_matrix, 1)]
        level_weights10 = pweights((level_matrix^10)[1,:])
        new(level1, level2, level_matrix, levels, level_weights, level_weights10, level_variance, observation_variance)
    end
end

function forecast(::Val{LocalLevelJump}, values, horizon; maxtime=10.0, size=500, forecast_percentiles=0.5)
    fcs = fit(Val{LocalLevelJump}(), values; maxtime=maxtime, size=size)
    smc = SMC{SizedVector{3, Float64}, LocalLevelJump}(fcs, 1_000)
    filter!(smc, values; record=false)
    obs, weights = predict_observations(smc, horizon)
    if isa(forecast_percentiles, Real)
        return percentiles(forecast_percentiles, obs, weights)
    else
        return [percentiles(p, obs, weights) for p in forecast_percentiles]
    end
end

function fit(::Val{LocalLevelJump}, values; maxtime=10, regularization=0.0, size=100, min_observation_variance=0.00001)
    xs = SMCForecast.bboptimize2(get_loss_function(Val{LocalLevelJump}(), values; regularization=regularization, size=size),
                    [values[1], 0.00001, var(values), 0.9, 0.9, 0.5],
                    Dict(
                    :SearchRange => [(0, maximum(values)), (0.00001, mean(values) / 2), 
                                     (0.00001, var(values)), (min_observation_variance, 0.99999), 
                                     (0.0001, .9999), (0.0001, .9999)], 
                    :NumDimensions => dim, 
                    :MaxTime => maxtime,
                    :MaxStepsWithoutProgress => 2000)
                    )
    
    fcs2 = LocalLevelJump(xs[1], 
                          xs[2],
                          [1-xs[5] xs[5]; 
                           1-xs[6] xs[6]],
                          abs(xs[3]),
                          abs(xs[4]))
    return fcs2
end

function get_loss_function(::Val{LocalLevelJump}, values; regularization=0.0, size=1000)
    return xs -> begin
        fcs2 = LocalLevelJump(xs[1], 
                            xs[2],
                            [1-xs[5] xs[5]; 
                            1-xs[6] xs[6]],
                            abs(xs[3]),
                            abs(xs[4]))
        smc = SMC{SizedVector{3, Float64, Vector{Float64}}, LocalLevelJump}(fcs2, size)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false)
        return -likelihood + regularization * sum(x^2 for x in xs)
    end
end

function sample_initial_state(system::LocalLevelJump, count; rng=Random.default_rng())
    states = sample(rng, [1,2], system.level_weights10, count)
    return [SizedVector{3, Float64, Vector{Float64}}(1.0, system.level1, states[i]) for i in eachindex(states)]
end

function sample_states(system::LocalLevelJump, 
                       current_states::Vector{SizedVector{3, Float64, Vector{Float64}}}, next_observation::Union{Missing, Float64}, 
                       new_states, sampling_probabilities; happy_only=false, rng=Random.default_rng())
    time = Int(current_states[1][1])

    for (i, current_state) in enumerate(current_states)
        value = current_state[2]
        state = Int(current_state[3])
        
        n = Normal(0, sqrt(system.level_variance))

        new_state = sample(rng, system.levels, system.level_weights[state])
        if happy_only
            while new_state == 2
                new_state = sample(rng, system.levels, system.level_weights[state])
            end        
        end
        ϵ = rand(rng, n)
        new_value = value + ϵ

        new_states[i][1] = time + 1
        new_states[i][2] = new_value
        new_states[i][3] = new_state

        sampling_probabilities[i] = 1
    end
end

function sample_observation(system::LocalLevelJump, current_state::SizedVector{3}; rng=Random.default_rng())
    value::Float64 = current_state[2]
    state = Int(current_state[3])

    if state == 2
        return rand(rng, Poisson(system.level2))
    end

    p = system.observation_variance
    return rand(rng, NegativeBinomial((system.level2 + exp(value)) * p / (1 - p), p))
end

function transition_probability(system::LocalLevelJump, state1::SizedVector{3}, new_observation, state2::SizedVector{3})::Float64
    time = state1[1]
    value = state1[2]
    state = Int(state1[3])

    new_value = state2[2]
    new_state = Int(state2[3])

    n = Normal(0, sqrt(system.level_variance))
    
    p = pdf(n, new_value - value)
    probability = system.level_matrix[state, new_state] * p
    if isinf(probability) || isnan(probability)
        #println("$probability, $state1, $state2")
    end
    return probability
end

function observation_probability(system::LocalLevelJump, current_state::SizedVector{3}, current_observation)::Float64
    time = current_state[1]
    value = current_state[2]
    state = Int(current_state[3])

    if state == 2
        return pdf(Poisson(system.level2), current_observation)
    end

    p = system.observation_variance
    #println("$value, $p, $current_observation, $(pdf(NegativeBinomial(value * p / ( 1 - p), p), current_observation))")
    return pdf(NegativeBinomial((system.level2 + exp(value)) * p / (1 - p), p), current_observation) 
end

function average_state(system::LocalLevelJump, states, weights)
    return SizedVector{3, Float64, Vector{Float64}}([states[1][1], 
                           sum(states[i][2] * weights[i] for i in eachindex(weights)),
                           sum(states[i][3] * weights[i] for i in eachindex(weights))])
end