struct LocalLevelChange <: SMCSystem{SizedVector{3, Float64, Vector{Float64}}}
    level::Float64
    change::Float64
    
    level_variance::Float64
    change_variance::Float64
    observation_variance::Float64

    prior_distribution

    function LocalLevelChange(last_level, last_change, level_variance, change_variance, observation_variance)
        new(last_level, last_change, 
            level_variance, min(change_variance, level_variance), observation_variance,
            () -> Uniform(last_level-0.0001, last_level+0.0001)            )
    end
end

function forecast(::Val{LocalLevelChange}, values, horizon; maxtime=10.0, size=500, forecast_percentiles=0.5)
    fcs = fit(Val{LocalLevelChange}(), values; maxtime=maxtime, size=size)
    smc = SMC{SizedVector{3, Float64, Vector{Float64}}, LocalLevelChange}(fcs, 1_000)
    filter!(smc, values; record=false)
    obs, weights = predict_observations(smc, horizon)
    if isa(forecast_percentiles, Real)
        return percentiles(forecast_percentiles, obs, weights)
    else
        return [percentiles(p, obs, weights) for p in forecast_percentiles]
    end
end

function fit(::Val{LocalLevelChange}, values; maxtime=10.0, size=100, regularization=0.0)
    dim = 5
    xs = SMCForecast.bboptimize2(get_loss_function(Val{LocalLevelChange}(), values; size=size, regularization=regularization),
                    [values[1], values[2]-values[1], var(values) / length(values), 0.01, var(values) / length(values)],
                    Dict(
                        :SearchRange => [(minimum(values), maximum(values)), 
                                         (-10, +10), 
                                         (0.00001, var(values)), 
                                         (0.00001, var(values) / 1000), 
                                         (0.00001, var(values))], 
                        :NumDimensions => dim, 
                        :MaxStepsWithoutProgress => 15000,
                        :MaxTime => maxtime); quiet=false
            )
    
    fcs2 = LocalLevelChange(xs[1], xs[2], abs(xs[3]), abs(xs[4]), abs(xs[5]))
    return fcs2
end

function get_loss_function(::Val{LocalLevelChange}, values; size=100, regularization=0.0)
    return xs -> begin
        fcs2 = LocalLevelChange(xs[1], xs[2], abs(xs[3]), abs(xs[4]), abs(xs[5]))
        smc = SMC{SizedVector{3, Float64, Vector{Float64}}, LocalLevelChange}(fcs2, size)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false)
        return -likelihood + regularization * sum(x^2 for x in xs)
    end
end

function sample_initial_state(system::LocalLevelChange, count; rng=Random.default_rng())
    states = [SizedVector{3, Float64, Vector{Float64}}(0, r, system.change) for r in rand(rng, system.prior_distribution(), count)]
    return states
end

function sample_states(system::LocalLevelChange, 
                      current_states::Vector{SizedVector{3, Float64, Vector{Float64}}}, next_observation::Union{Missing, Float64}, 
                      new_states::Vector{SizedVector{3, Float64, Vector{Float64}}}, sampling_probabilities::Array{Float64, 1}; rng=Random.default_rng())
    time = Int(current_states[1][1])
    levels = [current_state[2] for current_state in current_states]
    changes = [current_state[3] for current_state in current_states]

    n = Normal(0, sqrt(system.level_variance))
    level_ϵs = rand(rng, n, length(new_states))

    n = Normal(0, sqrt(system.change_variance))
    change_ϵs = rand(rng, n, length(new_states))

    for i in 1:length(new_states)
        new_states[i][1] = time + 1
        new_states[i][2] = levels[i] + changes[i] + level_ϵs[i]
        new_states[i][3] = changes[i] + change_ϵs[i]
    end
    sampling_probabilities .= 1
end

function sample_observation(system::LocalLevelChange, current_state::SizedVector{3}; rng=Random.default_rng())
    value::Float64 = current_state[2]
    n = Normal(value, sqrt(system.observation_variance))
    return rand(rng, n)
end

function transition_probability(system::LocalLevelChange, 
                                state::SizedVector{3, Float64, Vector{Float64}}, 
                                new_observation,
                                new_state::SizedVector{3, Float64, Vector{Float64}})::Float64
    time = state[1]
    level::Float64 = state[2]
    change::Float64 = state[3]
    
    new_level::Float64 = new_state[2]
    new_change::Float64 = new_state[3]

    level_p = pdf(Normal(0, sqrt(system.level_variance)), new_level - (level + change))
    change_p = pdf(Normal(0, sqrt(system.change_variance)), new_change - change)
    return  level_p * change_p
end

function observation_probability(system::LocalLevelChange, state::SizedVector{3, Float64, Vector{Float64}}, observation::Float64)::Float64
    value::Float64 = state[2]
    t = Normal(value, sqrt(system.observation_variance))
    return pdf(t, observation)
end

function average_state(system::LocalLevelChange, states, weights)
    return SizedVector{3, Float64, Vector{Float64}}([states[1][1], sum(states[i][2] * weights[i] for i in eachindex(weights)), sum(states[i][3] * weights[i] for i in eachindex(weights))])
end