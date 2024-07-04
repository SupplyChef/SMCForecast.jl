"""
Local level model
"""
struct LocalLevel <: SMCSystem{SizedVector{2, Float64}}
    level::Float64
    level_variance::Float64
    observation_variance::Float64

    prior_distribution

    function LocalLevel(level, level_variance, observation_variance)
        new(level, level_variance, observation_variance,
            () -> Uniform(level-0.0001, level+0.0001))
    end
end

function forecast(::Val{LocalLevel}, values, horizon; maxtime=10.0, size=500, forecast_percentiles=0.5)
    fcs = fit(Val{LocalLevel}(), values; maxtime=maxtime, size=size)
    smc = SMC{SizedVector{2, Float64}, LocalLevel}(fcs, size)
    filter!(smc, values; record=false)
    obs, weights = predict_observations(smc, horizon)
    if isa(forecast_percentiles, Real)
        return percentiles(forecast_percentiles, obs, weights)
    else
        return [percentiles(p, obs, weights) for p in forecast_percentiles]
    end
end

function fit(::Val{LocalLevel}, values; maxtime=10.0, size=100)
    dim = 3
    xs = SMCForecast.bboptimize2(get_loss_function(Val{LocalLevel}(), values; size=size),
                    [values[1], var(values), var(values)],
                    Dict(
                        :SearchRange => [(minimum(values), maximum(values)), (0.0001, var(values)), (0.0001, var(values))], 
                        :NumDimensions => dim, 
                        :MaxStepsWithoutProgress => 5000,
                        :MaxTime => maxtime)
            )

    fcs2 = LocalLevel(xs[1], 
                      abs(xs[2]),
                      abs(xs[3]))
    return fcs2
end

function get_loss_function(::Val{LocalLevel}, values; size=100)
    return xs -> begin
        fcs2 = LocalLevel(xs[1], 
                          abs(xs[2]),
                          abs(xs[3]))
        smc = SMC{SizedVector{2, Float64}, LocalLevel}(fcs2, size)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false)
        return -likelihood
    end
end

function sample_initial_state(system::LocalLevel, count; rng=Random.default_rng())
    states = [SizedVector{2, Float64}(0, r) for r in rand(rng, system.prior_distribution(), count)]
    return states
end

function sample_states(system::LocalLevel, 
                       current_states::Vector{SizedVector{2, Float64}}, next_observation::Union{Missing, Float64}, 
                      new_states, sampling_probabilities; rng=Random.default_rng())
    time = Int(current_states[1][1])
    levels = [current_state[2] for current_state in current_states]
    
    n = Normal(0, sqrt(system.level_variance))
    level_ϵs = rand(rng, n, length(new_states))

    for i in 1:length(new_states)
        new_states[i][1] = time + 1
        new_states[i][2] = levels[i] + level_ϵs[i]
    end
    sampling_probabilities .= 1
end

function sample_observation(system::LocalLevel, current_state::SizedVector{2}; rng=Random.default_rng())
    level::Float64 = current_state[2]
    n = Normal(level, sqrt(system.observation_variance))
    return rand(rng, n)
end

function transition_probability(system::LocalLevel, 
                                state::SizedVector{2, Float64}, 
                                new_observation,
                                new_state::SizedVector{2, Float64})::Float64
    time = state[1]
    value::Float64 = state[2]
    new_value::Float64 = new_state[2]
    return Distributions.normpdf(value, 
                                 sqrt(system.level_variance), 
                                 new_value)
end

function observation_probability(system::LocalLevel, state::SizedVector{2, Float64}, observation::Float64)::Float64
    level::Float64 = state[2]
    t = Normal(level, sqrt(system.observation_variance))
    return pdf(t, observation)
end

function average_state(system::LocalLevel, states, weights)
    return SizedVector{2, Float64}([states[1][1], sum(states[i][2] * weights[i] for i in eachindex(weights))])
end