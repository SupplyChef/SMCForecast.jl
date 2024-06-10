"""
Local level model with regressors
"""
struct LocalLevelRegressor <: SMCSystem{SizedVector{3, Float64}}
    level::Float64
    change::Float64
    regressor
    level_variance::Float64
    observation_variance::Float64

    prior_distribution

    function LocalLevelRegressor(last_level, last_change, regressor, level_variance, observation_variance)
        new(last_level, last_change, regressor, level_variance, observation_variance,
            () -> Uniform(last_level-0.0001, last_level+0.0001))
    end
end

"""
    forecast(::Val{LocalLevelRegressor}, values, horizon; maxtime=10.0, size=500, forecast_percentiles=0.5)

Forecasts future values at the given percentiles
"""
function forecast(::Val{LocalLevelRegressor}, values, horizon; maxtime=10.0, size=500, forecast_percentiles=0.5)
    fcs = fit(Val{LocalLevelRegressor}(), values; maxtime=maxtime, size=size)
    smc = SMC{SizedVector{3, Float64}, LocalLevelRegressor}(fcs, size)
    filter!(smc, values; record=false)
    obs, weights = predict_observations(smc, horizon)
    if isa(forecast_percentiles, Real)
        return percentiles(forecast_percentiles, obs, weights)
    else
        return [percentiles(p, obs, weights) for p in forecast_percentiles]
    end
end

"""
    fit(::Val{LocalLevelRegressor}, values; maxtime=10.0, size=100)

Fits the model.
"""
function fit(::Val{LocalLevelRegressor}, values; maxtime=10.0, size=100)
    labels = values[2:end] .- values[1:end-1]
    features = DataFrame(LastValue = values[1:end-1], LastChange=vcat([0], labels[1:end-1]))

    model = RandomForestRegressor(;max_depth=2, n_trees=Int(ceil(length(values) / 10)))
    mach = machine(model, features, labels) |> MLJ.fit!

    dim = 3
    xs = SMCForecast.bboptimize2(get_loss_function(Val{LocalLevelRegressor}(), mach, values; size=size),
                    [values[1], var(values), var(values)],
                    Dict(
                        :SearchRange => [(0.0, maximum(values)), (0.0001, var(values)), (0.0001, var(values))], 
                        :NumDimensions => dim, 
                        :MaxStepsWithoutProgress => 5000,
                        :MaxTime => maxtime)
            )
    
    fcs2 = LocalLevelRegressor(xs[1], 0.0, mach, abs(xs[2]), abs(xs[3]))
    return fcs2
end

function get_loss_function(::Val{LocalLevelRegressor}, mach, values; size=100)
    return xs -> begin
        fcs2 = LocalLevelRegressor(xs[1], 0.0, mach, abs(xs[2]), abs(xs[3]))
        smc = SMC{SizedVector{3, Float64}, LocalLevelRegressor}(fcs2, size)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false)
        return -likelihood
    end
end

function sample_initial_state(system::LocalLevelRegressor, count; rng=Random.default_rng())
    states = [SizedVector{3, Float64}(0, r, 0.0) for r in rand(rng, system.prior_distribution(), count)]
    return states
end

function sample_states(system::LocalLevelRegressor, 
                       current_states::Vector{SizedVector{3, Float64}}, next_observation::Union{Missing, Float64}, 
                       new_states, sampling_probabilities)
    time = Int(current_states[1][1])
    values = [current_state[2] for current_state in current_states]
    changes = [current_state[3] for current_state in current_states]

    new_changes = [MLJ.predict(system.regressor, Tables.table([value, change]'))[1] for (value, change) in zip(values, changes)]
    #println(new_change)
    
    n = Normal(0, sqrt(system.level_variance))
    ϵs = rand(n, length(new_states))
    ps = pdf(n, ϵs)

    for i in 1:length(new_states)
        new_states[i][1] = time + 1
        new_states[i][2] = values[i] + new_changes[i] + ϵs[i]
        new_states[i][3] = ϵs[i]
    end
    sampling_probabilities .= 1
end

function sample_observation(system::LocalLevelRegressor, current_state::SizedVector{3})
    value::Float64 = current_state[2]
    n = Normal(value, sqrt(system.observation_variance))
    return rand(n)
end

function transition_probability(system::LocalLevelRegressor, state::SizedVector{3, Float64}, new_state::SizedVector{3, Float64})::Float64
    time = state[1]
    value::Float64 = state[2]
    change::Float64 = state[3]
    
    new_value::Float64 = new_state[2]
    
    new_change = MLJ.predict(system.regressor, Tables.table([value, change]'))[1]
    
    return Distributions.normpdf(value + new_change, 
                                 sqrt(system.level_variance), 
                                 new_value)
end

function observation_probability(system::LocalLevelRegressor, state::SizedVector{3, Float64}, observation::Float64)::Float64
    value::Float64 = state[2]
    t = Normal(value, sqrt(system.observation_variance))
    return pdf(t, observation)
end

function average_state(system::LocalLevelRegressor, states, weights)
    return SizedVector{3, Float64}([states[1][1], sum(states[i][2] * weights[i] for i in eachindex(weights)), sum(states[i][3] * weights[i] for i in eachindex(weights))])
end