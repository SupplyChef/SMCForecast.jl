struct LocalLevelCountStockoutExplanatory <: SMCSystem{SizedVector{3, Float64, Vector{Float64}}}
    exogenous::Matrix{Float64}

    level1::Float64
    level2::Float64
    level_matrix::Array{Float64, 2}
    levels::Array{Int64, 1}
    
    coefficients::Array{Float64, 1}

    level_variance::Float64
    
    zero_inflation::Float64
    overdispersion::Float64

    level2_exp::Float64

    level_weights::Array{ProbabilityWeights, 1}
    level_weights10::ProbabilityWeights
    level_equal_weights::ProbabilityWeights

    adjust_sampling::Bool

    function LocalLevelCountStockoutExplanatory(;exogenous, level1, level2, level_matrix, coefficients, level_variance, zero_inflation, overdispersion, adjust_sampling=false)
        levels = [1, 2]
        level_weights = [pweights(level_matrix[i,:]) for i in 1:size(level_matrix, 1)]
        level_weights10 = pweights((level_matrix^10)[1,:])
        level_equal_weights = pweights([0.5, 0.5])

        new(exogenous, 
            level1, 
            level2, 
            level_matrix, 
            levels, 
            coefficients, 
            level_variance, 
            zero_inflation,
            overdispersion,
            exp(-level2), level_weights, level_weights10, level_equal_weights, 
            adjust_sampling)
    end
end

function forecast(::Val{LocalLevelCountStockoutExplanatory}, exogenous, values, horizon; maxtime=10.0, size=500, forecast_percentiles=0.5)
    fcs = fit(Val{LocalLevelCountStockoutExplanatory}(), exogenous, values; maxtime=maxtime, size=size)
    smc = SMC{SizedVector{3, Float64}, LocalLevelCountStockoutExplanatory}(fcs, 1_000)
    filter!(smc, values; record=false)
    obs, weights = predict_observations(smc, horizon)
    if isa(forecast_percentiles, Real)
        return percentiles(forecast_percentiles, obs, weights)
    else
        return [percentiles(p, obs, weights) for p in forecast_percentiles]
    end
end

function fit(::Val{LocalLevelCountStockoutExplanatory}, exogenous, values; maxtime=10, regularization=0.0, size=100,
                                                                        min_stay_outofstock_probability=0.0001)
    loss_function = get_loss_function(Val{LocalLevelCountStockoutExplanatory}(), exogenous, values; regularization=regularization, size=size)
    
    dim = 7 + Base.size(exogenous, 1)

    xs = bboptimize2(loss_function,
                    vcat([values[1], 
                        0.00001, 
                        max((var(values) - (length(values) * mean(values))) / length(values),  0.001), 
                        0.0, 
                        0.0, 
                        0.1, 
                        max(0.00001, 0.9)],
                        zeros(Base.size(exogenous, 1))
                    ),
                    Dict(:SearchRange => vcat([(0, maximum(values)), 
                                                (0.00001, mean(values) / 5), 
                                                (0.00001, var(values) / length(values)),
                                                (0.00001, .9999),
                                                (0.00001, .9999), 
                                                (0.00001, .9999), 
                                                (min_stay_outofstock_probability, .9999)],
                                                repeat([(-1.0, 2.0)], Base.size(exogenous, 1)),
                                        ),
                         :NumDimensions => dim, 
                         :MaxTime => maxtime)
                         )

    fcs2 = LocalLevelCountStockoutExplanatory(;exogenous=exogenous, 
                                            level1=xs[1], 
                                            level2=xs[2],
                                            level_variance=abs(xs[3]), 
                                            zero_inflation=abs(xs[4]),
                                            overdispersion=abs(xs[5]),
                                            level_matrix=[1-xs[6] xs[6]; 
                                                        1-xs[7] xs[7]],
                                            coefficients=xs[8:(7 + Base.size(exogenous, 1))])
    return fcs2
end

function get_loss_function(::Val{LocalLevelCountStockoutExplanatory}, exogenous, values; regularization=0.0, size=1000)
    return xs -> begin
        fcs2 = LocalLevelCountStockoutExplanatory(;exogenous=exogenous, 
                                                level1=xs[1], 
                                                level2=xs[2],
                                                level_variance=abs(xs[3]), 
                                                zero_inflation=abs(xs[4]),
                                                overdispersion=abs(xs[5]),
                                                level_matrix=[1-xs[6] xs[6]; 
                                                            1-xs[7] xs[7]],
                                                coefficients=xs[8:(7 + Base.size(exogenous, 1))]
                              )
        smc = SMC{SizedVector{3, Float64, Vector{Float64}}, LocalLevelCountStockoutExplanatory}(fcs2, size)
        rng = MersenneTwister(1)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false, rng=rng)
        return -likelihood + regularization * sum(x^2 for x in xs[8:end])
    end
end

function sample_initial_state(system::LocalLevelCountStockoutExplanatory, count; rng=Random.default_rng())::Array{SizedVector{3, Float64, Vector{Float64}}, 1}
    states = sample(rng, [1,2], pweights([0.9, 0.1]), count)
    return [SizedVector{3, Float64, Vector{Float64}}(1.0, system.level1, states[i]) for i in eachindex(states)]
end

function sample_states(system::LocalLevelCountStockoutExplanatory, 
                      current_states::Vector{SizedVector{3, Float64, Vector{Float64}}},
                      next_observation::Union{Missing, Float64}, 
                      new_states, sampling_probabilities; happy_only=false, rng=Random.default_rng())
    time = Int(current_states[1][1])

    for (i, current_state) in enumerate(current_states)
        value = de_exogenous_multiplicative(current_state, system.exogenous, system.coefficients)
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
        new_states[i][2] = re_exogenous_multiplicative(new_value, time + 1, system.exogenous, system.coefficients)
        new_states[i][3] = new_state
    end
end

function sample_observation(system::LocalLevelCountStockoutExplanatory, current_state::SizedVector{3}; rng=Random.default_rng())
    value::Float64 = current_state[2]
    state = Int(current_state[3])

    if state == 2
        return rand(rng, Poisson(system.level2))
    end

    value = value * (1 - system.overdispersion) / (1 - system.zero_inflation)
    return sample_zigp(value, system.overdispersion, system.zero_inflation)
end

function transition_probability(system::LocalLevelCountStockoutExplanatory, state1::SizedVector{3, Float64, Vector{Float64}}, new_observation, state2::SizedVector{3, Float64, Vector{Float64}})::Float64
    time = Int(state1[1])
    value = de_exogenous_multiplicative(state1, system.exogenous, system.coefficients)
    state = Int(state1[3])

    new_value = de_exogenous_multiplicative(state2, system.exogenous, system.coefficients)
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

function observation_probability(system::LocalLevelCountStockoutExplanatory, current_state::SizedVector{3, Float64, Vector{Float64}}, current_observation)::Float64
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

function average_state(system::LocalLevelCountStockoutExplanatory, states, weights)
    return SizedVector{3, Float64, Vector{Float64}}([states[1][1], 
                           sum(states[i][2] * weights[i] for i in eachindex(weights)), 
                           sum(states[i][3] * weights[i] for i in eachindex(weights))])
end

function de_exogenous_multiplicative(state::SizedVector, exogenous::Matrix{Float64}, coefficients::Vector{Float64})::Float64
    time = Int(state[1])
    value = state[2]
    if time == 0
        return value
    end
    prod = 1.0
    @inbounds @views for (i, e) in enumerate(exogenous[:, time])
        if e > 0
            prod *=  (coefficients[i] + 1)
        end
    end
    return value / prod
end

function re_exogenous_multiplicative(value, time, exogenous, coefficients)::Float64
    if time == 0
        return value
    end
    prod = 1.0
    @inbounds @views for (i, e) in enumerate(exogenous[:, time])
        if e > 0
            prod *=  (coefficients[i] + 1)
        end
    end
    return value * prod
end
