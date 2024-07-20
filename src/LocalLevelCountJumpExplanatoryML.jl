using DecisionTree: apply_tree

struct LocalLevelCountJumpExplanatoryML <: SMCSystem{SizedVector{3, Float64, Vector{Float64}}}
    exogenous::Matrix{Float64}

    level1::Float64
    level2::Float64
    level_matrix::Array{Float64, 2}
    levels::Array{Int64, 1}
    
    machine

    level_variance::Float64
    
    zero_inflation::Float64
    overdispersion::Float64

    level2_exp::Float64

    level_weights::Array{ProbabilityWeights, 1}
    level_weights10::ProbabilityWeights
    level_equal_weights::ProbabilityWeights

    adjust_sampling::Bool

    texogenous

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
            adjust_sampling,
            collect(exogenous'))
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
                                                                        min_stay_outofstock_probability=0.0001)

    DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
    model = DecisionTreeRegressor(max_depth=5, min_samples_split=30)
    mach = machine(model, table(exogenous[:,1:length(values)]'), values) |> MLJ.fit!

    loss_function = get_loss_function(Val{LocalLevelCountJumpExplanatoryML}(), exogenous, values, mach; regularization=regularization, size=size)
    
    dim = 7

    xs = bboptimize2(loss_function,
                       [values[1], 
                        0.00001, 
                        max((var(values) - (length(values) * mean(values))) / length(values),  0.001), 
                        0.0, 
                        0.0, 
                        0.1, 
                        max(0.00001, 0.9)],
                    Dict(:SearchRange => [(0, maximum(values)), 
                                            (0.00001, mean(values) / 5), 
                                            (0.00001, var(values) / length(values)),
                                            (0.00001, .9999),
                                            (0.00001, .9999), 
                                            (0.00001, .9999), 
                                            (min_stay_outofstock_probability, .9999)],
                         :NumDimensions => dim, 
                         :MaxTime => maxtime)
                    )

    fcs2 = LocalLevelCountJumpExplanatoryML(;exogenous=exogenous, 
                                            machine = mach,
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
    return xs -> begin
        fcs2 = LocalLevelCountJumpExplanatoryML(;exogenous=exogenous, 
                                                machine = mach,
                                                level1=xs[1], 
                                                level2=xs[2],
                                                level_variance=abs(xs[3]), 
                                                zero_inflation=abs(xs[4]),
                                                overdispersion=abs(xs[5]),
                                                level_matrix=[1-xs[6] xs[6]; 
                                                            1-xs[7] xs[7]]
                              )
        smc = SMC{SizedVector{3, Float64, Vector{Float64}}, LocalLevelCountJumpExplanatoryML}(fcs2, size)
        rng = MersenneTwister(1)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false, rng=rng)
        return -likelihood 
    end
end

function sample_initial_state(system::LocalLevelCountJumpExplanatoryML, count; rng=Random.default_rng())::Array{SizedVector{3, Float64, Vector{Float64}}, 1}
    states = sample(rng, [1, 2], pweights([0.9, 0.1]), count)
    return [SizedVector{3, Float64, Vector{Float64}}(0.0, system.level1, states[i]) for i in eachindex(states)]
end

function sample_states(system::LocalLevelCountJumpExplanatoryML, 
                      current_states::Vector{SizedVector{3, Float64, Vector{Float64}}},
                      next_observation::Union{Missing, Float64}, 
                      new_states, sampling_probabilities; happy_only=false, rng=Random.default_rng())
    time = Int(current_states[1][1])

    for (i, current_state) in enumerate(current_states)
        state = Int(current_state[3])
        sampling_probabilities[i] = 1

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

        #println(size(system.exogenous))
        #@views new_value = MLJ.predict(system.machine, Tables.table(system.exogenous[:, time + 1]'))[1]
        @views new_value = apply_tree(system.machine.fitresult[1], system.exogenous[:, time + 1])

        new_states[i][1] = time + 1
        new_states[i][2] = new_value
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

function transition_probability(system::LocalLevelCountJumpExplanatoryML, state1::SizedVector{3, Float64, Vector{Float64}}, new_observation, state2::SizedVector{3, Float64, Vector{Float64}})::Float64
    time = Int(state1[1])
    state = Int(state1[3])

    new_state = Int(state2[3])
    
    probability = system.level_matrix[state, new_state] 

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