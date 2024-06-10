struct LocalLevelExplanatory <: SMCSystem{SizedVector{2, Float64}}
    exogenous::Matrix{Float64}

    intercept::Float64
    coefficients::Array{Float64, 1}

    transition_variance::Float64
    observation_variance::Float64

    prior_distribution

    function LocalLevelExplanatory(exogenous, level0, intercept, coefficients, transition_variance, observation_variance)
        new(exogenous, intercept, coefficients, transition_variance, observation_variance,
            () -> Uniform(level0-0.0001, level0+0.0001))
    end
end

function fit(::Val{LocalLevelExplanatory}, exogenous, values; regularization=0.0, maxtime=120)
    
    loss_function = get_loss_function(Val{LocalLevelExplanatory}(), exogenous, values; regularization=regularization)
    
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

    fcs2 = LocalLevelExplanatory(exogenous, 
                              xs[1],
                              xs[2], 
                              xs[3:(2 + size(exogenous, 1))],
                              abs(xs[2 + size(exogenous, 1) + 1]),
                              abs(xs[2 + size(exogenous, 1) + 2]))
    return fcs2
end

function get_loss_function(::Val{LocalLevelExplanatory}, exogenous, values; regularization=regularization)
    return xs -> begin
        fcs2 = LocalLevelExplanatory(exogenous, 
                              xs[1], 
                              xs[2],
                              xs[3:(2 + size(exogenous, 1))],
                              abs(xs[2 + size(exogenous, 1) + 1]),
                              abs(xs[2 + size(exogenous, 1) + 2]))
        smc = SMC{SizedVector{2, Float64}, LocalLevelExplanatory}(fcs2, 30)
        filtered_states, likelihood = SMCForecast.filter!(smc, values)
        return -likelihood + regularization * sum(x^2 for x in xs)
    end
end

function prodsum(x, y) 
if length(x) == 0 return 0.0 end
    v = x[1] * y[1]
    for i in 2:length(x)
        @inbounds v += x[i] * y[i]
    end
    return v
end

function de_exogenous(state::SizedVector, exogenous::Matrix{Float64}, intercept::Float64, coefficients::Vector{Float64})::Float64
    time = Int(state[1])
    value = state[2]
    if time == 0
        return value - intercept
    end
    return @inbounds @views value - prodsum(exogenous[:, time], coefficients) - intercept
end

function re_exogenous(value, time, exogenous, intercept, coefficients)::Float64
    if time == 0
        return value + intercept
    end
    return @views value + dot(exogenous[:, time], coefficients) + intercept
end

function sample_initial_state(system::LocalLevelExplanatory, count; rng=Random.default_rng())
    states = [SizedVector{2, Float64}(0, r) for r in rand(rng, system.prior_distribution(), count)]
    #println(states)
    return states
end

function sample_states(system::LocalLevelExplanatory, 
                       current_states::Vector{SizedVector{2, Float64}}, 
                       next_observation::Union{Missing, Float64}, 
                       new_states, sampling_probabilities)
    time = Int(current_states[1][1])
    values = [de_exogenous(current_state, system.exogenous, system.intercept, system.coefficients) for current_state in current_states]

    d = Normal(0, sqrt(system.transition_variance))
    ϵs = rand(d, length(new_states))
    
    for i in 1:length(new_states)
        new_states[i][1] = time + 1
        new_states[i][2] = re_exogenous(values[i] + ϵs[i], time + 1, system.exogenous, system.intercept, system.coefficients)
    end
    sampling_probabilities .= 1
end

function transition_probability(system::LocalLevelExplanatory, state::SizedVector{2}, observation, new_state::SizedVector{2})::Float64
    time = state[1]
    value = state[2]

    return Distributions.normpdf(de_exogenous(state, system.exogenous, system.intercept, system.coefficients), 
                                 sqrt(system.transition_variance), 
                                 de_exogenous(new_state, system.exogenous, system.intercept, system.coefficients))
end

function sample_observation(system::LocalLevelExplanatory, current_state::SizedVector{2})
    time = Int(current_state[1])
    value = current_state[2]
    
    return rand(Normal(value, sqrt(system.observation_variance)))
end

function observation_probability(system::LocalLevelExplanatory, state::SizedVector{2}, observation)::Float64
    time = Int(state[1])
    value = state[2]

    d = Normal(value, sqrt(system.observation_variance))
    p = pdf(d, observation)
    #println(p)
    return p
end

function average_state(system::LocalLevelExplanatory, states, weights)
    return SizedVector{2}([states[1][1], sum(states[i][2] * weights[i] for i in eachindex(weights))])
end