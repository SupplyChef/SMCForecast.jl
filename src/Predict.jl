using Random


function expectile(obs, weights, q)
   p = sortperm(obs)

    total_values = sum(obs .* weights)

    set_value = 0.0
    moving_weight = sum(weights)

    j = 1
    for k in 1:maximum(obs)
        while j <= length(obs) && obs[p[j]] <= k
            set_value += obs[p[j]] * weights[p[j]]
            moving_weight -= weights[p[j]]
            j = j + 1
        end
        if set_value + moving_weight * k >= q * total_values
            return k
        end
    end
end

function cum_expectiles(percentile::Real, states::Array{Array{S, 1}, 1}, weights) where S <: SizedVector
    values = cumsum([map(s -> s[2], states[i]) for i in 1:length(states)])
    return [expectile(values[i], weights[i], percentile) for i in 1:length(values)]
end

function cum_expectiles(percentile::Real, obs::Array{Array{R, 1}, 1}, weights) where R <: Real
    obs = cumsum(obs)
    return [expectile(obs[i], weights[i], percentile) for i in 1:length(obs)]
end

function avg(states::Array{Array{S, 1}, 1}, weights) where S <: SizedVector
    return [sum(map(s -> s[2], states[i]) .* weights[i]) for i in 1:length(states)]
end

function percentiles(percentile::Real, states::Array{Array{S, 1}, 1}, weights) where S <: SizedVector
    return [quantile(map(s -> s[2], states[i]), pweights(weights[i]), percentile) for i in 1:length(states)]
end

function avg(obs::Array{Array{R, 1}, 1}, weights) where R <: Real
    return [sum(obs[i] .* weights[i]) for i in 1:length(obs)]
end

function percentiles(percentile::Real, obs::Array{Array{R, 1}, 1}, weights) where R <: Real
    return [quantile(obs[i], pweights(weights[i]), percentile) for i in 1:length(obs)]
end

function cum_percentiles(percentile::Real, states::Array{Array{S, 1}, 1}, weights) where S <: SizedVector
    values = cumsum([map(s -> s[2], states[i]) for i in 1:length(states)])
    return [quantile(values[i], pweights(weights[i]), percentile) for i in 1:length(values)]
end

function cum_percentiles(percentile::Real, obs::Array{Array{R, 1}, 1}, weights) where R <: Real
    obs = cumsum(obs)
    return [quantile(obs[i], pweights(weights[i]), percentile) for i in 1:length(obs)]
end

function predict_observations(smc::SMC{T, U}, horizon; happy_only=true, rng=Random.default_rng()) where {T <: SizedVector, U <: SMCSystem{T}}
    states, weights = predict_states(smc, horizon; happy_only=happy_only, rng=rng)
    #println("state mean: $(avg(states, weights))")
    #println("state median: $(percentiles(0.5, states, weights))")
    observations = [[sample_observation(smc.system, state) for state in states[i]] for i in 1:length(states)]
    #println("obs mean: $(avg(observations, weights))")
    #println("obs median: $(percentiles(0.5, observations, weights))")
    return observations, weights
end

function predict_states(smc::SMC{T, U}, horizon::Int64; happy_only=true, rng=Random.default_rng()) where {T <: SizedVector, U <: LocalLevelCountJump} 
    states = Array{T, 1}[]
    weights = Array{Float64, 1}[]

    smcstates = deepcopy(smc.states)
    smcweights = deepcopy(smc.weights)

    new_states = [T(zeros(length(T))) for i in 1:length(smc.states)]
    sampling_probabilities = [1.0 for i in 1:length(smc.states)]
    for j in 1:horizon
        weight_sum::Float64 = 0.0

        sample_states(smc.system, smcstates, missing, new_states, sampling_probabilities; happy_only=happy_only)
        
        for i in eachindex(smcstates)
            state = smcstates[i]
            
            observation_weight::Float64 = sampling_probabilities[i]

            weight::Float64 = smcweights[i] * observation_weight
            #println("$new_state; $probability, $new_state_transition_probability; $(smc.weights[i]), $observation_weight")
            copyto!(smcstates[i], new_states[i])
            smcweights[i] = weight

            weight_sum = weight_sum + weight

            #println("$j $i, $state $observation, $observation_weight")
        end

        smcweights .= smcweights ./ weight_sum

        push!(states, deepcopy(smcstates))
        push!(weights, deepcopy(smcweights))
    end

    return states, weights
end

function predict_states(smc::SMC{T, U}, horizon::Int64; happy_only=true, rng=Random.default_rng()) where {T <: SizedVector, U <: SMCSystem{T}} 
    states = Array{T, 1}[]
    weights = Array{Float64, 1}[]

    smcstates = deepcopy(smc.states)
    smcweights = deepcopy(smc.weights)

    new_states = T[T(zeros(size(T, 1))) for i in 1:length(smc.states)]
    sampling_probabilities = [1.0 for i in 1:length(smc.states)]
    for j in 1:horizon
        weight_sum::Float64 = 0.0

        sample_states(smc.system, smcstates, missing, new_states, sampling_probabilities)
        
        for i in eachindex(smcstates)
            state = smcstates[i]
            
            observation_weight::Float64 = sampling_probabilities[i]

            weight::Float64 = smcweights[i] * observation_weight
            #println("$new_state; $probability, $new_state_transition_probability; $(smc.weights[i]), $observation_weight")
            copyto!(smcstates[i], new_states[i])
            smcweights[i] = weight

            weight_sum = weight_sum + weight

            #println("$j $i, $state $observation, $observation_weight")
        end

        smcweights .= smcweights ./ weight_sum

        push!(states, deepcopy(smcstates))
        push!(weights, deepcopy(smcweights))
    end

    return states, weights
end
