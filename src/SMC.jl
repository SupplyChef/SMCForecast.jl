using Random

"""
Particle filter
"""
mutable struct SMC{T <: SizedVector, U <: SMCSystem{T}} 
    system::U
    
    states::Array{T, 1}
    weights::Array{Float64, 1}

    observation_likelihood

    historical_states::Array{Array{T, 1}, 1}
    historical_weights::Array{Array{Float64, 1}, 1}

    historical_proposed_states::Array{Array{T, 1}, 1}
    historical_proposed_weights::Array{Array{Float64, 1}, 1}

    function SMC{T, U}(system::U, count::Int64) where U <: SMCSystem{T} where T <: SizedVector
        smc = new{T, U}(system, 
                        T[T(zeros(size(T, 1))) for i in 1:count], 
                        zeros(Float64, count), 
                        Array{T, 1}[], 
                        Array{Float64, 1}[], 
                        Array{T, 1}[], 
                        Array{Float64, 1}[])
        return smc
    end
end

"""
    initialize!(smc::SMC{T, U}) where {T <: SizedVector, U <: SMCSystem{T}} 

Initializes the particle filter.
"""
function initialize!(smc::SMC{T, U}; rng=Random.default_rng()) where {T <: SizedVector, U <: SMCSystem{T}} 
    smc.states .= sample_initial_state(smc.system, length(smc.states); rng=rng)
    #smc.weights .= repeat([1.0 / length(smc.states)], length(smc.states))
    smc.weights .= repeat([1.0], length(smc.weights))
end

"""
    filter!(smc::SMC{T, U}, observations::Array{Float64, 1}; record=true) where {T <: SizedVector, U <: SMCSystem{T}}

Compute the filtered distribution.
"""
function filter!(smc::SMC{T, U}, observations::Array{Float64, 1}; record=true) where {T <: SizedVector, U <: SMCSystem{T}} 
    observations = convert(Array{Union{Float64, Missing}, 1}, observations)
    return filter!(smc, observations; record=record)
end

"""
    filter!(smc::SMC{T, U}, observations::Array{Union{Float64, Missing}, 1}; record=true) where {T <: SizedVector, U <: SMCSystem{T}}

Compute the filtered distribution.
"""
function filter!(smc::SMC{T, U}, observations::Array{Union{Float64, Missing}, 1}; record=true) where {T <: SizedVector, U <: SMCSystem{T}} 
    historical_states = Array{T, 1}[]
    historical_weights = Array{Float64, 1}[]
    historical_proposed_states = Array{T, 1}[]
    historical_proposed_weights = Array{Float64, 1}[]
    
    #initialize!(smc)
    #if record
    #    push!(historical_states, [copyto!(T(zeros(size(T, 1))), state) for state in smc.states])
    #    push!(historical_weights, deepcopy(smc.weights))
    #end

    filtered_states = []
    smc.observation_likelihood = zeros(length(observations))
    
    new_states = T[T(zeros(size(T, 1))) for i in 1:length(smc.states)]
    sampling_probabilities = [1.0 for i in 1:length(smc.states)]
    for (j, observation) in enumerate(observations)
        weight_sum::Float64 = 0.0

        if j == 1
            new_states = sample_initial_state(smc.system, length(new_states))
            smc.weights .= 1.0
            sampling_probabilities .= 1.0
        else
            sample_states(smc.system, smc.states, observation, new_states, sampling_probabilities)
            #println("New states: $new_states")
            #println("New states sampling probabilities: $sampling_probabilities")
        end

        for i in eachindex(smc.states)
            new_state_observation_probability = 1.0
            if !ismissing(observation)
                @inbounds new_state_observation_probability = observation_probability(smc.system, new_states[i], observation)
            end
            
            observation_weight::Float64 = sampling_probabilities[i] * new_state_observation_probability

            if isinf(sampling_probabilities[i])
                #println("Warning: Inf")
                observation_weight = new_state_observation_probability
            end
            if isnan(observation_weight) || isinf(observation_weight) || (observation_weight == 0)
                #println("Problem with observation weight: $(smc.states[i]) -> $(new_states[i]); $(smc.weights[i]), $new_state_observation_probability, $observation")
                observation_weight = 0
            end

            weight::Float64 = smc.weights[i] * observation_weight
            
            @inbounds copyto!(smc.states[i], new_states[i])
            #println("$(Base.objectid(smc.states[i])) $(Base.objectid(new_state))")
            smc.weights[i] = weight
            #println("$(smc.states[i]) $(smc.weights[i])")

            weight_sum = weight_sum + weight

            #println("$j $i, $state $observation, $observation_weight")
        end

        #println("step: $j, likelihood: $weight_sum")
        smc.observation_likelihood[j] = weight_sum

        #println("Weights: $(smc.weights)")

        smc.weights .= smc.weights ./ weight_sum .* length(smc.weights)

        #if record
        #    push!(historical_proposed_states, [copyto!(T(zeros(size(T, 1))), state) for state in smc.states])
        #    push!(historical_proposed_weights, deepcopy(smc.weights))
        #end

        multinomial_resample!(smc)

        push!(filtered_states, average_state(smc.system, smc.states, smc.weights ./ length(smc.weights)))

        #println([(p.state, p.weight) for p in smc.particles])
        #println(sum(p.state * p.weight for p in smc.particles))
    
        if record
            push!(historical_states, [copyto!(T(zeros(size(T, 1))), state) for state in smc.states])
            push!(historical_weights, deepcopy(smc.weights))
        end
    end
    #println(sum(log.(likelihood ./ length(smc.states))))

    if record
        smc.historical_states = historical_states
        smc.historical_weights = historical_weights
        smc.historical_proposed_states = historical_proposed_states
        smc.historical_proposed_weights = historical_proposed_weights
    end

    #return filtered_states, sum(log.(smc.observation_likelihood ./ length(smc.states)))
    return filtered_states, sum(log.(smc.observation_likelihood) .- 1 * log(length(smc.states)))
end

function multinomial_resample!(smc::SMC{T, U}) where {T <: SizedVector, U <: SMCSystem{T}} 
    #effective_size = 1 / sum(weight^2 for weight in smc.weights)
    effective_size = length(smc.weights)^2 / sum(weight^2 for weight in smc.weights)

    if effective_size < length(smc.states) / 4
        # avoid memory allocations by re-using the vectors that have not been resampled
        states = 1:length(smc.states)
        samples = Array{Int64}(undef, length(smc.states))
        sampled = zeros(Int64, length(smc.states))
        
        StatsBase.sample!(1:length(states), pweights(smc.weights), samples)
        for sample in samples
            sampled[sample] = sampled[sample] + 1
        end

        frees = Deque{Int64}()
        for state in states
            if sampled[state] == 0
                push!(frees, state)
            end
        end

        for state in states
            while sampled[state] > 1
                sampled[state] = sampled[state] - 1
                free = pop!(frees)
                copyto!(smc.states[free], smc.states[state])
            end
        end
        #smc.weights .= 1 / length(states)
        smc.weights .= 1
    end
end

function smooth(smc::SMC{T, U}, count::Int64) where {T <: SizedVector, U <: SMCSystem{T}} 
    if isnothing(smc.historical_states) || isnothing(smc.historical_weights)
        throw(ErrorException("The filter! method must be called first with record parameter set to true."))
    end

    all_smoothed_states = []
    weights = zeros(Float64, length(smc.historical_weights[1]))
    for c in 1:count
        smoothed_states = []
        
        smoothed_state = StatsBase.sample(smc.historical_states[end], pweights(smc.historical_weights[end]))
        push!(smoothed_states, smoothed_state)
        for i in (length(smc.historical_states)-1):-1:1
            #TODO: pass in the observation instead of missing
            weights .= smc.historical_weights[i] .* [transition_probability(smc.system, state, missing, smoothed_state) for state in smc.historical_states[i]]
            smoothed_state = StatsBase.sample(smc.historical_states[i], pweights(weights))
            push!(smoothed_states, smoothed_state)
        end

        reverse!(smoothed_states)
        push!(all_smoothed_states, smoothed_states)
    end
    return all_smoothed_states
end

function bboptimize2(f, x0, params; quiet=false, pool_size=20, best_callback=nothing)
    start = Dates.now()
    latest = start

    best_f = f(x0)
    best_x = copy(x0)

    if !quiet
        #println(x0)
        #println("** $best_f")
        #println(params)
    end
    
    last_progress = 0
    
    candidate_pool = vcat([x0], [[rand() .* (params[:SearchRange][j][2] - params[:SearchRange][j][1]) .+ params[:SearchRange][j][1] for j in 1:length(x0)] for i in 1:pool_size-1])
    #candidate_pool = [[rand() .* (params[:SearchRange][j][2] - params[:SearchRange][j][1]) .+ params[:SearchRange][j][1] for j in 1:length(x0)] for i in 1:pool_size]
    #println(candidate_pool)
    pool_f = [f(candidate) for candidate in candidate_pool]
    #println(pool_f)

    t = max(0.1, min(0.9, 6 / length(x0)))
    #println(t)

    for i in 1:get(params, :MaxFuncEvals, typemax(Int64))
        if i > last_progress + get(params, :MaxStepsWithoutProgress, Inf) || (Dates.now() - start) > Second(get(params, :MaxTime, typemax(Int64)))
            println("$i, $(Dates.now() - start), $best_f, $best_x")
            break
        end

        i1 = rand(1:pool_size)
        i2 = rand(1:pool_size)
        i3 = rand(1:pool_size)

        candidate = copy(candidate_pool[i1])
        @inbounds for j in eachindex(candidate)
            r = rand()
            if r < 0.01
                candidate[j] = params[:SearchRange][j][1]
            elseif r < 0.02
                candidate[j] = candidate_pool[i1][j] + 0.1 * (randn())
            elseif r < 0.08
                candidate[j] = candidate_pool[i1][j] + (candidate_pool[i2][j]-candidate_pool[i3][j]) * (randn())
                #candidate[k] = candidate_pool[i1][j] + 1 * (randn())
            elseif r < 0.12
                candidate[j] = candidate_pool[i2][j] + 0.1 * (randn())
            elseif r < t + 0.12
                candidate[j] = candidate_pool[i1][j] + (rand() + 0.2)^2 * (best_x[j] - candidate_pool[i3][j])
            end

            if candidate[j] < params[:SearchRange][j][1]
                candidate[j] = params[:SearchRange][j][1] + rand()^3 * (params[:SearchRange][j][2] - params[:SearchRange][j][1])
            end
            if candidate[j] > params[:SearchRange][j][2]
                candidate[j] = params[:SearchRange][j][2] - rand()^3 * (params[:SearchRange][j][2] - params[:SearchRange][j][1])
            end
        end
        candidate_f = 0
        try 
            candidate_f = f(candidate)
        catch e
            println(candidate)
            rethrow(e)
        end
        #println("$i, $(Dates.now() - start), $candidate_f, $candidate")
        if (candidate_f ≈ pool_f[i1]) && (sum(candidate_f) < sum(pool_f[i1]))
            pool_f[i1] = candidate_f
            candidate_pool[i1] = candidate
        end
        if candidate_f < pool_f[i1] || isnan(pool_f[i1])
            pool_f[i1] = candidate_f
            candidate_pool[i1] = candidate
        end
        if (candidate_f ≈ best_f) && (sum(candidate_f) < sum(best_f))
            best_f = candidate_f
            best_x = copy(candidate)
            if !quiet
                #println("*- $i, $(Dates.now() - start), $best_f, $best_x")
            end
        end
        if candidate_f < best_f || isnan(best_f)
            best_f = candidate_f
            best_x = copy(candidate)
            last_progress = i
            if !isnothing(best_callback)
                try
                    best_callback(best_f, best_x)
                catch
                end
            end
            if !quiet
                println("** $i, $(Dates.now() - start), $best_f, $best_x")
            end
        end

        #if i % 50 == 0
        #    if !quiet
        #        println("$i, $(Dates.now() - start), $(Dates.now() - latest), $best_f")#, $best_x")
        #        for k in 1:length(candidate_pool)
        #            println("$(pool_f[k]), $(candidate_pool[k])")
        #        end
        #    end
        #    latest = Dates.now()
        #end
    end

    #if !quiet        
    #    for k in 1:length(candidate_pool)
    #        println("$(pool_f[k]), $(candidate_pool[k])")
    #    end
    #end
    
    return best_x
end