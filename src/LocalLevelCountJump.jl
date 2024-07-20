
struct LocalLevelCountJump <: SMCSystem{SizedVector{3, Float64, Vector{Float64}}}
    level1::Float64
    level2::Float64
    
    level_matrix::Array{Float64, 2}
    levels::Array{Int64, 1}
    
    level_variance::Float64

    zero_inflation::Float64
    overdispersion::Float64

    # the members below are used to speed up computation
    level2_exp::Float64

    level_weights::Array{ProbabilityWeights, 1}
    level_weights10::ProbabilityWeights
    level_equal_weights::ProbabilityWeights

    adjust_sampling::Bool

    function LocalLevelCountJump(;level1, 
                             level2, 
                             level_matrix, 
                             level_variance, 
                             zero_inflation,
                             overdispersion,
                             adjust_sampling=true)
        levels = [1, 2]
        level_weights = [pweights(level_matrix[i,:]) for i in 1:size(level_matrix, 1)]
        level_weights10 = pweights((level_matrix^10)[1,:])
        level_equal_weights = pweights([0.5, 0.5])

        new(level1, 
            level2, 
            level_matrix, 
            levels, 
            level_variance, 
            zero_inflation,
            overdispersion, 
            exp(-level2), level_weights, level_weights10, level_equal_weights, 
            adjust_sampling)
    end
end

function forecast(::Val{LocalLevelCountJump}, values, horizon; maxtime=10.0, size=500, forecast_percentiles=0.5)
    fcs = fit(Val{LocalLevelCountJump}(), values; maxtime=maxtime, size=size)
    smc = SMC{SizedVector{3, Float64}, LocalLevelCountJump}(fcs, 1_000)
    filter!(smc, values; record=false)
    obs, weights = predict_observations(smc, horizon)
    if isa(forecast_percentiles, Real)
        return percentiles(forecast_percentiles, obs, weights)
    else
        return [percentiles(p, obs, weights) for p in forecast_percentiles]
    end
end

function fit(::Val{LocalLevelCountJump}, values; maxtime=10, regularization=0.0, size=100, 
                                    min_overdispersion=0.00001, min_stay_outofstock_probability=0.0001,
                                    adjust_sampling=true,
                                    best_callback=nothing, rng=Random.default_rng())
    println("mean: $(mean(values)) var: $(var(values)) est: $((var(values) - (length(values) * mean(values))) / length(values))")                
    xs = SMCForecast.bboptimize2(get_loss_function(Val{LocalLevelCountJump}(), values; regularization=regularization, size=size),
                    [mean(values), 
                     0.00001, 
                     max((var(values) - (length(values) * mean(values))) / length(values),  0.001), 
                     0.0, 
                     0.0, 
                     0.1, 
                     max(min_stay_outofstock_probability, 0.9)],
                    Dict(
                        :SearchRange => [(0, maximum(values)), 
                                        (0.00001, mean(values) / 5), 
                                        (0.00001, var(values) / length(values)),
                                        (0.00001, .9999),
                                        (min_overdispersion, .9999), 
                                        (0.0001, .9999), 
                                        (min_stay_outofstock_probability, .9999)], 
                        :NumDimensions => 7, 
                        :MaxTime => maxtime,
                        :MaxStepsWithoutProgress => 2000),
                    best_callback = best_callback,
                    rng=rng
                    )
    
    fcs2 = LocalLevelCountJump(; level1=xs[1], 
                            level2=xs[2],
                            level_variance=abs(xs[3]), 
                            zero_inflation=abs(xs[4]),
                            overdispersion=abs(xs[5]),
                            level_matrix=[1-xs[6] xs[6]; 
                                          1-xs[7] xs[7]],
                            adjust_sampling=adjust_sampling)
    return fcs2
end

function get_loss_function(::Val{LocalLevelCountJump}, values; regularization=0.0, size=1000, adjust_sampling=false)
    return xs -> begin
        fcs2 = LocalLevelCountJump(level1=xs[1], 
                              level2=xs[2],
                              level_variance=abs(xs[3]), 
                              zero_inflation=abs(xs[4]),
                              overdispersion=abs(xs[5]), 
                              level_matrix=[1-xs[6] xs[6]; 
                                            1-xs[7] xs[7]],
                              adjust_sampling=adjust_sampling)
        smc = SMC{SizedVector{3, Float64, Vector{Float64}}, LocalLevelCountJump}(fcs2, size)
        rng = MersenneTwister(1)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false, rng=rng)
        return -likelihood + regularization * sum(x^2 for x in xs)
    end
end

function sample_initial_state(system::LocalLevelCountJump, count; rng=Random.default_rng())
    states = sample(rng, [1,2], system.level_weights10, count)
    return [SizedVector{3, Float64, Vector{Float64}}(1.0, system.level1, states[i]) for i in eachindex(states)]
end

function sample_states(system::LocalLevelCountJump, 
                       current_states::Vector{SizedVector{3, Float64, Vector{Float64}}}, 
                       next_observation::Union{Missing, Float64}, 
                       new_states, sampling_probabilities; happy_only=false, rng=Random.default_rng())
    time = Int(current_states[1][1])

    for (i, current_state) in enumerate(current_states)
        value = current_state[2]
        state = Int(current_state[3])
        
        sampling_probabilities[i] = 1

        n = Normal(0, sqrt(system.level_variance))

        #new_state = sample(rng, system.levels, system.level_weights[state])
        @inbounds new_state = (rand(rng) > system.level_matrix[state, 1]) + 1
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
        new_states[i][2] = new_value
        new_states[i][3] = new_state
    end
end

function sample_observation(system::LocalLevelCountJump, current_state::SizedVector{3}; rng=Random.default_rng())
    value::Float64 = current_state[2]
    state = Int(current_state[3])

    if state == 2
        return rand(rng, Poisson(system.level2))
    end

    value = value * (1 - system.overdispersion) / (1 - system.zero_inflation)
    return sample_zigp(value, system.overdispersion, system.zero_inflation)
end

function transition_probability(system::LocalLevelCountJump, state1::SizedVector{3, Float64, Vector{Float64}}, new_observation, state2::SizedVector{3, Float64, Vector{Float64}})::Float64
    #time = state1[1]
    value = state1[2]
    state = Int(state1[3])

    new_value = state2[2]
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

function observation_probability(system::LocalLevelCountJump, current_state::SizedVector{3, Float64, Vector{Float64}}, current_observation)::Float64
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

function average_state(system::LocalLevelCountJump, states, weights)
    return SizedVector{3, Float64, Vector{Float64}}([states[1][1], 
                           sum(states[i][2] * weights[i] for i in eachindex(weights)),
                           sum(states[i][3] * weights[i] for i in eachindex(weights))])
end

function negative_binomial_pmf(r, p, logp, logonep, current_observation::Int64)
    return binomial_coefficient(current_observation + r - 1, current_observation) * exp(logp*r + logonep*current_observation)
end

function binomial_coefficient(n::Float64, k::Int64)::Float64
    if k == 0
        return 1.0
    end
    binomial_coefficient = n
    @inbounds for i in 2:k
        binomial_coefficient *= (n + 1 - i) / i
    end
    return binomial_coefficient
end

# Compute the log of the PMF of the Generalized Poisson Distribution
function log_generalized_poisson_pmf(k::Int, lambda::Float64, theta::Float64)::Float64
    if k == 0
        return -lambda
    elseif k == 1
        log_lambda = log(lambda)
        log_term3 = -(lambda + theta)

        log_pmf = log_lambda + log_term3
        return log_pmf 
    else
        log_lambda = log(lambda)
        log_term2 = (k - 1) * log(lambda + k * theta)
        log_term3 = -(lambda + k * theta)
        
        #TODO: precompute the factorial
        log_k_factorial = logfactorial(k)
        
        log_pmf = log_lambda + log_term2 + log_term3 - log_k_factorial
        return log_pmf
    end
end

# Compute the PMF of the Zero-Inflated Generalized Poisson Distribution
function zigp_pmf(k::Int, lambda::Float64, theta::Float64, pi::Float64)
    if k == 0
        p_zero = pi + (1 - pi) * exp(-lambda)
        return p_zero
    else
        p_k = (1 - pi) * exp(log_generalized_poisson_pmf(k, lambda, theta))
        return p_k
    end
end

function sample_zigp(lambda::Float64, theta::Float64, pi::Float64; rng=Random.default_rng())
    u = rand(rng)

    k = 0
    cum_pmf = zigp_pmf(k, lambda, theta, pi)
    while cum_pmf <= u && k <= 1000
        k = k+1
        cum_pmf += zigp_pmf(k, lambda, theta, pi)
        #println("$k $cum_pmf $u")
    end
    return k
end
