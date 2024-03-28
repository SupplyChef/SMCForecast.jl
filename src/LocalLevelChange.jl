struct LocalLevelChange <: SMCSystem{SizedVector{3, Float64, Vector{Float64}}}
    level::Float64
    change::Float64
    
    level_variance::Float64
    change_variance::Float64
    observation_variance::Float64

    prior_distribution

    mixture_distribution::MixtureModel{Univariate, Distributions.Continuous, Distribution{Univariate, Distributions.Continuous}, Categorical{Float64, Vector{Float64}}}

    function LocalLevelChange(last_level, last_change, level_variance, change_variance, observation_variance, use_mixture=false)
        new(last_level, last_change, 
            level_variance, min(change_variance, level_variance), observation_variance,
            () -> Uniform(last_level-0.0001, last_level+0.0001), 
            MixtureModel([Normal(0, sqrt(level_variance)), Uniform(-25000, 25000)], [0.999, 0.001])
            #MixtureModel([Normal(0, sqrt(level_variance)), Uniform(-0.0001, 0.0001)], [0.99, 0.01])
            )
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
                    [values[1], values[2]-values[1], var(values) / length(values), 0.01, 0.01],
                    Dict(
                        :SearchRange => [(minimum(values), maximum(values)), (-10, +10), (0.0001, var(values)), (0.0001, var(values) / 1000), (0.0001, var(values))], 
                        :NumDimensions => dim, 
                        :MaxStepsWithoutProgress => 15000,
                        :MaxTime => maxtime); quiet=false
            )
    #println(best_fitness(res))
    #println(best_candidate(res))
    #xs = best_candidate(res)
    
    # res = optimize(get_loss_function(fcs, values), 
    #                [values[1], var(values), var(values)], 
    #                Optim.NelderMead(),
    #                Optim.Options( f_tol=1e-6, g_tol=1e-6, iterations = 10000, show_trace = true))
    # xs = Optim.minimizer(res)
    
    fcs2 = LocalLevelChange(xs[1], xs[2], abs(xs[3]), abs(xs[4]), abs(xs[5]))
    return fcs2
end

function get_loss_function(::Val{LocalLevelChange}, values; size=100, regularization=0.0)
    return xs -> begin
        fcs2 = LocalLevelChange(xs[1], xs[2], abs(xs[3]), abs(xs[4]), abs(xs[5]), true)
        smc = SMC{SizedVector{3, Float64, Vector{Float64}}, LocalLevelChange}(fcs2, size)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false)
        return -likelihood + regularization * sum(x^2 for x in xs)
    end
end

function sample_initial_state(system::LocalLevelChange, count)
    states = [SizedVector{3, Float64, Vector{Float64}}(0, r, 0.0) for r in rand(system.prior_distribution(), count)]
    return states
end

function sample_states(system::LocalLevelChange, 
                      current_states::Vector{SizedVector{3, Float64, Vector{Float64}}}, next_observation::Union{Missing, Float64}, 
                      new_states::Vector{SizedVector{3, Float64, Vector{Float64}}}, sampling_probabilities::Array{Float64, 1})
    time::Int64 = Int(current_states[1][1])
    
    finish::Int64 = length(new_states)
    
    level_n = Normal(0, sqrt(system.level_variance))
    #level_ϵs::Array{Float64, 1} = Float64[]
    if ismissing(next_observation)
        #level_ϵs = rand(level_n, finish)
        sampling_probabilities .= rand(level_n, finish)
    else
        level_n = system.mixture_distribution
        level_ns::Array{Distribution{Univariate, Distributions.Continuous}, 1} = sample(components(system.mixture_distribution), pweights(probs(system.mixture_distribution)), finish)
        #level_ϵs = rand.(level_ns)
        sampling_probabilities .= rand.(level_ns)
    end

    change_n = Normal(0, sqrt(system.change_variance))
    change_ϵs::Array{Float64, 1} = rand(change_n, finish)

    @inbounds for i in 1:finish
        level::Float64 = current_states[i][2]
        change::Float64 = current_states[i][3]

        new_states[i][1] = time + 1
        #new_states[i][2] = level + change + level_ϵs[i]
        new_states[i][2] = level + change + sampling_probabilities[i]
        new_states[i][3] = change + change_ϵs[i]
    end

    if  ismissing(next_observation)
        @inbounds sampling_probabilities .= pdf.(level_n, sampling_probabilities) .* pdf.(change_n, change_ϵs)
    else #!ismissing(next_observation)
        #p::Array{Float64, 1} = probs(system.mixture_distribution)
        #n::Normal = components(system.mixture_distribution)[1]
        #u::Uniform = components(system.mixture_distribution)[2]
        #@inbounds sampling_probabilities .= (p[1] .* pdf.(n, sampling_probabilities) .+ 
        #                                     p[2] .* pdf.(u, sampling_probabilities)) .* pdf.(change_n, change_ϵs) 
        @inbounds sampling_probabilities .= 0.999 .* pdf.(Normal(0, sqrt(system.level_variance)), sampling_probabilities) .+ 0.001 .* pdf.(Uniform(-25000, 25000), sampling_probabilities) 
    end

    if !ismissing(next_observation)
        new_states[1][1] = time + 1
        new_states[1][2] = next_observation
        new_states[1][3] = current_states[1][3]
        sampling_probabilities[1] = 1.0 / length(current_states)
    end
end

function sample_observation(system::LocalLevelChange, current_state::SizedVector{3})
    value::Float64 = current_state[2]
    n = Normal(value, sqrt(system.observation_variance))
    return rand(n)
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

    level_p = 0.0
    if ismissing(new_observation)
        level_p = pdf(Normal(0, sqrt(system.level_variance)), new_level - (level + change))
    else #!ismissing(new_observation)
        #p::Array{Float64, 1} = probs(system.mixture_distribution)
        #n::Normal = components(system.mixture_distribution)[1]
        #u::Uniform = components(system.mixture_distribution)[2]
        #n::Normal = components(system.mixture_distribution)[1]
        level_p = 0.999 * pdf(Normal(0, sqrt(system.level_variance)), new_level - (level + change)) + 0.001 * pdf(Uniform(-25000, 25000), new_level - (level + change))
        #level_p = p[1] * pdf(n, new_level - (level + change)) + p[2] * pdf(u, new_level - (level + change)) 
    end
    
    return  level_p * 
            Distributions.normpdf(change, 
                                 sqrt(system.change_variance), 
                                 new_change)
end

function observation_probability(system::LocalLevelChange, state::SizedVector{3, Float64, Vector{Float64}}, observation::Float64)::Float64
    value::Float64 = state[2]
    t = Normal(value, sqrt(system.observation_variance))
    return pdf(t, observation)
end

function average_state(system::LocalLevelChange, states, weights)
    return SizedVector{3, Float64, Vector{Float64}}([states[1][1], sum(states[i][2] * weights[i] for i in eachindex(weights)), sum(states[i][3] * weights[i] for i in eachindex(weights))])
end