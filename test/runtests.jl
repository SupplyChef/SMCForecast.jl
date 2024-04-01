using Test

using SMCForecast

using CSV
using DataFrames
using Distributions
using BlackBoxOptim
using PlotlyJS
using StaticArrays
using StatsBase

@testset "Base" begin
@test begin
    system = System{SizedVector{1}}(x -> Normal(x, 1),
                                    x -> Normal(x, 1),
                                    Uniform(0, 1))
    smc = SMC{SizedVector{1}, System{SizedVector{1}}}(system, 10)

    observations = [0 + 0.01 * i for i in 1:200]
    SMCForecast.filter!(smc, observations)

    #println(observations)
    #println([(p.state, p.weight) for p in smc.particles])

    true
end

@test begin
    system_generator(θ) = begin 
                    return System{SizedVector{1}}(x -> θ * x + Normal(0, 1),
                                                  x -> x + Normal(0, 1),
                                                  Uniform(1, 1.001))
    end

    system = system_generator(0.9)
    observations = rand(system, 100)
    
    for θ in 0.5:0.1:1
        print("$θ ")
        system = system_generator(θ)
        smc = SMC{SizedVector{1}, System{SizedVector{1}}}(system, 100)
        SMCForecast.filter!(smc, observations)
    end

    true
end

@test begin
    system = System{SizedVector{1}}(x -> Normal(x, 1),
                                    x -> Normal(x, 1),
                                    Uniform(0, 1))
    smc = SMC{SizedVector{1}, System{SizedVector{1}}}(system, 10)

    observations = [0 + 0.01 * i for i in 1:200]
    SMCForecast.filter!(smc, observations)
    states, weights = SMCForecast.smooth(smc, 100)

    #println(states)
    true
end

@test begin
    system = System{SizedVector{1}}(x -> Normal(x, 1),
                                    x -> Normal(x, 1),
                                    Uniform(0, 1))
    smc = SMC{SizedVector{1}, System{SizedVector{1}}}(system, 10)

    observations = [0 + 0.01 * i for i in 1:200]
    SMCForecast.filter!(smc, observations)

    states = predict_states(smc, 10)
    #println(states)
    true
end

@test begin
    system = System{SizedVector{1}}(x -> Normal(x, 1),
                                    x -> Normal(x, 1),
                                    Uniform(0, 1))
    smc = SMC{SizedVector{1}, System{SizedVector{1}}}(system, 10)

    observations = [0 + 0.01 * i for i in 1:200]
    SMCForecast.filter!(smc, observations)

    observations = predict_observations(smc, 10)
    #println(observations)
    true
end
end

include("test_locallevelcountjump.jl")
include("test_locallevelexplanatory.jl")

# @test begin
#     using Dates

#     history = 1400
#     horizon = 500
#     scenario_count = 1
#     particle_count = 300
    
#     dates = Date(2022, 1, 1) .+ Day.(1:history)
#     values = rand(Poisson.(20), history)

#     values[month.(dates) .== 11] .+= rand(Poisson.(10), sum(month.(dates) .== 11))
#     values[month.(dates) .== 12] .+= rand(Poisson.(20), sum(month.(dates) .== 12))

#     event_dates = Date(2022, 1, 1) .+ Day.(1:history+horizon)
#     events = monthname.(event_dates)

#     values[100] += 30
#     push!(event_dates, dates[100])
#     push!(events, "promo");

#     values = values[sortperm(dates)] * 1.0
#     values = convert(Vector{Union{Missing, Float64}}, values)
#     dates = sort(dates)
    
#     future_dates = maximum(dates) .+ Day.(1:horizon+1)
    
#     event_set = Set([(event_dates[i], events[i]) for i in 1:length(event_dates)])

#     unique_events = collect(unique(events))
    
#     one_hot_events = Float64[(date, event) ∈ event_set ? 1.0 : 0.0 for date in dates, event in unique_events]
#     future_one_hot_events = Float64[(date, event) ∈ event_set ? 1.0 : 0.0 for date in future_dates, event in unique_events, k in 1:scenario_count]
    
#     all_dates = vcat(dates, future_dates)
#     exogenous = Float64[(date, event) ∈ event_set ? 1.0 : 0.0 for event in unique_events, date in all_dates]
    
#     intercept = 10.0
#     coefficients = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 30]
#     ϕ = 1.0
#     transition_variance = 0.1
#     observation_variance = 3.0
#     fcs = ForecastSystem(exogenous, intercept, coefficients, ϕ, transition_variance, observation_variance)

#     dim = 4 + size(exogenous, 1)
#     res = bboptimize(get_loss_function(fcs, values); 
#                     SearchRange = (0.0, 15.0), 
#                     NumDimensions = dim, 
#                     MaxTime = 160)

#     println(best_fitness(res))
#     println(unique_events)
#     println(best_candidate(res))

#     xs = best_candidate(res)
#     fcs2 = ForecastSystem(fcs.exogenous, 
#                               xs[1], 
#                               xs[2:(1 + size(fcs.exogenous, 1))],
#                               xs[1 + size(fcs.exogenous, 1) + 1],
#                               abs(xs[1 + size(fcs.exogenous, 1) + 2]),
#                               abs(xs[1 + size(fcs.exogenous, 1) + 3]))
#     smc = SMC{SizedVector{2}, ForecastSystem}(fcs2, particle_count)

#     likelihood = SMCForecast.filter!(smc, values)
    
#     states, weights = SMCForecast.predict(smc, horizon);

#     #println(exogenous)
#     #println(values)

#     #println(smc.states)
#     #println(smc.weights)
#     plot(vcat([scatter(;x=dates,y=values,line_color="red")],
#           [scatter(;x=future_dates, y=[states[i][j][2] for i in eachindex(states)], opacity=0.07, line_color="blue") for j in eachindex(states[1])],
#           [scatter(;x=future_dates, y=[percentile([states[i][j][2] for j in eachindex(states[i])], 90) for i in eachindex(states)])],
#           [scatter(;x=future_dates, y=[percentile([states[i][j][2] for j in eachindex(states[i])], 50) for i in eachindex(states)])],
#           [scatter(;x=future_dates, y=[percentile([states[i][j][2] for j in eachindex(states[i])], 10) for i in eachindex(states)])]
#           ))  
#     println(likelihood)

#     smc = SMC{SizedVector{2}, ForecastSystem}(fcs, particle_count)
#     values2 = copy(values)
#     values2[500:700] .= missing
#     likelihood = SMCForecast.filter!(smc, values2)
#     smoothed_states = smooth(smc, 100)
    
#     states, weights = SMCForecast.predict(smc, horizon);

#     #println(exogenous)
#     plot(vcat([scatter(;x=dates,y=values,line_color="red")],
#           [scatter(;x=future_dates, y=[states[i][j][2] for i in eachindex(states)], opacity=0.07, line_color="blue") for j in eachindex(states[1])],
#           [scatter(;x=future_dates, y=[percentile([states[i][j][2] for j in eachindex(states[i])], 90) for i in eachindex(states)])],
#           [scatter(;x=future_dates, y=[percentile([states[i][j][2] for j in eachindex(states[i])], 50) for i in eachindex(states)])],
#           [scatter(;x=future_dates, y=[percentile([states[i][j][2] for j in eachindex(states[i])], 10) for i in eachindex(states)])]
#           ))  

#     println(likelihood)
#     true
# end

#plot(vcat([scatter(;x=dates,y=values2), scatter(;x=dates,y=[mean(smoothed_states[j][i][2] for j in 1:length(smoothed_states)) for i in 1:length(dates)])],
#          [scatter(;x=dates,y=[smc.historical_states[i][j]] for j in 1:length(dates)) for i in 1:length(smc.historical_states)]))