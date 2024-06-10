@testitem "LocalLevelCountJumpExplanatory" begin
    using SMCForecast
    using CSV
    using DataFrames
    using Distributions
    using Random
    using StaticArrays

    @test begin
        low = rand(Poisson(1), 1200) * 1.0
        low[800:1100] .= rand(Poisson(0.2), 301)
        low[200] = 5 * maximum(low)
        
        exogenous = [mod1(j, 12) == i for i in 2:12, j in 1:length(low)+200] * 1.0

        fcs2 = SMCForecast.fit(Val{LocalLevelCountJumpExplanatory}(), exogenous, low; regularization=0.01, maxtime=60)

        smc = SMC{SizedVector{3, Float64}, LocalLevelCountJumpExplanatory}(fcs2, 1000)
        filtered_states, loglikelihood = SMCForecast.filter!(smc, low)
        println(loglikelihood)

        smoothed_states = smooth(smc, 100);

        states, states_weights = predict_states(smc, 199)
        obs, obs_weights = predict_observations(smc, 199)

        true
    end
end