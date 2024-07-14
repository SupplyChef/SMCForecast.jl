@testitem "LocalLevel" begin
    using SMCForecast
    using Distributions
    using Random
    using StaticArrays
    
    @test begin
        low = rand(Poisson(1), 1200)
        low[800:1100] .= rand(Poisson(0.2), 301)
        low[200] = 5 * maximum(low)

        fcs2 = SMCForecast.fit(Val{LocalLevelChange}(), low * 1.0; maxtime=30, size=200)

        smc = SMC{SizedVector{3, Float64, Vector{Float64}}, LocalLevelChange}(fcs2, 1000)
        filtered_states, loglikelihood2 = SMCForecast.filter!(smc, low * 1.0)
        println(loglikelihood2)

        smoothed_states = smooth(smc, 100);

        states, states_weights = predict_states(smc, 200)
        obs, obs_weights = predict_observations(smc, 200)

        true
    end
end