@testset "LocalLevelExplanatory" begin
    @test begin
        drivers = CSV.read(raw"..\datasets\front_rear_seat.csv", DataFrame);
        exogenous = [mod1(j, 12) == i for i in 2:12, j in 1:length(drivers.british_drivers_KSI)+200] * 1.0

        fcs2 = SMCForecast.fit(Val{LocalLevelExplanatory}(), exogenous, drivers.british_drivers_KSI; regularization=0.01, maxtime=60)

        smc = SMC{SizedVector{2, Float64, Vector{Float64}}, LocalLevelExplanatory}(fcs2, 1000)
        filtered_states, loglikelihood = SMCForecast.filter!(smc, drivers.british_drivers_KSI)
        println(loglikelihood)

        smoothed_states = smooth(smc, 100);

        states, states_weights = predict_states(smc, 199)
        obs, obs_weights = predict_observations(smc, 199)

        true
    end
end