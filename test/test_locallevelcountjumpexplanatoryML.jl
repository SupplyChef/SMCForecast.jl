@testitem "LocalLevelCountJumpExplanatoryML" begin
    using SMCForecast
    using CSV
    using DataFrames
    using Distributions
    using MLJ
    using Random
    using StaticArrays

    @test begin
        low = rand(Poisson(1), 500) * 1.0
        #low[800:1100] .= rand(Poisson(0.2), 301)
        #low[200] = 5 * maximum(low)
        
        exogenous = [mod1(j, 12) == i for i in 2:12, j in 1:length(low)] * 1.0

        fcs = SMCForecast.fit(Val{LocalLevelCountJump}(), low[1:20]; maxtime=5)

        f = SMCForecast.get_loss_function(Val{LocalLevelCountJump}(), low; size=10)

        DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
        model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10)
        mach = MLJ.machine(model, table(exogenous'), low) |> MLJ.fit!

        leaves = SMCForecast.get_leaves(mach.fitresult[1])

        g = SMCForecast.get_loss_function(Val{LocalLevelCountJumpExplanatoryML}(), exogenous, low, mach; regularization=0.0, size=10)

        f1= f([ fcs.level1, 
            fcs.level2, 
            fcs.level_variance,
            fcs.zero_inflation, 
            fcs.overdispersion, 
            fcs.level_matrix[1, 2], 
            fcs.level_matrix[2, 2]])

        g1 = g(vcat([fcs.level1, 
                fcs.level2, 
                fcs.level_variance,
                fcs.zero_inflation, 
                fcs.overdispersion, 
                fcs.level_matrix[1, 2], 
                fcs.level_matrix[2, 2]],
                [0.0 for leaf in leaves]
            ))

        println("$f1 vs $g1")
        f1 ≈ g1
    end

    @test begin
        low = rand(Poisson(1), 1200) * 1.0
        low[800:1100] .= rand(Poisson(0.2), 301)
        low[200] = 5 * maximum(low)
        
        exogenous = [mod1(j, 12) == i for i in 2:12, j in 1:length(low)] * 1.0

        fcs2 = SMCForecast.fit(Val{LocalLevelCountJumpExplanatoryML}(), exogenous, low[1:1000]; regularization=0.01, maxtime=60)

        smc = SMC{SizedVector{3, Float64, Vector{Float64}}, LocalLevelCountJumpExplanatoryML}(fcs2, 1000)
        filtered_states, loglikelihood = SMCForecast.filter!(smc, low[1:1000])
        println(loglikelihood)

        smoothed_states = smooth(smc, 100);

        states, states_weights = predict_states(smc, 199)
        obs, obs_weights = predict_observations(smc, 199)

        true
    end
end