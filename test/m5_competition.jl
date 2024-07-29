@testitem "M5" begin
    using CSV
    using DataFrames
    using JSON3
    using Random
    using StaticArrays
    using XLSX
    using SMCForecast

    println(pwd())
    # sales = CSV.read(raw"C:\Users\renau\source\repos\SMCForecast\datasets\m5-forecasting-uncertainty\sales_train_evaluation.csv", DataFrame)
    # sales2 = stack(sales, 7:1947)

    # calendar = CSV.read(raw"C:\Users\renau\source\repos\SMCForecast\datasets\m5-forecasting-uncertainty\calendar.csv", DataFrame)
    # sales3 = innerjoin(sales2, calendar, on=:variable=>:d)

    # prices = CSV.read(raw"C:\Users\renau\source\repos\SMCForecast\datasets\m5-forecasting-uncertainty\sell_prices.csv", DataFrame)
    # sales4 = innerjoin(sales3, prices, on=[:store_id, :item_id, :wm_yr_wk])

    xf =  XLSX.openxlsx(raw"../datasets/sample_data.xlsx") 
    sales4 = DataFrame(XLSX.gettable(xf["data"]))

    gd = first(groupby(sales4, :id), 30)
    historical_values = Dict(k.id => select(gd[k], :date, :value) for k in keys(gd))

    @test begin
        time_series = "HOBBIES_1_029_CA_1_evaluation"

        v = historical_values[time_series][1:end-28, :]
        t = historical_values[time_series][end-28:end, :]
        #f = forecast(Val{LocalLevelCountStockout}(), 
        #            v.value * 1.0, 
        #            nrow(t); 
        #            forecast_percentiles=[0.025, 0.5, 0.975], size=20, maxtime=30)

        f = []
        forecast_percentiles=[0.025, 0.5, 0.975]
        fcs = SMCForecast.fit(Val{LocalLevelCountStockout}(), v.value * 1.0; maxtime=130, size=300)
        println(fcs)
        smc = SMC{SizedVector{3, Float64, Vector{Float64}}, LocalLevelCountStockout}(fcs, 10)
        filtered_states, likelihood = SMCForecast.filter!(smc, v.value * 1.0; record=false)
        obs, weights = SMCForecast.predict_observations(smc, nrow(t))
        println("obs: $obs")
        println("weights: $weights")
        if isa(forecast_percentiles, Real)
            f = SMCForecast.percentiles(forecast_percentiles, obs, weights)
        else
            f = [SMCForecast.percentiles(p, obs, weights) for p in forecast_percentiles]
        end

        true
    end 

#     @test begin
#         s = raw"""{
#                     "level1": 0.3124214750820895,
#                     "level2": 0.028502217874561857,
#                     "level_matrix": [
#                                         0.920018575346042,
#                                         0.1964475107783209,
#                                         0.07998142465395798,
#                                         0.8035524892216791
#                                     ],
#                             "levels": [
#                                         1,
#                                         2
#                                     ],
#                     "level_weights": [
#                                         [
#                                         0.920018575346042,
#                                         0.07998142465395798
#                                         ],
#                                         [
#                                         0.1964475107783209,
#                                         0.8035524892216791
#                                         ]
#                                     ],
#                     "level_weights10": [
#                                         0.7220438667626469,
#                                         0.2779561332373533
#                                     ],
#                     "level_variance": 3.328419783102148e-5,
#             "observation_variance": 0.7987808522272768
# }"""        

#     model = JSON3.read(s)
                
#     fcs = LocalLevelCountStockout(model[:level1], model[:level2], reshape(collect(model[:level_matrix]), (2,2)), model[:level_variance], model[:observation_variance])
#     smc = SMC{SizedVector{3, Float64, Vector{Float64}}, LocalLevelCountStockout}(fcs, 500)

#     rng = Random.default_rng()
#     states, weights = predict_states(smc, horizon; happy_only=true, rng=rng)
#     #println("state mean: $(avg(states, weights))")
#     #println("state median: $(percentiles(0.5, states, weights))")
#     observations = [[sample_observation(smc.system, state) for state in states[i]] for i in 1:length(states)]
#     #println("obs mean: $(avg(observations, weights))")
#     #println("obs median: $(percentiles(0.5, observations, weights))")
#     end
end