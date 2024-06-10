sales = CSV.read(raw"C:\Users\renau\Downloads\m5-forecasting-uncertainty\sales_train_evaluation.csv", DataFrame)
sales2 = stack(sales, 7:1947)

calendar = CSV.read(raw"C:\Users\renau\Downloads\m5-forecasting-uncertainty\calendar.csv", DataFrame)
sales3 = innerjoin(sales2, calendar, on=:variable=>:d)

prices = CSV.read(raw"C:\Users\renau\Downloads\m5-forecasting-uncertainty\sell_prices.csv", DataFrame)
sales4 = innerjoin(sales3, prices, on=[:store_id, :item_id, :wm_yr_wk])

gd = first(groupby(sales4, :id), 30)
historical_values = Dict(k.id => select(gd[k], :date, :value) for k in keys(gd))

@test begin
    time_series = "HOBBIES_1_029_CA_1_evaluation"

    v = historical_values[time_series][1:end-28, :]
    t = historical_values[time_series][end-28:end, :]
    #f = forecast(Val{LocalLevelJump}(), 
    #            v.value * 1.0, 
    #            nrow(t); 
    #            forecast_percentiles=[0.025, 0.5, 0.975], size=20, maxtime=30)

    f = []
    forecast_percentiles=[0.025, 0.5, 0.975]
    fcs = SMCForecast.fit(Val{LocalLevelJump}(), v.value * 1.0; maxtime=130, size=300)
    println(fcs)
    smc = SMC{SizedVector{3, Float64}, LocalLevelJump}(fcs, 10)
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