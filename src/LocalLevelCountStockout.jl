const LocalLevelCountStockout = LocalLevelCountStockoutModel{IdentityAdjustment}

function LocalLevelCountStockout(;level1,
                         level2,
                         level_matrix,
                         level_variance,
                         zero_inflation,
                         overdispersion,
                         adjust_sampling=true)
    return LocalLevelCountStockoutModel(IdentityAdjustment();
                                         level1=level1,
                                         level2=level2,
                                         level_matrix=level_matrix,
                                         level_variance=level_variance,
                                         zero_inflation=zero_inflation,
                                         overdispersion=overdispersion,
                                         adjust_sampling=adjust_sampling)
end

function forecast(::Val{LocalLevelCountStockout}, values, horizon; maxtime=10.0, size=500, forecast_percentiles=0.5)
    fcs = fit(Val{LocalLevelCountStockout}(), values; maxtime=maxtime, size=size)
    smc = SMC{MVector{3, Float64}, LocalLevelCountStockout}(fcs, 1_000)
    filter!(smc, values; record=false)
    obs, weights = predict_observations(smc, horizon)
    if isa(forecast_percentiles, Real)
        return percentiles(forecast_percentiles, obs, weights)
    else
        return [percentiles(p, obs, weights) for p in forecast_percentiles]
    end
end

function fit(::Val{LocalLevelCountStockout}, values; maxtime=10, regularization=0.0, size=100,
                                    min_overdispersion=0.00001, min_stay_outofstock_probability=0.0001,
                                    adjust_sampling=true,
                                    best_callback=nothing, rng=Random.default_rng())
    println("mean: $(mean(values)) var: $(var(values)) est: $((var(values) - (length(values) * mean(values))) / length(values))")
    xs = SMCForecast.bboptimize2(get_loss_function(Val{LocalLevelCountStockout}(), values; regularization=regularization, size=size),
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

    fcs2 = LocalLevelCountStockout(; level1=xs[1],
                            level2=xs[2],
                            level_variance=abs(xs[3]),
                            zero_inflation=abs(xs[4]),
                            overdispersion=abs(xs[5]),
                            level_matrix=[1-xs[6] xs[6];
                                          1-xs[7] xs[7]],
                            adjust_sampling=adjust_sampling)
    return fcs2
end

function get_loss_function(::Val{LocalLevelCountStockout}, values; regularization=0.0, size=1000, adjust_sampling=false)
    return xs -> begin
        fcs2 = LocalLevelCountStockout(level1=xs[1],
                              level2=xs[2],
                              level_variance=abs(xs[3]),
                              zero_inflation=abs(xs[4]),
                              overdispersion=abs(xs[5]),
                              level_matrix=[1-xs[6] xs[6];
                                            1-xs[7] xs[7]],
                              adjust_sampling=adjust_sampling)
        smc = SMC{MVector{3, Float64}, LocalLevelCountStockout}(fcs2, size)
        rng = MersenneTwister(1)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false, rng=rng)

        return -likelihood + regularization * sum(x^2 for x in xs)
    end
end
