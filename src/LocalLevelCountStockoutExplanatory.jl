const LocalLevelCountStockoutExplanatory = LocalLevelCountStockoutModel{LinearAdjustment}

function LocalLevelCountStockoutExplanatory(;exogenous, level1, level2, level_matrix, coefficients, level_variance, zero_inflation, overdispersion, adjust_sampling=false)
    return LocalLevelCountStockoutModel(LinearAdjustment(exogenous, coefficients);
                                         level1=level1,
                                         level2=level2,
                                         level_matrix=level_matrix,
                                         level_variance=level_variance,
                                         zero_inflation=zero_inflation,
                                         overdispersion=overdispersion,
                                         adjust_sampling=adjust_sampling,
                                         # LocalLevelCountStockoutExplanatory has historically sampled its initial
                                         # hidden state from a fixed [0.9, 0.1] prior (unlike LocalLevelCountStockout
                                         # and LocalLevelCountStockoutExplanatoryML, which derive it from
                                         # level_matrix^10) and has not applied the exogenous adjustment to its
                                         # initial value. Preserved here rather than "fixed" so this refactor does
                                         # not change existing fitted behavior.
                                         initial_state_weights=pweights([0.9, 0.1]),
                                         adjust_initial_value=false)
end

function forecast(::Val{LocalLevelCountStockoutExplanatory}, exogenous, values, horizon; maxtime=10.0, size=500, forecast_percentiles=0.5)
    fcs = fit(Val{LocalLevelCountStockoutExplanatory}(), exogenous, values; maxtime=maxtime, size=size)
    smc = SMC{MVector{3, Float64}, LocalLevelCountStockoutExplanatory}(fcs, 1_000)
    filter!(smc, values; record=false)
    obs, weights = predict_observations(smc, horizon)
    if isa(forecast_percentiles, Real)
        return percentiles(forecast_percentiles, obs, weights)
    else
        return [percentiles(p, obs, weights) for p in forecast_percentiles]
    end
end

function fit(::Val{LocalLevelCountStockoutExplanatory}, exogenous, values; maxtime=10, regularization=0.0, size=100,
                                                                        min_stay_outofstock_probability=0.0001)
    loss_function = get_loss_function(Val{LocalLevelCountStockoutExplanatory}(), exogenous, values; regularization=regularization, size=size)

    dim = 7 + Base.size(exogenous, 1)

    xs = bboptimize2(loss_function,
                    vcat([values[1],
                        0.00001,
                        max((var(values) - (length(values) * mean(values))) / length(values),  0.001),
                        0.0,
                        0.0,
                        0.1,
                        max(0.00001, 0.9)],
                        zeros(Base.size(exogenous, 1))
                    ),
                    Dict(:SearchRange => vcat([(0, maximum(values)),
                                                (0.00001, mean(values) / 5),
                                                (0.00001, var(values) / length(values)),
                                                (0.00001, .9999),
                                                (0.00001, .9999),
                                                (0.00001, .9999),
                                                (min_stay_outofstock_probability, .9999)],
                                                repeat([(-1.0, 2.0)], Base.size(exogenous, 1)),
                                        ),
                         :NumDimensions => dim,
                         :MaxTime => maxtime)
                         )

    fcs2 = LocalLevelCountStockoutExplanatory(;exogenous=exogenous,
                                            level1=xs[1],
                                            level2=xs[2],
                                            level_variance=abs(xs[3]),
                                            zero_inflation=abs(xs[4]),
                                            overdispersion=abs(xs[5]),
                                            level_matrix=[1-xs[6] xs[6];
                                                        1-xs[7] xs[7]],
                                            coefficients=xs[8:(7 + Base.size(exogenous, 1))])
    return fcs2
end

function get_loss_function(::Val{LocalLevelCountStockoutExplanatory}, exogenous, values; regularization=0.0, size=1000)
    return xs -> begin
        fcs2 = LocalLevelCountStockoutExplanatory(;exogenous=exogenous,
                                                level1=xs[1],
                                                level2=xs[2],
                                                level_variance=abs(xs[3]),
                                                zero_inflation=abs(xs[4]),
                                                overdispersion=abs(xs[5]),
                                                level_matrix=[1-xs[6] xs[6];
                                                            1-xs[7] xs[7]],
                                                coefficients=xs[8:(7 + Base.size(exogenous, 1))]
                              )
        smc = SMC{MVector{3, Float64}, LocalLevelCountStockoutExplanatory}(fcs2, size)
        rng = MersenneTwister(1)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false, rng=rng)
        return -likelihood + regularization * sum(x^2 for x in xs[8:end])
    end
end
