const LocalLevelCountStockoutExplanatoryML = LocalLevelCountStockoutModel{TreeAdjustment}

function LocalLevelCountStockoutExplanatoryML(;exogenous, level1, level2, level_matrix, machine, level_variance, zero_inflation, overdispersion, adjust_sampling=false)
    return LocalLevelCountStockoutModel(TreeAdjustment(exogenous, machine);
                                         level1=level1,
                                         level2=level2,
                                         level_matrix=level_matrix,
                                         level_variance=level_variance,
                                         zero_inflation=zero_inflation,
                                         overdispersion=overdispersion,
                                         adjust_sampling=adjust_sampling,
                                         adjust_initial_value=true)
end

function forecast(::Val{LocalLevelCountStockoutExplanatoryML}, exogenous, values, horizon; maxtime=10.0, size=500, forecast_percentiles=0.5)
    fcs = fit(Val{LocalLevelCountStockoutExplanatoryML}(), exogenous, values; maxtime=maxtime, size=size)
    smc = SMC{MVector{3, Float64}, LocalLevelCountStockoutExplanatoryML}(fcs, 1_000)
    filter!(smc, values; record=false)
    obs, weights = predict_observations(smc, horizon)
    if isa(forecast_percentiles, Real)
        return percentiles(forecast_percentiles, obs, weights)
    else
        return [percentiles(p, obs, weights) for p in forecast_percentiles]
    end
end

function fit(::Val{LocalLevelCountStockoutExplanatoryML}, exogenous, values; maxtime=10, regularization=0.0, size=100,
                                                                        min_stay_outofstock_probability=0.0001,
                                                                        rng=Random.default_rng(),
                                                                        adjust_sampling=true,)

    fcs = SMCForecast.fit(Val{LocalLevelCountStockout}(), values * 1.0; maxtime=20, size=120, rng=rng)

    smc = SMC{MVector{3, Float64}, LocalLevelCountStockout}(fcs, 500)
    filtered_states, likelihood = SMCForecast.filter!(smc, values * 1.0; record=true, rng=rng)
    smoothed_states = smooth(smc, 200; rng=rng)

    smoothed_values = [mean(smoothed_states[j][i][2] for j in 1:length(smoothed_states)) for i in 1:length(smoothed_states[1])]
    stockouts = [mean(smoothed_states[j][i][3] for j in 1:length(smoothed_states)) for i in 1:length(smoothed_states[1])] .> 1.90

    filtered_values = (values .- smoothed_values)[stockouts .== false]
    filtered_exogenous = exogenous[:, 1:length(values)][:, stockouts .== false]

    DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=1)
    mach = MLJ.machine(model, table(filtered_exogenous'), filtered_values) |> MLJ.fit!

    print_tree(mach.fitresult[1])

    leaves = get_leaves(mach.fitresult[1])

    loss_function = get_loss_function(Val{LocalLevelCountStockoutExplanatoryML}(), exogenous, values, mach; regularization=regularization, size=size)

    dim = 7

    xs = bboptimize2(loss_function,
                       vcat([fcs.level1,
                        fcs.level2,
                        fcs.level_variance,
                        fcs.zero_inflation,
                        fcs.overdispersion,
                        fcs.level_matrix[1, 2],
                        fcs.level_matrix[2, 2]],
                        [leaf.majority for leaf in leaves]
                       ),
                        Dict(:SearchRange => vcat([(0, maximum(values)),
                                                (0.00001, mean(values) / 5),
                                                (0.00001, var(values) / length(values)),
                                                (0.00001, .9999),
                                                (0.00001, .9999),
                                                (0.00001, .9999),
                                                (min_stay_outofstock_probability, .9999)],
                                                [(min(leaf.majority, 0.0), max(leaf.majority, 0.0)) for leaf in leaves]),
                        :NumDimensions => dim,
                        :MaxTime => maxtime),
                        rng=rng,
                        x1 = vcat([fcs.level1,
                                    fcs.level2,
                                    fcs.level_variance,
                                    fcs.zero_inflation,
                                    fcs.overdispersion,
                                    fcs.level_matrix[1, 2],
                                    fcs.level_matrix[2, 2]],
                                    [0.0 for leaf in leaves]),
                    )

    machine = copy_tree1(mach.fitresult[1], IdDict(l => xs[7 + i] for (i, l) in enumerate(leaves)))
    fcs2 = LocalLevelCountStockoutExplanatoryML(;exogenous=exogenous,
                                            machine = machine,
                                            level1=xs[1],
                                            level2=xs[2],
                                            level_variance=abs(xs[3]),
                                            zero_inflation=abs(xs[4]),
                                            overdispersion=abs(xs[5]),
                                            level_matrix=[1-xs[6] xs[6];
                                                        1-xs[7] xs[7]],
                                            adjust_sampling=adjust_sampling)
    return fcs2
end

function get_loss_function(::Val{LocalLevelCountStockoutExplanatoryML}, exogenous, values, mach; regularization=0.0, size=1000, adjust_sampling=false)
    leaves = get_leaves(mach.fitresult[1])

    return xs -> begin
        machine = copy_tree1(mach.fitresult[1], IdDict(l => xs[7 + i] for (i, l) in enumerate(leaves)))

        fcs2 = LocalLevelCountStockoutExplanatoryML(;exogenous=exogenous,
                                                machine = machine,
                                                level1=xs[1],
                                                level2=xs[2],
                                                level_variance=abs(xs[3]),
                                                zero_inflation=abs(xs[4]),
                                                overdispersion=abs(xs[5]),
                                                level_matrix=[1-xs[6] xs[6];
                                                            1-xs[7] xs[7]],
                                                adjust_sampling=adjust_sampling)
        smc = SMC{MVector{3, Float64}, LocalLevelCountStockoutExplanatoryML}(fcs2, size)
        rng = MersenneTwister(1)
        filtered_states, likelihood = SMCForecast.filter!(smc, values; record=false, rng=rng)

        return -likelihood + regularization * sum(x^2 for x in xs[8:end])
    end
end
