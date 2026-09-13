@testitem "CountStockout shared model: type structure" begin
    using SMCForecast

    # LocalLevelCountStockout, LocalLevelCountStockoutExplanatory, and
    # LocalLevelCountStockoutExplanatoryML must remain distinct concrete types
    # (each parameterized by a different MeanAdjustment) so that SMC{T, U}'s
    # `U <: SMCSystem{T}` dispatch, and the Predict.jl method specialized on
    # `U <: LocalLevelCountStockout`, keep selecting exactly the same methods
    # they did before this became a shared implementation.
    @test LocalLevelCountStockout <: LocalLevelCountStockoutModel
    @test LocalLevelCountStockoutExplanatory <: LocalLevelCountStockoutModel
    @test LocalLevelCountStockoutExplanatoryML <: LocalLevelCountStockoutModel

    @test LocalLevelCountStockout != LocalLevelCountStockoutExplanatory
    @test LocalLevelCountStockout != LocalLevelCountStockoutExplanatoryML
    @test LocalLevelCountStockoutExplanatory != LocalLevelCountStockoutExplanatoryML

    @test !(LocalLevelCountStockoutExplanatory <: LocalLevelCountStockout)
    @test !(LocalLevelCountStockoutExplanatoryML <: LocalLevelCountStockout)
end

@testitem "CountStockout shared model: neutral adjustments reproduce the base model" begin
    using SMCForecast
    using Distributions
    using Random
    using StaticArrays
    using Statistics

    # LocalLevelCountStockout, LocalLevelCountStockoutExplanatory, and
    # LocalLevelCountStockoutExplanatoryML now share one SMCSystem
    # implementation (LocalLevelCountStockoutModel) parameterized by how the
    # latent level is adjusted for exogenous effects. Given the same core
    # parameters and an adjustment that is a no-op (zero coefficients / an
    # all-zero regression tree), all three must produce identical
    # observation/transition probabilities and statistically identical
    # sampled observations -- i.e. the refactor must not change the model
    # given the same parameters, only how the exogenous adjustment is
    # plugged in.
    level1 = 6.0
    level2 = 0.05
    level_matrix = [0.95 0.05; 0.4 0.6]
    level_variance = 0.03
    zero_inflation = 0.05
    overdispersion = 0.1

    base = LocalLevelCountStockout(;level1=level1, level2=level2, level_matrix=level_matrix,
                                    level_variance=level_variance, zero_inflation=zero_inflation,
                                    overdispersion=overdispersion, adjust_sampling=false)

    n_periods = 20
    n_regressors = 3
    exogenous = zeros(n_regressors, n_periods) # all-zero: LinearAdjustment/TreeAdjustment(0) are exact no-ops regardless of coefficients/tree

    explanatory = LocalLevelCountStockoutExplanatory(;exogenous=exogenous, level1=level1, level2=level2,
                                                      level_matrix=level_matrix, coefficients=[0.7, -0.3, 1.5],
                                                      level_variance=level_variance, zero_inflation=zero_inflation,
                                                      overdispersion=overdispersion, adjust_sampling=false)

    zero_tree = SMCForecast.MutableRoot(SMCForecast.MutableLeaf(0.0))
    ml = LocalLevelCountStockoutExplanatoryML(;exogenous=exogenous, level1=level1, level2=level2,
                                               level_matrix=level_matrix, machine=zero_tree,
                                               level_variance=level_variance, zero_inflation=zero_inflation,
                                               overdispersion=overdispersion, adjust_sampling=false)

    state1_instock = MVector{3, Float64}(3.0, level1 + 1.2, 1.0)
    state2_instock = MVector{3, Float64}(4.0, level1 - 0.5, 1.0)
    state1_outofstock = MVector{3, Float64}(3.0, level2, 2.0)

    @test begin
        all(
            SMCForecast.observation_probability(base, state1_instock, k) ≈
            SMCForecast.observation_probability(explanatory, state1_instock, k) ≈
            SMCForecast.observation_probability(ml, state1_instock, k)
            for k in 0:15
        )
    end

    @test begin
        SMCForecast.observation_probability(base, state1_outofstock, 0) ≈
        SMCForecast.observation_probability(explanatory, state1_outofstock, 0) ≈
        SMCForecast.observation_probability(ml, state1_outofstock, 0)
    end

    @test begin
        SMCForecast.transition_probability(base, state1_instock, missing, state2_instock) ≈
        SMCForecast.transition_probability(explanatory, state1_instock, missing, state2_instock) ≈
        SMCForecast.transition_probability(ml, state1_instock, missing, state2_instock)
    end

    @test begin
        # sample_observation's in-stock branch draws via sample_zigp, which -- in the
        # original code for all three variants alike, preserved as-is here -- doesn't
        # actually consume the `rng` keyword passed to sample_observation; it always
        # draws from the global default RNG. So reproducibility here comes from
        # reseeding that global RNG (Random.seed!), not from passing a local `rng`.
        Random.seed!(42)
        base_draws = [SMCForecast.sample_observation(base, state1_instock) for _ in 1:20_000]

        Random.seed!(42)
        explanatory_draws = [SMCForecast.sample_observation(explanatory, state1_instock) for _ in 1:20_000]

        Random.seed!(42)
        ml_draws = [SMCForecast.sample_observation(ml, state1_instock) for _ in 1:20_000]

        # sample_observation doesn't consult the adjustment at all (it only ever reads
        # state[2], which is already in observed scale), so with the same rng seed the
        # three variants must draw *identically*, not just from the same distribution.
        base_draws == explanatory_draws == ml_draws
    end
end

@testitem "CountStockout shared model: particle filter recovers an injected stockout window" begin
    using SMCForecast
    using Distributions
    using Random
    using StaticArrays
    using Statistics

    function stockout_posterior(smc)
        return [
            sum((smc.historical_weights[t][j] for j in eachindex(smc.historical_states[t]) if smc.historical_states[t][j][3] == 2); init=0.0) /
            sum(smc.historical_weights[t])
            for t in eachindex(smc.historical_states)
        ]
    end

    in_stock_mean = 6.0
    out_of_stock_mean = 0.05
    stockout_window = 51:80
    n = 120

    rng = MersenneTwister(2024)
    values = zeros(n)
    for t in 1:n
        mean_t = t in stockout_window ? out_of_stock_mean : in_stock_mean
        values[t] = rand(rng, Poisson(mean_t))
    end

    level_matrix = [0.97 0.03; 0.3 0.7]

    for (label, system) in [
        ("base", LocalLevelCountStockout(;level1=in_stock_mean, level2=out_of_stock_mean, level_matrix=level_matrix,
                                          level_variance=0.02, zero_inflation=0.0, overdispersion=0.0, adjust_sampling=false)),
        ("explanatory (zero exogenous)", LocalLevelCountStockoutExplanatory(;exogenous=zeros(2, n), level1=in_stock_mean,
                                          level2=out_of_stock_mean, level_matrix=level_matrix, coefficients=[0.0, 0.0],
                                          level_variance=0.02, zero_inflation=0.0, overdispersion=0.0, adjust_sampling=false)),
        ("ML (zero tree)", LocalLevelCountStockoutExplanatoryML(;exogenous=zeros(2, n), level1=in_stock_mean,
                                          level2=out_of_stock_mean, level_matrix=level_matrix,
                                          machine=SMCForecast.MutableRoot(SMCForecast.MutableLeaf(0.0)),
                                          level_variance=0.02, zero_inflation=0.0, overdispersion=0.0, adjust_sampling=false)),
    ]
        @test begin
            smc = SMC{MVector{3, Float64}, typeof(system)}(system, 500)
            filter_rng = MersenneTwister(7)
            filtered_states, loglikelihood = SMCForecast.filter!(smc, values; rng=filter_rng)

            posterior = stockout_posterior(smc)
            inside = mean(posterior[stockout_window])
            outside = mean(posterior[setdiff(1:n, stockout_window)])

            println("$label: P(stockout) inside window = $inside, outside window = $outside")

            isfinite(loglikelihood) && inside > 0.5 && outside < 0.2 && (inside - outside) > 0.4
        end
    end
end

@testitem "CountStockout shared model: LinearAdjustment matches the original multiplicative formula" begin
    using SMCForecast

    # Pins the exact math that used to live in de_exogenous_multiplicative /
    # re_exogenous_multiplicative: prod_i (coefficients[i] + 1) over active
    # (> 0) regressors, applied multiplicatively.
    exogenous = [1.0 0.0 1.0; 0.0 1.0 1.0] # 2 regressors x 3 periods
    coefficients = [0.5, -0.2]

    adjustment = SMCForecast.LinearAdjustment(exogenous, coefficients)

    row1 = SMCForecast.exogenous_row(adjustment, 1) # only regressor 1 active -> (0.5+1) = 1.5
    @test SMCForecast.readjust(adjustment, 10.0, row1) ≈ 10.0 * 1.5
    @test SMCForecast.deadjust(adjustment, 15.0, row1) ≈ 15.0 / 1.5

    row2 = SMCForecast.exogenous_row(adjustment, 2) # only regressor 2 active -> (-0.2+1) = 0.8
    @test SMCForecast.readjust(adjustment, 10.0, row2) ≈ 10.0 * 0.8

    row3 = SMCForecast.exogenous_row(adjustment, 3) # both active -> 1.5 * 0.8
    @test SMCForecast.readjust(adjustment, 10.0, row3) ≈ 10.0 * 1.5 * 0.8

    @test SMCForecast.deadjust(adjustment, SMCForecast.readjust(adjustment, 7.0, row3), row3) ≈ 7.0
end

@testitem "CountStockout shared model: TreeAdjustment matches apply_tree1" begin
    using SMCForecast

    # A tiny hand-built tree: feature 1 < 0.5 -> -2.0, else -> 3.0
    tree = SMCForecast.MutableRoot(SMCForecast.MutableNode(1, 0.5, SMCForecast.MutableLeaf(-2.0), SMCForecast.MutableLeaf(3.0)))
    adjustment = SMCForecast.TreeAdjustment(zeros(1, 1), tree)

    @test SMCForecast.readjust(adjustment, 10.0, [0.1]) ≈ 10.0 - 2.0
    @test SMCForecast.readjust(adjustment, 10.0, [0.9]) ≈ 10.0 + 3.0
    @test SMCForecast.deadjust(adjustment, 10.0, [0.1]) ≈ 10.0 + 2.0
    @test SMCForecast.deadjust(adjustment, 10.0, [0.9]) ≈ 10.0 - 3.0
end

@testitem "CountStockout shared model: filter! throughput" begin
    using SMCForecast
    using Distributions
    using Random
    using StaticArrays

    # Regression guard, not a tight benchmark: this only needs to catch an
    # accidental algorithmic regression (e.g. an O(n^2) reintroduction) in the
    # shared sample_states/observation_probability/transition_probability
    # implementation, not to pin an exact throughput number. Includes JIT
    # warm-up cost. Tighten the ceiling once real CI timings are available.
    rng = MersenneTwister(11)
    n = 400
    values = Float64.(rand(rng, Poisson(5), n))

    system = LocalLevelCountStockout(;level1=5.0, level2=0.05, level_matrix=[0.97 0.03; 0.3 0.7],
                                      level_variance=0.02, zero_inflation=0.0, overdispersion=0.0, adjust_sampling=false)

    smc = SMC{MVector{3, Float64}, LocalLevelCountStockout}(system, 2000)

    elapsed = @elapsed SMCForecast.filter!(smc, values; rng=MersenneTwister(11))
    println("filter! with 2000 particles over $n periods took $(elapsed)s")

    @test elapsed < 60.0
end
