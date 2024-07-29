module SMCForecast

using CategoricalArrays
using DataFrames
using DataStructures
using Dates
using Distributions
import Distributions:rand
using LinearAlgebra
using MLJ
using StatsBase
using StaticArrays
using SpecialFunctions
using Tables

export SMC
export System
export ForecastSystem
export LocalLevel
export LocalLevelChange
export LocalLevelExplanatory
export LocalLevelCountStockout
export LocalLevelCountStockoutExplanatory
export LocalLevelCountStockoutExplanatoryML

export forecast
export initialize!
export filter!
export smooth
export predict_states
export predict_observations
export percentiles
export cum_percentiles
export cum_expectiles

export MSIS
export MASE
export sMAPE
export coverage

export single_order
export evaluate_order

export get_loss_function

RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree

abstract type ForecastSystem end

include("System.jl")
include("SMC.jl")

include("LocalLevel.jl")
include("LocalLevelChange.jl")
include("LocalLevelExplanatory.jl")
include("LocalLevelCountStockout.jl")
include("LocalLevelCountStockoutExplanatory.jl")
include("LocalLevelCountStockoutExplanatoryML.jl")

include("Metrics.jl")
include("Order.jl")
include("Predict.jl")

function rand(system::System{SizedVector{1}}, count::Integer)
    ys = zeros(count)
    x = rand(system.prior_distribution)
    #ys[1] = rand(system.observation_distribution(x))
    for i in 1:count
        x = rand(system.transition_distribution(x))
        ys[i] = rand(system.observation_distribution(x))
    end
    return ys
end

end # module SMCForecast