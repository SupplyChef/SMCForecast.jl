module SMCForecast

using DataFrames
using DataStructures
using Dates
using LinearAlgebra
using MLJ
using Optim
using StatsBase
using StaticArrays
using SpecialFunctions
using Distributions
import Distributions:rand

export SMC
export System
export ForecastSystem
export LocalLevel
export LocalLevelChange
export LocalLevelJump
export LocalLevelCountJumpExplanatory
export LocalLevelRegressor
export LocalLevelExplanatory
export LocalLevelJumpExplanatory

export forecast
export initialize!
export filter!
export smooth
export predict_states
export predict_observations
export percentiles
export cum_percentiles

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
include("LocalLevelRegressor.jl")
include("LocalLevelExplanatory.jl")
include("LocalLevelCountJump.jl")
#include("LocalLevelJumpExplanatory.jl")
include("LocalLevelCountJumpExplanatory.jl")

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