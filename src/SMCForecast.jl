module SMCForecast

using BlackBoxOptim
using PRIMA
using DataFrames
using DataStructures
using Dates
using LinearAlgebra
using MLJ
using Optim
using StatsBase
using StaticArrays
using Distributions
import Distributions:rand
import BlackBoxOptim:bboptimize

export SMC
export System
export ForecastSystem
export LocalLevel
export LocalLevelChange
export LocalLevelJump
export LocalLevelRegressor
export LocalLevelExplanatory
export LocalLevelJumpExplanatory

export forecast
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

export get_loss_function

RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree

abstract type ForecastSystem end

include("Metrics.jl")
include("System.jl")
include("SMC.jl")
include("LocalLevel.jl")
include("LocalLevelChange.jl")
include("LocalLevelRegressor.jl")
include("LocalLevelExplanatory.jl")
include("LocalLevelCountJump.jl")
include("LocalLevelJumpExplanatory.jl")
include("SMC_predict.jl")

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