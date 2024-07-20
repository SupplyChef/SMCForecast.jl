using TestItems
using Test

using SMCForecast

using CSV
using DataFrames
using Dates
using Distributions
using MLJ
using Parsers
using PlotlyJS
using Random
using StaticArrays
using StatsBase
using Statistics

using TestItemRunner

@run_package_tests

@testitem "Generation" begin
    using SMCForecast
    using Distributions
    using Random
    using StaticArrays
        
    @test begin
        system = System{SizedVector{1}}(x -> Normal(x, 1),
                                        x -> Normal(x, 1),
                                        Uniform(0, 1))
        smc = SMC{SizedVector{1}, System{SizedVector{1}}}(system, 10)

        rng=Random.default_rng()
        Random.seed!(rng, 1)
        
        initialize!(smc; rng=rng)
        s1 = predict_states(smc, 20; rng=rng)

        Random.seed!(rng, 1)
        
        initialize!(smc; rng=rng)
        s2 = predict_states(smc, 20; rng=rng)

        all(s1 .== s2)
    end

    @test begin
        system = System{SizedVector{1}}(x -> Normal(x, 1),
                                        x -> Normal(x, 1),
                                        Uniform(0, 1))
        smc = SMC{SizedVector{1}, System{SizedVector{1}}}(system, 10)

        rng=Random.default_rng()
        Random.seed!(rng, 1)
        
        initialize!(smc; rng=rng)
        s1 = predict_states(smc, 20; rng=rng)

        Random.seed!(rng, 2)
        
        initialize!(smc; rng=rng)
        s2 = predict_states(smc, 20; rng=rng)

        !all(s1 .== s2)
    end
end

@testitem "Filtering" begin
    using SMCForecast
    using Distributions
    using Random
    using StaticArrays

    @test begin
        system = System{SizedVector{1}}(x -> Normal(x, 1),
                                        x -> Normal(x, 1),
                                        Uniform(0, 1))
        smc = SMC{SizedVector{1}, System{SizedVector{1}}}(system, 10)

        observations = [0 + 0.1 * i for i in 1:200]
        filtered_states, likelihood = SMCForecast.filter!(smc, observations)

        filtered_states[end][1] > 19 && filtered_states[end][1] < 21
    end

    @test begin
        system = System{SizedVector{1}}(x -> Normal(x, 1),
                                        x -> Normal(x, 1),
                                        Uniform(0, 1))
        smc = SMC{SizedVector{1}, System{SizedVector{1}}}(system, 10)

        observations = [0 + 0.1 * i for i in 1:200]
        filtered_states, likelihood = SMCForecast.filter!(smc, observations; record=false)

        filtered_states[end][1] > 19 && filtered_states[end][1] < 21
    end
end

@testitem "Smoothing" begin
    using SMCForecast
    using Distributions
    using Random
    using StaticArrays
    using Statistics

    @test begin
        system = System{SizedVector{1}}(x -> Normal(x, 1),
                                        x -> Normal(x, 1),
                                        Uniform(0, 1))
        smc = SMC{SizedVector{1}, System{SizedVector{1}}}(system, 10)

        observations = [0 + 0.1 * i for i in 1:200]
        filtered_states, likelihood = SMCForecast.filter!(smc, observations)
        smoothed_states = smooth(smc, 100)
        
        mean(x -> x[end][1], smoothed_states) > 19 && mean(x -> x[end][1], smoothed_states) < 21
    end
end

@testitem "Likelihood" begin
    using CSV
    using DataFrames
    using Random
    using StatsBase
    using StaticArrays

    @test begin
        println(pwd())
        data = CSV.read(raw"..\datasets\nile.csv", DataFrame).flow

        fcs = SMCForecast.fit(Val{LocalLevel}(), data; maxtime=60, size=100)

        #@test get_constrained_value(model, "sigma2_ε") ≈ 15099 rtol = 1e-3
        #@test get_constrained_value(model, "sigma2_η") ≈ 1469.1 rtol = 1e-3

        println(fcs)
        true
    end

    @test begin
        system = LocalLevel(100, 10, 30)
        smc = SMC{SizedVector{2, Float64, Vector{Float64}}, LocalLevel}(system, 10)
        
        rng=Random.default_rng()
        Random.seed!(rng, 1)
        initialize!(smc; rng=rng)
        
        obs, weights = predict_observations(smc, 100; rng=rng)
        #data = sample(rng, obs, pweights(weights))
        data = map(o -> o[1], obs)

        fcs = SMCForecast.fit(Val{LocalLevel}(), data; maxtime=60, size=100)

        println(fcs)
        true
    end

    @test begin
        system = LocalLevel(100, 10, 30)
        smc = SMC{SizedVector{2, Float64, Vector{Float64}}, LocalLevel}(system, 10)
        
        rng=Random.default_rng()
        Random.seed!(rng, 1)
        initialize!(smc; rng=rng)
        
        obs, weights = predict_observations(smc, 100; rng=rng)
        #data = sample(rng, obs, pweights(weights))
        data = map(o -> o[1], obs)

        fcs = SMCForecast.fit(Val{LocalLevelChange}(), data; maxtime=60, size=100)

        println(fcs)
        true
    end

    @test begin
        system = LocalLevel(100, 10, 30)
        smc = SMC{SizedVector{2, Float64, Vector{Float64}}, LocalLevel}(system, 10)
        
        rng=Random.default_rng()
        Random.seed!(rng, 1)
        initialize!(smc; rng=rng)
        
        obs, weights = predict_observations(smc, 100; rng=rng)
        #data = sample(rng, obs, pweights(weights))
        data = map(o -> round(o[1]), obs)

        fcs = SMCForecast.fit(Val{LocalLevelCountJump}(), data; maxtime=60, size=100)

        println(fcs)
        true
    end
end

include("test_locallevel.jl")
include("test_locallevelchange.jl")
include("test_locallevelcountjump.jl")
include("test_locallevelexplanatory.jl")

#include("m5_competition.jl")

DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree

function add_exogenous(data)
    data1 = DataFrame(date=data.date, 
                      value=data.value, 
                      day=categorical(day.(data.date)), 
                      dow=categorical(dayofweek.(data.date)), 
                      month=categorical(month.(data.date)), 
                      #year=year.(d), 
                      #t=map(d -> d.value, Day.(d .- d[1]))
                      )

    ohe = machine(OneHotEncoder(), data1)  |> MLJ.fit!
    data2 = MLJ.transform(ohe, data1)
    return data2
end

function create_model(data2)
    model = DecisionTreeRegressor(max_depth=1, min_samples_split=3)
    mach = machine(model, data2[:, Not([:date, :value])], data2.value) |> MLJ.fit!
    return mach
end


function load_data()
    xf =  XLSX.openxlsx(raw"C:\Users\renau\source\repos\SMCForecast\datasets\sample_data.xlsx") 
    sales4 = DataFrame(XLSX.gettable(xf["data"]))

    gd = first(groupby(sales4, :id), 50)
    historical_values = Dict(k.id => select(gd[k], :date, :value) for k in keys(gd))

    return historical_values
end


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
#
# Example of usage
# loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe("TSForecasting/tsf_data/sample.tsf")
function convert_tsf_to_dataframe(full_file_path_and_name::String;
    replace_missing_vals_with="NaN",
    value_column_name="series_value")
    col_names = String[]
    col_types = String[]
    all_data = Dict{String, Any}()
    all_series = Vector{Any}(undef, 0)
    line_count = 0
    frequency = nothing
    forecast_horizon = nothing
    contain_missing_values = nothing
    contain_equal_length = nothing
    found_data_tag = false
    found_data_section = false
    started_reading_data_section = false

    open(full_file_path_and_name, "r") do file
        for line in eachline(file)
            line = strip(line)

            if !isempty(line)
                if startswith(line, "@")  # Read meta-data
                    if !startswith(line, "@data")
                        line_content = split(line, " ")
                        if startswith(line, "@attribute")
                            if length(line_content) != 3  # Attributes have both name and type
                                throw("Invalid meta-data specification.")
                            end

                            push!(col_names, line_content[2])
                            push!(col_types, line_content[3])
                        else
                            if length(line_content) != 2  # Other meta-data have only values
                                throw("Invalid meta-data specification.")
                            end

                            if startswith(line, "@frequency")
                                frequency = line_content[2]
                            elseif startswith(line, "@horizon")
                                forecast_horizon = parse(Int, line_content[2])
                            elseif startswith(line, "@missing")
                                contain_missing_values = line_content[2] == "true"
                            elseif startswith(line, "@equallength")
                                contain_equal_length = line_content[2] == "true"
                            end
                        end
                    else
                        if isempty(col_names)
                            throw("Missing attribute section. Attribute section must come before data.")
                        end

                        found_data_tag = true
                    end
                elseif !startswith(line, "#")
                    if isempty(col_names)
                        throw("Missing attribute section. Attribute section must come before data.")
                    elseif !found_data_tag
                        throw("Missing @data tag.")
                    else
                        if !started_reading_data_section
                            started_reading_data_section = true
                            found_data_section = true

                            for col in col_names
                                all_data[col] = Any[]
                            end
                        end

                        full_info = split(line, ":")

                        if length(full_info) != (length(col_names) + 1)
                            throw("Missing attributes/values in series.")
                        end

                        numeric_series = Vector{Union{Float64, String}}(undef, 0)

                        for val in eachsplit(full_info[end], ",")
                            if val == "?"
                                push!(numeric_series, replace_missing_vals_with)
                            else
                                push!(numeric_series, Parsers.parse(Float64, val))
                            end
                        end

                        if isempty(numeric_series)
                            throw("A given series should contain a set of comma-separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol")
                        end
                        if count(==(replace_missing_vals_with), numeric_series) == length(numeric_series)
                            throw("All series values are missing. A given series should contain a set of comma-separated numeric values. At least one numeric value should be there in a series.")
                        end

                        push!(all_series, numeric_series)

                        for i in eachindex(col_names)
                            att_val = nothing
                            if col_types[i] == "numeric"
                                att_val = parse(Int, full_info[i])
                            elseif col_types[i] == "string"
                                att_val = full_info[i]
                            elseif col_types[i] == "date"
                                att_val = DateTime(full_info[i], dateformat"yyyy-mm-dd HH-MM-SS")
                            else
                                throw("Invalid attribute type.")
                            end

                            if att_val == nothing
                                throw("Invalid attribute value.")
                            else
                                push!(all_data[col_names[i]], att_val)
                            end
                        end
                    end
                end
            end
            line_count += 1
        end
    end

    if line_count == 0
        throw("Empty file.")
    end
    if isempty(col_names)
        throw("Missing attribute section.")
    end
    if !found_data_section
        throw("Missing series information under data section.")
    end

    all_data[value_column_name] = all_series
    loaded_data = DataFrame(all_data)

    return (loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length)
end