function evaluate_order(order::Real, current_inventory::Int64, order_period::Int64, lead_time::Int64, cover_until_period::Int64, service_level::Float64, future_observations, weights::Vector{Float64})
    lost_sales_during_coverage::Float64 = 0.0
    sales_during_coverage::Float64 = 0.0
    inventory::Vector{Float64} = repeat([current_inventory] * 1.0, length(weights))

    tmp::Vector{Float64} = zeros(Float64, length(weights))
    rounded_order::Int64 = Int(floor(order))
    for i in 1:length(future_observations)
        if i == order_period
            inventory .+= rounded_order 
        end
        if i <= order_period + lead_time
            inventory .= inventory .- future_observations[i]
            inventory .= max.(inventory, 0)
        end
        if i >= order_period + lead_time
            tmp .= future_observations[i]
            tmp .= tmp .- inventory
            tmp .= max.(tmp, 0)
            tmp .= weights .* tmp
            lost_sales_during_coverage += sum(tmp)
            
            tmp .= future_observations[i]
            tmp .= weights .* tmp
            sales_during_coverage += sum(tmp)
            
            inventory .= inventory .- future_observations[i]
            inventory .= max.(inventory, 0)
        end
        if i == cover_until_period
            break
        end
    end

    return lost_sales_during_coverage, sales_during_coverage
end

function single_order(current_inventory::Int64, order_period::Int64, lead_time::Int64, cover_until_period::Int64, service_level::Float64, future_observations, weights; maxtime=10)
    loss_function = x -> begin
        
        lost_sales_during_coverage, sales_during_coverage = evaluate_order(x[1], current_inventory, order_period, lead_time, cover_until_period, service_level, future_observations, weights)

        return x[1] + 10_000 * max(0, lost_sales_during_coverage - sales_during_coverage * (1 - service_level))
    end 

    order = bboptimize2(loss_function,
                [0.0],
                Dict(
                :SearchRange => [(0.0, 10_000.0)], 
                :NumDimensions => 1, 
                :MaxStepsWithoutProgress => 15000,
                :MaxTime => maxtime))

    return Int(floor(order[1]))
end

