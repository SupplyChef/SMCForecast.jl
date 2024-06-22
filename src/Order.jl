function evaluate_order(order, current_inventory, order_period, lead_time, cover_until_period, service_level, future_observations, weights)
    lost_sales_during_coverage = 0
    sales_during_coverage = 0
    inventory = repeat([current_inventory], length(weights))
    for i in 1:length(future_observations)
        if i == order_period
            inventory .= inventory .+ Int(floor(order))
        end
        if i <= order_period + lead_time
            inventory .= inventory .- future_observations[i]
            inventory .= max.(inventory, 0)
        end
        if i >= order_period + lead_time
            lost_sales_during_coverage += sum(weights .* max.(future_observations[i] .- inventory, 0))
            sales_during_coverage += sum(weights .* future_observations[i])
            inventory .= inventory .- future_observations[i]
            inventory .= max.(inventory, 0)
        end
        if i == cover_until_period
            break
        end
    end

    return lost_sales_during_coverage, sales_during_coverage
end

function single_order(current_inventory, order_period, lead_time, cover_until_period, service_level, future_observations, weights)
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
                :MaxTime => 30))

    return order[1]
end

