@testitem "Order" begin
    using SMCForecast
    
    #single_order(current_inventory, order_period, lead_time, cover_until_period, service_level, future_observations, weights)
    @test begin
        current_inventory = 0
        order_period = 1
        lead_time = 0
        cover_until_period = 10
        service_level = 1.0
        future_observations = [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
        weights = [0.1, 0.8, 0.1]

        order = single_order(current_inventory, order_period, lead_time, cover_until_period, service_level, future_observations, weights)

        println(order)
        order == 33
    end

    @test begin
        current_inventory = 10
        order_period = 1
        lead_time = 0
        cover_until_period = 10
        service_level = 1.0
        future_observations = [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
        weights = [0.1, 0.8, 0.1]

        order = single_order(current_inventory, order_period, lead_time, cover_until_period, service_level, future_observations, weights)

        println(order)
        order == 23
    end

    @test begin
        current_inventory = 0
        order_period = 1
        lead_time = 0
        cover_until_period = 10
        service_level = 0.9
        future_observations = [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
        weights = [0.1, 0.8, 0.1]

        order = single_order(current_inventory, order_period, lead_time, cover_until_period, service_level, future_observations, weights)

        println(order)
        println(evaluate_order(order, current_inventory, order_period, lead_time, cover_until_period, service_level, future_observations, weights))
        order == 21
    end
end