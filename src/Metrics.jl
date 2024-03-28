function sMAPE(forecast, true_values)
    horizon = length(forecast)

    average_forecast_error = sum(abs.(forecast - true_values)) / horizon

    return average_forecast_error * 2 / (sum(abs.(forecast)) + sum(abs.(true_values)))
end

function MASE(forecast, true_values, historical_values, frequency)
    horizon = length(forecast)

    average_forecast_error = sum(abs.(forecast - true_values)) / horizon

    average_naive_error = sum(abs(historical_values[t] - historical_values[t-frequency]) for t in (frequency+1):length(historical_values)) / (length(historical_values) - frequency)

    return average_forecast_error / average_naive_error
end

function coverage(forecast_low, forecast_high, true_values)
    horizon = length(forecast_low)

    cover = 0
    for t in 1:horizon
        if true_values[t] >= forecast_low[t] && true_values[t] <= forecast_high[t]
            cover = cover + 1
        end
    end
    return cover / horizon
end

function MSIS(forecast_low, forecast_high, true_values, historical_values, frequency)
    horizon = length(forecast_low)

    average_forecast_error = 1 / horizon * (
        sum(forecast_high .- forecast_low) + 
        2 / 0.05 * sum(true_values[i] - forecast_high[i] for i in 1:horizon if true_values[i] > forecast_high[i]; init=0.0) +
        2 / 0.05 * sum(forecast_low[i] - true_values[i] for i in 1:horizon if true_values[i] < forecast_low[i]; init=0.0)
    )

    average_naive_error = sum(abs(historical_values[t] - historical_values[t-frequency]) for t in (frequency+1):length(historical_values)) / (length(historical_values) - frequency)

    return average_forecast_error / average_naive_error
end