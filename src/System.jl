abstract type SMCSystem{T <: SizedVector} end

struct System{T} <: SMCSystem{T}
    transition_distribution
    observation_distribution
    prior_distribution
end

function sample_states(system::System{SizedVector{1}}, 
                       current_states::Vector{SizedVector{1}}, 
                       next_observation::Union{Missing, Float64}, 
                       new_states, sampling_probabilities)

    for (i, current_state) in enumerate(current_states)
        d = system.transition_distribution(current_state[1])
        x = rand(d)
        p = pdf(d, x)

        new_states[i][1] = x
        sampling_probabilities[i] = p
    end
end

function sample_observation(system::System{SizedVector{1}}, current_state::SizedVector{1})
    return rand(system.observation_distribution(current_state[1]))
end

function transition_probability(system, state, observation, new_state)
    throw(ErrorException("Invalid"))
end

function sample_initial_state(system::System{SizedVector{1}}, count)
    return SizedVector{1}.(rand(system.prior_distribution, count))
end

function transition_probability(system::System{SizedVector{1}}, state::SizedVector{1}, observation, new_state::SizedVector{1}) 
    return pdf(system.transition_distribution(state[1]), new_state[1])
end

function observation_probability(system::System{SizedVector{1}}, state::SizedVector{1}, observation) 
    return pdf(system.observation_distribution(state[1]), observation)
end

function average_state(system::System{SizedVector{1}}, states, weights)
    return SizedVector{1}([sum(states[i][1] * weights[i] for i in eachindex(weights))])
end