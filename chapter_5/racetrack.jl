using DelimitedFiles
import Plots as plt
using Random
using Dates


struct RaceTrack
    race_track_map::Matrix{Int64}
    start_points::Vector{CartesianIndex{2}}
end

GRASS = 0
TRACK = 1
START = 2
FINISH = 3
MAX_VELOCITY = 4
ACTION_SPACE = [[i, j] for i in range(-1, 1) for j in range(-1, 1)]

function load_race_track(file_path::String)
    race_track_map = reverse(readdlm(joinpath(@__DIR__, file_path), ',', Int, '\n'), dims=1)
    start_points = findall(race_track_map .== START)
    return RaceTrack(race_track_map, start_points)
end

race_track_1 = load_race_track("./data/figure05-05-left.csv");
# race_track_1 = load_race_track("./data/test.csv")

int(x::Number) = floor(Int, x)

function render(env::RaceTrack, state_history::Union{Vector{Vector{Int64}},Nothing}=nothing)
    map_size = size(env.race_track_map)
    xs = 0.5:map_size[2]+2
    ys = 0.5:map_size[1]+2
    fig = plt.heatmap(
        xs[2:end-1],
        ys[2:end-1],
        (x, y) -> env.race_track_map[int(y), int(x)];
        color=["white", "lightgray", "pink", "lightgreen"],
        colorbar=false,
        aspect_ratio=:equal,
        size=(400, 800),
        xlims=(0, map_size[2] + 1),
        ylims=(0, map_size[1] + 1)
    )
    plt.hline!(floor.(ys), l=(0.2, :black), legend=false)
    plt.vline!(floor.(xs), l=(0.2, :black), legend=false)
    if state_history !== nothing
        xs = [p[2] for p in state_history] .+ 0.5
        ys = [p[1] for p in state_history] .+ 0.5
        plt.plot!(fig, xs, ys, linewidth=2, markersize=2, markershape=:circle)
    end
    return fig
end

render(race_track_1)

function generate_discrete_points(pos::Vector{Int64}, velocity::Vector{Int64})
    if velocity[1] == 0
        return [[pos[1], pos[2] + step] for step in range(0, velocity[2])]
    end

    # The map origin is on top left corner
    slope = velocity[2] / velocity[1]
    intercept = pos[2] - slope * pos[1]

    start_range = pos[1]
    end_range = pos[1] + velocity[1]
    if start_range > end_range
        start_range = end_range
        end_range = pos[1]
    end
    return [
        [round(Int, x), round(Int, intercept + slope * x)]
        for x in range(start_range, end_range)
    ]
end

begin
    generate_discrete_points([7, 7], [2, 2])
    generate_discrete_points([7, 7], [2, 1])
    generate_discrete_points([7, 7], [0, 1])
end

struct CarState
    pos::Vector{Int64}
    velocity::Vector{Int64}
end

function reset_car(env::RaceTrack)
    rand_idx = rand(1:length(env.start_points))
    start_point = env.start_points[rand_idx]
    return CarState([start_point[1], start_point[2]], zeros(2))
end

function step_env(env::RaceTrack, current_state::CarState, action::Vector{Int64})
    next_velocity = clamp.(
        current_state.velocity + action, 0, MAX_VELOCITY
    )
    path = generate_discrete_points(current_state.pos, next_velocity)
    map_size = size(env.race_track_map)
    valid_points = [
        [p[1], p[2]]
        for p in path
        if 1 <= p[1] <= map_size[1]
        &&
        1 <= p[2] <= map_size[2]
    ]
    out_of_track_points = [
        p
        for p in valid_points[2:end]
        if env.race_track_map[p[1], p[2]] == GRASS
    ]
    if length(out_of_track_points) > 0
        next_car_state = reset_car(env)
        return -1.0, next_car_state, false
    end

    finish_points_on_path = [
        p
        for p in valid_points[2:end]
        if env.race_track_map[p[1], p[2]] == FINISH
    ]
    if length(finish_points_on_path) != 0
        finish_point = finish_points_on_path[1]
        next_car_state = CarState(
            finish_point, finish_point - current_state.pos
        )
        return -1.0, next_car_state, true
    elseif length(path) > length(valid_points)
        next_car_state = reset_car(env)
        return -1.0, next_car_state, false
    else
        next_car_state = CarState(valid_points[end], next_velocity)
        return -1.0, next_car_state, false
    end
end

function sample_random_episode(env::RaceTrack, policy::Array{Int64,4}; max_time::Int, epsilon=0.1)
    current_state = reset_car(env)
    trajectories = []
    next_state = nothing
    for t in 1:max_time
        state = [current_state.pos..., (current_state.velocity .+ 1)...]
        next_velocities = [action + current_state.velocity for action in ACTION_SPACE]
        valid_action_idxs = [
            idx
            for (idx, next_velocity) in enumerate(next_velocities)
            if all((next_velocity .>= 0) .&& (next_velocity .<= MAX_VELOCITY))
            && !all(next_velocity .== 0)
        ]
        num_acts = length(valid_action_idxs)

        action_idx = policy[state...]
        πa_valid = action_idx in valid_action_idxs
        action_prob = 0
        if rand() >= epsilon
            if πa_valid
                action_prob = 1 - epsilon + epsilon / num_acts
            else
                action_idx = rand(valid_action_idxs)
                action_prob = 1 / num_acts
            end
        else
            action_idx = rand(valid_action_idxs)
            action_prob = (πa_valid ? epsilon : 1) / num_acts
        end

        action = ACTION_SPACE[action_idx]
        reward, next_state, finished = step_env(env, current_state, action)
        pushfirst!(
            trajectories,
            [current_state, action_idx, reward, action_prob]
        )
        current_state = next_state
        if finished
            break
        end
    end
    return trajectories, next_state
end

begin
    policy = ones(Int64, (size(race_track_1.race_track_map)..., MAX_VELOCITY + 1, MAX_VELOCITY + 1))
    trajectories, current_state = sample_random_episode(race_track_1, policy; max_time=10000)
    state_history = append!([current_state.pos], [trajectory[1].pos for trajectory in trajectories[1:25]])
    render(race_track_1, state_history)
end

function off_policy_mc(env::RaceTrack; discount=1.0, max_eps=100, max_time=10000)
    @info "$(now()) Start learning..."
    # Initialize
    action_dim = length(ACTION_SPACE)
    state_action_dims = (
        size(env.race_track_map)...,
        MAX_VELOCITY + 1,
        MAX_VELOCITY + 1,
        action_dim
    )
    Q = -400 * rand(state_action_dims...)
    C = zeros(state_action_dims...)
    policy = ones(Int64, state_action_dims[1:end-1]...)
    for ep_num in 1:max_eps
        trajectories, final_state = sample_random_episode(env, policy; max_time)
        G = 0.0
        W = 1.0
        for (current_state, action_idx, reward, action_prob) in trajectories
            # Julia is 1-based => velocity 0 has index 1
            state = CartesianIndex(current_state.pos..., (current_state.velocity .+ 1)...)
            state_action = CartesianIndex(state, action_idx)
            G = discount * G + reward
            C[state_action] += W
            Q[state_action] += W * (G - Q[state_action]) / C[state_action]
            best_action_idx = argmax(Q[state, :])
            policy[state] = best_action_idx
            if action_idx != best_action_idx
                break
            end
            W /= action_prob
        end
        if ep_num % 100 == 0
            @info "$(now()) Episode $ep_num/$max_eps: W=$W, G=$G, $(length(trajectories))"
        end
    end
    @info "$(now()) Finished learning"
    return policy
end

policy = off_policy_mc(race_track_1; max_eps=100000, max_time=50000);

function sample_policy_episode(env::RaceTrack, policy::Array{Int64,4}, max_time::Int)
    current_state = reset_car(env)
    state_history = [current_state.pos]
    for _ in 1:max_time
        state = [current_state.pos..., (current_state.velocity .+ 1)...]
        action_idx = policy[state...]
        reward, next_state, finished = step_env(env, current_state, ACTION_SPACE[action_idx])
        append!(state_history, [next_state.pos])
        current_state = next_state
        if finished
            break
        end
    end
    return render(env, state_history)
end

begin
    for _ in 1:3
        display(sample_policy_episode(race_track_1, policy, 30))
    end
end