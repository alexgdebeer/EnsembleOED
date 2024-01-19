function generate_B_is(selected_sensors, candidate_sensors, n_obs)

    n_sensors = length(selected_sensors) + 1

    B_is = [
        zeros(n_sensors, n_obs) 
        for _ ∈ 1:length(candidate_sensors)
    ]

    for (B_i, cand_i) ∈ zip(B_is, candidate_sensors)

        sensors = [selected_sensors..., cand_i]
        for (i, s) ∈ enumerate(sensors)
            B_i[i, s] = 1.0
        end

    end

    return B_is

end

function run_oed(
    ensembles::Vector{Ensemble},
    B::AbstractMatrix,
    ys::Vector{AbstractVector},
    C_ϵ::AbstractMatrix,
    n_sensors::Int
)

    n_obs = length(ys[1])
    selected_sensors = Int[]

    for i ∈ 1:n_sensors 

        candidate_sensors = [
            i for i ∈ 1:n_sensors 
            if i ∉ selected_sensors
        ]

        B_is = generate_B_is(selected_sensors, candidate_sensors, n_obs)

        opt_ind = select_sensor(B, B_is, ensembles, ys, C_ϵ)
        push!(selected_sensors, candidate_sensors[opt_ind])

    end

    return selected_sensors

end

function select_sensor(
    B::AbstractMatrix,
    B_is::Vector{AbstractMatrix},
    ensembles::Vector{Ensemble},
    ys::Vector{AbstractVector},
    C_ϵ::AbstractMatrix,
)

    n_runs = length(ys)
    traces = Float64[]

    for B_i ∈ B_is

        ensembles_i = deepcopy(ensembles)

        for (y, ens_i) ∈ zip(ys, ensembles_i)

            # Compute predictions based on current ensemble and observation operator
            compute_Gs!(ens_i, B_i * B)

            run_eki_dmc!(ens_i, B_i * B, B_i * y, B_i * C_ϵ * B_i')
            C_post = compute_C_uu(ens_i) # NOTE: transformed covariance

            push!(traces, tr(C_post))

        end

    end

    traces = reshape(traces, n_runs, :)
    mean_traces = mean(traces, dims=1)
    min_ind = argmin(mean_traces)

    return min_ind

end