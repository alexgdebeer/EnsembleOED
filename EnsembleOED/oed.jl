function generate_B_i(n_sensors, n_obs, design)

    B_i = zeros(n_sensors, n_obs)
    for (i, s) ∈ enumerate(design)
        B_i[i, s] = 1.0
    end
    return B_i

end

function generate_B_is(selected_sensors, candidate_sensors, n_obs)

    n_sensors = length(selected_sensors) + 1

    B_is = []

    for cand_i ∈ candidate_sensors

        design = [selected_sensors..., cand_i]
        B_i = generate_B_i(n_sensors, n_obs, design)
        push!(B_is, B_i)

    end

    return B_is

end

function run_oed(
    ensembles::Vector{Ensemble},
    B::AbstractMatrix,
    ys::AbstractMatrix,
    C_ϵ::AbstractMatrix,
    max_sensors::Int
)

    n_obs = size(ys, 1)
    n_sensors = size(ys, 1)

    selected_sensors = Int[]
    traces_list = []

    for i ∈ 1:max_sensors 

        candidate_sensors = [
            i for i ∈ 1:n_sensors 
            if i ∉ selected_sensors
        ]

        B_is = generate_B_is(selected_sensors, candidate_sensors, n_obs)

        traces, opt_ind = select_sensor(B, B_is, ensembles, ys, C_ϵ)
        
        push!(traces_list, traces)
        push!(selected_sensors, candidate_sensors[opt_ind])

        @info "Selected sensor: $(candidate_sensors[opt_ind])."

    end

    return traces_list, selected_sensors

end

function select_sensor(
    B::AbstractMatrix,
    B_is::Vector,
    ensembles::Vector{Ensemble},
    ys::AbstractMatrix,
    C_ϵ::AbstractMatrix,
)

    n_runs = size(ys, 2)
    traces = Float64[]

    for (i, B_i) ∈ enumerate(B_is)

        ensembles_i = deepcopy(ensembles)

        for (y, ens_i) ∈ zip(eachcol(ys), ensembles_i)

            compute_Gs!(ens_i, B_i * B)
            run_eki_dmc!(ens_i, B_i * B, B_i * y, B_i * C_ϵ * B_i')
            
            C_post = compute_C_uu(ens_i)
            push!(traces, tr(C_post))

        end

        println("Candidate sensor $i: $(mean(traces[end-4:end])).")

    end

    traces = reshape(traces, n_runs, :)
    mean_traces = vec(mean(traces, dims=1))
    min_ind = argmin(mean_traces)
    return traces, min_ind

end

function read_design(
    fname::AbstractString
)

    f = h5open(fname, "r")
    design = f["sensors"][:]
    traces_list = []
    for i ∈ 1:length(design)
        push!(traces_list, f["traces_$i"][:, :])
    end
    close(f)

    return traces_list, design

end

function validate_designs(
    designs::AbstractVector,
    ensembles::Vector{Ensemble},
    B::AbstractMatrix,
    ys::AbstractMatrix,
    C_ϵ::AbstractMatrix,
    n_rand_designs::Int=20
)

    n_obs = size(ys, 1)
    n_sensor_locs = size(ys, 1)
    n_sensors = length(designs[1])

    # Generate a set of random designs
    rand_designs = [
        randperm(n_sensor_locs)[1:n_sensors] 
        for _ ∈ 1:n_rand_designs
    ]

    all_designs = [designs..., rand_designs...]
    n_designs = length(all_designs)

    traces = Float64[]
    μs_post = []

    # Generate observation operators
    for design ∈ all_designs

        ensembles_i = deepcopy(ensembles)
        B_i = generate_B_i(n_sensors, n_obs, design)
        
        for (y, ens_i) ∈ zip(eachcol(ys), ensembles_i)

            compute_Gs!(ens_i, B_i * B)
            run_eki_dmc!(ens_i, B_i * B, B_i * y, B_i * C_ϵ * B_i')
                
            C_post = compute_C_uu(ens_i)
            μ_post = compute_μ_u(ens_i)
            push!(traces, tr(C_post))
            push!(μs_post, μ_post)

            # println(tr(C_post))
        
        end

        println(mean(traces[end-19:end]))

    end

    traces = reshape(traces, :, n_designs)
    return traces, μs_post

end