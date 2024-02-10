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

function select_sensor(
    B::AbstractMatrix,
    B_is::Vector,
    ensembles::Vector{Ensemble},
    ys::AbstractMatrix,
    C_ϵ::AbstractMatrix
)

    n_runs = size(ys, 2)
    n_sensors = length(B_is)

    traces = zeros(n_sensors, n_runs)

    ens = [deepcopy(ensembles) for _ ∈ 1:n_sensors]
    ys = [copy(ys) for _ ∈ 1:n_sensors]

    Threads.@threads for i ∈ 1:n_sensors

        for (j, (y_ij, ens_ij)) ∈ enumerate(zip(eachcol(ys[i]), ens[i]))

            compute_Gs!(ens_ij, B_is[i] * B)
            run_eki_dmc!(ens_ij, B_is[i] * B, B_is[i] * y_ij, B_is[i] * C_ϵ * B_is[i]')
            
            C_post = compute_C_uu(ens_ij)
            traces[i, j] = tr(C_post)

        end

        println("Candidate sensor $i: $(mean(traces[i, :])).")

    end

    mean_traces = vec(mean(traces, dims=2))
    display(mean_traces)
    min_ind = argmin(mean_traces)
    return traces, min_ind

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
    save_steps::AbstractVector
)

    n_runs = size(ys, 2)
    n_sensors = length(B_is)

    a_opt_objs = zeros(n_sensors, n_runs)
    n_opt_objs = zeros(n_sensors, n_runs)

    ens = [deepcopy(ensembles) for _ ∈ 1:n_sensors]
    ys = [copy(ys) for _ ∈ 1:n_sensors]

    Threads.@threads for i ∈ 1:n_sensors

        for (j, (y_ij, ens_ij)) ∈ enumerate(zip(eachcol(ys[i]), ens[i]))

            compute_Gs!(ens_ij, B_is[i] * B)
            θs, means, covs = run_eks!(
                ens_ij, B_is[i] * B, B_is[i] * y_ij, 
                B_is[i] * C_ϵ * B_is[i]', 
                save_steps
            )

            C_θθ = compute_C_θθ(ens_ij)
            a_opt_objs[i, j] = tr(C_θθ)
            n_opt_objs[i, j] = measure_gaussianity(θs, means, covs)

        end

        println("Candidate sensor $i: $(mean(a_opt_objs[i, :])).")
        println("Candidate sensor $i: $(mean(n_opt_objs[i, :])).")

    end

    mean_a_opt_objs = vec(mean(a_opt_objs, dims=2))
    mean_n_opt_objs = vec(mean(n_opt_objs, dims=2))
    min_ind = argmin(mean_a_opt_objs + mean_n_opt_objs)
    return a_opt_objs, n_opt_objs, min_ind

end

function run_oed(
    ensembles::Vector{Ensemble},
    B::AbstractMatrix,
    ys::AbstractMatrix, 
    C_ϵ::AbstractMatrix, 
    save_steps::AbstractVector,
    max_sensors::Int
)

    n_obs = size(ys, 1)
    n_sensors = size(ys, 1)

    selected_sensors = Int[]
    a_opt_list = []
    n_opt_list = []

    for i ∈ 1:max_sensors 

        candidate_sensors = [
            i for i ∈ 1:n_sensors 
            if i ∉ selected_sensors
        ]

        B_is = generate_B_is(selected_sensors, candidate_sensors, n_obs)

        a_opt_objs, n_opt_objs, opt_ind = select_sensor(B, B_is, ensembles, ys, C_ϵ, save_steps)
        
        push!(a_opt_list, a_opt_objs)
        push!(n_opt_list, n_opt_objs)
        push!(selected_sensors, candidate_sensors[opt_ind])

        @info "Selected sensor: $(candidate_sensors[opt_ind])."

    end

    return a_opt_list, n_opt_list, selected_sensors

end

function read_designs(fname::AbstractString)

    f = h5open(fname, "r")
    designs = [f["design_25"][:], f["design_50"][:], f["design_100"][:]]
    close(f)

    return designs

end

function validate_designs(
    designs::AbstractVector,
    ensembles::Vector{Ensemble},
    B::AbstractMatrix,
    us::AbstractMatrix,
    ys::AbstractMatrix,
    C_ϵ::AbstractMatrix,
    n_rand_designs::Int=20
)

    n_obs = size(ys, 1)
    n_sensor_locs = size(ys, 1)
    n_sensors = length(designs[1])
    n_ens = length(ensembles)

    # Generate a set of random designs
    rand_designs = [
        randperm(n_sensor_locs)[1:n_sensors] 
        for _ ∈ 1:n_rand_designs
    ]

    all_designs = [designs..., rand_designs...]
    n_designs = length(all_designs)

    traces = zeros(n_designs, n_ens)
    norms = zeros(n_designs, n_ens)

    Threads.@threads for i ∈ 1:n_designs

        ens_i = deepcopy(ensembles)
        B_i = generate_B_i(n_sensors, n_obs, all_designs[i])
        
        for j ∈ 1:n_ens

            compute_Gs!(ens_i[j], B_i * B)
            run_eki_dmc!(ens_i[j], B_i * B, B_i * ys[:, j], B_i * C_ϵ * B_i')
                
            C_post = compute_C_uu(ens_i[j])
            μ_post = compute_μ_u(ens_i[j])

            traces[i, j] = tr(C_post)
            norms[i, j] = norm(μ_post .- us[:, j]) / norm(us[:, j])
        
        end

        println("Design $i:\n" * 
                " - Mean trace: $(mean(traces[i, :])) \n" *
                " - Mean norm: $(mean(norms[i, :]))")

    end

    return traces, norms

end

function measure_gaussianity(
    θs::AbstractMatrix,
    means::AbstractVector,
    covs::AbstractVector
)

    nθ = size(θs, 2)

    # Form Gaussian mixture
    dists = [MvNormal(m, Hermitian(c)) for (m, c) ∈ zip(means, covs)]

    # Compute the maximum likelihood Gaussian based on samples
    ml_gaussian = MvNormal(vec(mean(θs, dims=2)), Hermitian(cov(θs, dims=2)))

    kl_div = 0.0

    for θ_i ∈ eachcol(θs)
        kl_div += log((1 / nθ) * sum([pdf(d, θ_i) for d ∈ dists]))
        kl_div -= logpdf(ml_gaussian, θ_i)
    end

    kl_div /= nθ
    return kl_div

end