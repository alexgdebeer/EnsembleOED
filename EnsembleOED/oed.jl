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

        # println("Candidate sensor $i: $(mean(traces[i, :])).")

    end

    mean_traces = vec(mean(traces, dims=2))
    # display(mean_traces)
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

        # @info "Selected sensor: $(candidate_sensors[opt_ind])."

    end

    return traces_list, selected_sensors

end

function select_sensor(
    B::AbstractMatrix,
    B_is::Vector,
    ensembles::Vector{Ensemble},
    ys::AbstractMatrix,
    C_ϵ::AbstractMatrix,
    pri::Distribution,
    save_steps::AbstractVector
)

    n_runs = size(ys, 2)
    n_sensors = length(B_is)

    d_opt_objs = zeros(n_sensors, n_runs)
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

            d_opt_objs[i, j] = compute_eig(means, covs, pri)
            n_opt_objs[i, j] = measure_gaussianity(means, covs)

        end

        println("Candidate sensor $i: $(mean(d_opt_objs[i, :])).")
        println("Candidate sensor $i: $(mean(n_opt_objs[i, :])).")

    end

    return d_opt_objs, n_opt_objs

end

function run_oed(
    ensembles::Vector{Ensemble},
    B::AbstractMatrix,
    ys::AbstractMatrix, 
    C_ϵ::AbstractMatrix,
    pri::Distribution, 
    save_steps::AbstractVector
)

    n_obs = size(ys, 1)
    n_sensors = size(ys, 1)

    candidate_sensors = collect(1:n_sensors)

    B_is = generate_B_is([], candidate_sensors, n_obs)

    d_opt_objs, n_opt_objs = select_sensor(
        B, B_is, ensembles, 
        ys, C_ϵ, pri, save_steps
    )

    return d_opt_objs, n_opt_objs

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

function compute_mixture(means, covs)

    components = [(m, Hermitian(c)) for (m, c) ∈ zip(means, covs)]
    m = MixtureModel(MvNormal, components)
    return m
    
end

function evaluate_mixture(mixture, θ)
    return log(1 / length(mixture) * sum([pdf(c, θ) for c ∈ mixture]))
end

function compute_eig(
    means::AbstractVector,
    covs::AbstractVector,
    pri::Distribution
)

    nθ = 250

    mixture = compute_mixture(means, covs)
    θs = rand(mixture, nθ)

    kl_div = 0.0

    for θ_i ∈ eachcol(θs)
        kl_div += logpdf(mixture, θ_i)
        kl_div -= logpdf(pri, θ_i)
    end

    return kl_div / nθ

end

function measure_gaussianity(
    means::AbstractVector,
    covs::AbstractVector
)

    nθ = 250

    # Form Gaussian mixture
    mixture = compute_mixture(means, covs)
    θs = rand(mixture, nθ)

    # Compute mean and covariance of "overall" Gaussian
    mean_ml = vec(mean(hcat(means...), dims=2))
    cov_ml = (1 / length(means)) * sum([c + m * m' for (m, c) in zip(means, covs)]) - mean_ml * mean_ml'
    cov_ml = Hermitian(cov_ml)

    # Compute the maximum likelihood Gaussian based on samples
    gaussian_ml = MvNormal(mean_ml, cov_ml)

    kl_div = 0.0

    for θ_i ∈ eachcol(θs)
        kl_div += logpdf(mixture, θ_i)
        kl_div -= logpdf(gaussian_ml, θ_i)
    end

    return kl_div / nθ

end