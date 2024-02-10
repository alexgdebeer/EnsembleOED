const CONV_TOL = 1e-8

const α₁ = 1_000
const γ = 0.839

mutable struct Ensemble

    θs::AbstractMatrix
    us::Union{AbstractMatrix, Nothing}
    Fs::Union{AbstractMatrix, Nothing}
    Gs::Union{AbstractMatrix, Nothing}

    channel::AbstractChannel
    solve::Function

    J::Int
    nθ::Int

    function Ensemble(
        channel::AbstractChannel, 
        solve::Function, 
        J::Int
    )

        θs = rand(channel, J)
        us = nothing 
        Fs = nothing 
        Gs = nothing

        ens = new(θs, us, Fs, Gs, channel, solve, J, size(θs, 1))
        transform_ensemble!(ens)
        run_ensemble!(ens)

        return ens

    end

end

function resample_particles!(
    ens::Ensemble, 
    inds_det::AbstractVector
)

    n_det = length(inds_det)
    @info "Resampling $(n_det) detatched particles."

    inds = (1:ens.J)[1:ens.J .∉ inds_det]
    inds_res = rand(inds, n_det)
    
    ens.θs[:, inds_det] = ens.θs[:, inds_res]
    ens.us[:, inds_det] = ens.us[:, inds_res]
    ens.Fs[:, inds_det] = ens.Fs[:, inds_res]
    ens.Gs[:, inds_det] = ens.Gs[:, inds_res]

    return

end

function compute_covs(
    ens::Ensemble
)

    Δθ = ens.θs .- mean(ens.θs, dims=2)
    ΔG = ens.Gs .- mean(ens.Gs, dims=2)

    C_θG = (Δθ * ΔG') ./ (ens.J-1)
    C_GG = (ΔG * ΔG') ./ (ens.J-1)

    return C_θG, C_GG

end

function compute_C_θθ(
    ens::Ensemble
)

    Δθ = ens.θs .- mean(ens.θs, dims=2)
    C_θθ = (Δθ * Δθ') ./ (ens.J-1)
    return C_θθ

end

function compute_C_uu(
    ens::Ensemble
)

    Δu = ens.us .- mean(ens.us, dims=2)
    C_uu = (Δu * Δu') ./ (ens.J-1)
    return C_uu

end

function compute_μ_u(ens::Ensemble)
    return mean(ens.us, dims=2)
end

function update_ensemble_eki!(
    ens::Ensemble, 
    α::Real, 
    y::AbstractVector, 
    C_ϵ::AbstractMatrix
)

    C_θG, C_GG = compute_covs(ens)
    ϵs = rand(MvNormal(α * C_ϵ), ens.J)
    ens.θs += C_θG * inv(C_GG + α * C_ϵ) * (y .+ ϵs .- ens.Gs)

    return

end

function compute_Δt_eks(
    D::AbstractMatrix,
    Δt₀::Real
)

    return Δt₀ / (norm(D) + 1e-8)

end

function transform_ensemble!(
    ens::Ensemble
)

    ens.us = hcat([transform(ens.channel, θ) for θ ∈ eachcol(ens.θs)]...)
    return

end

function run_ensemble!(
    ens::Ensemble
)

    ens.Fs = hcat([ens.solve(u) for u ∈ eachcol(ens.us)]...)
    return

end

function compute_Gs!(
    ens::Ensemble,
    B::AbstractMatrix
)

    ens.Gs = B * ens.Fs
    return

end

function compute_α_dmc(
    t::Real, 
    ens::Ensemble, 
    y::AbstractVector, 
    C_ϵ_invsqrt::AbstractMatrix,
    M::Int,
    N_it::Int
)

    φs = 0.5 * sum((C_ϵ_invsqrt * (ens.Gs .- y)).^2, dims=1)

    μ_φ = mean(φs)
    var_φ = var(φs)
    
    α_inv = max(M / 2μ_φ, √(M / 2var_φ))
    # Ensure number of iterations is no more than 30
    α_inv = max(α_inv, (α₁ * γ^N_it)^-1)
    α_inv = min(α_inv, 1-t)

    return 1.0/α_inv

end

function run_eki_dmc!(
    ens::Ensemble,
    B::AbstractMatrix,
    y::AbstractVector,
    C_ϵ::AbstractMatrix
)

    M = length(y)
    C_ϵ_invsqrt = sqrt(inv(C_ϵ))
    t = 0.0
    i = 0

    while true 

        α = compute_α_dmc(t, ens, y, C_ϵ_invsqrt, M, i)
        i += 1
        t += α^-1

        # println(t)

        update_ensemble_eki!(ens, α, y, C_ϵ)
        transform_ensemble!(ens)

        if abs(t - 1.0) < CONV_TOL
            # @info "Converged in $(i) iterations."
            return
        end

        run_ensemble!(ens)
        compute_Gs!(ens, B)

    end

end

function compute_particle_weights(
    ens::Ensemble, 
    γ::Real
)

    # Compute the squared distances between particles (weighted w.r.t. Euclidean norm)
    ds = pairwise(Euclidean(), ens.θs).^2

    # Compute the normalised weights associated with each particle
    ws = exp.(-(1/2γ) * ds)
    
    for c ∈ eachcol(ws)
        c ./= sum(c)
    end

    inds_det = findall(>(0.98), diag(ws))
    
    if !isempty(inds_det)
        resample_particles!(ens, inds_det)
    end

    # println(maximum(ws)

    return ws

end

function update_ensemble_eks_loc!(
    ens::Ensemble,
    y::AbstractVector,
    C_ϵ::AbstractMatrix,
    Δt::Real,
    γ::Real
)

    ws = compute_particle_weights(ens, γ)

    means = []
    covs = []

    for j ∈ 1:ens.J 

        ws_j = ws[:, j]
        θ_j = ens.θs[:, j]
        G_j = ens.Gs[:, j]

        μθ_j = sum(ws_j' .* ens.θs, dims=2)
        μG_j = sum(ws_j' .* ens.Gs, dims=2)

        Δθ_j = ens.θs .- μθ_j 
        ΔG_j = ens.Gs .- μG_j

        # Compute weighted covariance matrices
        C_θθ_j = (ws_j' .* Δθ_j) * Δθ_j'
        C_θG_j = (ws_j' .* Δθ_j) * ΔG_j'

        # Compute correction term
        cor = ws_j[j] * (ens.nθ + 1) * (θ_j - μθ_j)

        for k ∈ 1:ens.J 

            θ_k = ens.θs[:, k]
            ∇w_jk = (ws_j[k] / γ) * (θ_k - μθ_j)

            cor += (θ_k * θ_k' * ∇w_jk)
            cor -= (μθ_j * θ_k' * ∇w_jk)
            cor -= (θ_k * μθ_j' * ∇w_jk)

        end

        # Mean and covariance of distribution particle j is sampled from
        mean = θ_j + Δt * (
            - C_θG_j * (C_ϵ \ (G_j - y))
            - C_θθ_j * θ_j
            + cor
        )

        cov = 2Δt * C_θθ_j

        # Sample noise to add to the particle
        ζ_dist = MvNormal(Hermitian(cov))
        ζ_j = rand(ζ_dist)

        ens.θs[:, j] = mean + ζ_j

        push!(means, mean[:])
        push!(covs, cov)

    end

    return means, covs

end

function run_eks!(
    ens::Ensemble,
    B::AbstractMatrix,
    y::AbstractVector,
    C_ϵ::AbstractMatrix,
    save_steps::AbstractVector;
    Δt::Real=0.01,
    tmax::Real=1.0,
    γ::Real=0.5
)

    t = 0.0
    i = 0

    means = []
    covs = []
    particles = []

    while true 

        i += 1

        means_i, covs_i = update_ensemble_eks_loc!(ens, y, C_ϵ, Δt, γ)
        transform_ensemble!(ens)

        if i ∈ save_steps
            push!(particles, copy(ens.θs))
            push!(means, means_i...)
            push!(covs, covs_i...)
        end

        t += Δt

        if t ≥ tmax
            # @info "Converged."
            return hcat(particles...), means, covs
        end

        run_ensemble!(ens)
        compute_Gs!(ens, B)

        μ_misfit = mean(abs.(ens.Gs .- y))
        @printf "%3i | %.2e | %.2e | %.2e \n" i Δt t μ_misfit

    end

end