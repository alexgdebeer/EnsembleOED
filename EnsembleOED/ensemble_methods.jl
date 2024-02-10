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

function compute_particle_weights(ens, γ)

    # Compute the squared distances between particles (weighted w.r.t. Euclidean norm)
    ds = pairwise(Euclidean(), ens.θs).^2

    # Compute the normalised weights associated with each particle
    ws = exp.(-(1/2γ) * ds)
    
    for c ∈ eachcol(ws)
        c ./= sum(c)
    end

    return ws

end

function update_ensemble_eks_loc!(
    ens::Ensemble,
    y::AbstractVector,
    C_ϵ::AbstractMatrix,
    Δt₀::Real,
    γ::Real
)

    μ_G = mean(ens.Gs, dims=2)
    D = (1.0 / ens.J) * (ens.Gs .- μ_G)' * (C_ϵ \ (ens.Gs .- y))
    Δt = compute_Δt_eks(D, Δt₀)
    Δt = 0.01

    ws = compute_particle_weights(ens, γ)
    # display(ws[1:10, 1:10])

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

        # Sample noise to add to the particle
        ζ_dist = MvNormal(Hermitian(C_θθ_j + 1e-3 * Diagonal(diag(C_θθ_j))))
        ζ_j = rand(ζ_dist)

        # Update particle j 
        ens.θs[:, j] = θ_j + Δt * (
            - C_θG_j * (C_ϵ \ (G_j - y))
            - C_θθ_j * θ_j
            + cor
        ) + √(2Δt) * ζ_j

    end

    # error("Stop here...")

    return Δt

end

function update_ensemble_eks!(
    ens::Ensemble,
    y::AbstractVector,
    C_ϵ::AbstractMatrix,
    Δt₀::Real
)

    μ_G = mean(ens.Gs, dims=2)
    μ_θ = mean(ens.θs, dims=2)
       
    C_θθ = cov(ens.θs, dims=2, corrected=false)
    D = (1.0 / ens.J) * (ens.Gs .- μ_G)' * (C_ϵ \ (ens.Gs .- y))
    ζ = rand(MvNormal(C_θθ + 1e-3 * Diagonal(diag(C_θθ))), ens.J)
    
    Δt = compute_Δt_eks(D, Δt₀)
    
    ens.θs = ens.θs + Δt * (
        - (ens.θs .- μ_θ) * D
        - (C_θθ * ens.θs)
        + ((ens.nθ + 1) / ens.J) * (ens.θs .- μ_θ)
    ) + √(2Δt) * ζ

    return Δt

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

function run_eks!(
    ens::Ensemble,
    B::AbstractMatrix,
    y::AbstractVector,
    C_ϵ::AbstractMatrix;
    Δt₀::Real=2.0,
    tmax::Real=2.0,
    localised::Bool=true,
    γ::Real=1.0
)

    t = 0.0
    Δt = 0.0
    i = 0

    while true 

        i += 1

        # if i == 50 
        #     γ /= 10
        # end

        if localised
            Δt = update_ensemble_eks_loc!(ens, y, C_ϵ, Δt₀, γ)
        else 
            Δt = update_ensemble_eks!(ens, y, C_ϵ, Δt₀)
        end

        transform_ensemble!(ens)

        t += Δt
        if t ≥ tmax || i ≥ 200
            @info "Converged in $(i) iterations."
            return
        end

        run_ensemble!(ens)
        compute_Gs!(ens, B)

        μ_misfit = mean(abs.(ens.Gs .- y))
        @printf "%3i | %.2e | %.2e | %.2e \n" i Δt t μ_misfit

    end

end