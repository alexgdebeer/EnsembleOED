const CONV_TOL = 1e-8

mutable struct Ensemble

    θs::AbstractMatrix
    us::Union{AbstractMatrix, Nothing}
    Fs::Union{AbstractMatrix, Nothing}
    Gs::Union{AbstractMatrix, Nothing}

    channel::Channel
    solve::Function

    J::Int

    function Ensemble(
        channel::Channel, 
        solve::Function, 
        J::Int
    )

        θs = rand(channel, J)
        us = nothing 
        Fs = nothing 
        Gs = nothing

        ens = new(θs, us, Fs, Gs, channel, solve, J)
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

function compute_C_uu(
    ens::Ensemble
)

    Δu = ens.us .- mean(ens.us, dims=2)
    C_uu = (Δu * Δu') ./ (ens.J-1)
    return C_uu

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
    M::Int
)

    φs = 0.5 * sum((C_ϵ_invsqrt * (ens.Gs .- y)).^2, dims=1)

    μ_φ = mean(φs)
    var_φ = var(φs)
    
    α_inv = max(M / 2μ_φ, √(M / 2var_φ))
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

        α = compute_α_dmc(t, ens, y, C_ϵ_invsqrt, M)
        t += α^-1

        # println(t)

        update_ensemble_eki!(ens, α, y, C_ϵ)
        transform_ensemble!(ens)

        i += 1

        if abs(t - 1.0) < CONV_TOL
            # @info "Converged in $(i) iterations."
            return
        end

        run_ensemble!(ens)
        compute_Gs!(ens, B)

    end

end