UNIT_NORM = Normal()

abstract type AbstractChannel end

struct Channel <: AbstractChannel

    μ_int::Real 
    μ_ext::Real 

    L_int::AbstractMatrix 
    L_ext::AbstractMatrix

    bnds_geom::AbstractVector

    cs::AbstractVector

    nx::Int
    nθ::Int

    function Channel(
        g::Grid, 
        μ_int::Real, 
        μ_ext::Real, 
        σ_int::Real,
        σ_ext::Real,
        l_int::Real,
        l_ext::Real,
        bnds_geom::AbstractVector 
    )

        fname_int = "data/chol_$(g.nx)_int.h5"
        fname_ext = "data/chol_$(g.nx)_ext.h5"

        L_int = generate_chol(σ_int, l_int, g, fname_int)
        L_ext = generate_chol(σ_ext, l_ext, g, fname_ext)

        nx = g.nx
        nθ = length(bnds_geom) + 2nx^2

        return new(μ_int, μ_ext, L_int, L_ext, bnds_geom, g.cs, nx, nθ)

    end

end

struct ChannelGeom <: AbstractChannel

    lnk_int::Real 
    lnk_ext::Real 

    bnds_geom::AbstractVector 

    cs::AbstractVector 
    
    nx::Int 
    nθ::Int

    function ChannelGeom(
        g::Grid,
        lnk_int::Real,
        lnk_ext::Real,
        bnds_geom::AbstractVector
    )

        nθ = length(bnds_geom)
        return new(lnk_int, lnk_ext, bnds_geom, g.cs, g.nx, nθ)

    end

end

function gauss_to_unif(
    x::Real, 
    lb::Real, 
    ub::Real
)

    return lb + (ub - lb) * cdf(UNIT_NORM, x)

end

function Base.rand(c::Channel, n::Int=1)

    ωs = rand(UNIT_NORM, c.nθ, n)
    ωs[1:c.nx^2, :] = c.μ_int .+ c.L_int' * ωs[1:c.nx^2, :]
    ωs[c.nx^2+1:2c.nx^2, :] = c.μ_ext .+ c.L_ext' * ωs[c.nx^2+1:2c.nx^2, :]

    return ωs
end

function Base.rand(c::ChannelGeom, n::Int=1)

    ωs = rand(UNIT_NORM, c.nθ, n)
    return ωs

end

function in_channel(::ChannelGeom, x, m, p)
    centre = 800 * sin(2π*x[1] / p) + m * x[1] + 2500
    return centre - 800 ≤ x[2] ≤ centre + 800
end

function in_channel(::Channel, x, m, c, a, p, w)
    centre = a * sin(2π*x[1] / p) + m * x[1] + c 
    return centre - w ≤ x[2] ≤ centre + w
end

function inds_in_channel(
    c::AbstractChannel,
    ωs::AbstractVector
)

    us = [
        gauss_to_unif(ω, bnds...) 
        for (ω, bnds) ∈ zip(ωs, c.bnds_geom)
    ]

    is_int = [
        i for (i, coord) ∈ enumerate(c.cs) 
        if in_channel(c, coord, us...)
    ]

    return is_int
    
end

function transform(c::Channel, ωs::AbstractVecOrMat)

    lnks_int = ωs[1:c.nx^2]
    lnks_ext = ωs[c.nx^2+1:2c.nx^2]
    ωs_geom = ωs[2c.nx^2+1:end]

    is_int = inds_in_channel(c, ωs_geom)

    lnks = copy(lnks_ext)
    lnks[is_int] = lnks_int[is_int]

    return lnks

end

function transform(c::ChannelGeom, ωs::AbstractVecOrMat)

    is_int = inds_in_channel(c, vec(ωs))

    lnks = fill(c.lnk_ext, c.nx^2)
    lnks[is_int] .= c.lnk_int
    return lnks 

end