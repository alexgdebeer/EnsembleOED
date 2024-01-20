UNIT_NORM = Normal()

function gauss_to_unif(
    x::Real, 
    lb::Real, 
    ub::Real
)

    return lb + (ub - lb) * cdf(UNIT_NORM, x)

end

struct Channel

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

function Base.rand(c::Channel, n::Int=1)
    return rand(UNIT_NORM, c.nθ, n)
end

function inds_in_channel(
    c::Channel,
    ωs::AbstractVector
)

    function in_channel(x, m, c, a, p, w)
        centre = a * sin(2π*x[1] / p) + m * x[1] + c 
        centre - w ≤ x[2] ≤ centre + w
    end


    us = [
        gauss_to_unif(ω, bnds...) 
        for (ω, bnds) ∈ zip(ωs, c.bnds_geom)
    ]

    is_int = [
        i for (i, coord) ∈ enumerate(c.cs) 
        if in_channel(coord, us...)
    ]

    return is_int
    
end

function transform(c::Channel, ωs::AbstractVecOrMat)

    ωs_int = ωs[1:c.nx^2]
    ωs_ext = ωs[c.nx^2+1:2c.nx^2]
    ωs_geom = ωs[2c.nx^2+1:end]

    lnks_int = c.μ_int .+ c.L_int' * ωs_int 
    lnks_ext = c.μ_ext .+ c.L_ext' * ωs_ext

    is_int = inds_in_channel(c, ωs_geom)

    lnks = lnks_ext
    lnks[is_int] = lnks_int[is_int]

    return lnks

end