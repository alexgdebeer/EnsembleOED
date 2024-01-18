struct Channel

    grf_int::MaternField
    grf_ext::MaternField

    bnds_geom::AbstractVector

    cs::AbstractVector

    nx::Int
    nθ::Int

    function Channel(
        g::Grid, 
        μ_int::Real, 
        μ_ext::Real, 
        bnds_int::AbstractVector,
        bnds_ext::AbstractVector,
        bnds_geom::AbstractVector 
    )

        grf_int = MaternField(g, μ_int, bnds_int...)
        grf_ext = MaternField(g, μ_ext, bnds_ext...)

        nx = g.nx
        nθ = length(bnds_int) + length(bnds_ext) + length(bnds_geom) + 2nx^2

        return new(grf_int, grf_ext, bnds_geom, g.cs, nx, nθ)

    end

end

function Base.rand(c::Channel, n::Int=1)
    return rand(UNIT_NORM, n, c.nθ)
end

function inds_in_channel(
    c::Channel,
    ηs::AbstractVector
)

    function in_channel(x, m, c, a, p, w)
        centre = a * sin(2π*x[1] / p) + m * x[1] + c 
        centre - w ≤ x[2] ≤ centre + w
    end


    us = [
        gauss_to_unif(η, bnds...) 
        for (η, bnds) ∈ zip(ηs, c.bnds_geom)
    ]

    is_int = [
        i for (i, coord) ∈ enumerate(c.cs) 
        if in_channel(coord, us...)
    ]

    return is_int
    
end

function transform(c::Channel, ωs::AbstractVecOrMat)

    ωs_int = ωs[1:c.nx^2+2]
    ωs_ext = ωs[c.nx^2+3:2c.nx^2+4]
    ωs_geom = ωs[2c.nx^2+5:end]

    lnks_int = transform(c.grf_int, ωs_int)
    lnks_ext = transform(c.grf_ext, ωs_ext)

    is_int = inds_in_channel(c, ωs_geom)

    lnks = lnks_ext
    lnks[is_int] = lnks_int[is_int]

    return lnks

end