function generate_B(
    g::Grid, 
    cs::AbstractVector
)

    is = Int[]
    js = Int[]
    vs = Float64[]

    for (i, (x, y)) ∈ enumerate(cs)

        ix0 = findfirst(g.xs .> x) - 1
        iy0 = findfirst(g.xs .> y) - 1
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        x0, x1 = g.xs[ix0], g.xs[ix1]
        y0, y1 = g.xs[iy0], g.xs[iy1]

        inds = [(ix0, iy0), (ix0, iy1), (ix1, iy0), (ix1, iy1)]
        cell_inds = [i[1] + g.nx * (i[2]-1) for i ∈ inds]

        push!(is, i, i, i, i)
        push!(js, cell_inds...)
        push!(vs,
            (x1-x) * (y1-y) / g.Δ^2, 
            (x1-x) * (y-y0) / g.Δ^2, 
            (x-x0) * (y1-y) / g.Δ^2, 
            (x-x0) * (y-y0) / g.Δ^2
        )

    end

    B = sparse(is, js, vs, length(cs), g.nx^2)
    return B

end

function generate_data(
    F::Function,
    c::Channel,
    B::AbstractMatrix,
    C_ϵ::AbstractMatrix;
    n_runs::Int=5
)

    θs = rand(c, n_runs)
    us = hcat([transform(c, θ) for θ ∈ eachcol(θs)]...)
    hs = hcat([F(u) for u ∈ eachcol(us)]...)
    
    ys = B * hs
    ϵs = rand(MvNormal(C_ϵ), n_runs)
    ys += ϵs

    return θs, us, hs, ys

end