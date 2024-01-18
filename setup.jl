using Distributions
using LinearAlgebra
using LinearSolve
using Random: seed!

include("DarcyFlow/DarcyFlow.jl")

seed!(16)

xmax = 6.0
nx_f, nx_c = 100, 60
Δ_f, Δ_c = xmax/nx_f, xmax/nx_c

grid_f = Grid(nx_f, Δ_f)
grid_c = Grid(nx_c, Δ_c)

function generate_bcs(g::Grid)
    bcs = Dict(
        :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> -500.0 * g.Δ), # TODO: check sign
        :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
        :y0 => BoundaryCondition(:y0, :dirichlet, (x, y) -> 100.0), 
        :y1 => BoundaryCondition(:y1, :neumann, (x, y) -> 0.0)
    )
    return bcs 
end

function build_f(g::Grid)
    f = zeros(g.nx^2)
    for (i, c) ∈ enumerate(g.cs)
        if 4 ≤ c[2] < 5
            f[i] = -137 * g.Δ^2 # TODO: check these...
        elseif 5 ≤ c[2] ≤ 6 
            f[i] = -274 * g.Δ^2
        end
    end
    return f
end

bcs_f = generate_bcs(grid_f)
bcs_c = generate_bcs(grid_c)

f_f = build_f(grid_f)
f_c = build_f(grid_c)

# ----------------
# Prior setup
# ----------------

μ_int = 1
μ_ext = 4

# Bounds for standard deviation and lengthscale of log-permeability of 
# each random field
bnds_int = [(0.25, 0.75), (1.0, 2.0)]
bnds_ext = [(0.25, 0.75), (1.0, 2.0)]

# Bounds for slope, intercept, amplitude, period, width of channel 
bnds_geom = [
    (-0.3, 0.3), (1.5, 4.5), (0.3, 1.2), (2.0, 6.0), (0.5, 1.2)
]

pr = Channel(grid_c, μ_int, μ_ext, bnds_int, bnds_ext, bnds_geom)

ω = rand(pr)
lnks = transform(pr, ω)

ps = solve(grid_c, lnks, bcs_c, f_c)

lnks = reshape(lnks, grid_c.nx, grid_c.nx)
ps = reshape(ps, grid_c.nx, grid_c.nx)