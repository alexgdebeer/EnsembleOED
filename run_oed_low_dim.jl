using LinearAlgebra
using Random: seed!

include("EnsembleOED/EnsembleOED.jl")

seed!(16)

# ----------------
# Discretisation, boundary conditions, forcing function
# ----------------

xmax = 6000
nx = 61
Δ = xmax/(nx-1)

grid = Grid(nx, Δ)

bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 250 * 1e-3),
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0),
    :y0 => BoundaryCondition(:y0, :dirichlet, (x, y) -> 100), 
    :y1 => BoundaryCondition(:y1, :neumann, (x, y) -> 0)
)

function build_f(g::Grid)
    f = zeros(g.nx^2)
    for (i, c) ∈ enumerate(g.cs)
        if i ∈ g.is_inner
            if 4000 ≤ c[2] < 5000
                f[i] = 0.137 * 1e-3
            elseif 5000 ≤ c[2] ≤ 6000
                f[i] = 0.274 * 1e-3
            end
        end
    end
    return f
end

f = build_f(grid)

# ----------------
# Prior
# ----------------

lnperm_int = 2.0
lnperm_ext = 4.0

# Means and standard deviations of slope and period of channel
μs_geom = [0, 5000]
σs_geom = [0.15, 1500]

channel = ChannelGeom(grid, lnperm_int, lnperm_ext, μs_geom, σs_geom)
pri = MvNormal(Matrix(I, channel.nθ, channel.nθ))

# ----------------
# Measurement generation
# ----------------

n_data = 5

# Candidate locations
xs_cand = LinRange(500, 5_500, 8)
ys_cand = LinRange(500, 5_500, 8)
cs_cand = [(x, y) for y ∈ ys_cand for x ∈ xs_cand]

M = length(cs_cand)

B = generate_B(grid, cs_cand)

F(u) = solve(grid, u, bcs, f)

σ_ϵ = 0.02 * 150
C_ϵ = σ_ϵ^2 * Matrix(1.0I, M, M)

# ----------------
# OED
# ----------------

θs, us, hs, ys = generate_data(F, channel, B, C_ϵ, n_data)

J = 100
ensembles = [Ensemble(channel, F, J) for _ ∈ 1:n_data]

save_steps = 75:125
max_sensors = 1

d_opt_list, n_opt_list, design = run_oed(ensembles, B, ys, C_ϵ, pri, save_steps, max_sensors)

h5write("data/low_dim/results.h5", "d_opt", d_opt_list[1])
h5write("data/low_dim/results.h5", "n_opt", n_opt_list[1])