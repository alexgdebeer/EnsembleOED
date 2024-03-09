using LinearAlgebra
using Random: seed!

include("EnsembleOED/EnsembleOED.jl")

seed!(16)

fname_designs = "data/designs.h5"

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

μ_int = 2.0
μ_ext = 4.0

σ_int = 0.5
σ_ext = 0.25

l_int = 500
l_ext = 2000

# Bounds for slope, intercept, amplitude, period, width of channel 
bnds_geom = [
    (-0.3, 0.3), (1500, 4500), (300, 1200), (4000, 6000), (750, 1000)
]

channel = Channel(grid, μ_int, μ_ext, σ_int, σ_ext, l_int, l_ext, bnds_geom)

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

J = 50
ensembles = [Ensemble(channel, F, J) for _ ∈ 1:n_data]

max_sensors = 5
traces_list, design = run_oed(ensembles, B, ys, C_ϵ, max_sensors)
# designs = read_designs(fname_designs)

# ----------------
# Validation
# ----------------

# n_data_v = 20
# J_v = 200

# designs = [d[1:20] for d ∈ designs]

# θs_v, us_v, hs_v, ys_v = generate_data(F, channel, B, C_ϵ, n_data_v)
# ensembles_v = [Ensemble(channel, F, J_v) for _ ∈ 1:n_data_v]

# traces, norms = validate_designs(designs, ensembles_v, B, us_v, ys_v, C_ϵ)