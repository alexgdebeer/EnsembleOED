using Distributions
using LinearAlgebra
using Random: seed!

include("EnsembleOED/EnsembleOED.jl")

seed!(16)

xmax = 6000
nx_f, nx_c = 81, 61
Δ_f, Δ_c = xmax/(nx_f-1), xmax/(nx_c-1)

grid_f = Grid(nx_f, Δ_f)
grid_c = Grid(nx_c, Δ_c)

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

f_f = build_f(grid_f)
f_c = build_f(grid_c)

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

channel_c = Channel(grid_c, μ_int, μ_ext, σ_int, σ_ext, l_int, l_ext, bnds_geom)
channel_f = Channel(grid_f, μ_int, μ_ext, σ_int, σ_ext, l_int, l_ext, bnds_geom)

# ω = rand(channel_c)
# lnks = transform(channel_c, ω)

# ps = solve(grid_c, lnks, bcs, f_c)

# lnks = reshape(lnks, grid_c.nx, grid_c.nx)
# ps = reshape(ps, grid_c.nx, grid_c.nx)

# ----------------
# Measurement generation
# ----------------
n_data = 5

# Candidate locations
xs_cand = LinRange(500, 5_500, 8)
ys_cand = LinRange(500, 5_500, 8)
cs_cand = [(x, y) for y ∈ ys_cand for x ∈ xs_cand]

M = length(cs_cand)

B_c = generate_B(grid_c, cs_cand)
B_f = generate_B(grid_f, cs_cand)

solve_c(u) = solve(grid_c, u, bcs, f_c)
solve_f(u) = solve(grid_f, u, bcs, f_f)

σ_ϵ = 0.02 * 150
C_ϵ = σ_ϵ^2 * Matrix(1.0I, M, M)

θs, us, hs, ys = generate_data(solve_f, channel_f, B_f, C_ϵ, n_data)

# ----------------
# Ensemble generation
# ----------------
J = 100
ensembles = [
    Ensemble(channel_c, solve_c, J) for _ ∈ 1:n_data
]

max_sensors = 10
traces_list, sensors = run_oed(ensembles, B_c, ys, C_ϵ, max_sensors)

# ----------------
# Test
# ----------------

# θ_t = rand(channel_f)
# u_t = transform(channel_f, θ_t)
# h_t = solve_f(u_t)
# y_t = B_f * h_t # Add noise?

# J = 100
# ens = Ensemble(channel_c, solve_c, J)
# compute_Gs!(ens, B_c)
# run_eki_dmc!(ens, B_c, y_t, C_ϵ)