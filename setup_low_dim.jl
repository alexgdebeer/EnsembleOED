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

# Bounds for slope and period of channel
bnds_geom = [(-0.3, 0.3), (2000, 8000)]

channel = ChannelGeom(grid, lnperm_int, lnperm_ext, bnds_geom)

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

save_steps = 81:100
max_sensors = 1

a_opt_list, n_opt_list, design = run_oed(ensembles, B, ys, C_ϵ, save_steps, max_sensors)

# ----------------
# Testing 
# ----------------

# θ_t = rand(channel, 1)
# u_t = transform(channel, θ_t)
# F_t = F(u_t)
# G_t = B * F_t
# y = G_t + rand(MvNormal(C_ϵ))

# J = 100
# save_steps = 81:100

# ens = Ensemble(channel, F, J)

# compute_Gs!(ens, B)
# θs, means, covs = run_eks!(ens, B, y, C_ϵ, save_steps)

# n_grid = 50
# xis_grid = LinRange(-4, 4, n_grid)
# xjs_grid = LinRange(-4, 4, n_grid)
# zs_grid = zeros(n_grid, n_grid)

# for (i, xi) ∈ enumerate(xis_grid), (j, xj) ∈ enumerate(xjs_grid)

#     G_ij = B * F(transform(channel, [xi, xj]))
#     zs_grid[i, j] = 0.5norm([xi, xj])^2 + 0.5((G_ij - y)' * (C_ϵ \ (G_ij - y)))

# end

# compute_C_θθ(ens)

# dists = [MvNormal(m[:], Hermitian(c)) for (m, c) in zip(means, covs)]

# zs_ens = zeros(n_grid, n_grid)

# for (i, xi) ∈ enumerate(xis_grid), (j, xj) ∈ enumerate(xjs_grid)

#     zs_ens[i, j] = (1 / ens.J) * sum([pdf(d, [xi, xj]) for d ∈ dists])

# end

# measure_gaussianity(θs, means, covs)