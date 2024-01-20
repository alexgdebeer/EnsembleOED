EPS = 1e-8


"""Generates the Cholesky factorisation of a Matérn covariance matrix 
with regularity parameter ν=1."""
function generate_chol(
    σ::Real, 
    l::Real, 
    g::Grid,
    fname::AbstractString
)

    if isfile(fname)
        f = h5open(fname)
        return f["L"][:, :]
    end

    @info "Existing Cholesky not found. Generating..."

    cxs = repeat(g.xs, outer=g.nx)
    cys = repeat(g.xs, inner=g.nx)

    ds = ((cxs.-cxs').^2 + (cys.-cys').^2).^0.5

    C = σ^2 .* (ds ./ l) .* besselk.(1, ds ./ l)
    C[diagind(C)] .= σ^2 + EPS
    C = Hermitian(C)

    L = Matrix(cholesky(C).U)
    h5write(fname, "L", L)
    return L

end