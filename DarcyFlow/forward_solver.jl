exp_avg(x, y) = exp(0.5(x+y))

struct Grid

    Δ::Real
    nx::Int
    xs::AbstractVector

    is_corner::AbstractVector 
    is_bounds::AbstractVector 
    bs_bounds::AbstractVector
    is_inner::AbstractVector

    cs::AbstractVector

    function Grid(nx::Int, Δ::Real)

        function add_point_type!(i)

            if i ∈ [1, nx, nx^2-nx+1, nx^2]
                push!(is_corner, i)
                return
            end

            if (i < nx) || (i > nx^2-nx+1) || (i % nx ∈ [0, 1])
                i % nx == 1   && push!(bs_bounds, :x0)
                i % nx == 0   && push!(bs_bounds, :x1)
                i < nx        && push!(bs_bounds, :y0)
                i > nx^2-nx+1 && push!(bs_bounds, :y1)
                push!(is_bounds, i)
                return
            end
            
            push!(is_inner, i)
            return

        end
        
        xs = 0:Δ:(Δ*nx)
        nx += 1 # A bit hacky, fix...

        is_corner = []
        is_bounds = []
        bs_bounds = []
        is_inner  = []

        for i ∈ 1:nx^2
            add_point_type!(i)
        end

        cs = [(x, y) for y ∈ xs for x ∈ xs]

        return new(
            Δ, nx, xs, 
            is_corner, is_bounds, 
            bs_bounds, is_inner,
            cs
        )

    end

end

struct BoundaryCondition

    name::Symbol
    type::Symbol
    func::Function
    
end

function construct_A(
    g::Grid, 
    lnks::AbstractVector, 
    bcs::Dict{Symbol, BoundaryCondition}
)

    function add_corner_point!(
        is::Vector{Int}, 
        js::Vector{Int}, 
        vs::Vector{<:Real}, 
        i::Int
    )

        push!(is, i)
        push!(js, i)
        push!(vs, 1.0)

        return

    end

    function add_dirichlet_point!(
        is::Vector{Int}, 
        js::Vector{Int}, 
        vs::Vector{<:Real}, 
        i::Int
    )

        push!(is, i)
        push!(js, i)
        push!(vs, 1.0)

        return

    end

    function add_neumann_point!(
        is::Vector{Int}, 
        js::Vector{Int}, 
        vs::Vector{<:Real}, 
        i::Int,
        g::Grid, 
        bc::BoundaryCondition,
    )

        push!(is, i, i, i)

        if bc.name == :x0
            push!(js, i, i+1, i+2)
            push!(vs, 3.0 / 2g.Δ, -4.0 / 2g.Δ, 1.0 / 2g.Δ)
        elseif bc.name == :x1
            push!(js, i, i-1, i-2)
            push!(vs, -3.0 / 2g.Δ, 4.0 / 2g.Δ, -1.0 / 2g.Δ)
        elseif bc.name == :y0
            push!(js, i, i+g.nx, i+2g.nx)
            push!(vs, 3.0 / 2g.Δ, -4.0 / 2g.Δ, 1.0 / 2g.Δ)
        elseif bc.name == :y1 
            push!(js, i, i-g.nx, i-2g.nx)
            push!(vs, -3.0 / 2g.Δ, 4.0 / 2g.Δ, -1.0 / 2g.Δ)
        end

        return

    end

    function add_boundary_point!(
        is::Vector{Int}, 
        js::Vector{Int}, 
        vs::Vector{<:Real}, 
        i::Int,
        g::Grid, 
        bc::BoundaryCondition
    )

        bc.type == :dirichlet && add_dirichlet_point!(is, js, vs, i)
        bc.type == :neumann && add_neumann_point!(is, js, vs, i, g, bc)

        return

    end

    function add_interior_point!(
        is::Vector{Int}, 
        js::Vector{Int}, 
        vs::Vector{<:Real}, 
        i::Int,
        g::Grid, 
        lnks::AbstractVector
    )

        push!(is, i, i, i, i, i)
        push!(js, i, i+1, i-1, i+g.nx, i-g.nx)

        push!(
            vs,
            (exp_avg(lnks[i], lnks[i+1])    + exp_avg(lnks[i], lnks[i-1]))    / g.Δ^2 + 
            (exp_avg(lnks[i], lnks[i+g.nx]) + exp_avg(lnks[i], lnks[i-g.nx])) / g.Δ^2,
            -exp_avg(lnks[i], lnks[i+1]) / g.Δ^2,
            -exp_avg(lnks[i], lnks[i-1]) / g.Δ^2,
            -exp_avg(lnks[i], lnks[i+g.nx]) / g.Δ^2,
            -exp_avg(lnks[i], lnks[i-g.nx]) / g.Δ^2
        )

        return

    end

    is = Int[]
    js = Int[]
    vs = Float64[]

    for i ∈ g.is_corner
        add_corner_point!(is, js, vs, i)
    end

    for (i, b) ∈ zip(g.is_bounds, g.bs_bounds)
        add_boundary_point!(is, js, vs, i, g, bcs[b])
    end

    for i ∈ g.is_inner
        add_interior_point!(is, js, vs, i, g, lnks)
    end

    return sparse(is, js, vs, g.nx^2, g.nx^2)

end

function construct_b(
    g::Grid, 
    lnks::AbstractVector,
    bcs::Dict{Symbol, BoundaryCondition},
    q::AbstractVector
)

    b = copy(q)

    for (i, bc) ∈ zip(g.is_bounds, g.bs_bounds)

        if bcs[bc].type == :neumann
            b[i] += bcs[bc].func(g.cs[i]...) / exp(lnks[i])
        else 
            b[i] += bcs[bc].func(g.cs[i]...)
        end

    end

    return b

end

function SciMLBase.solve(
    g::Grid,
    lnks::AbstractVector,
    bcs::Dict{Symbol, BoundaryCondition},
    q::AbstractVector
)

    A = construct_A(g, lnks, bcs)
    b = construct_b(g, lnks, bcs, q)

    u = solve(LinearProblem(A, b)).u
    return u

end