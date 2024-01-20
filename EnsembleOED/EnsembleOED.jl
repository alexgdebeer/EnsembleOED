using Distributions
using HDF5
using LinearAlgebra
using LinearSolve
using SparseArrays
using SpecialFunctions: gamma

include("forward_solver.jl")
include("matern.jl")
include("channel.jl")

include("data_generation.jl")
include("eki.jl")
include("oed.jl")