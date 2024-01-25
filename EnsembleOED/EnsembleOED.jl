using Distributions
using HDF5
using LinearAlgebra
using LinearSolve
using Random
using SparseArrays
using SpecialFunctions

include("forward_solver.jl")
include("matern.jl")
include("channel.jl")

include("data_generation.jl")
include("eki.jl")
include("oed.jl")