#!/usr/bin/env julia

using SparseArrays
using Laplacians
using NPZ

using Pkg
Pkg.status()


if length(ARGS) < 2
    error("Usage: julia run_sparsify.jl input_csc.npz output_csc.npz [epsilon]")
end

infile  = ARGS[1]
outfile = ARGS[2]
ep = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.1
matrixConcConst = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 4.0
JLfac   = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 4.0


println("Reading CSC bundle from $infile ...")
D = npzread(infile)  # Dict{String,Any} of numeric arrays only

data   = Vector{Float64}(D["data"])
rowind = Vector{Int}(D["indices"]) .+ 1   # convert to 1-based indexing
colptr = Vector{Int}(D["indptr"])  .+ 1
m, n   = Tuple(D["shape"])

println("Reconstructing CSC matrix...")
A = SparseMatrixCSC(m, n, colptr, rowind, data)

println("Sparsifying with ep = $ep ...")
using Random
Random.seed!(12345)
As = sparsify(A; ep=ep, matrixConcConst=matrixConcConst, JLfac=JLfac)

println("Saving CSC bundle to $outfile ...")
# Convert back to 0-based for SciPy
npzwrite(outfile, Dict(
    "data"    => As.nzval,
    "indices" => As.rowval .- 1,
    "indptr"  => As.colptr .- 1,
    "shape"   => [size(As,1), size(As,2)]
))

println("Done.")