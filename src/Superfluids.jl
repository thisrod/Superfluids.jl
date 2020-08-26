module Superfluids

export Superfluid, Discretisation, FDDiscretisation, PinnedVortices, discretise, argand, cloud, steady_state, sample, find_vortex
using LinearAlgebra, BandedMatrices, Optim
using Statistics: mean

# Default parameters

_defaults = Dict{Symbol,Any}(:g_tol=>1e-4, :iterations=>1000)
default(k::Symbol) = get(_defaults, k, nothing)
default!(k::Symbol, v) = _defaults[k] = v
      

"""
    struct Superfluid{N}

N-dimensional superfluid
"""
struct Superfluid{N}
    C::Float64				# repulsion constant
    V				# (x,y) -> V
end

Superfluid{N}(C::Float64) where N = Superfluid{N}(C, (x...) -> 0.0)

default!(v::Superfluid) = default!(:superfluid, v)

"""
    abstract type Discretisation{N}

A domain with a way to discretise an order parameter on it.
"""
abstract type Discretisation{N} end

default!(v::Discretisation) = default!(:discretisation, v)

"""
    sample(f, d::Discretisation)

Return an array discretising the function f.

For now we assume that L² normalised functions give l² normalised arrays.
"""
function sample(f, d::Discretisation) end

"""
    D(u, axis, d::Discretisation, n=1)

Return the nth derivative of the field u
"""
D(u, axis, d::Discretisation) = D(u, axis, d, 1)

"""
    operators(s::Superfluid, d::Discretisation, syms...)

Return operator functions that act on sample(ψ, d)

By default returns a dictionary, with symbols returns those operators.
"""
function operators(s::Superfluid, d::Discretisation) end
operators(syms::Vararg{Symbol}) =
    operators(default(:superfluid), default(:discretisation), syms...)
operators(s::Superfluid, syms::Vararg{Symbol}) =
    operators(s, default(:discretisation), syms...)
operators(d::Discretisation, syms::Vararg{Symbol}) =
    operators(default(:superfluid), d, syms...)

# argand(d::Discretisation) = sample(Complex, d)
argand(d::Discretisation) = sample((x,y)->x+1im*y, d)
argand() = argand(default(:discretisation))


coords(d::Discretisation{N}) where N =
    [sample((r...)->r[j], d) for j = 1:N]

include("findiff.jl")
include("plotting.jl")
include("vortices.jl")
include("relaxation.jl")
include("bogoliubov.jl")

end # module
