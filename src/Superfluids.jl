module Superfluids

export Superfluid, Discretisation, FDDiscretisation, PinnedVortices, discretise, argand, cloud, relaxed_state, sample, find_vortex, relax_orbit
using LinearAlgebra, BandedMatrices, Optim, Arpack
using Statistics: mean
import DifferentialEquations, Interpolations


# Default parameters

default(k::Symbol) = get(_defaults, k, nothing)
default() = copy(_defaults)
default!(k::Symbol, v) = (_defaults[k] = v; nothing)
default!(d::Dict{Symbol,Any}) = (_defaults=copy(d); nothing)

_defaults = Dict{Symbol,Any}(
    :hbm=>1,
    :g_tol=>1e-6,
    :iterations=>1000,
    :dt=>1e-4,
    :relaxer=>Optim.ConjugateGradient,
    :formula=>DifferentialEquations.RK4,
    :solver=>nothing	# eigenproblem solver
)
      

"""
    struct Superfluid{N}

N-dimensional superfluid
"""
struct Superfluid{N}
    C::Float64				# repulsion constant
    hbm::Float64
    V				# (x,y) -> V
end

Superfluid{N}(C::Real, V=(x...)->0.0; hbm::Real=default(:hbm)) where N =
    Superfluid{N}(convert(Float64, C), convert(Float64, hbm), V)

default!(v::Superfluid) = default!(:superfluid, v)

"""
    abstract type Discretisation{N}

A domain with a way to discretise an order parameter on it.
"""
abstract type Discretisation{N} end
Discretisation() = default(:discretisation)
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
    L, H, T, V = operators(s::Superfluid, d::Discretisation)

Return operator functions that act on sample(ψ, d)

TODO specify which operators are returned when s and d are missing

TODO does s.hbm affect J?  Presumably not after we divide by hbar
"""
function operators(s::Superfluid, d::Discretisation)
    Δ, U, J = operators(d)
    W = sample(s.V, d)
    V(ψ) = W.*ψ
    
    T(ψ) = -s.hbm/2*Δ(ψ)
    L(ψ) = T(ψ)+V(ψ)+s.C*U(ψ)
    L(ψ,Ω) = L(ψ)-Ω*J(ψ)
    H(ψ) = T(ψ)+V(ψ)+s.C/2*U(ψ)
    H(ψ,Ω) = H(ψ)-Ω*J(ψ)
    L, H, T, V
end

operators() =
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
include("dynamics.jl")

end # module
