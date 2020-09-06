module Superfluids

export Superfluid, Discretisation, FDDiscretisation, PinnedVortices,
    steady_state, find_vortex, relax_orbit

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
    o1, ... = operators(s::Superfluid, d::Discretisation, s1, ...)

Return Gross-Pitaevskii operators specified by symbols

Let `u` be a normalised vector, that discretises the normalised
wave function ``\\psi(x,y,z)``.  The returned functions are selected
by symbols as follows:

* `:L` for the GPE dynamics operator 

* `:H` for the GPE energy operator

* `:T` for the kinetic energy

* `:V` for the trap potential

* `:U` for the nonlinear repulsion

* `:J` for the angular momentum operator

TODO mutating versions
"""
function operators(s::Superfluid, d::Discretisation, syms::Vararg{Symbol})
    Δ, W, J = primitive_operators(d)
    Vmat = sample(s.V, d)
    
    T(ψ) = -s.hbm/2*Δ(ψ)
    V(ψ) = Vmat.*ψ
    U(φ,ψ) = s.C*W(φ,ψ)
    U(ψ) = U(ψ,ψ)
    L(ψ) = T(ψ)+V(ψ)+U(ψ)
    L(ψ,Ω) = L(ψ)-Ω*J(ψ)
    H(ψ) = T(ψ)+V(ψ)+U(ψ)/2
    H(ψ,Ω) = H(ψ)-Ω*J(ψ)
    ops = Dict(:V=>V, :T=>T, :U=>U, :J=>J, :L=>L, :H=>H)
    [ops[j] for j in syms]
end

operators() =
    operators(default(:superfluid), default(:discretisation), syms...)
operators(s::Superfluid, syms::Vararg{Symbol}) =
    operators(s, default(:discretisation), syms...)
operators(d::Discretisation, syms::Vararg{Symbol}) =
    operators(default(:superfluid), d, syms...)

"""
    Δ, U, J = primitive_operators(d::Discretisation)

Return primitive operators

* `Δ(u)` is the Lagrangian ``\\psi_{xx}+\\psi_{yy}+\\psi_zz``

* `U(v, u)` is the nonlinear repulsion ``|\\phi|^2\\psi``.  This is
where ``\\phi`` is converted from vector to wave function normalisation.

* `J(u)` is the angular momentum operator ``-i{\\bf r}\\times\\nabla``.
TODO specify how this works in 2D and 3D.
"""
function primitive_operators(::Discretisation) end


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
