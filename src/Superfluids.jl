module Superfluids

export Superfluid, Discretisation, FDDiscretisation, PinnedVortices,
    steady_state, find_vortex, relax_orbit, hartree_modes, bdg_modes

using LinearAlgebra, BandedMatrices, LinearMaps, Optim, Arpack
using Statistics: mean
import DifferentialEquations, Interpolations


# Default parameters

default(k::Symbol) = get(_defaults, k, nothing)
default() = copy(_defaults)
default!(k::Symbol, v) = (_defaults[k] = v; nothing)
default!(d::Dict{Symbol,Any}) = (_defaults=copy(d); nothing)

# define methods for default superfluid and discretisation
macro defaults(m)
    # TODO look at types instead of names s and d
    strip_arg(x::Symbol) = x
    strip_arg(x) = strip_arg(x.args[1])
            
    if (m isa Expr && 
            m.head in [:(=), :function] &&
            m.args[1].head == :call)
        "foo"
    else
        error("Not a method definition")
    end
    f = m.args[1].args[1]
    ps = m.args[1].args[2:end]
    qs = strip_arg.(ps)
    if length(ps) ≥ 2 && ps[1] == :s && ps[2] == :d
        quote
            $f($(ps[3:end]...)) = $f(default(:superfluid), default(:discretisation), $(qs[3:end]...))
            $f(s::Superfluid, $(ps[3:end]...)) = $f(s, default(:discretisation), $(qs[3:end]...))
            $f(d::Discretisation, $(ps[3:end]...)) = $f(default(:superfluid), d, $(qs[3:end]...))
            $m
        end
    elseif length(ps) ≥1 && ps[1] == :s
        quote
            $f($(ps[2:end]...)) = $f(default(:superfluid), $(qs[2:end]...))
            $m
        end
    elseif length(ps) ≥1 && ps[1] == :d
        quote
            $f($(ps[2:end]...)) = $f(default(:discretisation), $(qs[2:end]...))
            $m
        end
    else
        m
    end
end

_defaults = Dict{Symbol,Any}(
    :hbm=>1,
    :g_tol=>1e-6,
    :iterations=>1000,
    :dt=>1e-4,
    :relaxer=>Optim.ConjugateGradient,
    :integrator=>DifferentialEquations.RK4,
    :diagonizer=>nothing	# eigenproblem solver
)

const OPT_ARGS = (:g_tol, :iterations, :relaxer)
const DIFEQ_ARGS = (:dt, :formula)
      

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
sample(f) = sample(f, Discretisation())

function dif! end
function dif2! end

let s = """
    dif!(y, d, a, u; axis)
    dif2!(y, d, a, u; axis)

Derivatives in axpy form

Add `a .* u_axis` to y in place.
"""
    @doc s dif!
    @doc s dif2!
end

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

The `L` and `H` operators accept a `Ω` keyword argument, which
evaluates them in a rotating frame.  The nonlinear `L`, `H` and `U`
operators have a two-argument version, taking a separate density,
such that `L(ψ) = L(ψ,|ψ|²)`.  This is required for eigenproblems.
The density can be complex, e.g. in the BdG operator.  The supplied density
is normalised as a vector, but the operators are evaluated
with it normalised as a wave function.

All operators have mutating versions.  For `L!` and `H!`, these are
the usual type, `L!(u, ρ, ψ; Ω)`.  The others, `V!`, `T!`, `U!` and
`J!`, are in gaxpy form, e.g,, `T!(u,ψ)` adds `T(ψ)` to `u` in place,
similarly `V!`.  A scalar `a` is provided to `U!(u,a,ρ,ψ)` to add
`a*U(ρ,ψ)` to `u`, similarly with `J!(u,a,ψ)`.

TODO DiscretisedSuperfluid memoizes the operators, can record an order
parameter.  `discretise!`
"""
function operators end

function operators(s::Superfluid, d::Discretisation, syms::Vararg{Symbol})
    # TODO does primitive_operators need @inline?
    Δ!, W!, J! = primitive_operators(d)
    Vmat = sample(s.V, d)
    
    V!(u,ψ) = (@. u += Vmat*ψ; u)
    U!(u,a,ψ,ρ=abs2.(ψ)) = (W!(u, s.C*a, ρ, ψ); u)
    T!(u,ψ) = (Δ!(u,-s.hbm/2,ψ); u)
    L!(u,ψ,ρ=abs2.(ψ); Ω=0) = LH!(u,1,ψ,ρ,Ω)
    H!(u,ψ,ρ=abs2.(ψ); Ω=0) = LH!(u,1/2,ψ,ρ,Ω)
    
    function LH!(u,a,ψ,ρ,Ω)
        u .= 0
        T!(u,ψ)
        V!(u,ψ)
        U!(u,a,ψ,ρ)
        J!(u,-Ω,ψ)
        u
    end
    
    T(ψ) = T!(zero(ψ),ψ)
    V(ψ) = V!(zero(ψ),ψ)
    # The primitive J! might not return u
    J(ψ) = (u=zero(ψ); J!(u,1,ψ); u)
    U(ψ,ρ=abs2.(ψ)) = U!(zero(ψ),1,ψ,ρ)
    L(ψ,ρ=abs2.(ψ); Ω=0) = L!(similar(ψ), ψ,ρ; Ω)
    H(ψ,ρ=abs2.(ψ); Ω=0) = H!(similar(ψ), ψ,ρ; Ω)
    
    # Return those requested
    ops = Dict(
        :V=>V, :T=>T, :U=>U, :J=>J, :L=>L, :H=>H,
        :V! =>V!, :T! =>T!, :U! =>U!, :J! =>J!, :L! =>L!, :H! =>H!
    )
    [ops[j] for j in syms]
end


"""
    Δ!, U!, J! = primitive_operators(d::Discretisation)

Return primitive operators

The operators are mutating axpy forms, such that `J!(u,a,ψ)` adds
`a*J(ψ)` to `y`.  These are not required to return `u`.

* `Δ!(u, a, ψ)` is the Lagrangian ``\\psi_{xx}+\\psi_{yy}+\\psi_zz``

* `U!(u, a, ψ, ρ)` is the nonlinear repulsion ``\\rho\\psi``.  This
is where ``\\rho`` is converted from vector to wave function
normalisation.  (The `a` constant could be absorbed into `ρ`.  Should
it be?)

* `J!(u, a, ψ)` is the angular momentum operator ``-i{\\bf r}\\times\\nabla``.

TODO specify how `J` works in 2D and 3D.
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
