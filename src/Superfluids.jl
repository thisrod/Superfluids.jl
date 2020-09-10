module Superfluids

export Superfluid, Discretisation, FDDiscretisation, PinnedVortices,
    steady_state, find_vortex, relax_orbit

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
operators have a two-argument version, which is linear in the second
argument.

All operators have mutating versions.  For `L!` and `H!`, these are
the usual type, `L!(y, u, v; Ω)`.  The others, `V!`, `T!`, `U!` and
`J!`, are in axpy form.  E.g, `T!(y,u)` adds `T(u)` to `y` in place,
similarly `V!`.  A scalar `a` is provided to `U!(y,a,v,u)` to add
`a*U(v,u)`, similarly with `J!(y,a,u)`.

TODO DiscretisedSuperfluid memoizes the operators, can record an order
parameter.  `discretise!`
"""
function operators end

function operators(s::Superfluid, d::Discretisation, syms::Vararg{Symbol})
    # TODO does primitive_operators need @inline?
    Δ!, W!, J! = primitive_operators(d)
    Vmat = sample(s.V, d)
    
    V!(y,u) = (@. y += Vmat*u; y)
    U!(y,a,v,u=v) = W!(y, s.C*a, v, u)
    T!(y,u) = Δ!(y,-s.hbm/2,u)
    L!(y,v,u=v; Ω=0) = LH!(y,v,u,Ω,1)
    H!(y,v,u=v; Ω=0) = LH!(y,v,u,Ω,1/2)
    
    function LH!(y, u, v, Ω, c)
        y .= 0
        T!(y,u)
        V!(y,u)
        U!(y,c,v,u)
        J!(y,-Ω,u)
    end
    
    T(u) = T!(zero(u),u)
    V(u) = V!(zero(u),u)
    J(u) = J!(zero(u),1,u)
    U(v, u=v) = U!(zero(u),1,v,u)
    L(v, u=v; Ω=0) = L!(similar(u), u, v; Ω)
    H(v, u=v; Ω=0) = H!(similar(u), u, v; Ω)
    
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

The operators are mutating axpy forms, such that `J!(y,a,u)` adds `a*J(u)` to `y`.

* `Δ!(y, a, u)` is the Lagrangian ``\\psi_{xx}+\\psi_{yy}+\\psi_zz``

* `U!(y, a, v, u)` is the nonlinear repulsion ``|\\phi|^2\\psi``.  This is
where ``\\phi`` is converted from vector to wave function normalisation.

* `J!(y, a, u)` is the angular momentum operator ``-i{\\bf r}\\times\\nabla``.

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
