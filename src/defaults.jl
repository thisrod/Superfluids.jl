# Default parameters

default(k::Symbol) = get(_defaults, k, nothing)
default() = copy(_defaults)
default!(k::Symbol, v) = (_defaults[k] = v)
default!(d::Dict{Symbol,Any}) = (_defaults = copy(d); nothing)
default!(v::Superfluid) = default!(:superfluid, v)
default!(v::Discretisation) = default!(:discretisation, v)

# TODO extend Documenter to format docstrings for default symbols, and
# warn about symbols in the default table that are not documented.

# TODO ask why optional positional arguments must come at the end.
# (Kind of similar to why Julia doesn't add multiple dispatch when it already handles ambiguity.)

# Abstract syntax types

struct Xpn{S} <: AbstractVector{Any}
    args
end

const MethodDefinition = Union{Xpn{:(=)}, Xpn{:function}}

# Generate default arguments
#
# If the expression m is a call, define the function and methods that allow
# initial arguments of types Superfluid and Discretisation to be elided.
#
# If m is a method definition, keyword arguments with no explicit
# default are looked up in defaults

macro defaults(E) defaults(xpn(E)) |> esc end

defaults(E::Xpn{:call}) = 
    quote
        function $(called_function(E)) end
        $(default_methods(E)...)
    end

defaults(E::MethodDefinition) = default_kwargs(E)

# The whole generated code is escaped from hygene, so default is
# inserted as a function, not as a symbol to be evaluated in the
# module where @defaults is expanded.

default_methods(E) = default_methods(E, method_types(E))
default_methods(E, ts) = [elided(E, e) for e in power_set(elisions(ts...)...)]

elisions(::Type{Superfluid}, _...) = [1=>:superfluid]
elisions(::Type{Discretisation}, _...) = [1=>:discretisation]
elisions(::Type{Superfluid}, ::Type{Discretisation}, _...) =
    [1=>:superfluid, 2=>:discretisation]
elisions(_, t::Union{Type{Superfluid}, Type{Discretisation}}, ts...) =
    [(j+1)=>s for (j,s) in elisions(t, ts...)]

function elided(E, ds)
    # TODO convert xs::Vararg{T} to xs...
    f(E::Symbol) = E
    f(E::Xpn{:(::)}) = 
        if E[2] isa Xpn{:curly} && E[2][1] == :Vararg
            :( $(E[1])... ) |> xpn
        else
            E[1]
        end
    f(E::Xpn{:(...)}) = E
    f(E::Xpn{:kw}) = f(E[1])
    as = args(E)
    bs = Any[f(a) for a in as]
    for d in ds
        bs[d[1]] = :( $default($(QuoteNode(d[2]))) ) |> xpn
    end
    as = as[setdiff(eachindex(as), first.(ds))]
    Xpn{:function}([method_call(E, as), method_call(E, bs)]) |> expr
end

power_set() = []
power_set(a) = [[a]]
power_set(a,b) = [[a], [b], [a,b]]

"Add defaults for keyword arguments" 
function default_kwargs(E)
    as, ps = aps(E)
    Xpn{:function}([method_call(E, as, add_defaults(ps)), body(E)]) |> expr
end

# This doesn't handle typed keyword parameters
add_defaults(ps) = [
    (p isa Symbol && haskey(_defaults, p)) ?
    Xpn{:kw}([p, xpn(:( $default($(QuoteNode(p))) ))]) :
    p
    for p in ps]

# Abstract syntax

xpn(E) = E
xpn(E::Expr) = Xpn(E)
Xpn(E::Expr) = Xpn{E.head}(xpn.(E.args))
Base.size(E::Xpn) = size(E.args)
Base.getindex(E::Xpn, i) = E.args[i]
expr(E) = E
expr(E::Xpn{S}) where S = Expr(S, expr.(E)...)

called_function(E::Xpn{:call}) = E[1]
called_function(E::MethodDefinition) = called_function(E[1])
body(E::MethodDefinition) = E[2]

args(E::Xpn) = aps(E)[1]
kwargs(E::Xpn) = aps(E)[2]
aps(E::MethodDefinition) = aps(E[1])
function aps(E::Xpn{:call})
    arg_list = E[2:end]
    if arg_list[1] isa Xpn{:parameters}
        arg_list[2:end], collect(arg_list[1])
    else
        arg_list, Xpn[]
    end
end

method_call(E::MethodDefinition, xs...) = method_call(E[1], xs...)
method_call(E::Xpn{:call}, args, kwargs=[Xpn(:(kwargs...))]) =
    :( $(called_function(E))(
        $(expr.(args)...);
        $(expr.(kwargs)...))) |> Xpn

function method_types(m::Xpn)
    t(x::Symbol) = :Any
    t(E::Xpn{:(::)}) = E[2]
    t(E::Xpn{:(...)}) = t(only(E))
    t(E::Xpn{:kw}) = t(E[1])
    # In principle, type identifiers should be evaluated in the module
    # where @defaults is expanded.  In practice that is always Superfluids.
    eval.(t.(args(m)))
end

_defaults = Dict{Symbol,Any}(
    :hbm => 1,
    :g_tol => 1e-6,
    :iterations => 1000,
    :dt => 1e-4,
    :relaxer => Optim.ConjugateGradient,
    :integrator => DifferentialEquations.RK4,
    :diagonizer => nothing,# eigenproblem solver
    :xlims => nothing,
    :ylims => nothing,
)

const OPT_ARGS = (:g_tol, :iterations, :relaxer)
const DIFEQ_ARGS = (:dt, :formula)
