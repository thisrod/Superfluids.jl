# Default parameters

default(k::Symbol) = get(_defaults, k, nothing)
default() = copy(_defaults)
default!(k::Symbol, v) = (_defaults[k] = v; nothing)
default!(d::Dict{Symbol,Any}) = (_defaults = copy(d); nothing)
default!(v::Superfluid) = default!(:superfluid, v)
default!(v::Discretisation) = default!(:discretisation, v)

# TODO extend Documenter to format docstrings for default symbols, and
# warn about symbols in the default table that are not documented.

# Generate methods with default arguments
#
# Initial arguments s and d can be elided for superfluid and discretisation
# Keyword arguments with no explicit default are looked up in defaults

# Dispatch on the types of the parameters

const SoD = Union{Superfluid, Discretisation}

macro defaults(m)
    ismethod(m) || error("Not a method definition")
    # first, write out function f end to hook the docstring
    quote
        $(default_methods(m, method_types(m)...))
    end
#     signature = m.args[1]
#     f = signature.args[1]
#     p = signature.args[2:end]
#     q = strip_arg.(ps)
#     if length(ps) ≥ 2 && isarg(p[1], :Superfluid) && isarg(p[2], :Discretisation)
#         ps = p[3:end]
#         qs = q[3:end]
#         quote
#             $f($ps...) = $f(default(:superfluid), default(:discretisation), $qs...)
#             $f($(p[1]), $ps...) = $f($(q[1]), default(:discretisation), $qs...)
#             $f($(p[2]), $ps...) = $f(default(:superfluid), $(q[2]), $qs...)
#             $m
#         end
#     elseif length(ps) ≥ 1 && isarg(ps[1], :Superfluid, :Discretisation)
#         quote
#             $f($(p[2:end])) = $f(default(:superfluid), $qs...)
#             $m
#         end
#     elseif length(ps) ≥ 1 && ps[1] == :d
#         quote
#             $f($(ps[2:end]...)) = $f(default(:discretisation), $(qs[2:end]...))
#             $m
#         end
#     else
#         m
#     end
end

default_methods(m, ::Type{Superfluid}, _...) =
    method_aps(m) do as, _
        quote
            $(method_call(m, as[2:end])) =
                $(method_call(m, [:(default(:superfluid)), as[2:end]...]))
        end
    end

# TODO implement abstract syntax à la SICP in Base.Meta
# even better, replace ex.head with subtypes of expression

ismethod(m) = false
ismethod(m::Expr) =
    m.head in [:(=), :function] &&
    m.args[1].head == :call

method_call(m) = m.args[1]
method_call(m, arguments, parameters=[:(kwargs...)]) =
    :( $(method_function(m))(
        $(arguments...);
        $(parameters...)))
method_function(m) = method_call(m).args[1]
method_arguments(m) = method_aps(m)[1]
method_parameters(m) = method_aps(m)[2]

function method_aps(m)
    arg_list = method_call(m).args[2:end]
    if arg_list[1].head == :parameters
        arg_list[2:end], arg_list[1].args
    else
        arg_list, []
    end
end
method_aps(f, m) = f(method_aps(m)...)

function method_types(m::Expr)
    t(x::Symbol) = Any
    t(x::Expr) = eval(x.args[2])
    t.(method_arguments(m))
end

# strip_arg(x::Symbol) = x
# strip_arg(x) = strip_arg(x.args[1])

# isarg(p, syms...) = (p.head == :(::) && p.args[2] in syms)

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
