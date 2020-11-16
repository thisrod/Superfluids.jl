# Default parameters

default(k::Symbol) = get(_defaults, k, nothing)
default() = copy(_defaults)
default!(k::Symbol, v) = (_defaults[k] = v; nothing)
default!(d::Dict{Symbol,Any}) = (_defaults = copy(d); nothing)

# Generate methods with default arguments
#
# Initial arguments s and d can be elided for superfluid and discretisation
# Keyword arguments with no explicit default are looked up in defaults
macro defaults(m)
    ismethod(m) || error("Not a method definition")
    signature = m.args[1]
    f = signature.args[1]
    p = signature.args[2:end]
    q = strip_arg.(ps)
    if length(ps) ≥ 2 && isarg(p[1], :Superfluid) && isarg(p[2], :Discretisation)
        ps = p[3:end]
        qs = q[3:end]
        quote
            $f($ps...) = $f(default(:superfluid), default(:discretisation), $qs...)
            $f($(p[1]), $ps...) = $f($(q[1]), default(:discretisation), $qs...)
            $f($(p[2]), $ps...) = $f(default(:superfluid), $(q[2]), $qs...)
            $m
        end
    elseif length(ps) ≥ 1 && isarg(ps[1], :Superfluid, :Discretisation)
        quote
            $f($(p[2:end])) = $f(default(:superfluid), $qs...)
            $m
        end
    elseif length(ps) ≥ 1 && ps[1] == :d
        quote
            $f($(ps[2:end]...)) = $f(default(:discretisation), $(qs[2:end]...))
            $m
        end
    else
        m
    end
end

ismethod(m) = m isa Expr && m.head in [:(=), :function] && m.args[1].head == :call

strip_arg(x::Symbol) = x
strip_arg(x) = strip_arg(x.args[1])

isarg(p, syms...) = (p.head == :(::) && p.args[2] in syms)

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
