# Rotating frame relaxation of vortex orbits and lattices

"""
    steady_state([s], [d]; Ω=0, rvs=[], initial, [Optim args])

Return a relaxed order parameter

If `Ω` is set to a non-zero value, the state is relaxed in a frame
rotating with that angular velocity.  If `rvs` is non empty, its
elements are interpreted as positions, and the relaxed state is
constrained to have vortices at those positions.

The state `initial` is used as a starting point for the relaxation.
By default, it is set to a smooth cloud with vortices at the `rvs`.
If both `rvs` and `initial` are provided, the initial wave function
should actually have vortices at the pinned locations.

See also: relax
"""
function steady_state(s::Superfluid, d::Discretisation; args...)
    result = relax(s, d; args...)
    Optim.converged(result) || error("Order parameter failed to converge")
    result.minimizer
end

steady_state(;args...) =
    steady_state(default(:superfluid), default(:discretisation); args...)
steady_state(s::Superfluid; args...) =
    steady_state(s, default(:discretisation); args...)
steady_state(d::Discretisation; args...) =
    steady_state(default(:superfluid), d; args...)
       

"""
    relax([s], [d]; Ω=0, rvs=[], initial, [Optim args])

Return an Optim result whose minimizer is `steady_state`
"""
function relax(s::Superfluid{2},
        d::Discretisation{2};
        Ω::Float64=0.0,
        rvs::Vector{Complex{Float64}}=Complex{Float64}[],
        initial=cloud(d, rvs...),
        g_tol=default(:g_tol),
        relaxer=default(:relaxer),
        iterations=default(:iterations)
    )
    
    L, H = operators(s, d, :L, :H)
    Optim.optimize(
        ψ -> dot(ψ,H(ψ,Ω)) |> real,
        (buf,ψ) -> copyto!(buf, 2*L(ψ,Ω)),
        initial,
        relaxer(manifold=PinnedVortices(d, rvs...)),
        Optim.Options(iterations=iterations, g_tol=g_tol, allow_f_increases=true)
    )
end


"""
    cloud(d::Discretisation, rvs...)

Sample a smooth field that will efficiently relax
"""
function cloud(d::FDDiscretisation{2}, rvs::Vararg{Complex{Float64}})
    f(x,y) = cos(π*x/(d.n+1)/d.h)*cos(π*y/(d.n+1)/d.h)
    z = argand(d)
    φ = similar(z)
    φ .= sample(f, d)
    for r in rvs
        @. φ *= (z-r)
    end
    normalize!(φ)
end

cloud(rvs::Vararg{Complex{Float64}}) = cloud(default(:discretisation), rvs...)


"""
    Ω, q = acquire(steady_state args)
    
Find a rotating frame frequency that minimises the residual for the specified vortices
"""

"""
    ps, q = relax_lattice(f, s, d, Ω, ps)

Optimise ps to find a steady lattice in the rotating frame

The lattice has vortices at `f(ps)`.
"""


"""
    Ω, q = relax_orbit(s, d, r; Ωs, g_tol, iterations)

Find a frequency and order parameter with a stable orbit at radius r

The rotating frame energy of a stable orbit is a maximum wrt changes
in the orbit radius, but it is not stationary with respect to Ω;
usually it increases monotonically.  Therefore the only way to find
the steady Ω is to minimize the residual, although this is a bit
inelegant.
"""
function relax_orbit(s, d, r; Ωs, g_tol, iterations)
    #TODO Convergence parameters for optimize
    r = Complex{Float64}[r]
    L = first(operators(s, d))
    function residual(Ω)
        q = relax_field(s,d,r, Ω; g_tol, iterations)
        μ = dot(L(q), q)
        norm(L(q)-μ*q)
    end
    Ω = optimize(residual, Ωs..., abs_tol=g_tol).minimizer
    Ω, relax_field(s,d,r, Ω; g_tol, iterations)
end

PinnedVortices(d::Discretisation, rvs::Vararg{Number}) =
    PinnedVortices(d, [convert(Complex{Float64}, r) for r in rvs]...)
PinnedVortices(rvs::Vararg{Number}) =
    PinnedVortices(default(:discretisation), rvs...)


# Helper functions for circular part winding number
# What if there is a vortex within h of r?

"""
    loopixs(d::Discretisation{2}, r)

Return indices of d circling the origin at radius r
"""
function loopixs(d::Discretisation{2}, r)
    z = argand(d)
    ixs = eachindex(z)
    ixs = ixs[@. r-d.h < abs(z[ixs]) < r+d.h]
    sort!(ixs, by=j->angle(z[j]))
end

function winding(u, ixs)
    hh = unroll(angle.(u[ixs]))/2π
    round(Int, hh[end]-hh[1])
end




