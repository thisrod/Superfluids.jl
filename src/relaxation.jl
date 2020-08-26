# Rotating frame relaxation of vortex orbits and lattices

"""
    steady_state(s, d; initial, Ω, g_tol, method, iterations)

Return a relaxed order parameter in a rotating  frame

The initial guess φ₀ is relaxed to residual (L-μ)ψ < g_tol.
"""
function steady_state(s::Superfluid, d::Discretisation;
        initial=cloud(d), Ω=0.0, g_tol=default(:g_tol),
        method=ConjugateGradient(manifold=Sphere()),
        iterations=default(:iterations))
    L, H = operators(s, d, :L, :H)
    result = optimize(
        ψ -> dot(ψ,H(ψ,Ω)) |> real,
        (buf,ψ) -> copyto!(buf, 2*L(ψ,Ω)),
        initial,
        method,
        Optim.Options(iterations=iterations, g_tol=g_tol, allow_f_increases=true)
    )
    Optim.converged(result) || error("Ground state failed to converge")
    result.minimizer
end

steady_state(;args...) =
    steady_state(default(:superfluid), default(:discretisation); args...)
steady_state(s::Superfluid; args...) =
    steady_state(s, default(:discretisation); args)
steady_state(d::Discretisation; args...) =
    steady_state(default(:superfluid), d; args)


"""
    cloud(d::Discretisation)

Sample a smooth field that will efficiently relax
"""
function cloud(d::FDDiscretisation{2})
    f(x,y) = cos(π*x/(d.n+1)/d.h)*cos(π*y/(d.n+1)/d.h)
    φ = sample(f, d)
    φ ./= norm(φ)
end

cloud() = cloud(default(:discretisation))

"""
    PinnedVortices([s], [d], rv...) <: Optim.Manifold

Constrain a field to have vortices centred at the rvs

Sampling at the grid points around each rv, let o be the component
of 1 that is orthogonal to (z-rv), and zero on the rest of the grid.
The space orthogonal to every o comprises the fields with a vortex
at every rv (and maybe at other points too).
"""
struct PinnedVortices <: Manifold
   ixs::Matrix{Int}		# 2D array, column of indices for each vortex
   U::Matrix{Complex{Float64}}		# U[i,j] is a coefficient for z[ixs[i,j]]
   function PinnedVortices(s::Superfluid, d::Discretisation, rvs::Vararg{Complex{Float64}})
        z = argand(d)
        ixs = Array{Int}(undef, 4, length(rvs))
        U = ones(eltype(z), size(ixs))
        for (j, rv) = pairs(rvs)
            ixs[:,j] = sort(eachindex(z), by=k->abs(z[k]-rv))[1:4]
            a = normalize!(z[ixs[:,j]].-rv)
            U[:,j] .-= a*(a'*U[:,j])
            # Orthonormalise as we go
            # TODO check for which order Gram-Schmidt is stable
            # Although these are only parallel if two vortices are within a pixel
            for k = 1:j
                a = zeros(eltype(z), 4)
                for m = 1:4
                    n = findfirst(isequal(ixs[m,j]), ixs[:,k])
                    isnothing(n) || (a[n] = U[n,k])
                end
                U[:,j] .-= a*(a'*U[:,j])
            end
            normalize!(U[:,j])
        end
        new(ixs, U)
   end
end

PinnedVortices(s::Superfluid, d::Discretisation, rvs::Vararg{Number}) =
    PinnedVortices(s, d, [convert(Complex{Float64}, r) for r in rvs]...)
PinnedVortices(rvs::Vararg{Number}) =
    PinnedVortices(default(:superfluid), default(:discretisation), rvs...)
PinnedVortices(s::Superfluid, rvs::Vararg{Number}) =
    PinnedVortices(s, default(:discretisation), rvs...)
PinnedVortices(d::Discretisation, rvs::Vararg{Number}) =
    PinnedVortices(default(:superfluid), d, rvs...)

function prjct!(M, q)
    for j = 1:size(M.ixs,2)
        q[M.ixs[:,j]] .-= M.U[:,j]*(M.U[:,j]'*q[M.ixs[:,j]])
    end
    q
end

# The "vortex at R" space is invariant under normalisation
Optim.retract!(M::PinnedVortices, q) =
    Optim.retract!(Sphere(), prjct!(M, q))
Optim.project_tangent!(M::PinnedVortices, dq, q) =
    Optim.project_tangent!(Sphere(), prjct!(M, dq),q)


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




