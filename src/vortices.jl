# Vortex detection and imprinting

"""
    P, Q = poles(u)

Return the Wirtinger derivatives of u, assuming a unit grid step
"""
function poles(u)
    N = size(u, 1)
    @assert N == size(u, 2)
    u = complex.(u)
    rs = (-1:1) .+ 1im * (-1:1)'
    rs /= sum(abs2.(rs))
    conv(u, rs) = [rs .* u[j:j+2, k:k+2] |> sum for j = 1:N-2, k = 1:N-2]
    P = zero(u)
    P[2:end-1, 2:end-1] .= conv(u, conj(rs))
    Q = zero(u)
    Q[2:end-1, 2:end-1] .= conv(u, rs)
    P, Q
end


"""
    unroll(θs)

Return θs adjusted mod 2π to be as continuous as possible
"""
function unroll(θ)
    Θ = similar(θ)
    Θ[1] = θ[1]
    poff = 0.0
    for j = 2:length(θ)
        jump = θ[j] - θ[j-1]
        if jump > π
            poff -= 2π
        elseif jump < -π
            poff += 2π
        end
        Θ[j] = θ[j] + poff
    end
    Θ
end

"""
    find_vortex(d, u)

Return the centre of the sole vortex in u
"""
function find_vortex(d::Discretisation, u, ixs = nothing)
    P, Q = poles(u)
    if isnothing(ixs)
        ixs = abs.(P) .> 0.5maximum(abs.(P))
    end
    regress_core(d, u, ixs)
end

find_vortex(u) = find_vortex(default(:discretisation), u)

function find_vortices(d::Discretisation, u)
    P, Q = poles(u)
    ixs = abs.(P) .> 0.5maximum(abs.(P))
    clusters = cluster_adjacent(adjacent_index, keys(ixs)[ixs])
    [find_vortex(d, u, [j ∈ C for j in keys(u)]) for C in clusters]
end

function find_moat(d::Discretisation, u)
    P, Q = poles(u)
    v = @. (R-w/2 < r < R+w/2)*abs(P+conj(Q))/abs(u)
    ixs = v .> 0.5maximum(v)
    regress_core(d, u, ixs)
end

function regress_core(d, u, ixs)
    z = argand(d)
    ixs = expand_box(box(ixs))[:]
    a, b, c = [z[ixs] conj(z[ixs]) ones(size(z[ixs]))] \ u[ixs]
    (b * conj(c) - conj(a) * c) / (abs2(a) - abs2(b))
end

function expand_box(bb::CartesianIndices{2})
    e1, e2 = ((1, 0), (0, 1)) .|> CartesianIndex
    # TODO raise issue on CartesianIndex half-pregnancy
    if length(bb) == 1
        bb = bb[]
        bb = [
            bb-e1-e2 bb-e1 bb-e1+e2
            bb-e2 bb bb+e2
            bb+e1-e2 bb+e1 bb+e1+e2
        ]
    elseif size(bb, 1) == 1
        bb = [bb .- e1; bb; bb .+ e1]
    elseif size(bb, 2) == 1
        bb = [bb .- e2 bb bb .+ e2]
    end
    bb
end

"Bounding box of cartesian indices"
function box(cixs::AbstractVector{CartesianIndex{2}})
    j1 = k1 = typemax(Int)
    j2 = k2 = typemin(Int)
    for c in cixs
        j1 = min(j1, c[1])
        j2 = max(j2, c[1])
        k1 = min(k1, c[2])
        k2 = max(k2, c[2])
    end
    CartesianIndices((j1:j2, k1:k2))
end

box(ixs::AbstractMatrix{Bool}) = box(keys(ixs)[ixs])


function cluster_adjacent(f, ixs)
    # function f determines adjacency
    clusters = Set()
    for i in ixs
        out = Set()
        ins = Set()
        for C in clusters
            if any(j -> f(i, j), C)
                push!(ins, C)
            else
                push!(out, C)
            end
        end
        push!(out, union(Set([i]), ins...))
        clusters = out
    end
    clusters
end

adjacent_index(j, k) = -1 ≤ j[1] - k[1] ≤ 1 && -1 ≤ j[2] - k[2] ≤ 1

"""
    PinnedVortices([d], rvs, as) <: Optim.Manifold

Constrain a field to have vortices centred at the rvs

TODO document the new point-estimates and Gaussians approach
"""
struct PinnedVortices <: Manifold
    # V'*ψ[:] is a vector of interpolated ψ.(rvs)
    # (1-U*V')*ψ projects the singularities back to the rvs
    # TODO make these Tensars
    # TODO solve a linear system in the u(rvs) to make the projection exact
    U::Matrix{Complex{Float64}}
    V::Matrix{Complex{Float64}}

    function PinnedVortices(
        d::Sampling{2},
        rvs::Vector{Complex{Float64}},
        as::Vector{Float64}
    )
        U = Array{Complex{Float64}}(undef, d.n^2, length(rvs))
        V = similar(U)
        for j in eachindex(rvs)
            U[:, j] = finterp(d, rvs[j], as[j])[:]
            V[:, j] = interpolant(d, rvs[j])[:]
        end
        new(U, V)
    end
end

"""
    finterp(d, r, a)

Return interpolant(d, r) convoluted with a Gaussian of width a.

Subtracting this is a stable way to shift a vortex by further than one pixel.
"""
function finterp end

# use coordinates instead of this

# PinnedVortices(d::Discretisation, rvs::Vararg{Number}; kwargs...) =
#     PinnedVortices(d, [convert(Complex{Float64}, r) for r in rvs]...; kwargs...)
# PinnedVortices(rvs::Vararg{Number}; kwargs...) =
#     PinnedVortices(default(:discretisation), rvs...; kwargs...)

function prjct!(M, q)
    w = q[:]
    q .-= reshape(M.U * (M.V' * w), size(q))
end

# The "vortex at R" space is invariant under normalisation
Optim.retract!(M::PinnedVortices, q) = Optim.retract!(Sphere(), prjct!(M, q))
Optim.project_tangent!(M::PinnedVortices, dq, q) =
    Optim.project_tangent!(Sphere(), prjct!(M, dq), q)
