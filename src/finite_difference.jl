# Finite difference discretisation

"""
    FDDiscretisation

Finite difference discretisation on square domain

TODO per-thread scratch space
"""
struct FDDiscretisation{N} <: Sampling{N}
    n::Int
    h::Float64# Axis steps
    Ds::Any
    scratch::Any

    function FDDiscretisation{2}(n::Int, h::Float64, order = 2)
        Ds = [fdop(n, h, 2order + 1, 1), fdop(n, h, 2order + 1, 2)]
        scratch = [Array{Complex{Float64}}(undef, n, n) for j = 1:Threads.nthreads()]
        new(n, h, Ds, scratch)
    end
end

function fdop(N, h, slength, order)
    ud = (slength - 1) ÷ 2
    D = BandedMatrix{Float64}(undef, (N, N), (ud, ud))
    weights = Array{Float64}(undef, N + 2)
    for j = 1:N
        stencil = (0:N+1) ∩ (j-ud:j+ud)
        weights .= 0
        weights[stencil.+1] = DiffEqOperators.calculate_weights(order, h * j, h * stencil)
        D[j, stencil∩(1:N)] = weights[(stencil∩(1:N)).+1]
    end
    D
end

# Maximum domain size l, limit maximum V to nyquist T
# TODO update to take maximum V, account for hbm
# TODO copy the keyword syntax for Range
FDDiscretisation(s::Superfluid{N}, n, l = Inf) where {N} =
    FDDiscretisation{N}(n, min(l / (n + 1), sqrt(√2 * π / n)))
FDDiscretisation(n, l) = FDDiscretisation(default(:superfluid), n, l)

Base.show(io::IO, ::MIME"text/plain", d::FDDiscretisation{N}) where {N} =
    print(io, "FDDiscretisation{$N}($(d.n), $(d.h))")

function D!(y, d::FDDiscretisation{2}, a, u, n, axis)
    if axis == 1
        y .+= a .* (d.Ds[n] * u)
    elseif axis == 2
        y .+= a .* (u * d.Ds[n]')
    else
        error("Non-existent axis")
    end
end

# TODO test that interpolating z at r gives r
interpolant(d::FDDiscretisation{2}, r::Complex{Float64}) = finterp(d, r, 0.0)

# TODO Sparse matrices for small a
function finterp(d::FDDiscretisation{2}, r::Complex{Float64}, a)
    z = argand(d)
    u = zero(z)
    # TODO less brute force
    ixs = sort(eachindex(z), by = k -> abs(z[k] - r))[1:4]
    for j in ixs
        @. u +=
            exp(-abs2(z - z[j]) / 2a^2) *
            (d.h - abs(real(z[j] - r))) *
            (d.h - abs(imag(z[j] - r))) / d.h^2
    end
end


"""
    matrices(s::Superfluid{2}, d::FDDiscretisation{2}, Ω, ψ, syms::Vararg{Symbol})

Like operators, but as matrices instead of functions.

The argument ψ is an order parameter to linearise U about.
This is the kind of place that TensArrays will be useful.
"""
function matrices(s::Superfluid{2}, d::FDDiscretisation{2}, Ω, ψ, syms::Vararg{Symbol})
    x, y = daxes(d)
    ∂, ∂² = d.Ds
    V = diagm(0 => sample(s.V, d)[:])
    eye = Matrix(I, d.n, d.n)
    ρ = diagm(0 => abs2.(ψ[:]))

    T = -s.hbm * kron(eye, ∂²) / 2 - s.hbm * kron(∂², eye) / 2
    U = s.C / d.h^2 * ρ
    J = -1im * (repeat(x, 1, d.n)[:] .* kron(∂, eye) - repeat(y, d.n, 1)[:] .* kron(eye, ∂))
    L = T + V + U - Ω * J
    H = T + V + U / 2 - Ω * J

    ops = Dict(:V => V, :T => T, :U => U, :J => J, :L => L, :H => H)
    isempty(syms) ? ops : [ops[j] for j in syms]
end
