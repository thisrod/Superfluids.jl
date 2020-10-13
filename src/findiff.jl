

"""
    matrices(s::Superfluid{2}, d::FDDiscretisation{2}, Ω, ψ, syms::Vararg{Symbol})

Like operators, but as matrices instead of functions.

The argument ψ is an order parameter to linearise U about.
This is the kind of place that TensArrays will be useful.
"""
function matrices(s::Superfluid{2}, d::FDDiscretisation{2}, Ω, ψ, syms::Vararg{Symbol})
    x, y = d.xyz
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

"""
    interpolate(d, ψ)

Return a function that interpolates ψ on (x,y) or z

TODO Make this consistent with the FD interpolants
"""
function interpolate(d::FDDiscretisation{2}, q)
    xs = d.h / 2 * (1-d.n:2:d.n-1)
    f = Interpolations.CubicSplineInterpolation((xs, xs), q / d.h)
    g(x, y) = f(x, y)
    g(z) = f(real(z), imag(z))
end
