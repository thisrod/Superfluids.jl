# Finite difference discretisation

"""
    FDDiscretisation

Finite difference discretisation on square domain
"""
struct FDDiscretisation{N} <: Discretisation{N}
    n::Int
    h::Float64			# Axis steps
    Ds
    xyz::NTuple{N,Array{Float64,N}}
    
    function FDDiscretisation{2}(n::Int, h::Float64)
        x = h/2*(1-n:2:n-1)
        x = reshape(x, n, 1)
        Ds = [(1/h).*op(n, Float64[-1/2, 0, 1/2]),
            (1/h^2).*op(n, Float64[1, -2, 1])]
        new(n, h, Ds, (x,x'))
    end
end

# Maximum domain size l, limit maximum V to nyquist T
FDDiscretisation(s::Superfluid{N}, n, l=Inf) where N =
    FDDiscretisation{N}(n, min(l/(n+1), sqrt(√2*π/n)))
FDDiscretisation(n, l) = FDDiscretisation(default(:superfluid), n, l)

function D(u, j, d::FDDiscretisation, n)
    if j == 1
        d.Ds[n]*u
    elseif j == 2
        u*d.Ds[n]'
    else
        error("NYI")
    end
end

function op(n, stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],n-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (n,n))
end

function primitive_operators(d::FDDiscretisation{2})
    x, y = d.xyz
    
    Δ(ψ) = D(ψ,1,d,2)+D(ψ,2,d,2)
    # φ is normalised as a number, so the density is φ/h^N
    U(φ,ψ) = @. abs2(φ)/d.h^2*ψ
    J(ψ) = -1im*(x.*D(ψ,2,d)-y.*D(ψ,1,d))
    Δ, U, J
end

"""
    matrices(s::Superfluid{2}, d::FDDiscretisation{2}, Ω, ψ, syms::Vararg{Symbol})

Like operators, but as matrices instead of functions.

The argument ψ is an order parameter to linearise U about.
This is the kind of place that TensArrays will be useful.
"""
function matrices(s::Superfluid{2}, d::FDDiscretisation{2}, Ω, ψ, syms::Vararg{Symbol})
    x, y = d.xyz
    ∂, ∂² = d.Ds
    V = diagm(0=>sample(s.V, d)[:])
    eye = Matrix(I,d.n,d.n)
    ρ = diagm(0=>abs2.(ψ[:]))
    
    T = -kron(eye, ∂²)/2 - kron(∂², eye)/2
    U = s.C/d.h*ρ
    J = -1im*(repeat(x,1,d.n)[:].*kron(∂,eye)-repeat(y,d.n,1)[:].*kron(eye,∂))
    L = T+V+U-Ω*J
    H = T+V+U/2-Ω*J

    ops = Dict(:V=>V, :T=>T, :U=>U, :J=>J, :L=>L, :H=>H)
    isempty(syms) ? ops : [ops[j] for j in syms]
end

sample(f, d::FDDiscretisation) = f.(d.xyz...)

"""
    interpolate(d, ψ)

Return a function that interpolates ψ on (x,y) or z

TODO Make this consistent with the FD interpolants
"""
function interpolate(d::FDDiscretisation{2}, q)
    xs = d.h/2*(1-d.n:2:d.n-1)
    f = Interpolations.CubicSplineInterpolation((xs, xs), q/d.h)
    g(x,y) = f(x,y)
    g(z) = f(real(z), imag(z))
end
