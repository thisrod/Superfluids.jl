# Discretisations based on pointwise samples


"""
    abstract type Sampling{N} <: Discretisation{N}

An interface that only requires D! to be defined.
"""
abstract type Sampling{N} <: Discretisation{N} end

dif!(y, d::Sampling{2}, a, u; axis) = D!(y, d, a, u, 1, axis)
dif2!(y, d::Sampling{2}, a, u; axis) = D!(y, d, a, u, 2, axis)

function primitive_operators(d::Sampling{2})
    # TODO confirm that using scratch storage this way works
    x, y = if d isa FDDiscretisation
        d.xyz
    elseif d isa FourierDiscretisation
        x = d.h / 2 * (1-d.n:2:d.n-1)
        x = reshape(x, d.n, 1)
        x, x'
    end

    function Δ!(w, a::Number, u)
        for axis = 1:2
            dif2!(w, d, a, u; axis)
        end
    end

    # φ is normalised as a number, so the density is φ/h^N
    U!(w, a, u, r) = (@. w += a * r / d.h^2 * u; w)

    function J!(w, a::Number, u)
        dif!(w, d, 1im * a * y, u, axis = 1)
        dif!(w, d, -1im * a * x, u, axis = 2)
    end

    Δ!, U!, J!
end


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
    xyz::NTuple{N,Array{Float64,N}}
    scratch::Any

    function FDDiscretisation{2}(n::Int, h::Float64, order = 2)
        x = h / 2 * (1-n:2:n-1)
        x = reshape(x, n, 1)
        Ds = [fdop(n, h, 2order + 1, 1), fdop(n, h, 2order + 1, 2)]
        scratch = [Array{Complex{Float64}}(undef, n, n) for j = 1:Threads.nthreads()]
        new(n, h, Ds, (x, x'), scratch)
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

sample(f, d::FDDiscretisation) = f.(d.xyz...)

function D!(y, d::FDDiscretisation{2}, a, u, n, axis)
    if axis == 1
        y .+= a .* (d.Ds[n] * u)
    elseif axis == 2
        y .+= a .* (u * d.Ds[n]')
    else
        error("Non-existent axis")
    end
end

# Fourier pseudospectral discretisation

struct FourierDiscretisation{N} <: Sampling{N}
    n::Int
    h::Float64
end

function D!(y, d::FourierDiscretisation{2}, a, u, n, axis)
    iks = 2π * complex.(0, fftfreq(d.n, 1 / d.h))
    if axis == 2
        iks = transpose(iks)
    end
    buf = copy(u)
    fft!(buf, axis)
    @. buf *= iks^(n)
    ifft!(buf, axis)
    @. y += a * buf
end

function sample(f, d::FourierDiscretisation{2})
    n, h = d.n, d.h
    x = h / 2 * (1-n:2:n-1)
    x = reshape(x, n, 1)
    f.(x, x')
end

"""
    finterp(d,r)

Fourier interpolation weights
"""
function finterp1(d, r::Float64, a=0.0)
    x = first(daxes(d))[:]
    a = max(a, eps())
    # Centre on a grid point in case a « h
    j = argmin(@. abs(x-r))
    x0 = x[j]
    u = similar(x, Complex{Float64})
    @. u = exp(-(x-x0)^2/2a^2)
    fft!(u)
    ks = 2π * fftfreq(d.n, 1 / d.h)
    @. u *= exp(-1im*ks*(r-x0))
    ifft!(u)
    real(u)
end

finterp(d, r, a) = finterp1(d, real(r), a) .* transpose(finterp1(d, imag(r), a))

function daxes(d::Sampling{2})
    x = d.h / 2 * (1-d.n:2:d.n-1)
    x = reshape(x, d.n, 1)
    (x, transpose(x))
end