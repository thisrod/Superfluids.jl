# Fourier spectral discretisation with periodic boundary conditions

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

# TODO test this with sin or something that's smooth at the boundary
interpolant(d::FourierDiscretisation{2}, r) = finterp(d, r, 0.0)

# Convolution is easy in a Fourier basis
finterp(d::FourierDiscretisation{2}, r, a) =
    finterp1(d, real(r), a) .* transpose(finterp1(d, imag(r), a))

function finterp1(d, r::Float64, a = 0.0)
    a < eps() && return finterp0(d, r)
    x = first(daxes(d))[:]
    # Centre on a grid point in case a « h
    j = argmin(@. abs(x - r))
    x0 = x[j]
    u = similar(x, Complex{Float64})
    @. u = exp(-(x - x0)^2 / 2a^2)
    fft!(u)
    ks = 2π * fftfreq(d.n, 1 / d.h)
    @. u *= exp(-1im * ks * (r - x0))
    ifft!(u)
    real(u)
end

# sinc shortcut for a = 0
function finterp0(d, r::Float64)
    x = first(daxes(d))[:]
    y = @. sin(π*(x-r)/d.h)/d.n/tan(π*(x-r)/d.n/d.h)
    @. y[abs(x-r) < eps()] = 1
    y
end

function dmats(d::FourierDiscretisation{2})
    # differentiate delta functions
    
    u = zeros(Complex{Float64}, d.n, 1)
    u[1] = 1
    D1 = Array{Complex{Float64}}(undef, d.n, d.n)
    y = zeros(Complex{Float64}, d.n, 1)
    dif!(y, d, 1, u; axis=1)
    for j = 1:d.n
        D1[:, j] = circshift(y, j-1)
    end
    D2 = Array{Complex{Float64}}(undef, d.n, d.n)
    y = zeros(Complex{Float64}, d.n, 1)
    dif2!(y, d, 1, u; axis=1)
    for j = 1:d.n
        D2[:, j] = circshift(y, j-1)
    end
    D1, D2
end

"""
    matrices(s::Superfluid{2}, d::FourierDiscretisation{2}, Ω, ψ, syms::Vararg{Symbol})

Like operators, but as matrices instead of functions.

The argument ψ is an order parameter to linearise U about.
This is the kind of place that TensArrays will be useful.

TODO Combine the FDDiscretisation and FourierDiscretisation methods
into a Sampling method (rename Sampling to Collocation?)
"""
function matrices(s::Superfluid{2}, d::FourierDiscretisation{2}, Ω, ψ, syms::Vararg{Symbol})
    x, y = daxes(d)
    ∂, ∂² = dmats(d)
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
