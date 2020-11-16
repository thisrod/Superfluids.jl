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
    x = first(daxes(d))[:]
    a = max(a, eps())
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
