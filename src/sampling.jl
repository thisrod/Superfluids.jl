# Discretisations based on pointwise samples

"""
    abstract type Sampling{N} <: Discretisation{N}

An interface that only requires D!, `sample(d)` and `interpolant` to be defined.
"""
abstract type Sampling{N} <: Discretisation{N} end

sample(f, d::Sampling) = f.(daxes(d)...)

argand(f, d::Sampling) = f.(argand(d))
# TODO figure out why Complex doesn't work
# argand(d::Discretisation) = sample(Complex, d)
argand(d::Sampling) = sample((x, y) -> x + 1im * y, d)

"""
    interpolate(d, u, r)
    
Return the value of u interpolated at r
"""
interpolate(d::Sampling, u, r) = dot(interpolant(d, r), u)

"""
    interpolant(d, r)

Return the dual vector that interpolates a discretised function at r.
"""
function interpolant(d::Sampling, r) end

dif!(y, d::Sampling{2}, a, u; axis) = D!(y, d, a, u, 1, axis)
dif2!(y, d::Sampling{2}, a, u; axis) = D!(y, d, a, u, 2, axis)

function daxes(d::Sampling{2})
    x = d.h / 2 * (1-d.n:2:d.n-1)
    x = reshape(x, d.n, 1)
    (x, transpose(x))
end

function primitive_operators(d::Sampling{2})
    # TODO confirm that using scratch storage this way works
    x, y = daxes(d)

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
