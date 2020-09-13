using LinearAlgebra
using Superfluids
using Superfluids: argand, cloud, operators

s = Superfluid{2}(3000, (x,y) -> (x^2+y^2)/2)



# Test values of discretisation and complex

fuzerate(::Type{Discretisation}) = [
    FDDiscretisation{2}(30,0.4),
    FDDiscretisation{2}(30,0.5),
    FDDiscretisation{2}(50,0.4),
    FDDiscretisation{2}(50,0.5)
]

fuzerate(::Type{Complex{Float64}}, n=7) =
    Complex.(randn(n),randn(n))

struct WaveFunction{N} end

function fuzerate(::Type{WaveFunction{2}}) = [
    (x,y) -> exp(-abs2(Complex(x,y))/2)),
    (x,y) -> x*exp(-abs2(Complex(x,y))/2)),
    (x,y) -> y*exp(-abs2(Complex(x,y))/2)),
    (x,y) -> (y+2im*x)*exp(-abs2(Complex(x,y))/2)),
    (x,y) -> (y+2im*x)*exp(-abs2(Complex(x,y))/2)),
    (x,y) -> (y+2im*x)*exp(-abs2(Complex(x,y))/2)),
    (x,y) -> Complex(x,y)*exp(-abs2(Complex(x,y))/2)),
    (x,y) -> Complex(x,-y)*exp(-abs2(Complex(x,y))/2))
]

# # Imprinted offset vortex test
# rv = Complex(1.5, 0.3)
# f1(z) = @. exp(-abs2(z)/18)
# f2(z) = @. f1(z)*(z-rv)/√(abs2(z-rv)+1)
# 
# q1 = f2.(argand(d1))
# normalize!(q1)
# q2 = f2.(argand(d2))
# normalize!(q2)
# 
# U1 = Superfluids.operators(s,d1,:U) |> only
# U2 = Superfluids.operators(s,d2,:U) |> only

"f(::Discretisation) = operator, g(z) is wave function"
function consistent_operator(f, g)
    zs = fuzerate(Complex{Float64})
    uzs = []
    for d = fuzerate(Discretisation)
        u = f(d)
        q = g.(argand(d))
        normalize!(q)
        uq = Superfluids.interpolate(d, u(q))
        push!(uzs, uq.(zs))
    end
    zs, uzs
end


function sho_error(d)
    s = Superfluid{2}(0, (x,y) -> (x^2+y^2)/2)
    z = argand(d)
    w = normalize(@. exp(-abs2(z)/2))
    q = relaxed_state(sho, d)
    norm(q - w*(w'*q))
end

"Expectation values of operators in SHO ground state"
function sho_energies(d)
    z = argand(d)
    w = normalize(@. exp(-abs2(z)/2))
    Δ, U, J = Superfluids.operators(d)
    Dict(
        :T=>dot(w,Δ(w)),
        :U=>dot(w,U(w)),
        :J=>dot(w,J(w))
    )
end

# check energies are (1/2, 0, 1/2) for SHO

# check derivatives of cloud(d)

# φ = ground_state(S,D)
# 
# zs, us = consistent_operator(d->only(operators(s, d, :U)), f2)