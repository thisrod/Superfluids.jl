using Superfluids
using LinearAlgebra: normalize!, dot, norm

s = Superfluid{2}(3000, (x,y) -> x^2+y^2)
d1 = FDDiscretisation(s, 30)
d2 = FDDiscretisation(s, 50)

rv = Complex(1.5, 0.3)
f1(z) = @. exp(-abs2(z)/18)
f2(z) = @. f1(z)*(z-rv)/√(abs2(z-rv)+1)

q1 = f2.(argand(d1))
normalize!(q1)
q2 = f2.(argand(d2))
normalize!(q2)

U1 = Superfluids.operators(s,d1,:U) |> only
U2 = Superfluids.operators(s,d2,:U) |> only

function interpolate_and_compare(op, q, ds...)
    zs = Complex.(randn(10),randn(10))
    
end

# check derivatives of cloud(d)

φ = ground_state(S,D)