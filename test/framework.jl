using Superfluids

s = Superfluid{2}(4900, (x,y) -> x^2+y^2)
d = FDDiscretisation(S, 110, 18.5)

# check derivatives of cloud(d)

φ = ground_state(S,D)