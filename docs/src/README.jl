# # Superfluids.jl
# 
# Solve the Gross-Pitaevskiii equation and Bogoliubov-de Gennes eigenproblem
# 
# To start, define a 2-dimensional harmonic trap with atomic repulsion

using Superfluids, Plots
s = Superfluid{2}(500, (x,y) -> x^2+y^2)

# and discretise it by a high order finite-difference formula on a
# moderate sized 66×66 grid

d = FDDiscretisation{2}(66, 0.3)

# Crop superfluid plots to the interesting part of the cloud, but leave other plots as is

Superfluids.default!(:xlims, (-5,5))
Superfluids.default!(:ylims, (-5,5))

# The ground state is the expected Thomas-Fermi cloud

ψ₀ = steady_state(s,d)
plot(d, ψ₀)
#hide savefig("sf001.png")

# ![plot of harmonic trap ground state cloud](sf001.png)

# ## GPE dynamics
# 
# The state `ψ₀` is simply an `Array{2}`, which can be given a momentum kick as follows

q = ψ₀ .* Superfluids.sample((x,y)->exp(1im*(x-y/2)), d)
plot(d,q)

# The dynamics can be solved as follows.  The range is the times
# to return the order paramter.

qs = Superfluids.integrate(s, d, q, 0:0.1:2π)
@animate for q in qs
    plot(d,q)
end
#hide gif(ans, "sf002.gif"; fps=3)

# ![animation of harmonic trap Kohn mode](sf002.gif)

# ## Bogoliubov de-Gennes modes
# 
# Find the Kohn mode statically

ωs, us, vs = bdg_modes(s, d, ψ₀, 0.0, 15, nev=50)
Superfluids.bdgspectrum(s, d, ωs, us, vs, leg=:none)

# 
# Animate and compare to GPE solution
# 
# ## Vortices
# 
# Show the energy landscape of a 7-vortex array, find a frame where there is a steady one
# 
# Relax to that steady lattice
# 
# Add a KT mode
# 
# Detect the vortex positions and plot their paths over time
# 
# 