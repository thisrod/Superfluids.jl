# Superfluids.jl

Solve the Gross-Pitaevskiii equation and Bogoliubov-de Gennes eigenproblem

To start, define a 2-dimensional harmonic trap with atomic repulsion

```julia
using Superfluids, Plots
s = Superfluid{2}(500, (x,y) -> x^2+y^2)
```

and discretise it by a high order finite-difference formula on a
moderate sized 66×66 grid

```julia
d = FDDiscretisation{2}(66, 0.3, 7)
```

Crop superfluid plots to the interesting part of the cloud, but leave other plots as is

```julia
Superfluids.default!(:xlims, (-5,5))
Superfluids.default!(:ylims, (-5,5))
```

The ground state is the expected Thomas-Fermi cloud

```julia
ψ₀ = steady_state(s,d)
plot(d, ψ₀, xlims=(-5,5), ylims=(-5,5))
#hide savefig("sf001.png")
```

![plot of harmonic trap ground state cloud](sf001.png)

## GPE dynamics

The state `ψ₀` is simply an `Array{2}`, which can be given a momentum kick as follows

Plotting dynamics gives an animation: `plot(::Discretisation{N}, ts, ψs)'.  Where `ψs` can be an `N+1` array, or a vector of `N` arrays.

## Bogoliubov de-Gennes modes

Find the Kohn mode statically

Animate and compare to GPE solution

## Vortices

Show the energy landscape of a 7-vortex array, find a frame where there is a steady one

Relax to that steady lattice

Add a KT mode

Detect the vortex positions and plot their paths over time

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

