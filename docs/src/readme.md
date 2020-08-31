## README for Github, uses Literate.jl

## Steady state relaxation

Set up a harmonic trap with l₀ and ω₀ units

`steady_state` 

Plotting wave functions

## GPE dynamics

Demonstrate momentum kick and Kohn mode

Plotting dynamics gives an animation: `plot(::Discretisation{N}, ts, ψs)'.  Where `ψs` can be an `N+1` array, or a vector of `N` arrays.

## Bogoliubov de-Gennes modes

Find the Kohn mode statically

Animate and compare to GPE solution

## Vortices

Show the energy landscape of a 7-vortex array, find a frame where there is a steady one

Relax to that steady lattice

Add a KT mode

Detect the vortex positions and plot their paths over time

