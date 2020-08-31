# Superfluids user manual

This is a library to solve the Gross-Pitaeveskii equation and the
Bogoliubov-de Gennes eigenproblem, and plot the resulting order
parameters.  It is currently most fully developed for the 2-dimensional
case, but the intention is to support 1 and 3 dimensions as well.

This library is not intended to implement any numerical methods.
Those should be in general packages such as `Optim`,
`DifferentialEquations` and `IterativeSolvers`.  The aim is to
configure those packages to best handle superfluid problems, and
let them at it.  Enhancements to the methods should be contributed
upstream.

Well, that's the ideal.  Currently, the julia system lacks some
parts we need, especially when it comes to discretisations and
transverse derivatives.  Those are currently part of this package,
but the intention is to get rid of them and use a general solution
as soon as an adequate one becomes available.

## Units

What GPE this solves

`default(:hbm)` to set hbar/m, but per-`Superfluid` override

Length units are up to you

How to set up a harmonic trap with l₀ and ω₀

## Superfluids

Optional (ψ₀, discretisation) pair

Potential can be a function, or a (V, discretisation) pair

## Rotating frames

A `Superfluid` can have a rotation rate, but this can be overridden by the `Ω` keyword argument.

## Discretisations

Finite difference, what's supported

Fourier spectral

### Wave functions

Everything is normalised with norm(q) = 1, no cleverness about discretisation.

### Operators

These scale with h to keep repulsion consistent

### Interpolation

Interpolating between discretisations

Interpolating time series between Superfluids with different hbm

Plotting automatically interpolates

## Loading and saving

Structure with source code, defaults, 

## Steady state relaxation

`steady_state` for TF cloud

`steady_state` for pinned vortices

Relaxing vortex positions with fixed rotation frequency

Finding a frequency where a set lattice is steady

## GPE dynamics

The `solve` function.  Easy.

Plotting dynamics gives an animation: `plot(::Discretisation{N}, ts, ψs)'.  Where `ψs` can be an `N+1` array, or a vector of `N` arrays. 

## Bogoliubov de-Gennes modes

The matrix functions

Normalisation with `norm2(u)-norm2(v) == 1`.

Plotting a mode set to give J vs ω

Animating modes

## Parameters

Which `Optim` parameters can be passed on

Which `DifferentialEquations` parameters

Almost every function in the Superfluids library must be passed a
`Superfluid`, or at least a `Discretisation`.  To avoid unnecessary
typing, these arguments are optional, and there is a mechanism
borrowed from `Plots.jl` to specify defaults.

```@docs
default
default!
```

Other arguments are in the same boat:

* `g_tol`  The residual to which order parameters are relaxed.

* `dt`  The time step for solving the GPE

# Internals

How the vortex detector works

How the vortex pinning works

Structured BdG matrices