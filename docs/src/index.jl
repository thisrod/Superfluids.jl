# # Superfluids user manual
# 
# This is a library to solve the Gross-Pitaeveskii equation and the
# Bogoliubov-de Gennes eigenproblem, and plot the resulting order
# parameters.  It is currently most fully developed for the 2-dimensional
# case, but the intention is to support 1 and 3 dimensions as well.
# 
# This library is not intended to implement any numerical methods.
# Those should be in general packages such as `Optim`,
# `DifferentialEquations` and `IterativeSolvers`.  The aim is to
# configure those packages to best handle superfluid problems, and
# let them at it.  Enhancements to the methods should be contributed
# upstream.
# 
# Well, that's the ideal.  Currently, the julia system lacks some
# parts we need, especially when it comes to discretisations and
# transverse derivatives.  Those are currently part of this package,
# but the intention is to get rid of them and use a general solution
# as soon as an adequate one becomes available.
# 
# ## Overview
# 
# To start, define a 2-dimensional harmonic trap with atomic repulsion

using Superfluids, Plots
s = Superfluid{2}(500, (x, y) -> x^2 + y^2)

# and discretise it by a high order finite-difference formula on a
# moderate sized 66×66 grid

d = FDDiscretisation{2}(66, 0.3)

# Crop superfluid plots to the interesting part of the cloud, but leave other plots as is

Superfluids.default!(:xlims, (-5, 5))
Superfluids.default!(:ylims, (-5, 5))

# The ground state is the expected Thomas-Fermi cloud

ψ₀ = steady_state(s, d)
plot(d, ψ₀)
#hide savefig("sf001.png")

# ![plot of harmonic trap ground state cloud](sf001.png)

# ### GPE dynamics
# 
# The state `ψ₀` is simply an `Array{2}`, which can be given a momentum kick as follows

q = ψ₀ .* Superfluids.sample((x, y) -> exp(1im * (x - y / 2)), d)
plot(d, q)

# The dynamics can be solved as follows.  The range is the times
# to return the order paramter.

qs = Superfluids.integrate(s, d, q, 0:0.1:2π)
@animate for q in qs
    plot(d, q)
end
#hide gif(ans, "sf002.gif"; fps=3)

# ![animation of harmonic trap Kohn mode](sf002.gif)

# ### Bogoliubov de-Gennes modes
# 
# Find the Kohn mode statically

ωs, us, vs = bdg_modes(s, d, ψ₀, 0.0, 15, nev = 50)
Superfluids.bdgspectrum(s, d, ωs, us, vs, leg = :none)

# 
# Animate and compare to GPE solution
# 
# ### Vortices
# 
# A novel feature of Superfluids is the ability to relax the order parameter, while constraining the positions of the vortices.  For example, we can plot the energy landscape of a vortex lattice, as a function of the lattice spacing.

Ω = 0.4
uu = @. exp(2π*1im*(0:5)/6)
uu = [0, uu...]
rr = 0.5:0.5:4
qs = [steady_state(s, d; Ω, rvs=r*uu) for r in rr]
H = Superfluids.operators(s, d, :H) |> only
Es = [dot(q, H(q;Ω)) |> real for q in qs]
scatter(rr, Es)

# The function `steady_lattice` relaxes the vortices to a steady lattice.

... = steady_lattice(r -> r*uu, s, d, 2.0)

# Add a KT mode
# 
# Detect the vortex positions and plot their paths over time
# 
# ## Scope
# 
# Superfluids are a class of physical systems, including Bose-Einstein
# condensates, that have global coherence described by an order parameter ``\psi``.
# This package computes order parameters that satisfy the Gross-Pitaevskii
# equation, and related quantities such as Bogoliubov sound wave
# modes.  This section will outline the theory, to define the
# conventions used in the package.  For a full treatment, refer to
# Fetter & Waleka or Pethick & Smith.
# 
# The order parameter theory can be extended to a Monte-Carlo phase
# space treatment, that includes more quantum mechanical effects.
# The package does not currently provide explicit support for this,
# but it would be an obvious extension for the future.
# 
# # The GPE, divided by ``\hbar`` to make every term a rate of change, is
# ```math
# i{\partial\psi\over\partial t} = \left(-{1\over2}{\hbar\over m}\nabla^2 + {V\over\hbar} +
#     {C\over\hbar}|\psi|^2\right)\psi = {\cal L}\psi.
# ```
# Where ``V`` is an external potential, typically from a trap
# containing a dilute gas, and ``C`` is a repulsion constant.  This
# library follows the normalisation convention ``\int |\psi|^2=1``,
# so the number of particles should be absorbed into ``C``.
# 
# For numerical purposes, the factors of 2 are an unsettled matter
# of taste.  Therefore the package provides a parameter `:hbm` for
# the user to set to ``\hbar/m`` in the unit scheme of their choice.

using Superfluids
Superfluids.default!(:hbm, 1)

# When ``\hbar/m`` is made dimensionless, time is identified with
# area, according to the quadratic dispersion relation for de Broglie
# waves.
# 
# In the case that ``V={1\over 2}m\omega x^2`` is a quadratic
# potential, a common convention is to define the time unit by
# ``\omega=1``, and use the characteristic trap length
# ``\sqrt{\hbar/m\omega}`` as a length unit.  With the setting `:hbm
# = 1`, which is the default and used in this manual, the GPE becomes
# ```math
# i\psi_t = {1\over2}\left(-\nabla^2 + x^2 +
#     {C\over\hbar}\left({m\omega\over\hbar}\right)^{N/2}|\psi|^2\right)\psi.
# ```
# Where ``N`` is the dimension of the space on which the GPE is
# being solved, and the repulsion constant is now dimensionless.
# 
# Steady states of the order parameter satisfy a static GPE,
# ```math
# {\cal L}\psi = {\mu\over\hbar}\psi
# ```
# 
# Around a steady state ``\psi_0``, small excitations have the form
# ```math
# \psi(t,x) = \psi_0(x) + \alpha u(x)e^{i\omega t} + \alpha^\ast v^\ast(x)e^{-i\omega t},
# ```
# where ``u`` and ``v`` satisfy the Bogoluibov-de Gennes eigenproblem (TODO check conjugates)
# ```math
# ({\cal L}+C|\psi_0|^2) u + C\psi^2v = \omega u
# ```
# ```math
# ({\cal L}+C|\psi_0|^2) v + C\psi^{2\ast}u = -\omega v
# ```
# 
# 
# ## Superfluids
# 
# Optional (ψ₀, discretisation) pair
# 
# hbm initialised from default
# 
# Potential can be a function, or a (V, discretisation) pair
# 
# ## Rotating frames
# 
# A `Superfluid` can have a rotation rate, but this can be overridden by the `Ω` keyword argument.
# 
# ## Discretisations
# 
# The `Superfluids` package represents order parameters, and other
# fields, as the standard Julia `Array` type.  The abstract type
# `Discretisation{N}`, and its subtypes `FDDiscretisation{N}` and
# `FourierDiscretisation{N}`, map these arrays to continuous fields.
# The discretisation interface is specified in the developer manual.
#
# The following defines a 2nd order finite-difference discretisation, on a 10×10 grid with step 0.3 along each axis

d = FDDiscretisation{2}(10, 0.3)

# Julia functions are discretised as follows

V(x,y) = x^2 + y^2
u = sample(V,d)
ψ(z) = z/√(1+abs2(z))
q = argand(ψ,d)

# and a plot recipe displays them as complex phase portraits, taking
# a discretisation where the axes would usually go

plot(d,q)

# Currently, only scalar fields are implemented.
# 
# TODO Docstrings for `FDDiscretisation` and `FourierDiscretisation`
# 
# For all types of discretisation that allow it, the elements of `u`
# and `q` are multiplied by their quadrature weights, so that `sum(u)`
# is the integral of `V`.  Normalised vectors are very widely supported
# in numerical analysis, so a great deal of glue code can be avoided
# by representing normalised functions as normalised arrays.
# 
# The inverse of `sample` is `interpolate`.

V1 = interpolate(u, d)
V(0.2,0.3) ≈ V1(0.2,0.3) ≈ interpolate(u, d, 0.2, 0.3)

# ### Operators
# 
# These scale with h to keep repulsion consistent
# 
# ### Interpolation
# 
# Interpolating between discretisations
# 
# Interpolating time series between Superfluids with different hbm
# 
# Plotting automatically interpolates
# 
# ## Loading and saving
# 
# Structure with source code, defaults, 
# 
# ## Steady state relaxation
# 
# `steady_state` for TF cloud
# 
# Mutating forms `steady_state!` etc. that store the order parameter in the superfluid.
# 
# `steady_state` for pinned vortices
# 
# ## Vortices and solitons
# 
# In 1D, soliton locations are specified as a vector of real numbers.
# 
# In 2D, vortex core locations are a vector of pairs or complex numbers.
# 
# TODO figure out how to specify vortex lines in 3D (and soliton lines in 2D, if they exist)
# 
# # 
# Relaxing vortex positions with fixed rotation frequency
# 
# Finding a frequency where a set lattice is steady
# 
# ## GPE dynamics
# 
# The `solve` function.  Easy.
# 
# Plotting dynamics gives an animation: `plot(::Discretisation{N}, ts, ψs)'.  Where `ψs` can be an `N+1` array, or a vector of `N` arrays. 
# 
# ## Bogoliubov de-Gennes modes
# 
# The matrix functions
# 
# Normalisation with `norm2(u)-norm2(v) == 1`.
# 
# Plotting a mode set to give J vs ω
# 
# Animating modes
# 
# ## Parameters
# 
# Which `Optim` parameters can be passed on
# 
# Which `DifferentialEquations` parameters
# 
# Almost every function in the Superfluids library must be passed a
# `Superfluid`, or at least a `Discretisation`.  To avoid unnecessary
# typing, these arguments are optional, and there is a mechanism
# borrowed from `Plots.jl` to specify defaults.
# 
# ```@docs
# default
# default!
# ```
# 
# Other arguments are in the same boat:
# 
# * `:g_tol`  The residual to which order parameters are relaxed.
# 
# * `:dt`  The time step for solving the GPE
# 
# * `:xlims`, `:ylims`  As for `Plots.jl`, but applied only when `plot` is called with a `Discretisation` argument. 
# 
# # Manual
#
# Copy the format of `Base` and `Documenter`.
# 
# # Internals
# 
# ## The discretisation interface
# 
# A discretised wave function is an array.  A normalised wave function should correspond to a normalised array, i.e. absorb the quadrature weights into the coefficients.
# 
# `primitive_operators`
# 
# `dif`
# `sample`
# 
# TODO how do vortices work with general discretisatons?  Presumably a function that takes `rv`, and returns the dual vector q -> ψ(rv).
# 
# # ## How the vortex detector works
# 
# ## How the vortex pinning works.  General idea: ψ(r_v) = 0 + ∇(r_v)[dψ]
# + O(dψ²).  Project dψ onto the null space of J.  Retraction is a
# multivariate optimization in concept, but one step in direction
# -∇(rv) works in practice.
# 
# Structured BdG matrices
# 
# ### Normalization
# 
# * Arrays that represent wave functions are normalised with ``l^2``.
# 
# * Arrays that represent energy-valued functions are samples.
# 
# * Wave functions returned by interpolation are ``L^2`` normalized.
# 
# * The tricky bit is ``|\psi|^2`` in ``U(\psi)``, which must be adjusted as if the ``L^2`` norm were 1.
# 
# # Roadmap
# 
# ## Version 1
# 
# * Support FD and Fourier spectral discretisation
# 
# * 1D and 2D are feature complete and documented
# 
# * 3D with vortices is underway
#
# * Multiple components are documented
# 
# * 100% test coverage
# 
# ## Version 2
# 
# * 3D and multiple components are fully implemented
#
# * Support for Wigner sampling initial states
# 
# * Julia has a ':LM' iterative solver, and the necessary sparse matrix inverses, for the BdG eigenproblem
# 
# * Efficient support for `DiffEqOperators`
# 
# ## Future
# 
# * Green's function preconditioning
# 
# * Dynamics can be solved by GPU
# 
# * Fornberg spectral discretisation
# 
# * Manage projection and aliasing for phase-space Monte-Carlo
