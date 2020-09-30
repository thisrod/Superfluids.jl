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
# Finite difference, what's supported
# 
# Fourier spectral
# 
# Complex coordinates in the Argand plane (quaternions in 3D?)
# 
# ### Wave functions
# 
# Everything is normalised with norm(q) = 1, no cleverness about discretisation.
#
# Sets of functions, us, vs, ψ(t), are stored as vectors of arrays.
# 
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
# * `g_tol`  The residual to which order parameters are relaxed.
# 
# * `dt`  The time step for solving the GPE
# 
# # Internals
# 
# ## The discretisation interface
# 
# A discretised wave function is an array.  A normalised wave function should correspond to a normalised array, i.e. absorb the quadrature weights into the coefficients.
# 
# `primitive_operators`
# 
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
