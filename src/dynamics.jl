# The time-dependent GPE

"""
    solve(s, d, ψ₀, (t0, t1))

Return a solution to the GPE

The state at t=0 is ψ₀, ts is a pair of start and stop times.
"""
function solve(s, d, ψ₀, ts; μ=nothing, dt=default(:dt))
    L = operators(s, d, :L) |> only
    isnothing(μ) && (μ = real(dot(L(ψ₀), ψ₀)))
    P = DifferentialEquations.ODEProblem((ψ,_,_)->-1im*(L(ψ)-μ*ψ), ψ₀, ts)
    S = DifferentialEquations.solve(P, DifferentialEquations.RK4(), adaptive=false; dt, saveat=ts[2]-ts[1])
    S[end]
end