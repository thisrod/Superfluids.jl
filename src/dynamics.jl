# The time-dependent GPE

"""
    solve(s, d, ψ₀, ts)

Return a vector of solutions to the GPE

The state at t=0 is ψ₀, solutions are for the times ts.
"""
function solve(s, d, ψ₀, ts; μ=nothing, dt=default(:dt))
    ts = sort(ts)
    L = operators(s, d) |> first
    isnothing(μ) && (μ = real(dot(L(q), q)))
    P = DifferentialEquations.ODEProblem((ψ,_,_)->-1im*(L(ψ)-μ*ψ), ψ₀, (0.0,ts[end]))
    S = DifferentialEquations.solve(P, DifferentialEquations.RK4(), adaptive=false, dt, saveat=ts)
    [S[j] for j = eachindex(S)]
end