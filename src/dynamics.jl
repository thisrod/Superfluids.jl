# The time-dependent GPE

"""
    integrate(s, d, ψ₀, ts; μ, dt)

Return a solution to the GPE

The state at t=0 is ψ₀, ts is a set of times to return states.
"""
function integrate(s, d, ψ₀, ts; μ=nothing, dt=default(:dt))
    # TODO amend DifferentialEquations so we can use saveat for this
    L = operators(s, d, :L) |> only
    isnothing(μ) && (μ = real(dot(L(ψ₀), ψ₀)))
    qs = [ψ₀]
    tt = ts[1] ≤ sqrt(eps()) ? ts : [0; ts]

    for j  = 2:length(tt)
        P = DifferentialEquations.ODEProblem((ψ,_,_)->-1im*(L(ψ)-μ*ψ), qs[j-1], tt[j-1:j])
        S = DifferentialEquations.solve(P, DifferentialEquations.RK4(), adaptive=false;
            dt, saveat = tt[j]-tt[j-1])
        push!(qs, S[end])
    end
    
    length(tt) == length(ts) ? qs : qs[2:end]
end