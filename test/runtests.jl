using Test

include("framework.jl")



"""
Consistency checks for operators and matrices:

julia> TT ≈ op2mat(T)

julia> VV ≈ diagm(0=>V[:])

julia> UU ≈ diagm(0=>U(ψ)[:])

julia> JJ ≈ op2mat(J)

function op2mat(f)
    M = similar(z,length(z),length(z))
    u = similar(z)
    for j = eachindex(u)
        u .= 0
        u[j] = 1
        M[:,j] = f(u)[:]
    end
    M
end

Check zero mode for BdGmat
"""

# Check that relaxed orbiting vortices and lattices have the right
# winding number around the trap edge.
