# Bogoliubov-de Gennes problems

function BdGmatrix(s, d, Ω, ψ)
    # TODO make this a BlockBandedMatrix
    T, V, U, J, L = matrices(s, d, Ω, ψ, :T, :V, :U, :J, :L)
    Q = diagm(0=>ψ[:])
    μ = sum(conj(ψ[:]).*(L*ψ[:])) |> real |> UniformScaling
    [
        T+V+2U-μ-Ω*J   -s.C/d.h*Q.^2;
        s.C/d.h*conj.(Q).^2    -T-V-2U+μ-Ω*J
    ]
end