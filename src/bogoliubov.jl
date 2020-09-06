# Bogoliubov-de Gennes problems

"""
    ωs, us, vs = modes(s, d, ψ, Ω, nev)
    
Return vectors of the lowest energy BdG modes

Normalised with ∥u∥² - ∥v∥² = 1.
"""
function modes(s, d, ψ, Ω, nev)
    N = d.n
    B = BdGmatrix(s,d, Ω, ψ)
    ωs,uvs = Arpack.eigs(B; nev=2*nev, which=:SM)
    @info "max imag frequency" iw=maximum(imag, ωs)
    umode(j) = reshape(uvs[1:N^2, j], N, N)
    vmode(j) = reshape(uvs[N^2+1:end, j], N, N)
    ωs = real(ωs)
    ixs = findall(ωs .> 0)
    ωs[ixs],
        [reshape(uvs[1:N^2, j], N, N) for j = ixs],
        [reshape(uvs[N^2+1:end, j], N, N) for j = ixs]
end

function BdGmatrix(s, d, Ω, ψ)
    # TODO make this a LinearMap
    apop(f) = x->f(reshape(x, d.n, d.n))[:]
    t, v, w, j, l = operators(s, d, :T, :V, :U, :J, :L)
    u(q) = w(ψ,q)
    qq(q) = @. ψ^2*q/d.h^2
    pp(q) = @. conj(ψ)^2*q/d.h^2
    
    μ = UniformScaling(real(dot(ψ, L(ψ))))
    T, V, J, U, QQ, PP = [
        LinearMap{Complex{Float64}}(apop(A), d.n^2) for
            A = t, v, j, u, qq, pp
    ]
        
    [
        T+V+2U-μ-Ω*J   -s.C*QQ;
        s.C*PP    -T-V-2U+μ-Ω*J
    ]
end