# Plot recipes

using RecipesBase, ComplexPhasePortrait

@recipe function f(d::Sampling{2}, u::Matrix{Complex{Float64}})
    x, y = if d isa FDDiscretisation
        d.xyz
    elseif d isa FourierDiscretisation
        x = d.h / 2 * (1-d.n:2:d.n-1)
        x = reshape(x, d.n, 1)
        x, x'
    end

    let xs = default(:xlims), ys = default(:ylims)
        xlims --> (isnothing(xs) ? (x[1], x[end]) : xs)
        ylims --> (isnothing(ys) ? (y[1], y[end]) : ys)
    end
    yflip := false
    aspect_ratio --> 1
    tick_direction --> :out
    x[:], y[:], transpose(saneportrait(u) .* abs2.(u) / maximum(abs2, u))
end

@recipe f(d::Discretisation{2}, u::Matrix{Float64}) = (d, Complex.(u))

saneportrait(u) = reverse(portrait(u), dims = 1)

# TODO plot recipes for animations

# BdG spectra

@userplot BdGSpectrum

@recipe function f(uvs::BdGSpectrum)
    # TODO don't require s for J operator
    s, d, ws, us, vs = uvs.args
    J = operators(s, d, :J) |> only
    js = [dot(u, J(u)) + dot(v, J(v)) |> real for (u, v) in zip(us, vs)]
    seriestype := :scatter
    js, ws
end
