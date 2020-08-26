# Old-style orbit and lattice finders, before the vortex pinning trick
#
# A good lattice optimiser will allow this cruft to be discarded

"""
    rv, Ω, q = acquire_orbit(s, d, r₀, g_tol, tol=h)
"""
function acquire_orbit(s, d, r₀, g_tol, tol=d.h)
    Ω, q = orbit_frequency(s, d, r₀, g_tol);
    rv = abs(find_vortex(d,q))
    a = r₀ - rv
    @assert a > 0
    
    # Intialise rs to [inside, outside]
    rs = [NaN, NaN]
    for j = 0:4
        r₁ = r₀
        rv = 0.0
        while rv < r₀
            rs[1] = r₁
            r₁ += a/2^j
            Ω, q = orbit_frequency(s, d, r₁, g_tol)
            isnan(Ω) && break
            rv = abs(find_vortex(d,q))
        end
        if isnan(Ω)
            continue
        elseif rv ≥ r₀
            rs[2] = r₁
            break
        end
    end
    any(isnan, rs) && return NaN, NaN, q
    @debug "Bracket" rs
    
    # bisect
    while abs(rv-r₀) > tol
        r₁ = mean(rs)
        Ω, q = orbit_frequency(s, d, r₁, g_tol)
        rv = abs(find_vortex(d, q))
        if rv < r₀
            @debug "Inside" r₁ rv Ω
            j = 1
        else
            @debug "outside" r₁ rv Ω
            j = 2
        end
        @debug "Imprint convergence" r=r₁ rtol=(rs[2]-rs[1])/2
        (rs[j] == r₁ || rs[1] ≥ rs[2]) && return NaN, NaN, q
        rs[j] = r₁
    end
    
    rv, Ω, q
end


"""
    p, q = bisect_parameter(f, s, d, l, c, r, g_tol)

The initial l, c, r are disposable, caller to copy if necessary
l, c, r = f(q, l, c, r, s, d, g_tol)
Find q = steady_state(s, d; ..., g_tol) for parameter c
When result is satisfactory, f returns same l, c, r it was given
In case of unsatisfiability, f returns c = nothing
"""
function bisect_parameter(f, s, d, l, c, r, g_tol)
    # Set up residual step down kludge
    # TODO add rejection to Optim and do this properly
    g_tols = 10 .^ (-3:-0.5:log(10,g_tol))
    if isempty(g_tols) || g_tols[end] ≉ g_tol
        push!(g_tols, g_tol)
    end
    
    # TODO something like similar
    q = Complex.(cloud(d))
    stack = [(l, c, r)]
    
   for g_tol = g_tols
        # Pop until l and r bracket success
        while !isempty(stack)
            l, c, r = pop!(stack)
            b, _, _ = f(q, prevfloat(l), l, r, g_tol)
            b == l || continue
            _, _, b = f(q, l, r, nextfloat(r), g_tol)
            b == r || continue
            push!(stack, (l, c, r))
            break
        end
        isempty(stack) &&
            error("No satisfactory solution between l and r")
        
        # Bisect until c is acceptable at this g_tol
        while true
            c1 = c
            l, c, r = f(q, l, c, r, g_tol)
            
            isnothing(c) && return NaN, q
            c == c1 && break
            push!(stack, (l, c, r))
        end
    end
    
    c, q
end


# Find a stable orbit with frequency Ω by imprinting vortices on q₀
function acquire_frequency(s, d, q₀, rmax, Ω, g_tol, R=nothing)
    z = argand(d)
    
    # find central vortex offset due to moat vortex
    u = z.*copy(q₀)
    u .= steady_state(s, d; initial=u, Ω=0.0, g_tol)
    r₀ = abs(find_vortex(d,u))
    @debug "Central vortex offset" r₀
    # Half pixel tolerance
    r₀ += d.h/2
    @assert r₀ < rmax
    ixs = loopixs(d, rmax)
    
    bisect_parameter(s, d, 2*d.h, rmax/2, rmax, g_tol) do q, l, c, r, g_tol
        l < c < r && r - l > 1e-6 ||
            return nothing, nothing, nothing
        @. q = (z-c)*q₀
        if !isnothing(R)
            @. q *= conj(z-R)
        end
        q .= steady_state(s, d; initial=q, Ω, g_tol)
        w = winding(q, ixs)
        
        if w>1	# Gained extra vortices, give up
            @debug "Gained vortices" l c r w
            nothing, nothing, nothing
        elseif w==0 || (rv = abs(find_vortex(d, q))) > rmax
            @debug "Lost vortex" l c r
            l, mean([l,c]), c
        elseif rv<r₀
            @debug "Central vortex" l c r rv
            c, mean([c,r]), r
        else
            @debug "Orbit" g_tol l c r rv
            l, c, r
        end
    end
end


"""
    Ω, q = bisect_frequency(p, s, d, q₀, Ω₋, Ω₊; residual=1e-4)

Return a frequency and wave function satisfying p

The function p(q, log_success) should return zero on success, a positive (negative) number if the frequency needs to increase (decrease), and NaN if the condition appears to be unsatisfiable.

"""
function bisect_frequency(p, s, d, q₀, Ωs; residual=1e-4)
    
    # Set up residual step down kludge
    # TODO add rejection to Optim and do this properly
    a = log(10,residual)
    g_tols = 10 .^ (-2:-0.5:a)
    if isempty(g_tols) || g_tols[end] ≉ residual
        push!(g_tols, residual)
    end
    
    q₀ /= norm(q₀)
    u = copy(q₀)
    
    mintol = Inf	# best residual yet, log when we beat it
    while true
        # TODO refactor the breaks and continues into tail recursion
        Ω = mean(Ωs)
        (Ωs[1] ≥ Ωs[2] || Ω ∈ Ωs) && return NaN, u
        Ωtol=(Ωs[2]-Ωs[1])/2
        u .= q₀
        
        pp = nothing
       for g_tol = g_tols
            u .= steady_state(s, d; initial=u, Ω, g_tol)
            log_data = (mintol=mintol, g_tol=g_tol, Ω=Ω, Ωtol=(Ωs[2]-Ωs[1])/2)
            @debug "Tried" g_tol Ω Ωtol
            pp = p(u, log_data)
            
            if isnan(pp)
                return NaN, u
            elseif iszero(pp)
                mintol = min(mintol, g_tol)
                continue
            elseif pp < 0
                Ωs[2] = Ω
            elseif pp > 0
                Ωs[1] = Ω
            end
            
            break
        end
        iszero(pp) && return Ω, u
    end
end


"""
    Ω, ψ = orbit_frequency(s, d, r₁, g_tol; Ωs = [0.0, 0.6], R=nothing)

Relax u₀ to a lattice ψ with the same number of vortices between r0 and r1

The function also returns a frequency Ω for a rotating frame in
which ψ is stationary, or NaN if there is no such frequency in the
range Ωs.
"""
function orbit_frequency(s, d, r₁, g_tol; Ωs = [0.0, 0.6], R=nothing)
    z = argand(d)

    # find central vortex offset due to moat vortex
    u = z.*cloud(d)
    u ./= norm(u)
    u .= steady_state(s, d; initial=u, Ω=0.0, g_tol)
    r₀ = abs(find_vortex(d,u))
    @debug "Central vortex offset" r₀
    # Half pixel tolerance
    r₀ += d.h/2
    @assert r₀ < r₁
    
    ixs = loopixs(d, r₁)
    
    q₀ = (z.-r₁).*cloud(d)
    if !isnothing(R)
        @. q₀ *= conj(z+R)
    end
    q₀ ./= norm(q₀)
    
    bisect_frequency(s, d, q₀, Ωs; residual=g_tol) do u, log_data
        w = winding(u, ixs)
    
        if w==0	# Lost the vortex, go faster
            @debug "Lost vortex"
            1
        elseif w>1	# Gained extra vortices, slow down
            @debug "Gained vortices" w
            -1
        elseif (r = abs(find_vortex(d, u))) > r₁
            @debug "Increased radius" r r₁
            1
        elseif r<r₀
            @debug "Central vortex" r
            -1
        else
            @debug "Orbit" r
            0
        end
    end
end
