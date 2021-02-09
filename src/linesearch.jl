# fdf(x, α, d) = f(x + α * d), ∇f(x + α * d)
function strong_backtracking!(fdf, x0, d, y0, grad0;
                              α = one(y0),
                              αmax = convert(typeof(y0), Inf),
                              β = convert(typeof(y0), 1e-4),
                              σ = convert(typeof(y0), 0.5)
                             )
    α = min(α, αmax / 2)
    α_prev = zero(α)
    y_prev = y0
    ylo = y_prev
    yhi = convert(typeof(y0), NaN)
    αlo = αhi = convert(typeof(y0), NaN)
    g0 = dot(grad0, d)
    @assert g0 < 0 "derivative is non-negative: $g0"

    glo = g_prev = g0
    ghi = convert(typeof(g0), NaN)
    # if g0 > 0
    #     g0 = -g0
    #     rmul!(d, -1)
    # end
    mag = max(abs(y0), -α * g0)
    ϵ = sqrt(eps(one(y0))) * mag

    wolfe1 = β * g0
    wolfe2 = -σ * g0

    # bracketing phase
    min_factor = 17/16 # min. factor to expand bracketing interval
    while true
        y, grad = fdf(x0, α, d)
        g = dot(grad, d)
        Δyp = (g + g0) * α / 2 # parabolic approximation
        if abs(Δyp) < ϵ
            Δy = Δyp
        else
            Δy = y - y0
        end
        if Δy > α * wolfe1 || y >= y_prev + ϵ # x >= NaN is always false
            αlo, αhi, ylo, yhi, glo, ghi = α_prev, α, y_prev, y, g_prev, g
            break
        end
        if abs(g) <= wolfe2
            return α
        elseif g >= 0
            αlo, αhi, ylo, yhi, glo, ghi = α_prev, α, y_prev, y, g_prev, g
            break
        end
        # cubic interpolation (Nocedal & Wright 2nd ed., p.59)
        Δα = α - α_prev
        d1 = g_prev + g - 3 * (y - y_prev) / Δα
        det = d1^2 - g * g_prev
        if det < 0
            αnew = min(2, sqrt(αmax / α)) * α
        else
            d2 = sqrt(det)
            αnew = α - Δα * (g + d2 - d1) / (g - g_prev + 2 * d2)
        end
        if α < αnew < αmax && min_factor * α < αnew
            α_prev, α = α, αnew
        else
            α_prev, α = α, α * min(2, sqrt(αmax / α))
        end
        y_prev, g_prev = y, g
    end

    # zoom phase
    # αlo is lower bound, αhi is higher
    small_α = sqrt(eps(one(αlo)))
    for nzoom in 1:200
        # cubic interpolation (Nocedal & Wright 2nd ed., p.59)
        Δα = αhi - αlo
        if Δα < eps(αhi) * 32
            @warn "Step too small; returning"
            return αlo + Δα / 2
        end
        d1 = ghi + glo - 3 * (yhi - ylo) / Δα
        det = d1^2 - ghi * glo
        if det < 0
            α = αlo + Δα / 2
        else
            d2 = sqrt(det)
            α = αlo - Δα * (ghi + d2 - d1) / (ghi - glo + 2 * d2)
            # ensure that new α is in (αlo; αhi) and not too close to bounds
            if !(α - αlo > small_α * αlo && αhi - α > small_α * αhi)
                α = αlo + Δα / 2
            end
        end
        y, grad = fdf(x0, α, d)
        g = grad ⋅ d
        Δyp = (g + g0) * α / 2 # parabolic approximation
        if abs(Δyp) < ϵ
            Δy = Δyp
        else
            Δy = y - y0
        end
        if Δy > α * wolfe1 || y >= ylo + ϵ
            αhi, yhi, ghi = α, y, g
        else
            abs(g) <= wolfe2 && return α
            if g > 0
                αhi, yhi, ghi = α, y, g
            else
                αlo, ylo, glo = α, y, g
            end
        end
    end
end

function strong_backtracking!(fdf, x0, d;
                              α = one(eltype(d)),
                              αmax = convert(eltype(d), Inf),
                              β = convert(eltype(d), 1e-4),
                              σ = convert(eltype(d), 0.5)
                             )
    y0, grad0 = fdf(x0, zero(eltype(d)), d)
    return strong_backtracking!(fdf, x0, d, y0, grad0, α = α, αmax = αmax, β = β, σ = σ)
end