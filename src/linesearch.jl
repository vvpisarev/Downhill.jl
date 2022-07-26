# fdf(x, α, d) = f(x + α * d), ∇f(x + α * d)
"""
    DescentMethods.strong_backtracking!(fdf, x₀, d, [y₀, g₀]; [α, αmax, β, σ])

Find `α` such that `f(x₀ + αd) ≤ f(x₀) + β α d⋅∇f(x)` and `|d⋅∇f(x₀ + αd)| ≤ σ|d⋅∇f(x₀)|`
    (strong Wolfe conditions) using the backtracking line search with cubic interpolation
    and Hager-Zhang approximation when steps are very small. Optionally, `y₀ = f(x₀)` and
    `g₀ = ∇f(x₀)` may be provided to avoid recalculation.

# Arguments
- `fdf`: a function `fdf(x, α, d)` returning a tuple `f(x + α * d), ∇f(x + α * d)` where `f`
    is the minimized function
- `x₀::AbstractVector`: the initial point
- `d::AbstractVector`: the search direction. Must be a descent direction, i.e. `d⋅∇f(x₀) < 0`
- `y₀`: (optional) the value of `f(x₀)` if it is known beforehand
- `g₀`: (optional) the value of `∇f(x₀)` if it is known beforehand

# Keywords
- `α=1`: the initial value of `α`
- `αmax=Inf`: the maximum allowed value of α
- `β=1e-4`: the coefficient in Wolfe conditions
- `σ=0.5`: the coefficient in Wolfe conditions
"""
function strong_backtracking!(
    fdf::F, x0, d, y0::T, grad0;
    α = one(y0),
    αmax = convert(T, Inf),
    β = convert(T, 1e-4),
    σ = convert(T, 0.5)
) where {F,T}
    @logmsg LSLogLevel """

    ==LINE SEARCH START==
    x0 = $(repr(x0))
     d = $(repr(d))
    y0 = $(repr(y0))
    g0 = $(repr(grad0))
    """
    α = min(α, αmax / 2)
    α_prev = zero(α)
    y_prev = y0
    ylo = y_prev
    yhi = convert(T, NaN)
    αlo = αhi = convert(T, NaN)
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

    nbracket_max = 200
    @logmsg LSLogLevel "==BRACKETING THE MINIMUM=="
    for nbracket in 1:nbracket_max
        local y, grad
        while true
            try
                y, grad = fdf(x0, α, d)
                break
            catch err
                if err isa DomainError
                    α_new = (α + α_prev) / 2
                    if α_new == α_prev || α_new == α
                        @warn "==BRACKETING FAIL, LAST STEP VALUE RETURNED==" α
                        @logmsg LSLogLevel "==LINEAR SEARCH INTERRUPTED=="
                        return zero(α)
                    end
                    α = α_new
                else
                    rethrow(err)
                end
            end
        end
        g = dot(grad, d)
        Δyp = (g + g0) * α / 2 # parabolic approximation
        if abs(Δyp) < ϵ
            @logmsg LSLogLevel "" """
                Δyp = $(Δyp) (*)
                Δy = $(y-y0)
            """
            Δy = Δyp
        else
            Δy = y - y0
            @logmsg LSLogLevel "" """
                Δyp = $(Δyp)
                Δy = $(Δy) (*)
            """
        end
        if Δy > α * wolfe1 || y >= y_prev + ϵ # x >= NaN is always false
            αlo, αhi, ylo, yhi, glo, ghi = α_prev, α, y_prev, y, g_prev, g
            @logmsg LSLogLevel "==BRACKETING SUCCESS: FUNCTION CHANGE==" y Δy α * wolfe1 y_prev + ϵ
            break
        elseif abs(g) <= wolfe2
            @logmsg LSLogLevel "==BRACKETING SUCCESS=="
            @logmsg LSLogLevel "==LINEAR SEARCH SUCCESS==" α
            return α
        elseif g >= 0
            αlo, αhi, ylo, yhi, glo, ghi = α_prev, α, y_prev, y, g_prev, g
            @logmsg LSLogLevel "==BRACKETING SUCCESS=="
            break
        end

        nbracket == nbracket_max && error("Failed to find bracketing")
        # cubic interpolation (Nocedal & Wright 2nd ed., p.59)
        Δα = α - α_prev

        # return 0 signaling that bracketing failed
        if iszero(Δα)
            @warn "==BRACKETING FAIL, LAST STEP VALUE RETURNED==" α
            @logmsg LSLogLevel "==LINEAR SEARCH INTERRUPTED=="
            return Δα
        end
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
    @logmsg LSLogLevel "==ZOOM PHASE==" αlo αhi
    for nzoom in 1:200
        # cubic interpolation (Nocedal & Wright 2nd ed., p.59)
        Δα = αhi - αlo
        if Δα < eps(αhi) * 32
            @warn "Step too small; interrupting line search" Δα αhi
            @logmsg LSLogLevel "==LINEAR SEARCH INTERRUPTED=="
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
            if abs(g) <= wolfe2
                @logmsg LSLogLevel "==ZOOM PHASE SUCCESS=="
                @logmsg LSLogLevel "==LINEAR SEARCH SUCCESS==" α
                return α
            elseif g > 0
                αhi, yhi, ghi = α, y, g
            else
                αlo, ylo, glo = α, y, g
            end
        end
        @logmsg LSLogLevel "Zoom iteration $(nzoom)" αlo αhi α
    end
end

function strong_backtracking!(
    fdf::F, x0::AbstractVector, d::AbstractVector{T};
    α = one(T),
    αmax = convert(T, Inf),
    β = convert(T, 1e-4),
    σ = convert(T, 0.5)
) where {F,T}
    y0, grad0 = fdf(x0, zero(T), d)
    return strong_backtracking!(fdf, x0, d, y0, grad0, α = α, αmax = αmax, β = β, σ = σ)
end
