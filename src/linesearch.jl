# fdf(x, α, d) = f(x + α * d), ∇f(x + α * d)
function strong_backtracking!(fdf, x0, d, y0, grad0;
                              α = one(eltype(x0)),
                              αmax = convert(eltype(x0), Inf),
                              β = convert(eltype(x0), 1e-4),
                              σ = convert(eltype(x0), 0.5)
                             )
    α_prev = zero(y0)
    y_prev = αlo = αhi = convert(typeof(y0), NaN)
    g0 = grad0 ⋅ d
    if g0 > 0
        g0 = -g0
        d .*= -1
    end
    mag = max(abs(y0), -α * g0)
    ϵ = sqrt(eps(one(y0))) * mag

    wolfe1 = β * g0
    wolfe2 = -σ * g0

    α = min(α, αmax / 2)
    while true
        y, grad = fdf(x0, α, d)
        if y > y0 + α * wolfe1 || y >= y_prev # x >= NaN is always false
            αlo, αhi = α_prev, α
            break
        end
        g = grad ⋅ d
        if abs(g) <= wolfe2
            return α
        elseif g >= 0
            αlo, αhi = α, α_prev
            break
        end
        y_prev, α_prev, α = y, α, α * min(2, sqrt(αmax/α))
    end

    # zoom phase
    ylo, = fdf(x0, αlo, d)
    for nzoom in 1:20_000
        α = (αlo + αhi) / 2
        y, grad = fdf(x0, α, d)
        g = grad ⋅ d
        Δyp = (g + g0) * α / 2 # parabolic approximation
        if abs(Δyp) < ϵ
            Δy = Δyp
        else
            Δy = y - y0
        end
        if Δy > α * wolfe1 || y >= ylo + ϵ
            αhi = α
        else
            if abs(g) <= wolfe2
                return α
            elseif sign(g) == sign(αhi - αlo)
                αhi = αlo
            end
            αlo = α
        end
    end
end

function strong_backtracking!(fdf, x0, d;
                              α = one(eltype(x0)),
                              αmax = convert(eltype(x0), Inf),
                              β = convert(eltype(x0), 1e-4),
                              σ = convert(eltype(x0), 0.5)
                             )
    y0, grad0 = fdf(x0, zero(eltype(x0)), d)
    return strong_backtracking!(fdf, x0, d, y0, grad0, α = α, αmax = αmax, β = β, σ = σ)
end