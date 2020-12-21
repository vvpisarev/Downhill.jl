using LinearAlgebra

const GAS_CONSTANT_SI = 8.31446261815324

"""
    solve_cubic(a::Real, b::Real, c::Real, d::Real)

Finds real roots of a cubic equation
```math
a x^3 + b x^2 + c x + d = 0
```

Return Tuple{Float64, Float64, Float64} where first roots are real,
complex roots are defined to `NaN`

Reference: J. F. Blinn, "How to Solve a Cubic Equation, Part 5: Back to Numerics,"
in IEEE Computer Graphics and Applications, vol. 27, no. 3, pp. 78-89, May-June 2007.
doi: 10.1109/MCG.2007.60 
"""
function solve_cubic(a::Real, b::Real, c::Real, d::Real)::NTuple{3,Float64}
    # convert to Ax³ + 3Bx² + 3Cx + D = 0
    A, B, C, D = Float64(a), b / 3.0, c / 3.0, Float64(d)
    δ₁ = A * C - B * B
    δ₂ = A * D - B * C
    lostbits2 = -log2(abs(A*D/(B*C) - 1))
    δ₃ = B * D - C * C
    d13 = δ₁ * δ₃
    d22 = δ₂ * δ₂
    Δ = 4 * d13 - d22
    Δ = 52 + log2(abs(Δ/d22)) < lostbits2+1 ? 0.0 : Δ

    if Δ < 0
        At, Cb, Db = 0., 0., 0. # A-tilde, C-bar, D-bar
        if B^3 * D >= A * C^3
            At, Cb, Db = A, δ₁, -2 * B * δ₁ + A * δ₂
        else
            At, Cb, Db = D, δ₃, -D * δ₂ + 2 * C * δ₃
        end
        T₀ = -copysign(At, Db) * sqrt(-Δ)
        T₁ = -Db + T₀
        p = cbrt(0.5 * T₁)
        q = T₁ == T₀ ? -p : -Cb / p
        x₁ = Cb <= 0 ? p + q : -Db / (p^2 + q^2 + Cb)
        x, w = B^3 * D >= A * C^3 ? (x₁ - B, A) : (-D, x₁ + C)
        return (x/w, NaN, NaN)
    else
        δ₁ == δ₂ == δ₃ == 0 && return (-B/A, -B/A, -B/A)
        sΔ = sqrt(Δ)
        θA, θD = (atan(A*sΔ, 2*B*δ₁ - A*δ₂), atan(D*sΔ, D*δ₂ - 2*C*δ₃)) ./ 3 .|> abs
        sCA, sCD = sqrt(-δ₁), sqrt(-δ₃)
        x₁A, x₁D = 2*sCA*cos(θA), 2*sCD*cos(θD)
        x₃A, x₃D = -sCA*(cos(θA)+sqrt(3)*sin(θA)), -sCD*(cos(θD)+sqrt(3)*sin(θD))
        xlt = (x₁A+x₃A > 2*B) ? x₁A : x₃A
        xst = (x₁D+x₃D < 2*C) ? x₁D : x₃D
        xl, wl = xlt - B, A
        xs, ws = -D, xst + C
        Δ == 0 && return (xs/ws, xl/wl, NaN)
        E = wl * ws
        F = -xl * ws - wl * xs
        G = xl * xs
        xm, wm = C * F - B * G, C * E - B * F
        return (xs/ws, xm/wm, xl/wl)
    end
end

struct BrusilovskyEoSComponent
    # meta information
    name::String
    
    # physical parameters
    Pc::Float64  # critical pressure
    Tc::Float64  # critical temperature
    acentric_factor::Float64
    RTc::Float64   # R * critical temperature
    molar_mass::Float64  # [kg mol⁻¹] molar mass 
    carbon_number::Int64  # [dimless] number of carbons

    # eos parameters
    ac::Float64     # explicit coefficient of the eos a_c
    b::Float64      # explicit coefficient of the eos b
    c::Float64      # explicit coefficient of the eos c
    d::Float64      # explicit coefficient of the eos d
    Psi::Float64    # primary coefficient of the eos - \Psi
    
    function BrusilovskyEoSComponent(
        ;
        name::String="no name",
        critical_pressure::Real=NaN,
        critical_temperature::Real=NaN,
        acentric_factor::Real=NaN,
        Omegac::Real=NaN,
        Zc::Real=NaN,
        Psi::Real=NaN,
        molar_mass::Real=NaN,
        carbon_number::Int64=0
    )
        alpha = Omegac^3
        beta = Zc + Omegac - 1.0
        ds = sqrt(Omegac - 0.75)
        sigma = -Zc + Omegac * (0.5 + ds)
        delta = -Zc + Omegac * (0.5 - ds)
        RTc = GAS_CONSTANT_SI * critical_temperature
        rtp = RTc / critical_pressure
        ac = alpha * RTc * rtp
        b = beta * rtp
        c = sigma * rtp
        d = delta * rtp
        new(
            name,
            critical_pressure,
            critical_temperature,
            acentric_factor,
            RTc,
            molar_mass,
            carbon_number,
            ac, b, c, d, Psi
        )
    end
end

"""
    brusilovsky_a_coef(substance::BrusilovskyEoSComponent, RT::Real)

Returns EoS coefficient ``a(T, Ψ) = a_c ϕ(T, Ψ)`` of `substance`.
```math
ϕ(T, psi) = [ 1 + Ψ (1 - T_r^0.5) ]^2
```
Reference: Brusylovsky2002[section: 5.5.2 (algorithm step 3), eq: (4.34), see p.142 and p.164]
"""
function brusilovsky_a_coef(substance::BrusilovskyEoSComponent, RT::Real)
    psi = substance.Psi
    phi = ( 1 + psi * (1 - sqrt(RT / substance.RTc)) )^2
    return substance.ac * phi
end

function brusilovsky_compressibility(substance::BrusilovskyEoSComponent,
                                     pressure::Real,
                                     RT::Real,
                                     phase::AbstractChar = 'g')

    prt = pressure / RT
    am = brusilovsky_a_coef(substance, RT) * prt / RT
    bm = substance.b * prt
    cm = substance.c * prt
    dm = substance.d * prt
    zf = solve_cubic(
        1.0,
        cm + dm - bm - 1.0,
        am - bm * cm + cm * dm - bm * dm - dm - cm,
        -bm * cm * dm - cm * dm - am * bm
    )
    if phase == 'g'
        return maximum(z for z in zf if z > 0.0) # NaNs are filtered too
    else
        return minimum(z for z in zf if z > 0.0) # NaNs are filtered too
    end
end

function logΦ(subst::BrusilovskyEoSComponent, V, RT, nmoles)
    b, c, d = subst.b, subst.c, subst.d
    a = brusilovsky_a_coef(subst, RT)
    B, C, D = nmoles .* (b, c, d)
    A = a * nmoles / RT
    return log(1 - B / V) - B / (V - B) + 
           A / (C - D) * log((V + C) / (V + D)) + 
           A * V / ((V + C) * (V + D))
end

function wilson_psat(subst, RT)
    return subst.Pc * exp(5.37 * (1 + subst.acentric_factor) * (1 - subst.RTc / RT))
end

"""
    brusilovsky_pressure(substance::BrusilovskyEoSComponent, mvol::Real, RT::Real)

Computes pressure of fluid component `substance` at given molar volume `mvol` 
and thermal energy `RT`.
"""
function brusilovsky_pressure(substance::BrusilovskyEoSComponent, mvol::Real, RT::Real)
    acoeff = brusilovsky_a_coef(substance, RT)
    b = substance.b
    c = substance.c
    d = substance.d
    return RT / (mvol - b) - acoeff / ((mvol + c) * (mvol + d))
end

const methane = BrusilovskyEoSComponent(
                    name = "CH4",
                    critical_pressure = 4.5992e6,
                    critical_temperature = 190.56,
                    acentric_factor = 0.01142,
                    Omegac = 0.7563,
                    Zc = 0.33294,
                    Psi = 0.37447,
                    molar_mass = 0.016043)

function vt_stability(subst, RT, molar_dens; 
                      optmethod::T = DescentMethods.BFGS
                     ) where T<:Type{<:DescentMethods.DescentMethod}
    chempot(rho) = log(rho) - logΦ(subst, 1.0, RT, rho)
    μ0 = chempot(molar_dens)
    p_orig = brusilovsky_pressure(subst, 1/molar_dens, RT) / RT
    function Dfunc!(rho, dro)
        Δμ = chempot(rho[1]) - μ0
        d = Δμ * rho[1] - brusilovsky_pressure(subst, 1/rho[1], RT) / RT
        dro[1] = Δμ
        return d, dro
    end

    P_init = wilson_psat(subst, RT)
    z1 = brusilovsky_compressibility(subst, P_init, RT, 'g')
    z2 = brusilovsky_compressibility(subst, P_init, RT, 'l')
    rho1, rho2 = P_init ./ (RT .* (z1, z2))

    rhovec = [rho1]
    dd = [0.0]
    
    function maxstep(x, d)
        to_zero = -x[1] / d[1]
        to_b = (1 / subst.b - x[1]) / d[1]
        return minimum(a for a in (to_zero, to_b) if a > 0)
    end
    opt = optmethod(rhovec)
    #opt.α0 = 1e-8
    optresult = DescentMethods.optimize!(opt,
                                         Dfunc!,
                                         rhovec,
                                         gtol = 1e-9,
                                         maxiter = 1000,
                                         maxcalls = 2000,
                                         constrain_step = maxstep)
    p_gas = brusilovsky_pressure(subst, 1/optresult.argument[1], RT) / RT
    if p_gas >= p_orig && 
        norm(x - y for(x, y) in zip(rhovec, optresult.argument)) > 1e-5 * norm(rhovec) #+ sqrt(eps(one(p_orig))) * abs(p_orig)
        return optresult, p_gas, p_orig
    end
    rhovec[1] = rho2
    optresult = DescentMethods.optimize!(opt, 
                                         Dfunc!,
                                         rhovec,
                                         gtol = 1e-9,
                                         maxiter = 1000,
                                         maxcalls = 2000, 
                                         constrain_step = maxstep)
    p_liq = brusilovsky_pressure(subst, 1/optresult.argument[1], RT) / RT
    return optresult, p_liq, p_orig
end

function vt_flash(subst, RT, molar_dens; 
                  optmethod::T = DescentMethods.BFGS
                 ) where T<:Type{<:DescentMethods.DescentMethod}
    chempot(rho) = log(rho) - logΦ(subst, 1.0, RT, rho)
    μ0 = chempot(molar_dens)
    p_orig = brusilovsky_pressure(subst, 1/molar_dens, RT) / RT
    
    A0 = μ0 * molar_dens - p_orig
    function twophaseA!(rhov, grad)
        mol1, V1 = rhov[1], rhov[2]
        mol2, V2 = molar_dens - mol1, 1 - V1
        p1 = brusilovsky_pressure(subst, V1/mol1, RT) / RT
        p2 = brusilovsky_pressure(subst, V2/mol2, RT) / RT
        μ1, μ2 = chempot(mol1 / V1), chempot(mol2 / V2)
        dA = μ1 * mol1 + μ2 * mol2 - (p1 * V1 + p2 * V2)
        grad[1] = μ1 - μ2
        grad[2] = p2 - p1
        return dA, grad
    end

    stabtest = vt_stability(subst, RT, molar_dens, optmethod = optmethod)

    if stabtest[2] < p_orig #+ sqrt(eps(one(p_orig))) * abs(p_orig)
        return
    end

    znew = stabtest[1].argument
    rhov = [float(molar_dens), one(molar_dens)]
    dd = [0.0, 0.0]
    
    function maxstep(x, d)
        to_zero = minimum(zip(x, d)) do (xi, di)
            a = -xi / di
            a > 0 ? a : convert(typeof(a), Inf)
        end
        to_one = minimum(zip(x, d, rhov)) do (xi, di, init)
            a = (init - xi) / di
            a > 0 ? a : convert(typeof(a), Inf)
        end
        to_b = 
            let Δ1 = subst.b * x[1] - x[2],
                Δ2 = subst.b * (rhov[1] - x[1]) - (1 - x[2]),
                Δ = subst.b * d[1] - d[2]
                
                minimum((-Δ1 / Δ, Δ2 / Δ)) do x
                    x > 0 ? x : convert(typeof(x), Inf)
                end
            end
        return min(to_zero, to_one, to_b)
    end

    opt = optmethod(rhov)
    #opt.α0 = 1e-8

    arg = DescentMethods.argumentvec(opt)
    gradA = DescentMethods.gradientvec(opt)

    let a = maxstep(rhov, (-znew[1], -1)) / 2
        while a > 1e-20
            arg[1] = rhov[1] - a * znew[1]
            arg[2] = 1 - a
            y, g = twophaseA!(arg, gradA)
            d = g ⋅ (znew[1], one(znew[1]))
            if y < A0 && d > 0
                if opt isa DescentMethods.BFGS
                    opt.xdiff .= arg .- rhov
                    DescentMethods.reset!(opt, arg, dot(opt.xdiff, g) / dot(g, g))
                end
                break
            end
            a /= 2
        end
    end

    track_file = open("track.vtflash.$(optmethod).$(RT / GAS_CONSTANT_SI)K.$(round(molar_dens, digits=2))mol_m3.txt", "w")
    optresult = DescentMethods.optimize!(opt,
                                         twophaseA!,
                                         arg,
                                         gtol = 1e-9,
                                         maxiter = 1000,
                                         maxcalls = 20000,
                                         constrain_step = maxstep,
                                         reset = false,
                                         track_io = track_file)

    close(track_file)
    xfinal = optresult.argument
    d1 = xfinal[1] / xfinal[2]
    d2 = (rhov[1] - xfinal[1]) / (1 - xfinal[2])
    p1 = brusilovsky_pressure(subst, 1/d1, RT)
    p2 = brusilovsky_pressure(subst, 1/d2, RT)
    if d1 < d2
        return (ρgas = d1, ρliq = d2, s = xfinal[2], pgas = p1, pliq = p2, opt = optresult)
    else
        return (ρgas = d2, ρliq = d1, s = 1 - xfinal[2], pgas = p2, pliq = p1, opt = optresult)
    end
end

let phase_split_cg = vt_flash(methane,
                              GAS_CONSTANT_SI * 110,
                              1000,
                              optmethod = DescentMethods.CGDescent),
    phase_split_bfgs = vt_flash(methane,
                                GAS_CONSTANT_SI * 110,
                                1000,
                                optmethod = DescentMethods.BFGS)

    println("""
    CG: $phase_split_cg
    BFGS: $phase_split_bfgs
    """)
end