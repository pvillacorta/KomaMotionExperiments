"""
This file generates a .phantom of a contracting and expanding 3D ring.
A ventricle of the heart is recreated, with a diameter equal to D 
and contraction and rotation motions.

The aim is to recreate the phantom used in:

Xanthis, C.G., Venetis, I.E. & Aletras, A.H.
High performance MRI simulations of motion on multi-GPU systems. 
J Cardiovasc Magn Reson 16, 48 (2014).
https://doi.org/10.1186/1532-429X-16-48
"""

using KomaMRI

function ring3D_phantom()
    D = 100e-3   # [m] outter diameter
    d = 50e-3    # [m] inner diameter
    L = 80e-3    # [m] length 

    Δx = Δy = 0.25e-3 # [m]
    Δz      = 1e-3    # [m]

    Ro = D/2 # outter radius
    Ri = d/2 # inner radius (initial)
    ri = 10e-3  # inner radius (final)

    x = range(-D/2, stop=D/2, step=Δx)
    y = range(-D/2, stop=D/2, step=Δy)
    z = range(-L/2, stop=L/2, step=Δz)

    xx = reshape(x, (length(x),1,1)) 
    yy = reshape(y, (1,length(y),1)) 
    zz = reshape(z, (1,1,length(z))) 

    x = 1*xx .+ 0*yy .+ 0*zz
    y = 0*xx .+ 1*yy .+ 0*zz
    z = 0*xx .+ 0*yy .+ 1*zz

    ⚪(R) = (x.^2 .+ y.^2 .<= R^2)
    ϵ = 1e-7 # To adjust to the exact same number of spins as in https://doi.org/10.1186/1532-429X-16-48
    ring = ⚪(Ro + ϵ) .- ⚪(Ri - ϵ)

    ρ = 1.0*ring #proton density
    T1 = (900*ring)*1e-3    # T1 = 0.9 s
    T2 = (50*ring)*1e-3     # T2 = 0.05 s

    x_values  = x[ρ .!= 0]
    y_values  = y[ρ .!= 0]
    z_values  = z[ρ .!= 0]

    T1_values = T1[ρ .!= 0]
    T2_values = T2[ρ .!= 0]
    ρ_values  = ρ[ρ .!= 0]

    # ----------------- TIME CURVE -------------------
    timecurve = TimeCurve(
        t        = [0.0, 0.35, 0.5, 0.7, 0.8],
        t_unit   = [0.0, 1.0,  1.0, 0.0, 0.0],
        periodic = true
    )

    # ------------------ MOTION ---------------------
    #  Heart motion model presented by Tecelao et al. in:
    # "Extended harmonic phase tracking of myocardial motion: 
    # Improved coverage of myocardium and its effect on strain results"

    # Cylindrical coordinates
    R = sqrt.(x_values .^ 2 + y_values .^ 2)
    Θ = atan.(y_values, x_values)
    Z = z_values

    λ = 1
    Φ = 0.556  * π/180 * 1e3 # [rad/m]
    Γ = 0.6    * π/180 * 1e3 # [rad/m]
    ϵ = 18.334 * π/180       # [rad]
    ω = 0.278
    δ = 4.167  * 1e-3        # [m]

    r = sqrt.(ri^2 .+ (R .^ 2 .- Ri^2) ./ (λ))
    θ = Φ .* R .+ Θ .+ Γ .* Z .+ ϵ
    z = ω .* R .+ λ .* Z .+ δ

    # Back to cartesian coordinates:
    Δxo = r .* cos.(θ) .- x_values
    Δyo = r .* sin.(θ) .- y_values
    Δzo = z .- z_values

    Nspins = length(r)
    dx = hcat(zeros(Nspins), Δxo)
    dy = hcat(zeros(Nspins), Δyo)
    dz = hcat(zeros(Nspins), Δzo) 

    motion = Path(dx, dy, dz, timecurve)

    obj = Phantom(
        name="3D Ring",
        x=x_values,
        y=y_values,
        z=z_values,
        T1=T1_values,
        T2=T2_values,
        ρ=ρ_values,
        motion = motion
    )

    return obj
end