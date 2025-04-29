function artery_phantom()
    R = 1e-2
    r = 5/11*R  
    L = 4e-2 
    Δx = 3e-4
    v = 1

    #POSITIONS
    x = -R:Δx:R
    y = -R:Δx:R 
    z = -L/2:Δx:L/2

    xx = reshape(x, (length(x),1,1)) 
    yy = reshape(y, (1,length(y),1)) 
    zz = reshape(z, (1,1,length(z))) 

    # Grid
    x = 1*xx .+ 0*yy .+ 0*zz
    y = 0*xx .+ 1*yy .+ 0*zz
    z = 0*xx .+ 0*yy .+ 1*zz

    #PHANTOM
    ⚪(R) =  (x.^2 .+ y.^2 .<= R^2) # circle of radius R

    # -------------- Tissue phantom -----------------
    ts = Bool.(⚪(R) - ⚪(r))

    PD = 1.0
    T1 = 1000e-3
    T2 = 42e-3

    tissue = Phantom(
        name="Tissue",
        x=x[ts],
        y=y[ts],
        z=z[ts],
        ρ=PD.*ones(length(x[ts])),
        T1=T1.*ones(length(x[ts])),
        T2=T2.*ones(length(x[ts]))
    )

    # -------------- Blood phantom -------------------
    bl  = Bool.(⚪(r))

    PD = 0.9
    T1 = 1200e-3
    T2 = 92e-3

    # Displacements
    Nt = 500

    dx = dy = zeros(length(z[bl]), Nt)
    dz =z[bl] .+ cumsum(L/Nt .+ zeros(1,Nt), dims=2)

    spin_reset = dz .> L/2
    for i in 1:size(spin_reset, 1)
        idx = findfirst(x -> x == 1, spin_reset[i, :])
        if idx !== nothing
            spin_reset[i, :]  .= 0
            spin_reset[i, idx] = 1 # Se pone a 1 en el nodo SIGUIENTE al salto (ya que en la función reset_spin_hace Constant{Next})
        end
    end

    dz[dz .> L/2] .-= L
    dz .-= z[bl]

    blood = Phantom(
        name="Blood",
        x=x[bl],
        y=y[bl],
        z=z[bl],
        ρ =PD.*ones(length(x[bl])),
        T1=T1.*ones(length(x[bl])),
        T2=T2.*ones(length(x[bl])),
        motion=FlowPath(dx, dy, dz, spin_reset, Periodic(L/v, 1.0-1e-6))
    )

    # ------------- tissue + blood phantom -----------------
    obj = tissue + blood
    return obj
end