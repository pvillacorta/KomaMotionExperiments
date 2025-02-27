function artery_double_phantom()
    R = 1e-2
    r = 5/11*R  
    L = 4e-2 
    Δx = 3e-4
    v = 10e-2

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
    tissue = Bool.(⚪(R) - ⚪(r))

    PD = 1.0
    T1 = 1000e-3
    T2 = 42e-3

    tissue = Phantom(
        name="Tissue",
        x=x[tissue],
        y=y[tissue],
        z=z[tissue],
        ρ=PD.*ones(length(x[tissue])),
        T1=T1.*ones(length(x[tissue])),
        T2=T2.*ones(length(x[tissue]))
    )

    # -------------- Blood phantom -------------------
    blood  = Bool.(⚪(r))

    PD = 0.9
    T1 = 1200e-3
    T2 = 92e-3

    # Displacements
    Nt = 100

    dx = dy = zeros(length(z[blood]), Nt)
    dz = z[blood] .+ cumsum(L/Nt .+ zeros(1,Nt), dims=2)

    spin_reset = dz .> L/2
    for i in 1:size(spin_reset, 1)
        idx = findfirst(x -> x == 1, spin_reset[i, :])
        if idx !== nothing
            spin_reset[i, :]  .= 0
            spin_reset[i, idx] = 1 # Se pone a 1 en el nodo SIGUIENTE al salto (ya que en la función reset_spin_hace Constant{Next})
        end
    end

    dz[dz .> L/2] .-= L
    dz .-= z[blood]

    blood = Phantom(
        name="Blood",
        x=x[blood],
        y=y[blood],
        z=z[blood],
        ρ =PD.*ones(length(x[blood])),
        T1=T1.*ones(length(x[blood])),
        T2=T2.*ones(length(x[blood])),
        motion=FlowPath(dx, dy, dz, spin_reset, Periodic(L/v, 1.0 - 1e-7))
    )

    # ------------- tissue + blood phantom -----------------
    obj1 = tissue + blood
    obj2 = copy(obj1)
    obj1.y .-= R + 1e-3
    obj2.y .+= R + 1e-3
    obj2.motion = MotionList(obj2.motion, Rotate(0.0, 180.0, 0.0, TimeRange(0.0,  1e-7)))

    obj = obj1 + obj2
    return obj

end