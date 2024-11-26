using KomaMRI

## 1. Phantom creation - Para crear el fichero artery.phantom

FOV = 2e-2   # [m] Artery diameter
L = 4e-2     # [m] Artery length 

v = 10e-2 #[m/s]

#PARAMETERS
N = 10
Δxr = FOV/(N-1) #Aprox rec resolution, use Δx_pix and Δy_pix
Ns = 5 #number of spins per voxel
Δx = Δxr/sqrt(Ns) #spin separation
#POSITIONS
FOVx = FOVy = FOV
FOVz = L



x = -FOVx/2:Δx:FOVx/2-Δx 
y = -FOVy/2:Δx:FOVy/2-Δx 
z = -FOVz/2:Δx:FOVz/2-Δx 

xx = reshape(x, (length(x),1,1)) 
yy = reshape(y, (1,length(y),1)) 
zz = reshape(z, (1,1,length(z))) 

# Grid
x = 1*xx .+ 0*yy .+ 0*zz
y = 0*xx .+ 1*yy .+ 0*zz
z = 0*xx .+ 0*yy .+ 1*zz

#PHANTOM
⚪(R) =  (x.^2 .+ y.^2 .<= R^2) #ircle of radius R
# Water spins
R = FOV/2
r = 8/11*FOV/2

# -------------- Tissue phantom -----------------
tissue = ⚪(R) - ⚪(r)
ρ = 1.0 .* tissue #proton density
T1 = ρ .* (1000 * 1e-3)  
T2 = ρ .* (42   * 1e-3)   

x_values  = x[ρ .!= 0]
y_values  = y[ρ .!= 0]
z_values  = z[ρ .!= 0]

T1_values = T1[ρ .!= 0]
T2_values = T2[ρ .!= 0]
ρ_values  = ρ[ρ .!= 0]

tissue = Phantom(
    name="Tissue",
    x=x_values,
    y=y_values,
    z=z_values,
    T1=T1_values,
    T2=T2_values,
    ρ=ρ_values
)

# -------------- Blood phantom -------------------
blood  = ⚪(r)

ρ = 0.9 .* blood #proton density
T1 = ρ .* (1026 * 1e-3)  
T2 = ρ .* (42  * 1e-3)   

x_values  = x[ρ .!= 0]
y_values  = y[ρ .!= 0]
z_values  = z[ρ .!= 0]

T1_values = T1[ρ .!= 0]
T2_values = T2[ρ .!= 0]
ρ_values  = ρ[ρ .!= 0]

# Displacements
Nt = 50
d_max = FOVz

dx = dy = zeros(length(z_values), Nt)
dz = zeros(length(z_values)) .+ (z_values .+ cumsum(d_max/Nt .+ zeros(1,Nt), dims=2)) 

spin_reset = dz .> FOVz/2
for i in 1:size(spin_reset, 1)
    idx = findfirst(x -> x == 1, spin_reset[i, :])
    if idx !== nothing
        spin_reset[i, :]  .= 0
        # spin_reset[i, idx-1] = 1 # Se pone a 1 en el nodo PREVIO al salto (ya que en la función reset_spin_hace Constant{Previous})
        spin_reset[i, idx] = 1 # Se pone a 1 en el nodo PREVIO al salto (ya que en la función reset_spin_hace Constant{Next})
    end
end


dz[dz .> FOVz/2] .-= FOVz
dz .-= z_values


blood = Phantom(
    name="Blood",
    x=x_values,
    y=y_values,
    z=z_values,
    T1=T1_values,
    T2=T2_values,
    ρ=ρ_values,
    motion=MotionList(
        FlowPath(dx, dy, dz, spin_reset, Periodic(d_max/v, 1.0))
    )
)


# ------------- tissue + blood phantom -----------------
obj = tissue + blood

obj.ρ  .= 1.0
obj.T1 .= 1026 * 1e-3  
obj.T2 .= 42  * 1e-3   

## 2. Write (and read) phantom into a file
write_phantom(obj, "../artery.phantom")