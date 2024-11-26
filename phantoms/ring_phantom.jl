"""
This file generates a .phantom of a contracting and expanding ring.
A ventricle of the heart is recreated, with a diameter equal to FOV 
and an expansion and contraction of Δ meters every T seconds.
"""

using KomaMRI

FOV = 10e-2   # [m] Diameter ventricule
Δ = 1e-2      # [m] Displacement
T = [1.0]     # [s] Period
K = 4

#PARAMETERS
N = 21
Δxr = FOV/(N-1) #Aprox rec resolution, use Δx_pix and Δy_pix
Ns = 20 #number of spins per voxel
Δx = Δxr/sqrt(Ns) #spin separation
#POSITIONS
x = y = -FOV/2:Δx:FOV/2-Δx #spin coordinates
x, y = x .+ y'*0, x*0 .+ y' #grid points
#PHANTOM
⚪(R) =  (x.^2 .+ y.^2 .<= R^2)*1. #Circle of radius R
v = FOV/4 #m/s 1/16 th of the FOV during acquisition
# Water spins
R = 9/10*FOV/2
r = 6/11*FOV/2
ring = ⚪(R) .- ⚪(r)

ρ = 0.9*ring #proton density
T1 = (1026*ring)*1e-3   #Myocardial T1
T2 = (42*ring)*1e-3     #T2 map [s]

x_values  = x[ρ .!= 0]
y_values  = y[ρ .!= 0]
T1_values = T1[ρ .!= 0]
T2_values = T2[ρ .!= 0]
ρ_values  = ρ[ρ .!= 0]

phantom = Phantom(
    name="Contracting Ring",
    x=x_values,
    y=y_values,
    T1=T1_values,
    T2=T2_values,
    ρ=ρ_values
)

phantom.motion = MotionList(
    HeartBeat(-0.4, 0.4, 0.0, Periodic(1.0, 0.3)),
    Rotate(0.0, 0.0, 30.0, Periodic(1.0, 0.3))
)

write_phantom(phantom, "contracting_ring.phantom")