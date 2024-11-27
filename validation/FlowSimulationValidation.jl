using KomaMRI
# This script is used to validate the flow simulation
# by replicating "The solution of Bloch equations for flowing
# spind during a selective pulse using a finite diference method", Yuan et al.

## Sequence
# Grad
T = 2.6794e-3     # 2.6794 ms
Gz0 = 1.0 * 1e-2  # 10 mT/m
z0 = 7e-3         # 7mm 
# RF
h = 8.9313e-7     # s
A = 0.1750 * 1e-4 # T
# Window
L = 4500
i = 0:floor(Int, 2L/3)
W = 0.42 .- 0.5 .* cos.(2π .* i ./(2L/3 - 1)) .+ 0.08 .* cos.(4π .* i ./(2L/3 - 1))
# Sinc
ω0 = 2π * γ * z0 * Gz0 / 2
H1 = A .* W .* sin.(ω0 .* (h .* i  .- T/2)) ./ (ω0 .* (h .* i .- T/2))
# Slice-selective RF pulse
seq =  Sequence([Grad(0,0); Grad(0,0); Grad(Gz0, T);;], [RF(H1, T);;])
seq += Sequence([Grad(0,0); Grad(0,0); Grad(-Gz0, T/2);;])
plot_seq(seq)

## Phantom
Nspins = 400
v = 80e-2  # m/s <--------- VELOCITY
Lt = 18e-3 # 18mm
dz = v * dur(seq)
obj = Phantom(x=zeros(Nspins), z=collect(range(-Lt/2-dz, Lt/2-dz, Nspins)))
obj.motion = MotionList(
    Translate(0.0, 0.0, dz, TimeRange(0.0, dur(seq)), AllSpins())
)
plot_phantom_map(obj, :T1)

## Scanner
sys = Scanner()

## Simulation
sim_params = KomaMRICore.default_sim_params()
sim_params["return_type"] = "state"
sim_params["Δt_rf"] = 1e-6 # <---- This is actually important !!!
M = simulate(obj, seq, sys; sim_params)

## Plot
using KomaMRIPlots.PlotlyJS
x, y, z = get_spin_coords(obj.motion, obj.x, obj.y, obj.z, dur(seq))
plot([
    scatter(x=z * 1e3, y=real.(M.xy), name="Mx"),
    scatter(x=z * 1e3, y=imag.(M.xy), name="My"),
    scatter(x=z * 1e3, y=M.z, name="Mz")
], Layout(title="Magnetization (v = $(v * 1e2) cm/s)", xaxis_title="z (mm)", yaxis_title="Magnetization"))