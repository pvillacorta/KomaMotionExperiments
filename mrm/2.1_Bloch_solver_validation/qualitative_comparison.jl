# This script is used to validate the flow simulation
# by replicating "The solution of Bloch equations for flowing
# spins during a selective pulse using a finite diference method", Yuan et al.

cd(@__DIR__)

import Pkg
Pkg.activate(".")
Pkg.instantiate()

using KomaMRICore, CUDA, Plots, Measures

include("yuan_sequence.jl")

## Results directory
results_dirname = "results/"
if !isdir(results_dirname)
    mkdir(results_dirname)
end



## Sequence
seq = yuan_sequence()

## Phantom
Nspins = 400
Lt = 30e-3 # 30mm

vs = [0, 80, 120, 160, 200] .* 1e-2
objs = Phantom[]
Ms = Mag[]

for (i, v) in enumerate(vs) # m/s <--------- VELOCITY
    dz = v * dur(seq)
    obj = Phantom(x=zeros(Nspins), z=collect(range(-Lt/2-dz, Lt/2-dz, Nspins)))
    obj.motion = MotionList(
        Translate(0.0, 0.0, dz, TimeRange(0.0, dur(seq)), AllSpins())
    )

    ## Scanner
    sys = Scanner()

    ## Simulation
    sim_params = KomaMRICore.default_sim_params()
    sim_params["return_type"] = "state"
    sim_params["Δt_rf"] = 1e-6 # <---- This is actually important !!!
    M = simulate(obj, seq, sys; sim_params)
    push!(Ms, M)
    push!(objs, obj)
end
## Plot
f_size = (350, 220)
# Mx
px = Plots.plot(;legend=:outertopright, ylabel="Mx", yrange=(-1.2, 0.9), xrange=((-Lt/2)*1e3, (Lt/2)*1e3), size=f_size)
for (i, v) in enumerate(vs)
    x, y, z = get_spin_coords(objs[i].motion, objs[i].x, objs[i].y, objs[i].z, dur(seq))
    Plots.plot!(z * 1e3, real.(Ms[i].xy), label="$(v * 1e2) cm/s", linewidth=2)
end
# My
py = Plots.plot(;legend=:outertopright, ylabel="My", yrange=(-1.0, 1.0), xrange=((-Lt/2)*1e3, (Lt/2)*1e3), size=f_size)
for (i, v) in enumerate(vs)
    x, y, z = get_spin_coords(objs[i].motion, objs[i].x, objs[i].y, objs[i].z, dur(seq))
    Plots.plot!(z * 1e3, imag.(Ms[i].xy), label="$(v * 1e2) cm/s", linewidth=2)
end
# My
pz = Plots.plot(;legend=:outertopright, xlabel="z (mm)", ylabel="Mz", yrange=(-0.2, 1.2), xrange=((-Lt/2)*1e3, (Lt/2)*1e3), size=f_size)
for (i, v) in enumerate(vs)
    x, y, z = get_spin_coords(objs[i].motion, objs[i].x, objs[i].y, objs[i].z, dur(seq))
    Plots.plot!(z * 1e3, Ms[i].z, label="$(v * 1e2) cm/s", linewidth=2)
end

pt = Plots.plot(px, py, pz, layout=(3,1), size=(f_size[1], f_size[2] * 3), left_margin=10mm)

## Save plot
Plots.savefig(pt, results_dirname*"magn_profiles.svg");

println("Plots saved to ", results_dirname*"magn_profiles.svg")