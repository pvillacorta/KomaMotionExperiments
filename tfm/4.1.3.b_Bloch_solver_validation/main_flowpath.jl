# This script is used to validate the flow simulation
# by replicating "The solution of Bloch equations for flowing
# spins during a selective pulse using a finite diference method", Yuan et al.

cd(@__DIR__)

using KomaMRI, CUDA

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
# plot_seq(seq)

## Phantom
Nspins = 400
Lt = 30e-3

vs = [0, 80, 120, 160, 200] .* 1e-2
objs = Phantom[]
Ms = Mag[]

z0 = collect(range(-Lt/2, Lt/2, Nspins))

for (i, v) in enumerate(vs) # m/s <--------- VELOCITY
    dz_max = v * dur(seq)
    Nt = 500
    dx = dy = zeros(Nspins, Nt)
    dz      = zeros(Nspins) .+ (dz_max/(Nt-1) .* collect(0:(Nt-1))')
    zt = z0 .+ dz

    spin_reset = zt .> Lt/2
    jump_spins = []
    for i in eachindex(spin_reset[:,1])
        idx = findfirst(x -> x == 1, spin_reset[i, :])
        if idx !== nothing
            push!(jump_spins, i)
            spin_reset[i, :]  .= 0
            spin_reset[i, idx] = 1 
        end
    end
    spin_reset = collect(spin_reset)

    zt[zt .> Lt/2] .-= Lt
    dz .= zt .- z0

    obj = Phantom(x=zeros(Nspins), z=collect(range(-Lt/2, Lt/2, Nspins)))
    obj.motion = FlowPath(dx, dy, dz, spin_reset, TimeRange(0.0, dur(seq)))

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
using Plots
# Mx
px = plot(;legend=:outertopright, ylabel="Mx", yrange=(-1.2, 0.9), xrange=((-Lt/2)*1e3, (Lt/2)*1e3))
for (i, v) in enumerate(vs)
    x, y, z = get_spin_coords(objs[i].motion, objs[i].x, objs[i].y, objs[i].z, dur(seq))
    plot!(sort(z) .* 1e3, real.(Ms[i].xy)[sortperm(z)], label="$(v * 1e2) cm/s", linewidth=2)
end
# My
py = plot(;legend=:outertopright, ylabel="My", yrange=(-1.0, 1.0), xrange=((-Lt/2)*1e3, (Lt/2)*1e3))
for (i, v) in enumerate(vs)
    x, y, z = get_spin_coords(objs[i].motion, objs[i].x, objs[i].y, objs[i].z, dur(seq))
    plot!(sort(z) .* 1e3, imag.(Ms[i].xy)[sortperm(z)], label="$(v * 1e2) cm/s", linewidth=2)
end
# Mz
pz = plot(;legend=:outertopright, xlabel="z (mm)", ylabel="Mz", yrange=(-0.2, 1.2), xrange=((-Lt/2)*1e3, (Lt/2)*1e3))
for (i, v) in enumerate(vs)
    x, y, z = get_spin_coords(objs[i].motion, objs[i].x, objs[i].y, objs[i].z, dur(seq))
    plot!(sort(z) .* 1e3, Ms[i].z[sortperm(z)], label="$(v * 1e2) cm/s", linewidth=2)
end
pt = plot(px, py, pz, layout=(3,1), size=(340, 700))

## Save plot
results_dirname = "results/"
if !isdir(results_dirname)
    mkdir(results_dirname)
end

Plots.savefig(pt, results_dirname*"m_motion_flowpath.svg")
