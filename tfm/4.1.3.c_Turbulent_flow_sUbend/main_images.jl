cd(@__DIR__)

using KomaMRI, CUDA, StatsBase, JLD2

include("../sequences/GRE.jl")

## ---- Phantom ----
filename = "sUbend" 
# filename = "sUbend_2M_stepsize_5e-4_numSteps_5"
obj = read_phantom(filename * "_rotated.phantom")
obj.z .-= (maximum(obj.z) + minimum(obj.z))/2
obj.y .-= (maximum(obj.y) + minimum(obj.y))/2
obj.motion = NoMotion()
## ---- Scanner ---- 
sys = Scanner(Gmax=80e-3, Smax=200, GR_Δt=1e-5, RF_Δt=1e-5, ADC_Δt=1e-5)

## ---- Sequence ----
# Overload KomaMRI `*` function to avoid the following: ---------
# Rotating an array of gradients (in x, y, and z) with 
# different durations causes all durations to become the same
# https://github.com/JuliaHealth/KomaMRI.jl/issues/545
## Execute this only if the GitHub issue above is not solved:
import KomaMRIBase: *
function *(α::Matrix, x::Array{Grad})
    y = deepcopy(x)
    A_values = [g.A for g in x]  
    A_result = α * A_values     
    for (i, g) in enumerate(y)
        g.A = A_result[i]     
    end
    return y
end
##
# ---------------------------------------------------------------

fov      = [0.345, 0.142]
N_matrix = [171, 71]
res = fov ./ N_matrix

TE = 5e-3
TR =  8.621e-3
flip_angle = 15.0

R = [0. 0. 1.; 0. -1. 0.; 1. 0. 0.]

#VENC
vs                  = ["vz", "vz",  "vz", "vy",  "vy", "vx",  "vx" ]
vencs               = [ Inf,  550.0, 50.0, 250.0, 50.0, 100.0, 50.0] .* 1e-2
venc_durations_flat = [ 0.8,  0.8,   0.0,  0.8,   0.0,  0.8,   0.0 ] .* 1e-3
venc_durations_rise = [ 0.1,  0.1,   0.5,  0.1,   0.5,  0.1,   0.5 ] .* 1e-3

##
kspace      = []
magnitude   = []
phase       = []
seqs        = Sequence[]

rg = 1:1

@time for (i, v) in enumerate(vs[rg])
    @info "Simulating PC sequence with VENC = $(vencs[i] * 1e2) cm/s \n Direction: $v"

    ## ---- Sequence ----
    seq, _, _ = PC_GRE(
        vencs[i],
        Float64.([v=="vx", v=="vy", v=="vz"]),
        fov,
        N_matrix,
        TE,
        TR,
        flip_angle,
        sys;
        R = R,
        pulse_duration = 1e-3,
        adc_duration = 4e-3,
        venc_duration_flat = venc_durations_flat[i],
        venc_duration_rise = venc_durations_rise[i],
        balanced = true,
        crusher_duration = 1e-3,
        crusher_area = 4 * π / (2 * π * γ) / res[1]
    )
    push!(seqs, seq)

    ## ---- Simulation ----
    sim_params = KomaMRICore.default_sim_params()
    sim_params["Nblocks"] = 5000
    sim_params["Δt"]    = 8e-4
    sim_params["Δt_rf"] = 1e-5
    raw = simulate(obj[1:end], seq, sys; sim_params=sim_params)
    
    ## ---- Reconstruction ----
    recParams = Dict{Symbol,Any}(:reco=>"direct")
    Nx, Ny = N_matrix
    recParams[:reconSize] = (Nx, Ny)
    recParams[:densityWeighting] = false
    acqData = AcquisitionData(raw)
    seq_no_rot = inv(R) * seqs[i]
    _, ktraj = get_kspace(seq_no_rot)
    # Kdata
    acqData.kdata[1] = reshape(acqData.kdata[1],(Nx*Ny,1))
    # Traj
    acqData.traj[1].circular = false
    acqData.traj[1].nodes = transpose(ktraj[:, 1:2])
    acqData.traj[1].nodes = acqData.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqData.traj[1].nodes[:]))
    acqData.traj[1].numProfiles = Ny
    acqData.traj[1].times = acqData.traj[1].times
    # subsampleIndices
    acqData.subsampleIndices[1] = acqData.subsampleIndices[1][1:Nx*Ny]
    # Reconstruction
    aux = @timed reconstruction(acqData, recParams)
    recon = reshape(aux.value.data, Nx, Ny, :)

    m = abs.(recon[end-111:end,:,1])
    percen = percentile(vec(m), 99.9)
    m[m .> percen] .= percen

    push!(magnitude, m)
    push!(phase,     angle.(recon[end-111:end,:,1]))
end

## Plot results
magnitude_ref = copy(magnitude[1])
phase_ref = copy(phase[1])

RdBu_matplotlib = [
    [0.0, "rgb(103,0,31)"],
    [0.125, "rgb(178,24,43)"],
    [0.25, "rgb(214,96,77)"],
    [0.375, "rgb(244,165,130)"],
    [0.5, "rgb(235,235,235)"],
    [0.625, "rgb(146,197,222)"],
    [0.75, "rgb(67,147,195)"],
    [0.875, "rgb(33,102,172)"],
    [1.0, "rgb(5,48,97)"]
]

magnitudes = []
phases = []
phase_diffs = []
turb_estimations = []

mask = load("sUbend_image_mask.jld2")["mask"]

for (i, v) in enumerate(vs[rg])
    mg = magnitude[i]
    ph = -phase[i]
    ph_diff_masked = map((x, m) -> m ? x : missing, -(phase[i] .- phase_ref), mask)
    tke_masked = (log.(abs.(magnitude_ref) ./ abs.(magnitude[i]))) .* mask .+ .!(mask)

    push!(magnitudes, plot_image(magnitude[i]; title="Magnitude $v, VENC = $(vencs[i]*1e2) cm/s", zmin=minimum(magnitude_ref), zmax=maximum(magnitude_ref)))
    push!(phases, plot_image(ph; colorscale=RdBu_matplotlib, title="ϕ $v, VENC = $(vencs[i]*1e2) cm/s", zmin=-π, zmax=π))
    push!(phase_diffs, plot_image(ph_diff_masked; colorscale=RdBu_matplotlib, title="ϕ-ϕREF $v, VENC = $(vencs[i]*1e2) cm/s", zmin=-π, zmax=π))
    push!(turb_estimations, plot_image(tke_masked; colorscale="Hot", title="TKE $v, VENC = $(vencs[i]*1e2) cm/s", zmin=0, zmax=1))
end

fig = KomaMRIPlots.make_subplots(
    rows=4, cols=length(magnitudes), 
    subplot_titles= hcat(
        vcat([trace.plot.layout.title for trace in magnitudes]...),
        vcat([trace.plot.layout.title for trace in phases]...),
        vcat([trace.plot.layout.title for trace in phase_diffs]...),
        vcat([trace.plot.layout.title for trace in turb_estimations]...)
        ),
    shared_yaxes=true, 
    shared_xaxes=true,
    vertical_spacing=0.05,
    horizontal_spacing=0.0
)

for (i, v) in enumerate(vs[rg])
    phase_diffs[i].plot.data[1].xaxis_showgrid = false
    KomaMRIPlots.add_trace!(fig, magnitudes[i].plot.data[1], row=1, col=i)
    KomaMRIPlots.add_trace!(fig, phases[i].plot.data[1], row=2, col=i)
    KomaMRIPlots.add_trace!(fig, phase_diffs[i].plot.data[1], row=3, col=i)
    KomaMRIPlots.add_trace!(fig, turb_estimations[i].plot.data[1], row=4, col=i)
end

display(fig)

## Save fig
using Dates
KomaMRIPlots.savefig(fig, "sUbend_result_$(filename)_$(now()).svg", width=2200, height=900)

## Save results as julia structs
using JLD2, Dates
save("sUbend_result_$(now()).jld2", Dict("magnitude" => magnitude, "phase" => phase))