cd(@__DIR__)

using KomaMRI, CUDA

include("../../sequences/GRE.jl")
include("artery_double_phantom.jl")

## ---- Phantom ---- 
obj = artery_double_phantom()
v = 10e-2
L = maximum(obj.z) - minimum(obj.z)
period = L/v

periodic = Periodic(period, 1.0 - 1e-7)
obj.motion.motions[1].time = periodic
obj.motion.motions[2].time = periodic

## ---- Scanner ---- 
sys = Scanner()


## ---- Sequence - PC-GRE ------- 
fov = 0.045
N_matrix = 100

TE = 10e-3
TR = 60e-3
flip_angle = 15
delta_f = -0

# Bipolar gradiens
venc = 18e-2
direction = [0, 0, 1.0]

# Sequence rotation
rot_x = 0
rot_y = 0
rot_z = 0

R = rotz(rot_z / 180 * π) * roty(rot_y / 180 * π) * rotx(rot_x / 180 * π)

# We multiply the venc by 2 since we will obtain the difference of phases 
# between the signals produced by gre_a and gre_b: ϕA - ϕB = π: 
seqs = PC_GRE(
    2*venc,
	direction,
    fov,
    N_matrix,
    TE,
    TR,
    flip_angle,
    sys;
    Δf = delta_f,
    R = R
)

## ---- Simulation ----
raws = []
for seq in seqs[1:2]
    sim_params = KomaMRICore.default_sim_params()
    sim_params["Nblocks"] = length(seq)
    sim_params["Δt"]    = 5e-4
    sim_params["Δt_rf"] = 1e-5
    push!(raws, simulate(obj, seq, sys; sim_params=sim_params))
end

## ---- Reconstruction ----
recParams = Dict{Symbol,Any}(:reco=>"direct")
Nx = Ny = N_matrix
recParams[:reconSize] = (Nx, Ny)
recParams[:densityWeighting] = false

recons = []
for (i, raw_signal) in enumerate(raws)
    acqData = AcquisitionData(raw_signal)
    seq_no_rot = inv(R) * seqs[i]
    _, ktraj = get_kspace(seq_no_rot)
    
    # Kdata
    acqData.kdata[1] = reshape(acqData.kdata[1],(N_matrix^2,1))
    
    # Traj
    acqData.traj[1].circular = false
    
    acqData.traj[1].nodes = transpose(ktraj[:, 1:2])
    acqData.traj[1].nodes = acqData.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqData.traj[1].nodes[:]))
    
    acqData.traj[1].numProfiles = N_matrix
    acqData.traj[1].times = acqData.traj[1].times
    
    # subsampleIndices
    acqData.subsampleIndices[1] = acqData.subsampleIndices[1][1:N_matrix^2]
    
    # Reconstruction
    aux = @timed reconstruction(acqData, recParams)
    push!(recons, reshape(aux.value.data, Nx, Ny, :))
end

## Plot and save results
# Magnitude
magnitude_mean = (abs.(recons[1][:,:,1]) .+   abs.(recons[2][:,:,1])) ./ 2
magnitude_mean_plot = plot_image(magnitude_mean,  title="Magnitude Mean(A, B)")

# Phase
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


phase_a_plot = plot_image(angle.(recons[1][:,:,1]), title="Phase A, venc = $(round(venc * 1e2)) cm/s", colorscale=RdBu_matplotlib)
phase_b_plot = plot_image(angle.(recons[2][:,:,1]), title="Phase B, venc = $(round(venc * 1e2)) cm/s", colorscale=RdBu_matplotlib)

phase_diff = angle.(recons[1][:,:,1]) .- angle.(recons[2][:,:,1])
phase_diff_plot = plot_image(phase_diff,       title="Phase(A) - Phase(B), venc = $(round(venc * 1e2)) cm/s", zmin=-π, zmax=π, colorscale=RdBu_matplotlib)

## Create the /results folder if it does not exist:
results_dirname = "results/"
if isdir(results_dirname)
    rm(results_dirname; recursive=true)
end
mkdir(results_dirname)

## Save results
KomaMRIPlots.PlotlyJS.savefig(phase_a_plot, results_dirname*"phase_a.png")
KomaMRIPlots.PlotlyJS.savefig(phase_b_plot, results_dirname*"phase_b.png")
KomaMRIPlots.PlotlyJS.savefig(magnitude_mean_plot, results_dirname*"magnitude_mean.png")
KomaMRIPlots.PlotlyJS.savefig(phase_diff_plot, results_dirname*"phase_diff.png")