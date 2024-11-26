# using Pkg
# Pkg.add("StatsBase")

using KomaMRI, LinearAlgebra, StatsBase

include("sequences/Sequences.jl")

phantom = brain_phantom3D(ss=4, start_end=[1,360])
phantom.Δw .= 0.0

sys = Scanner()

## Sequence
fov = 0.25
N_matrix = 64
TE = 30e-3
TR = 100e-3
flip_angle = 3
delta_f = 0

# EPI
ex = selective_rf(flip_angle, sys; Δf=delta_f)
epi = PulseDesigner.EPI(fov, N_matrix, sys)
TE_delay = TE - ( ex.DUR[1]/2 + ex.DUR[2] + dur(epi)/2 )
TE_delay = TE_delay > 0 ? TE_delay : 0.0
delay = Delay(TE_delay)
seq = ex + delay + epi

# GRE
# seq = GRE(fov, N_matrix, TE, TR, flip_angle, sys; G=[0,0,0], Δf=delta_f)
seq = bSSFP(fov, N_matrix, TR, flip_angle, sys; G=[0,0,0], Δf=delta_f)

## ROTATE
θ = 90
seq_rot = rotx(θ / 180 * π) * roty(θ / 180 * π) * seq #<--------------CAMBIO

## Simulation
simParams = Dict{String,Any}()
raw_signal = simulate(phantom, seq_rot, sys; sim_params=simParams)


## Reconstruction
_, ktraj = get_kspace(seq)
recParams = Dict{Symbol,Any}(:reco=>"direct")
acqData = AcquisitionData(raw_signal)
Nx = Ny = N_matrix
acqData.encodingSize = (Nx, Ny)
acqData.traj[1].circular = false #Removing circular window
acqData.traj[1].nodes = transpose(ktraj[:, 1:2]) #<----------------CAMBIO
acqData.traj[1].nodes = acqData.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqData.traj[1].nodes[:])) #Normalize k-space to -.5 to .5 for NUFFT
recParams[:reconSize] = (Nx, Ny)
recParams[:densityWeighting] = true

aux = @timed reconstruction(acqData, recParams)
image  = reshape(aux.value.data,Nx,Ny,:)
# global kspace = KomaMRI.fftc(reshape(aux.value.data,Nx,Ny,:))

plot_image(abs.(image)[:,:,1])
