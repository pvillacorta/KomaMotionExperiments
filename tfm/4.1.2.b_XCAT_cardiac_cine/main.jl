cd(@__DIR__)

using KomaMRI, StatsBase, LinearAlgebra
using CUDA

include("../utils/divide_spins_ranges.jl")
include("../utils/plot_cine.jl")
include("../sequences/bSSFP.jl")

## Phantom ------------------------------
obj = read_phantom("../phantoms/XCAT_heart.phantom") # This file must be downloaded from Zenodo: https://shorturl.at/G8Dsc
obj[abs.(obj.z) .< 0.005]  # 1 cm-high subset along the z dimension
plot_phantom_map(obj,:T1; time_samples=10)  # Plot the phantom

## Scanner ------------------------------
sys = Scanner()

## Sequence ------------------------------
heart_rate = 60   # [bpm]
N_matrix = 128    # image size 
N_phases = 10     # Number of cardiac phases
FOV = 15e-2       # [m]
TR = 100e-3       # [s]
flip_angle = 4    # [º]

seq = bSSFP_cine(FOV, N_matrix, TR, flip_angle, heart_rate, N_phases, sys)

## Simulation ----------------------------
# Set maximum number of spins per GPU (this is set to avoid a GPU overflow)
# Another possible solution would be to set sim_params["Nblocks"] to a high value (e.g. 2000)
MAX_SPINS_PER_GPU = 500_000
sequential_parts = divide_spins_ranges(length(obj), MAX_SPINS_PER_GPU)

if length(sequential_parts) > 1
    @info "Dividing phantom ($(length(obj)) spins) into $(length(sequential_parts)) parts that will be simulated sequentially"
end

raws = []
for (j, sequential_part) in enumerate(sequential_parts)
    if length(sequential_parts) > 1
        @info "Simulating phantom part $(j)/$(length(sequential_parts))"
    end
    push!(raws, simulate(obj[sequential_part], seq, sys))
end
raw_signal = reduce(+, raws)

frames = []

## Reconstruction ---------------------------
@info "Running reconstruction"
@time begin
    recParams = Dict{Symbol,Any}(:reco=>"direct")
    Nx = Ny = N_matrix
    recParams[:reconSize] = (Nx, Ny)
    recParams[:densityWeighting] = false

    acqData = AcquisitionData(raw_signal)

    _, ktraj = get_kspace(seq)

    for i in 1:N_phases
        acqAux = copy(acqData)
        range = reduce(vcat,[j*(N_matrix*N_phases).+((i-1)*N_matrix.+(1:N_matrix)) for j in 0:N_matrix-1])

        # Kdata
        acqAux.kdata[1] = reshape(acqAux.kdata[1][range],(N_matrix^2,1))

        # Traj
        acqAux.traj[1].circular = false

        acqAux.traj[1].nodes = transpose(ktraj[:, 1:2])[:,range]
        acqAux.traj[1].nodes = acqAux.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqAux.traj[1].nodes[:]))

        acqAux.traj[1].numProfiles = N_matrix
        acqAux.traj[1].times = acqAux.traj[1].times[range]

        # Reconstruction
        aux = @timed reconstruction(acqAux, recParams)
        image  = reshape(aux.value.data,Nx,Ny,:)
        image_aux = abs.(image[:,:,1])

        push!(frames,image_aux)
    end
end

## Post-processing ----------------------------
post = []
for frame in frames
    aux = copy(frame)
    perc = percentile(frame[:],98)
    aux[frame.>=perc].= perc;
    push!(post,aux)
end

## Create the /results folder if it does not exist:
results_dirname = "results/"
if isdir(results_dirname)
    rm(results_dirname; recursive=true)
end
mkdir(results_dirname)

## Plot ----------------------------
fps = 10
Δt  = (heart_rate/60) / N_phases
plot_cine(frames, fps; Δt=Δt, filename=results_dirname*"frames.gif")
plot_cine(post,   fps; Δt=Δt, filename=results_dirname*"post.gif")

## Save frames ---------------------
frames_dirname = results_dirname*"frames/"
if isdir(frames_dirname)
    rm(frames_dirname; recursive=true)
end
mkdir(frames_dirname)

for (i, frame) in enumerate(post)
    p = plot_image(frame)
    KomaMRIPlots.PlotlyJS.savefig(p, frames_dirname*"$(i).png")
end