cd(@__DIR__)

import Pkg
Pkg.activate(".")

using KomaMRI, CUDA, StatsBase

include("../utils/divide_spins_ranges.jl")
include("../utils/plot_cine.jl")

include("artery_phantom.jl")
include("bSFFP_cine.jl")

## Phantom --------------------------------
obj = artery_phantom()
v = 4e-2
L = maximum(obj.z) - minimum(obj.z)
period = L/v
obj.motion.time = Periodic(period, 1.0 - 1e-6)

## Scanner --------------------------------
sys = Scanner()

## Sequence --------------------------------
heart_rate = 60/period   # [bpm]
N_matrix   = 100         # image size = N x N
N_phases   = 30          # Number of cardiac phases
FOV        = 0.04        # [m]
TR         = 30e-3       # [s]
flip_angle = 50          # [º]

angles = [0, -90] # Sequence orientation angle (0: Axial, -90: Sagittal)

for (i, orientation) in enumerate(["axial", "sagittal"])
    @info "Simulating $(orientation) sequence"
    θ = angles[i]
    seq = bSSFP_cine(FOV, N_matrix, TR, flip_angle, heart_rate/2, N_phases, sys) # Divide heart_rate by 2 to get 1 out of 2 cycles
    seq_rot = roty(θ / 180 * π) * seq 

    ## Simualtion --------------------------------
    MAX_SPINS_PER_GPU = 250_000
    sequential_parts = divide_spins_ranges(length(obj), MAX_SPINS_PER_GPU)

    sim_params = KomaMRICore.default_sim_params()

    if length(sequential_parts) > 1
        @info "Dividing phantom ($(length(obj)) spins) into $(length(sequential_parts)) parts that will be simulated sequentially"
    end

    raws = []
    for (j, sequential_part) in enumerate(sequential_parts)
        if length(sequential_parts) > 1
            @info "Simulating phantom part $(j)/$(length(sequential_parts))"

        end
        push!(raws, simulate(obj[sequential_part], seq_rot, sys; sim_params=sim_params))
    end
    raw_signal = reduce(+, raws)

    ## Reconstruction
    frames = []
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

            # subsampleIndices
            acqAux.subsampleIndices[1] = acqAux.subsampleIndices[1][1:N_matrix^2]

            # Reconstruction
            aux = @timed reconstruction(acqAux, recParams)
            image  = reshape(aux.value.data,Nx,Ny,:)
            image_aux = abs.(image[:,:,1])

            push!(frames,image_aux)
        end
    end

    ## Create the /results folder if it does not exist:
    results_dirname = "results/"
    if !isdir(results_dirname)
        mkdir(results_dirname)
    end
    
    ## Plot
    plot_cine(frames, 2; Δt=TR, filename=results_dirname*"frames_$(orientation).gif")

    ## Save frames ---------------------
    frames_dirname = results_dirname*"frames_$(orientation)/"
    if !isdir(frames_dirname)
        mkdir(frames_dirname)
    end
    
    for (i, frame) in enumerate(frames)
        p = plot_image(frame)
        KomaMRIPlots.PlotlyJS.savefig(p, frames_dirname*"$(i).png")
    end
end