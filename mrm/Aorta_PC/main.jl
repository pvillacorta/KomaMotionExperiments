cd(@__DIR__)

using KomaMRI, CUDA, JLD2, StatsBase

include("../sequences/EPI.jl")
include("rotate_aorta.jl")

## ---- Phantom ---- 
obj = read_phantom("../phantoms/aorta.phantom") # This file must be downloaded from Zenodo: https://shorturl.at/G8Dsc
obj = rotate_aorta(obj)
## ---- Scanner ---- 
sys = Scanner()
for (i, orientation) in enumerate(["axial", "longitudinal"])
    for (j, v_direction) in enumerate(["vx", "vy", "vz", "novenc"])
        @info "Orientation: $(orientation), v_direction: $(v_direction)"

        ## ---- Sequence - PC-EPI -------
        fov = 0.11
        N_matrix = 128

        TE = 16e-3
        TR = 60e-3
        flip_angle = [50, 5][i]
        delta_f = [-2, 0.8][i] * 1e3

        # Sequence rotation
        rot_x = [0, 90][i]
        rot_y = 0
        rot_z = 0

        R = rotz(rot_z / 180 * π) * roty(rot_y / 180 * π) * rotx(rot_x / 180 * π)

        # Phase contrast parameters
        venc = 50e-2
        direction = Float64.([v_direction=="vx", v_direction=="vy", v_direction=="vz"])

        # We multiply the venc by 2 since we will obtain the difference of phases 
        # between the signals produced by gre_a and gre_b: ϕA - ϕB = π: 
        seqs = PC_EPI_multishot(2*venc, direction, TE, TR, flip_angle, fov, N_matrix, sys, 16; R = R, delta_f = delta_f)

        ## ---- Simulation ----
        raws = []
        for seq in seqs
            sim_params = KomaMRICore.default_sim_params()
            sim_params["Nblocks"] = 3000
            sim_params["Δt"]    = 8e-4
            sim_params["Δt_rf"] = 1e-5
            push!(raws, simulate(obj[1:2_000_000], seq, sys; sim_params=sim_params))
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
        magnitude_mean_plot = plot_image(magnitude_mean,  title="Magnitude Mean(A, B)", zmax=percentile(vec(magnitude_mean), 99.5), zmin=0)

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
        phase_diff_plot = plot_image(phase_diff,  title="Phase(A) - Phase(B), venc = $(round(venc * 1e2)) cm/s", zmin=-π, zmax=π, colorscale=RdBu_matplotlib)

        masks = load("masks.jld2")
        mask  = masks[orientation]

        phase_diff_masked = map((x, m) -> m ? x : missing, (mod.(phase_diff .+ π, 2π) .- π), mask)
        phase_diff_masked_plot = plot_image(phase_diff_masked .* 1.3, title="Phase(A) - Phase(B) masked, venc = $(round(venc * 1e2)) cm/s", zmin=-π, zmax=π, colorscale=RdBu_matplotlib)

        # Save results
        results_dir = "results"
        filename = "$(results_dir)/$(orientation)_$(v_direction)_"

        if !isdir(results_dir)
            mkdir(results_dir)
        end

        KomaMRIPlots.PlotlyJS.savefig(phase_a_plot, filename*"phase_a.png")
        KomaMRIPlots.PlotlyJS.savefig(phase_b_plot, filename*"phase_b.png")
        KomaMRIPlots.PlotlyJS.savefig(magnitude_mean_plot, filename*"magnitude_mean.png")
        KomaMRIPlots.PlotlyJS.savefig(phase_diff_plot, filename*"phase_diff.png")
        KomaMRIPlots.PlotlyJS.savefig(phase_diff_masked_plot, filename*"phase_diff_masked.svg") 
    end
end