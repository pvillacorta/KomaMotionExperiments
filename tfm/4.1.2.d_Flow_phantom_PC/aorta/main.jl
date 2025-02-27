cd(@__DIR__)

using KomaMRI, CUDA

include("../../sequences/EPI.jl")

## ---- Phantom ---- 
obj = read_phantom("../../phantoms/aorta.phantom")

rotate = Rotate(0f0, 0f0, 90f0, TimeRange(0f0, 1f-7))
obj.motion = MotionList(obj.motion, rotate)

## ---- Scanner ---- 
sys = Scanner()

for (i, orientation) in enumerate(["axial", "longitudinal"])
    for (j, v_direction) in enumerate(["vx", "vy", "vz"])
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
        direction = Float64.([j==1, j==2, j==3])

        # We multiply the venc by 2 since we will obtain the difference of phases 
        # between the signals produced by gre_a and gre_b: ϕA - ϕB = π: 
        seqs = PC_EPI_multishot(2*venc, direction, TE, TR, flip_angle, fov, N_matrix, sys, 16; R = R, delta_f = delta_f)

        ## ---- Simulation ----
        raws = []
        for seq in seqs
            sim_params = KomaMRICore.default_sim_params()
            # sim_params["Nblocks"] = 2*length(seq)
            sim_params["Nblocks"] = 800
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
        magnitude_mean_plot = plot_image(magnitude_mean,  title="Magnitude Mean(A, B)")

        # Phase
        custom_colorscale = [
            [0.0, "rgb(165,0,38)"],
            [0.111111111111, "rgb(215,48,39)"],
            [0.222222222222, "rgb(244,109,67)"],
            [0.5, "rgb(255,255,255)"],
            [0.777777777778, "rgb(116,173,209)"],
            [0.888888888889, "rgb(69,117,180)"],
            [1.0, "rgb(49,54,149)"]
        ]

        phase_a_plot = plot_image(angle.(recons[1][:,:,1]), title="Phase A, venc = $(round(venc * 1e2)) cm/s", colorscale=custom_colorscale)
        phase_b_plot = plot_image(angle.(recons[2][:,:,1]), title="Phase B, venc = $(round(venc * 1e2)) cm/s", colorscale=custom_colorscale)

        phase_diff = angle.(recons[1][:,:,1]) .- angle.(recons[2][:,:,1])
        phase_diff_plot = plot_image(phase_diff,  title="Phase(A) - Phase(B), venc = $(round(venc * 1e2)) cm/s", zmin=-π, zmax=π, colorscale=custom_colorscale)

        phase_diff_masked = phase_diff .* (normalize(magnitude_mean) .> 5.5e-3)
        phase_diff_masked[(phase_diff_masked .< -π/2) .| (phase_diff_masked .> π)] .= 0
        phase_diff_masked_plot = plot_image(phase_diff_masked .* 1.35, title="Phase(A) - Phase(B) masked, venc = $(round(venc * 1e2)) cm/s", zmin=-π, zmax=π, colorscale=custom_colorscale)

        # Save results
        results_dir = "results_aorta"
        filename = "$(results_dir)/$(orientation)_$(v_direction)_"

        if !isdir(results_dir)
            mkdir(results_dir)
        end

        KomaMRIPlots.PlotlyJS.savefig(phase_a_plot, "$(filename)phase_a.png")
        KomaMRIPlots.PlotlyJS.savefig(phase_b_plot, "$(filename)phase_b.png")
        KomaMRIPlots.PlotlyJS.savefig(magnitude_mean_plot, "$(filename)magnitude_mean.png")
        KomaMRIPlots.PlotlyJS.savefig(phase_diff_plot, "$(filename)phase_diff.png")
        KomaMRIPlots.PlotlyJS.savefig(phase_diff_masked_plot, "$(filename)phase_diff_masked.png") 

    end
end