using CUDA, Distributed

addprocs(length(devices()))

@everywhere begin
    using KomaMRI, CUDA, StatsBase

    include("plot_cine.jl")
    include("CardiacCine.jl")

    ## Read phantom
    folder = "/datos/work/phantomXCAT_myocardLV/x5resolution_3D/"

    ρ  = matread(folder*"mapaPD.mat")["PD"][:] .* 1f-2
    T1 = matread(folder*"mapaT1.mat")["T1"][:] .* 1f-3
    T2 = matread(folder*"mapaT2.mat")["T2"][:] .* 1f-3

    x = matread(folder*"X.mat")["X"][:] .* 1f-3; x .-= mean(x)
    y = matread(folder*"Y.mat")["Y"][:] .* 1f-3; y .-= mean(y)
    z = matread(folder*"Z.mat")["Z"][:] .* 1f-3; z .-= mean(z)

    deltaX = matread(folder*"deltaX.mat")["deltaX"].* 1f-3
    deltaY = matread(folder*"deltaY.mat")["deltaY"].* 1f-3
    deltaZ = matread(folder*"deltaZ.mat")["deltaZ"].* 1f-3

    motion = MotionList(
        Motion(
            Path(deltaX, deltaY, deltaZ),
            Periodic(1f0, 1f0)
        )
    )

    phantom = Phantom(
        x=x,
        y=y,
        z=z,
        ρ=ρ,
        T1=T1,
        T2=T2,
        motion=motion
    )

    ## Select phantom sub-range
    phantom = phantom[-0.005 .< z .< 0.005]

    ## Sequence parameters
    hr = 60        # [bpm]
    N_matrix = 128 # image size = N x N
    N_phases = 10  # Number of cardiac phases
    flip_angle = 10
    tagging = false
    Δf = 0
    dummy_cycles = 0

    sys = Scanner()

    FOV = 0.1
    TR  = (hr/60) / N_phases

    # ROTATE
    θ = 0
    R = rotx(θ / 180 * π) * roty(θ / 180 * π)

    RR = 60/hr

	# Sequence
	global seq = Sequence()


	# Tagging ----------------------------------
	tag = Sequence()
	if tagging

		# SPAMM
		hard_flip(T,α,sys) = begin
			B1 = α/(360*γ*T) 
			return PulseDesigner.RF_hard(B1, T, sys)
		end

		T_RF = 0.5e-3

		EX_22 = hard_flip(T_RF,22.5,sys)
		EX_45 = hard_flip(T_RF,45,sys)
		EX_90 = hard_flip(T_RF,90,sys)

		A = 3.5e-3
		T = 0.7e-3
		ζ = A / sys.Smax

		GR_x = Sequence(reshape([Grad(A,T,ζ);
								Grad(0,0);
								Grad(0,0)],(3,1)))

		GR_y = Sequence(reshape([Grad(0,0);
								Grad(A,T,ζ);
								Grad(0,0)],(3,1)))

		spamm_x =   EX_45 +
					GR_x +
					EX_45 +
					5*GR_x

		spamm_y =   EX_45 +
					GR_y +
					EX_45 +
					5*GR_y

		tag = spamm_y + Delay(5e-3)


		# DANTE 
		"""
		dante = PulseDesigner.RF_train(8, 1e-4, 1e-3, 25, sys; G = [0,4e-3,0])
		tag = dante + rotz(π/2)*dante + Delay(1e-3)
		"""
	end
	# -------------------------------------------------

	prospective = true
	if TR == RR/N_phases
		TR -= sum(dur(tag))/N_phases
		prospective = false
	end

	base_seq =  bSSFP(FOV, N_matrix, TR, flip_angle, sys; Δf=Δf)


	for i in 0:N_matrix-1
		global seq += tag

		line = base_seq[6*i .+ (1:6)]
		dummy_cycle = copy(line)
		dummy_cycle.ADC = [ADC(0,0) for i in 1:length(dummy_cycle)]
		# dummy_cycle.GR[1,:] = [Grad(0,0) for i in 1:length(dummy_cycle)]
		# dummy_cycle.GR[2,:] = [Grad(0,0) for i in 1:length(dummy_cycle)]

		for j in 1:dummy_cycles
			global seq += dummy_cycle
		end
		for j in 1:N_phases
			global seq += line
		end
		if prospective
			dead_space = RR - (N_phases*sum(dur(base_seq[6*i .+ (1:6)])) + sum(dur(tag)))
			global seq += Delay(dead_space)
		end
	end

    #Divide phantom
    parts = kfoldperm(length(phantom), nworkers())

    sim_params = KomaMRICore.default_sim_params()
	sim_params["Nblocks"] = 50
end

#Distribute simulation across workers
raw_signal = Distributed.@distributed (+) for i=1:nworkers()
    KomaMRICore.set_device!(i-1) #Sets device for this worker, note that CUDA devices are indexed from 0
    simulate(phantom[parts[i]], seq, sys; sim_params)
end

# Reconstruction
recParams = Dict{Symbol,Any}(:reco=>"direct")
Nx, Ny = raw_signal.params["reconSize"][1:2]
recParams[:reconSize] = (Nx, Ny)
recParams[:densityWeighting] = false

acqData = AcquisitionData(raw_signal)

_, ktraj = get_kspace(seq)

frames = []
for i in 1:N_phases
    acqAux = copy(acqData)
    range = reduce(vcat,[j*(N_matrix*N_phases).+((i-1)*N_matrix.+(1:N_matrix)) for j in 0:N_matrix-1])

    # Kdata
    acqAux.kdata[1] = reshape(acqAux.kdata[1][range],(N_matrix^2,1))

    # Traj
    acqAux.traj[1].circular = false

    acqAux.traj[1].nodes = acqAux.traj[1].nodes[:,range]
    # acqAux.traj[1].nodes = transpose(ktraj[i][:, 1:2]) #<----------------CAMBIO
    acqAux.traj[1].nodes = acqAux.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqAux.traj[1].nodes[:]))

    acqAux.traj[1].numProfiles = N_matrix
    acqAux.traj[1].times = acqAux.traj[1].times[range]

    # subsampleIndices
    acqAux.subsampleIndices[1] = acqAux.subsampleIndices[1][1:N_matrix^2]

    # Reconstruction
    aux = @timed reconstruction(acqAux, recParams)
    image  = reshape(aux.value.data,Nx,Ny,:)
    image_aux = abs.(image[:,:,1])
    # image_aux = round.(UInt8,255*(image_aux./maximum(image_aux)))

    push!(frames,image_aux)
end

## Plot
plot_cine(frames, 10; Δt=TR, filename="frames.gif")