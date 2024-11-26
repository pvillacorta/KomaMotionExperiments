using MAT

function cardiac_cine(
	FOV::Float64, heart_rate::Int, N_phases::Int, N::Int, obj::Phantom, sys::Scanner;
	Δf=0,
	flip_angle = 10,
	TR = (60/heart_rate)/N_phases,
	dummy_cycles = 0,
	tagging::Bool=false,
	rotation_matrix=[1. 0. 0.; 0. 1. 0.; 0. 0. 1.],
	max_spins=500_000
)

	RR = 60/heart_rate
	α = flip_angle

	R = rotation_matrix

	# Sequence
	seq = Sequence()


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

	base_seq =  bSSFP(FOV, N, TR, α, sys; Δf=Δf)


	for i in 0:N-1
		seq += tag

		line = base_seq[6*i .+ (1:6)]
		dummy_cycle = copy(line)
		dummy_cycle.ADC = [ADC(0,0) for i in 1:length(dummy_cycle)]
		# dummy_cycle.GR[1,:] = [Grad(0,0) for i in 1:length(dummy_cycle)]
		# dummy_cycle.GR[2,:] = [Grad(0,0) for i in 1:length(dummy_cycle)]

		for j in 1:dummy_cycles
			seq += dummy_cycle
		end
		for j in 1:N_phases
			seq += line
		end
		if prospective
			dead_space = RR - (N_phases*sum(dur(base_seq[6*i .+ (1:6)])) + sum(dur(tag)))
			seq += Delay(dead_space)
		end
	end

	seq_rot = R*seq


	# Simulation
	sim_params = KomaMRICore.default_sim_params()
	sim_params["Nblocks"] = 50

	raws = []

	parts = divide_spins_ranges(length(obj), max_spins)
	if length(parts) > 1
		@warn "Phantom subdivided in $(length(parts)) parts that will be simulated sequentially"
	end

	for (i,part) in enumerate(parts)
		if length(parts) > 1
			@info "Simulating phantom part $(i):"
		end
		push!(raws, simulate(phantom[part], seq_rot, sys))
	end

	raw_signal = reduce(+, raws)

	# Reconstruction
	recParams = Dict{Symbol,Any}(:reco=>"direct")
	Nx = Ny = N
	recParams[:reconSize] = (Nx, Ny)
	recParams[:densityWeighting] = false

	acqData = AcquisitionData(raw_signal)

	_, ktraj = get_kspace(seq)

	frames = []
	for i in 1:N_phases
		acqAux = copy(acqData)
		range = reduce(vcat,[j*(N*N_phases).+((i-1)*N.+(1:N)) for j in 0:N-1])

		# Kdata
		acqAux.kdata[1] = reshape(acqAux.kdata[1][range],(N^2,1))

		# Traj
		acqAux.traj[1].circular = false

		acqAux.traj[1].nodes = transpose(ktraj[:, 1:2])[:,range]
		acqAux.traj[1].nodes = acqAux.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqAux.traj[1].nodes[:]))

		acqAux.traj[1].numProfiles = N
		acqAux.traj[1].times = acqAux.traj[1].times[range]

		# subsampleIndices
		acqAux.subsampleIndices[1] = acqAux.subsampleIndices[1][1:N^2]

		# Reconstruction
		aux = @timed reconstruction(acqAux, recParams)
		image  = reshape(aux.value.data,Nx,Ny,:)
		image_aux = abs.(image[:,:,1])
		# image_aux = round.(UInt8,255*(image_aux./maximum(image_aux)))

		push!(frames,image_aux)
	end

	frames, seq_rot
end 


"""
Basic gradient-echo (GRE) Sequence
"""
GRE(FOV::Float64, N::Int, TE::Float64, TR::Float64, α, sys::Scanner; G=[0,0,0], Δf=0) = begin
	# Excitation (Sinc pulse) ----------------------------------
	# α = γ ∫(0-T) B1(t)dt 
	# ----------------------
	# We need to obtain B1 from flip angle α and a generic T=3ms duration
	# i.e. we need to resolve the equation above

	T_rf = 3e-3   		# Pulse duration
	Gss = 2e-3     	# Slice-select gradient

	# With T = 3ms, we need B1 = 8,69e-8 T to produce a flip angle α = 1°
	B_1° = 8.6938e-8
	B1 = α*B_1°
	EX = PulseDesigner.RF_sinc(B1,T_rf,sys;G=[0,0,Gss],Δf=Δf)

	# Acquisition ----------------------------------------------
	# Square acquisition (Nx = Ny = N) 
	# PHASE
	ζ_phase = EX[2].GR[1].rise
	T_phase = EX[2].GR[1].T

	Δk = (1/FOV)
	FOVk = (N-1)*Δk
	Gx = Gy = FOVk/(γ*(T_phase + ζ_phase))
	step = Δk/(γ*(T_phase + ζ_phase))

	"""
	print("Δk = ", Δk, " m⁻¹\n")
	print("FOVk = ", FOVk, " m⁻¹\n")
	print("Gx = ", Gx*1e3, " mT/m\n")
	print("step = ", step*1e3, " mT/m\n")
	"""

	# FE and Readout
	TE_min = (1/2) * ( sys.ADC_Δt*(N-1) + 2*((EX.DUR[1]/2) + EX.DUR[2]) )
	if TE < TE_min
		print("Error: TE must be greater than TE_min = ", TE_min*1e3, " ms\n")
		return
	end

	ACQ_dur = 2 * (TE - ( (EX.DUR[1]/2) + EX.DUR[2] ))
	G_ro = FOVk/(γ*ACQ_dur)
	ζ_ro = G_ro / sys.Smax
	T_ro = ACQ_dur - ζ_ro
	GR = reshape([Grad(G_ro,T_ro,ζ_ro), Grad(0,0), Grad(0,0)],(3,1))
	RO = Sequence(GR)
	RO.ADC[1] = ADC(N, T_ro, ζ_ro)
	delay_TR = TR - (EX.DUR[1] + EX.DUR[2] + RO.DUR[1])

	"""
	print("ACQ_dur = ", ACQ_dur*1e3, " ms\n")
	print("G_ro = ", G_ro*1e3, " mT/m\n")
	print("ζ = ", ζ_ro*1e3, " ms\n")
	"""

	gre = Sequence()
	for i in 0:(N-1)
		# Excitation and first phase 
		EX = PulseDesigner.RF_sinc(B1,T_rf,sys;G=[0,0,Gss],Δf=Δf)
		EX[end].GR[1].A = -Gx/2
		EX[end].GR[2].A = -Gy/2 + i*step
		gre += EX

		# FE and Readout
		gre += RO + Delay(delay_TR)
	end
	gre.DEF = Dict("Nx"=>N,"Ny"=>N,"Nz"=>1,"Name"=>"gre"*string(N)*"x"*string(N),"FOV"=>[FOV, FOV, 0])
	return gre
end

"""
b-SSFP Sequence
"""
bSSFP(FOV::Float64, N::Int, TR::Float64, α, sys::Scanner; G=[0,0,0], Δf=0) = begin
	# Excitation (Sinc pulse) ----------------------------------
	# α = γ ∫(0-T) B1(t)dt 
	# ----------------------
	# We need to obtain B1 from flip angle α and a generic T=3ms duration
	# i.e. we need to resolve the equation above

	T_rf = 3e-3   	# Pulse duration
	Gss = 2e-3     	# Slice-select gradient

	# With T = 3ms, we need B1 = 8,69e-8 T to produce a flip angle α = 1°
	B_1° = 8.6938e-8
	B1 = α*B_1°
	EX = PulseDesigner.RF_sinc(B1,T_rf,sys;G=[0,0,Gss],Δf=Δf)

	# Acquisition ----------------------------------------------
	# Square acquisition (Nx = Ny = N) 
	# PHASE
	ζ_phase = EX[2].GR[1].rise
	T_phase = EX[2].GR[1].T

	Δk = (1/FOV)
	FOVk = (N-1)*Δk
	Gx = Gy = FOVk/(γ*(T_phase + ζ_phase))
	step = Δk/(γ*(T_phase + ζ_phase))

	#=
	print("Δk = ", Δk, " m⁻¹\n")
	print("FOVk = ", FOVk, " m⁻¹\n")
	print("Gx = ", Gx*1e3, " mT/m\n")
	print("step = ", step*1e3, " mT/m\n")
	=#

	# FE and Readout
	delay = 0.1*TR # delay to "strech" readout time
	ACQ_dur = TR - (EX.DUR[1] + 2*EX.DUR[2] + 2*delay)
	G_ro = FOVk/(γ*ACQ_dur)
	ζ_ro = G_ro / sys.Smax
	T_ro = ACQ_dur - ζ_ro
	GR = reshape([Grad(G_ro,T_ro,ζ_ro), Grad(0,0), Grad(0,0)],(3,1))
	RO = Sequence(GR)
	RO.ADC[1] = ADC(N, T_ro, ζ_ro)

	#=
	print("ACQ_dur = ", ACQ_dur*1e3, " ms\n")
	print("G_ro = ", G_ro*1e3, " mT/m\n")
	print("ζ = ", ζ_ro*1e3, " ms\n")
	=#

	bssfp = Sequence()
	for i in 0:(N-1)
		# Excitation and first phase 
		EX = PulseDesigner.RF_sinc(B1,T_rf,sys;G=[0,0,Gss],Δf=Δf)
		EX[end].GR[1].A = -Gx/2
		EX[end].GR[2].A = -Gy/2 + i*step
		bssfp += EX

		# FE and Readout
		balance = Sequence(reshape([ EX[end].GR[1],
									-EX[end].GR[2],
									 EX[end].GR[3]],(3,1)))

		# balance = Sequence(reshape([  Grad(0,EX[end].GR[1].T),
		# 							  Grad(0,EX[end].GR[2].T),
		# 							  Grad(0,EX[end].GR[2].T)],(3,1)))	

		bssfp += Delay(delay) + RO + Delay(delay) + balance
	end
	bssfp.DEF = Dict("Nx"=>N,"Ny"=>N,"Nz"=>1,"Name"=>"bssfp"*string(N)*"x"*string(N),"FOV"=>[FOV, FOV, 0])
	return bssfp
end

function divide_spins_ranges(total_spins::Int, max_spins::Int)
	parts = []  # Inicializamos un arreglo vacío para almacenar los rangos
	start_idx = 1  # Índice inicial
	
	while total_spins > 0
		# Calculamos cuántos spins tendrá la siguiente parte
		spins_in_part = min(max_spins, total_spins)
		
		# Calculamos el índice final para esta parte
		end_idx = start_idx + spins_in_part - 1
		
		# Guardamos el rango en la lista
		push!(parts, start_idx:end_idx)
		
		# Actualizamos el total de spins restantes y el siguiente índice de inicio
		total_spins -= spins_in_part
		start_idx = end_idx + 1
	end

	return parts
end