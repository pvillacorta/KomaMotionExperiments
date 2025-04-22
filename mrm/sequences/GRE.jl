include("bipolar_gradients.jl")

"""
Basic gradient-echo (GRE) Sequence
"""
function GRE(
    FOV::Float64, 
    N::Int, 
    TE::Float64, 
    TR::Float64, 
    α, sys::Scanner; 
    G=[0,0,1e-3], 
    Δf=0,
	pulse_duration = 3e-3,
)
    # Excitation (Sinc pulse) ----------------------------------
	B_1° = 2.59947e-7 / (pulse_duration * 1e3)
	B1 = α*  B_1°
	EX = PulseDesigner.RF_sinc(-1im*B1,pulse_duration,sys;G=G,Δf=Δf)

	# Acquisition ----------------------------------------------
	# Square acquisition (Nx = Ny = N) 
	# PHASE
	ζ_phase = EX[2].GR[1].rise
	T_phase = EX[2].GR[1].T

	Δk = (1/FOV)
	FOVk = (N-1)*Δk
	Gx = Gy = FOVk/(γ*(T_phase + ζ_phase))
	step = Δk/(γ*(T_phase + ζ_phase))

	# print("Δk = ", Δk, " m⁻¹\n")
	# print("FOVk = ", FOVk, " m⁻¹\n")
	# print("Gx = ", Gx*1e3, " mT/m\n")
	# print("step = ", step*1e3, " mT/m\n")

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

	# print("ACQ_dur = ", ACQ_dur*1e3, " ms\n")
	# print("G_ro = ", G_ro*1e3, " mT/m\n")
	# print("ζ = ", ζ_ro*1e3, " ms\n")

	gre = Sequence()
	for i in 0:(N-1)
		# Excitation and first phase 
		EX = PulseDesigner.RF_sinc(-1im*B1,pulse_duration,sys;G=G,Δf=Δf)
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
PC_GRE
"""
function PC_GRE(
	venc::Real, 
	direction::Array,
	FOV::Union{Array,Real},
	N::Union{Array,Real},
	TE::Real, TR::Real, 
	α, sys::Scanner;
	pulse_duration = 3e-3,
	adc_duration = 3e-3,
	venc_duration_flat = 0.1e-3,
	venc_duration_rise = 1.8e-3,
	balanced = false,
	spoiled = false,
	flow_compensation = false,
	crusher_duration = 1e-3,
	crusher_area = 0.0,
	slice_thickness = 2e-3,
	z0 = 0.0,
	rf_phase_offset = 0.0,
	dummy_pulses = 0
)
	# If only fov and N are provided in one dimension: Square acquisition
	FOV = FOV isa Array ? FOV : [FOV,FOV]
	N = N isa Array ? N : [N,N]

	# SINC pulse parameters 
	Trf = pulse_duration
	B_1° = 2.59947e-7 / (pulse_duration * 1e3)
	B1 = α*  B_1°
	TBP = 4
	BW = TBP / Trf

	# Choose slice thickness dz and position z0
	dz = slice_thickness
	Gz = BW / (γ * dz)
	f0 = γ * z0 * Gz

	# Sequence
	get_b1(B1, rf_phase_offset) = begin
		if     mod(rf_phase_offset, 2π) == 0.0
			return -1im*B1
		elseif mod(rf_phase_offset, 2π) == π/2
			return B1
		elseif mod(rf_phase_offset, 2π) == π
			return 1im*B1
		elseif mod(rf_phase_offset, 2π) == 3π/2
			return -B1
		else
			return B1
		end
	end

	EX = PulseDesigner.RF_sinc(get_b1(B1, rf_phase_offset), Trf, sys; G=[0; 0; Gz], TBP=TBP, Δf=f0)

	# Acquisition ----------------------------------------------
	ζ_phase = 0.2e-3
	T_phase = 0.6e-3

	Δkx, Δky = 1 ./ FOV
	FOVkx, FOVky = (N .- 1) .* [Δkx, Δky]

	Gy = FOVky/(γ*(T_phase + ζ_phase))
	Gx = FOVkx/(γ*(T_phase + ζ_phase))

	step = Δky/(γ*(T_phase + ζ_phase))

	# Bipolar gradients
	bipolar = bipolar_gradients(venc,  direction, venc_duration_flat, venc_duration_rise) 

	# Flow compensation
	flow_compensated_ss = Sequence()
	flow_compensated_fe = Sequence()

	if flow_compensation
		# Slice selection
		EX[2].GR[3].A = 2*EX[2].GR[3].A
		flow_compensated_ss = -EX[2]/2
		# Frecuency encoding
		grad = Grad(1, T_phase, ζ_phase)
		flow_compensated_fe = Sequence([(Gx/2)*grad; 0*grad;  0*grad;;])
		Gx = 2*Gx
	end

	# FE and Readout
	adc_duration_min = sys.ADC_Δt*(N[1]-1)
	if adc_duration < adc_duration_min
		print("adc_duration too short. Setting it to adc_duration_min = ", adc_duration_min*1e3, " ms.")
		adc_duration = adc_duration_min
	end

	G_ro = FOVkx/(γ*adc_duration)
	ζ_ro = G_ro / sys.Smax
	T_ro = adc_duration - ζ_ro
	GR = reshape([Grad(G_ro,T_ro,ζ_ro), Grad(0,T_ro,ζ_ro), Grad(0,T_ro,ζ_ro)],(3,1))
	RO = Sequence(GR)
	RO.ADC[1] = ADC(N[1], T_ro, ζ_ro)

	TE_min = dur(RO)/2 + EX.DUR[1]/2 + EX.DUR[2] + dur(flow_compensated_ss) +  dur(bipolar) + 2*T_phase + 4*ζ_phase
	if TE < TE_min
		print("TE too short. Setting it to TE_min = ", TE_min*1e3, " ms\n")
		TE = TE_min
	end
	delay_TE = TE - TE_min

	t_balanced = balanced ? T_phase + 2*ζ_phase : 0.0
	crusher_duration = spoiled ? crusher_duration : 0.0
	TR_min = dur(EX) + dur(flow_compensated_ss) + delay_TE + dur(bipolar) + 2*T_phase + 4*ζ_phase + dur(RO) + max(t_balanced, crusher_duration)
	if TR < TR_min
		print("TR too short. Setting it to TR_min = ", TR_min*1e3, " ms\n")
		TR = TR_min
	end
	delay_TR = TR - TR_min

	crusher = spoiled ? Grad(crusher_area/(0.9*crusher_duration), 0.8*crusher_duration, 0.1*crusher_duration) : Grad(0., 0., 0.)

	gre_a = Sequence()
	gre_b = Sequence()
	gre   = Sequence()

	for i in 0:(dummy_pulses + N[2] - 1)
		gre_a += EX + flow_compensated_ss + Delay(delay_TE) 
		gre_b += EX + flow_compensated_ss + Delay(delay_TE) 
		gre   += EX + flow_compensated_ss + Delay(delay_TE) 

		grad = Grad(1, T_phase, ζ_phase)
		dephase          =            Sequence([(-Gx/2)*grad; (Gy/2 - i*step)*grad;  0*grad;;])
		balanced_dephase = balanced ? Sequence([0*crusher; 	  (-Gy/2 + i*step)*grad;   crusher;;]) : Sequence()

		if i in 0:dummy_pulses-1
			gre_a += Delay(dur(bipolar + flow_compensated_fe + dephase + RO + balanced_dephase) + delay_TR)
			gre_b += Delay(dur(bipolar + flow_compensated_fe + dephase + RO + balanced_dephase) + delay_TR)
			gre   += Delay(dur(bipolar + flow_compensated_fe + dephase + RO + balanced_dephase) + delay_TR)
		else
			gre_a += ( bipolar            + flow_compensated_fe + dephase + RO + balanced_dephase + Delay(delay_TR))
			gre_b += (-bipolar            + flow_compensated_fe + dephase + RO + balanced_dephase + Delay(delay_TR))
			gre   += (Delay(dur(bipolar)) + flow_compensated_fe + dephase + RO + balanced_dephase + Delay(delay_TR))
		end
	end

	gre_a.DEF = Dict("Nx"=>N[1],"Ny"=>N[2],"Nz"=>1,"Name"=>"gre_a_"*string(N[1])*"x"*string(N[2]),"FOV"=>[FOV[1], FOV[2], 0])
	gre_b.DEF = Dict("Nx"=>N[1],"Ny"=>N[2],"Nz"=>1,"Name"=>"gre_b_"*string(N[1])*"x"*string(N[2]),"FOV"=>[FOV[1], FOV[2], 0])
	gre.DEF   = Dict("Nx"=>N[1],"Ny"=>N[2],"Nz"=>1,"Name"=>"gre_"*string(N[1])*"x"*string(N[2]),"FOV"=>[FOV[1], FOV[2], 0])
	
	return gre_a, gre_b, gre
end