using LinearAlgebra, StatsBase

"""
Basic spin-echo (SE) Sequence
"""
SE(FOV::Float64, N::Int, TE::Float64, TR::Float64, sys::Scanner; G=[0,0,0], Δf=0) = begin
    # Excitation and inversion (Sinc pulse) ----------------------------------  

	"""
	# T_rf = 6e-3     # Pulse duration
	# Gss = 2e-3      # Slice-select gradient
	# With T = 6ms, we need B1 = 4.3469e-8 T to produce a flip angle α = 1°
	# B_1° = 4.3469e-8
	"""

	T_rf = 3e-3     # Pulse duration
	Gss = 2e-3      # Slice-select gradient

	# With T = 3ms, we need B1 = 8,69e-8 T to produce a flip angle α = 1°
	B_1° = 8.6938e-8

	# 90º pulse
	EX_90 = PulseDesigner.RF_sinc(90*B_1°,T_rf,sys;G=[0,0,Gss],Δf=Δf)

	# 180º pulse
	EX_180 = PulseDesigner.RF_sinc(180*B_1°*1im,T_rf,sys;G=[0,0,Gss],Δf=Δf)[1]

	# Acquisition ----------------------------------------------------
	# Square acquisition (Nx = Ny = N) 
	# PHASE
	ζ_phase = EX_90[2].GR[1].rise
	T_phase = EX_90[2].GR[1].T


	Δk = (1/FOV)
	FOVk = (N-1)*Δk
	Gx = Gy = FOVk/(γ*(T_phase + ζ_phase))
	step = Δk/(γ*(T_phase + ζ_phase))

	print("Δk = ", Δk, " m⁻¹\n")
	print("FOVk = ", FOVk, " m⁻¹\n")
	print("Gx = ", Gx*1e3, " mT/m\n")
	print("step = ", step*1e3, " mT/m\n")

	# FE and Readout
	TE_min = maximum(2*[(EX_90.DUR[1]/2 + EX_90.DUR[2] + EX_180.DUR[1]/2), 
						(EX_180.DUR[1]/2 + sys.ADC_Δt*(N-1)/2)])
	if TE < TE_min
		print("Error: TE must be greater than TE_min = ", TE_min*1e3, " ms\n")
		return
	end

	delay_TE = TE/2 -(EX_90.DUR[1]/2 + EX_90.DUR[2] + EX_180.DUR[1]/2)

	ACQ_dur = TE - EX_180.DUR[1]
	G_ro = FOVk/(γ*ACQ_dur)
	ζ_ro = G_ro / sys.Smax
	T_ro = ACQ_dur - ζ_ro
	GR = reshape([Grad(G_ro,T_ro,ζ_ro), Grad(0,0), Grad(0,0)],(3,1))
	RO = Sequence(GR)
	RO.ADC[1] = ADC(N, T_ro, ζ_ro)

	delay_TR = TR - TE - ACQ_dur/2 - EX_90.DUR[1]/2

	print("ACQ_dur = ", ACQ_dur*1e3, " ms\n")
	print("G_ro = ", G_ro*1e3, " mT/m\n")
	print("ζ = ", ζ_ro*1e3, " ms\n")

	se = Sequence()
	for i in 0:(N-1)
		# 90º pulse
		EX_90 = PulseDesigner.RF_sinc(90*B_1°,T_rf,sys;G=[0,0,Gss],Δf=Δf)
		EX_90[end].GR[1].A = Gx/2
		EX_90[end].GR[2].A = Gy/2 - i*step
		# 180º pulse
		EX_180 = RF_sinc(180*B_1°*1im,T_rf,sys;G=[0,0,Gss],Δf=Δf)[1]

		se += EX_90 + Delay(delay_TE) + EX_180 + RO + Delay(delay_TR)
	end

	R = rotation_matrix(G)
	se.DEF = Dict("Nx"=>N,"Ny"=>N,"Nz"=>1,"Name"=>"se"*string(N)*"x"*string(N),"FOV"=>[FOV, FOV, 0])
	R*se[2:end]
end


"""
Basic gradient-echo (GRE) Sequence
"""
GRE(FOV::Float64, N::Int, TE::Float64, TR::Float64, α, sys::Scanner; G=[0,0,0], Δf=0) = begin
	EX = selective_rf(α, sys; G=G, Δf=Δf)

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
		EX = selective_rf(α, sys; G=G, Δf=Δf)
		EX[end].GR[1].A = -Gx/2
		EX[end].GR[2].A = -Gy/2 + i*step
		gre += EX

		# FE and Readout
		gre += RO + Delay(delay_TR)
	end
	gre.DEF = Dict("Nx"=>N,"Ny"=>N,"Nz"=>1,"Name"=>"gre"*string(N)*"x"*string(N),"FOV"=>[FOV, FOV, 0])
	return gre
end



bSSFP(FOV::Float64, N::Int, TR::Float64, α, sys::Scanner; G=[0,0,0], Δf=0) = begin
	EX = selective_rf(α, sys; G=G, Δf=Δf)

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

	gre = Sequence()
	for i in 0:(N-1)
		# Excitation and first phase 
		EX = selective_rf(α, sys; G=G, Δf=Δf)
		EX[end].GR[1].A = -Gx/2
		EX[end].GR[2].A = -Gy/2 + i*step
		gre += EX

		# FE and Readout
		balance = Sequence(reshape([ EX[end].GR[1],
					    -EX[end].GR[2],
					     EX[end].GR[3]],(3,1)))


		gre += Delay(delay) + RO + Delay(delay) + balance
	end
	gre.DEF = Dict("Nx"=>N,"Ny"=>N,"Nz"=>1,"Name"=>"gre"*string(N)*"x"*string(N),"FOV"=>[FOV, FOV, 0])
	return gre
end


"""
	bipolar = bipolar_gradients(venc, direction, delay, sys)

Bipolar gracients for PC MRI. Taken from cmrsim
"""
bipolar_gradients(venc::Float64, direction::Array, sys::Scanner, G=sys.Gmax/4) = begin
	# desired M1
	M1 = π / (γ * venc)

	# find flat_duration for a G = Gmax/4
	a = G
	b = 3 * G^2 / sys.Smax
	c = 2 * G^3 / sys.Smax^2 - M1

	flat_time = (-b + sqrt(b^2 -4*a*c)) / (2*a)

	if flat_time > 0 # trapezoidal gradient
		rise = G / sys.Smax
		gr = Grad(G, flat_time, rise)
		lobe = Sequence((direction ./ norm(direction)) .* [gr; gr; gr;;])
		return lobe - lobe
	else # triangular gradient

	end
end


"""
PC_GRE
"""
function PC_GRE(
	venc::Float64, 
	direction::Array,
	FOV::Float64,
	N::Int,
	TE::Float64, TR::Float64, 
	α, sys::Scanner;
	G=[0, 0, 2e-3], 
	Δf=0
)
    EX = selective_rf(α, sys; G=G, Δf=Δf)

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

	# Bipolar gradients
	bipolar = bipolar_gradients(venc, direction, sys)

	# FE and Readout
	TE_min = sys.ADC_Δt*(N-1)/2 + (EX.DUR[1]/2) + EX.DUR[2] + dur(bipolar)
	if TE < TE_min
		print("Error: TE must be greater than TE_min = ", TE_min*1e3, " ms\n")
		return
	end

	ACQ_dur = 2 * (TE - ( EX.DUR[1]/2 + EX.DUR[2] + dur(bipolar) ))
	G_ro = FOVk/(γ*ACQ_dur)
	ζ_ro = G_ro / sys.Smax
	T_ro = ACQ_dur - ζ_ro
	GR = reshape([Grad(G_ro,T_ro,ζ_ro), Grad(0,0), Grad(0,0)],(3,1))
	RO = Sequence(GR)
	RO.ADC[1] = ADC(N, T_ro, ζ_ro)
	delay_TR = TR - (EX.DUR[1] + 2*EX.DUR[2] + dur(bipolar) + RO.DUR[1])

	# print("ACQ_dur = ", ACQ_dur*1e3, " ms\n")
	# print("G_ro = ", G_ro*1e3, " mT/m\n")
	# print("ζ = ", ζ_ro*1e3, " ms\n")

	gre = Sequence()
	for i in 0:(N-1)
		# Excitation and first phase 
		EX = PulseDesigner.RF_sinc(B1,T_rf,sys;G=[0,0,Gss],Δf=Δf)
		gre += EX

		# Bipolar gradients
		gre += bipolar

		# Position at the start of the line
		grad = Grad(1, T_phase, ζ_phase)
		dephase = Sequence([(-Gx/2)*grad; (-Gy/2 + i*step)*grad; 0*grad;;])
		gre += dephase

		# FE and Readout
		gre += RO + Delay(delay_TR)
	end
	gre.DEF = Dict("Nx"=>N,"Ny"=>N,"Nz"=>1,"Name"=>"gre"*string(N)*"x"*string(N),"FOV"=>[FOV, FOV, 0])
	return gre
end

"""
	Selective excitation
Arguments:
- Flip angle
- Scanner
Keywords
- Slice-selection gradients
- Frequency shift 
"""
function selective_rf(flip_angle, sys::Scanner; G::Array=[0,0,2e-3], Δf=0.0)
	# Excitation (Sinc pulse) ----------------------------------
	# α = γ ∫(0-T) B1(t)dt 
	# ----------------------
	# We need to obtain B1 from flip angle α and a generic T=3ms duration
	# i.e. we need to resolve the equation above
	T_rf = 3e-3   	# Pulse duration
	# With T = 3ms, we need B1 = 8,69e-8 T to produce a flip angle α = 1°
	B_1° = 8.6938e-8
	B1 = flip_angle*B_1°

	return PulseDesigner.RF_sinc(B1, T_rf, sys; G=G, Δf=Δf)
end

function EPI(FOV::Real, N::Integer, sys::Scanner; Δt=sys.ADC_Δt)
	Gmax = sys.Gmax
	Nx = Ny = N #Square acquisition
	Δx = FOV/(Nx-1)
	Ta = Δt*(Nx-1) #4-8 us
    	Δτ = Ta/(Ny-1)
	Ga = 1/(γ*Δt*FOV)
	ζ = Ga / sys.Smax
	if Ga > Gmax
		println("Ga=$(Ga*1e3) mT/m exceeds Gmax=$(Gmax*1e3) mT/m, increasing Δt to Δt_min="*string(round(1e6/(γ*Gmax*FOV),digits=2))*" us...")
		return EPI(FOV, N, sys; Δt=1/(γ*Gmax*0.99*FOV))
	end
	ϵ1 = Δτ/(Δτ+ζ)
	#EPI base
	epi = Sequence(vcat(
	    [mod(i,2)==0 ? Grad(Ga*(-1)^(i/2),Ta,ζ) : Grad(0.,Δτ,ζ) for i=0:2*Ny-2],  #Gx
	 	[mod(i,2)==1 ? ϵ1*Grad(Ga,Δτ,ζ) :         Grad(0.,Ta,ζ) for i=0:2*Ny-2])) #Gy
	epi.ADC = [mod(i,2)==1 ? ADC(0,Δτ,ζ) : ADC(N,Ta,ζ) for i=0:2*Ny-2]
	# Relevant parameters
	Δfx_pix = 1/Ta
	Δt_phase = (Ta+2ζ)*Ny + (Δτ+2ζ)*Ny
	Δfx_pix_phase = 1/Δt_phase
	# println("## EPI parameters ##")
	# println("Δx = $(round(Δx*1e3,digits=2)) mm")
	# println("Pixel Δf in freq. direction $(round(Δfx_pix,digits=2)) Hz")
	# println("Pixel Δf in phase direction $(round(Δfx_pix_phase,digits=2)) Hz")
	#Pre-wind and wind gradients
	ϵ2 = Ta/(Ta+ζ)
    PHASE =   Sequence(reshape(1/2*[Grad(      -Ga, Ta, ζ); ϵ2*Grad(-Ga, Ta, ζ)],:,1)) #This needs to be calculated differently
	DEPHASE = Sequence(reshape(1/2*[Grad((-1)^N*Ga, Ta, ζ); ϵ2*Grad(-Ga, Ta, ζ)],:,1)) #for even N
	seq = PHASE+epi+DEPHASE
	#Saving parameters
	seq.DEF = Dict("Nx"=>Nx,"Ny"=>Ny,"Nz"=>1,"Name"=>"epi")
	return seq
end
