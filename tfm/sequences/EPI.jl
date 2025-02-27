include("bipolar_gradients.jl")

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
	#Pre-wind and wind gradients
	ϵ2 = Ta/(Ta+ζ)
    PHASE =   Sequence(reshape(1/2*[Grad(      -Ga, Ta, ζ); ϵ2*Grad(-Ga, Ta, ζ)],:,1)) #This needs to be calculated differently
	DEPHASE = Sequence(reshape(1/2*[Grad((-1)^N*Ga, Ta, ζ); ϵ2*Grad(-Ga, Ta, ζ)],:,1)) #for even N
	seq = PHASE+epi+DEPHASE
	#Saving parameters
	seq.DEF = Dict("Nx"=>Nx,"Ny"=>Ny,"Nz"=>1,"Name"=>"epi")
	return seq
end

function PC_EPI_multishot(venc, direction, TE, TR, flip_angle, FOV::Real, N::Integer, sys::Scanner, ETL=N; Δt=sys.ADC_Δt, R = collect(I(3)), delta_f=0)
	@assert N%ETL == 0 "N must be  multiple of ETL"
	seq_A = Sequence()
	seq_B = Sequence()

	# Excitation
	T_rf = 3e-3   	# Pulse duration
	Gss = 4.5e-3    # Slice-select gradient	
	B_1° = 8.6649e-8
	B1 = flip_angle*B_1°
	EX = PulseDesigner.RF_sinc(-1im*B1,T_rf,sys;G=[0,0,Gss],Δf=delta_f)

	epi = EPI(FOV, N, sys)
	base_epi = epi[1:2*ETL] + epi[end]
	bipolar = bipolar_gradients(venc, direction, 1.8e-3, 0.1e-3)

	TE_min = ( EX.DUR[1]/2 + EX.DUR[2] + dur(bipolar) + dur(base_epi)/2 )
	@assert TE > TE_min "TE too short, please set it to at least TE_min = $(TE_min * 1e3) ms"
	delay_TE = Delay(TE - TE_min)

	TR_min = (dur(EX) + TE + dur(bipolar) + dur(base_epi))
	@assert TR > TR_min "TR too short, please set it to at least TR_min = $(TR_min * 1e3) ms"
	delay_TR = Delay(TR - TR_min)

	for i in 0:Int(N/ETL) - 1
		aux = copy(base_epi)
		Ga = -aux[end].GR[2].A
		step = 2*Ga*ETL/N
		aux[1].GR[2].A   += i*step
		aux[end].GR[2].A  = step - i*step
 		seq_A += R * EX + bipolar + delay_TE + R * aux + delay_TR
		seq_B += R * EX - bipolar + delay_TE + R * aux + delay_TR
	end

	return seq_A, seq_B
end