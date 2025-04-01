include("bipolar_gradients.jl")

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
	Gss = 4.5e-3    # Slice-select gradient

	# With T = 3ms, we need B1 = 8,69e-8 T to produce a flip angle α = 1°
	B_1° = 8.6649e-8
	B1 = α*B_1°
	EX = PulseDesigner.RF_sinc(B1,T_rf,sys;G=[0,0,Gss],Δf=Δf)

	# Acquisition ----------------------------------------------
	# Square acquisition (Nx = Ny = N) 
	# PHASE
	ζ_phase = EX[2].GR[3].rise
	T_phase = EX[2].GR[3].T

	Δk = (1/FOV)
	FOVk = (N-1)*Δk
	Gx = Gy = FOVk/(γ*(T_phase + ζ_phase))
	step = Δk/(γ*(T_phase + ζ_phase))

	# FE and Readout
	G_ro = 0.7 * sys.Gmax
	ζ_ro = G_ro / sys.Smax
	T_ro = (FOVk/(γ * G_ro)) - ζ_ro

	RO = Sequence([Grad(G_ro,T_ro,ζ_ro,ζ_ro,0); Grad(0,0); Grad(0,0);;])
	RO.ADC[1] = ADC(N, T_ro, ζ_ro)

	bssfp = Sequence()
	for i in 0:(N-1)
		# Excitation and first phase 
		EX = PulseDesigner.RF_sinc(B1,T_rf,sys;G=[0,0,Gss],Δf=Δf)
		EX[end].GR[1].A = -Gx/2
		EX[end].GR[2].A = -Gy/2 + i*step

		# FE and Readout
		balance = Sequence([EX[end].GR[1]; -EX[end].GR[2]; EX[end].GR[3];;])

		delay_TR = TR - (dur(EX) + dur(RO) + dur(balance))
		bssfp += (EX + Delay(delay_TR/2) + RO + Delay(delay_TR/2) + balance)
	end
	bssfp.DEF = Dict("Nx"=>N,"Ny"=>N,"Nz"=>1,"Name"=>"bssfp"*string(N)*"x"*string(N),"FOV"=>[FOV, FOV, 0])
	return bssfp
end


"""
PC_bSSFP
"""
function PC_bSSFP(
	venc::Float64, 
	direction::Array,
	FOV::Float64,
	N::Int,
	TR::Float64, 
	α, sys::Scanner;
	G=[0, 0, 2e-3], 
	Δf=0,
	R = collect(I(3))
)
    # Excitation (Sinc pulse) ----------------------------------
	# α = γ ∫(0-T) B1(t)dt 
	# ----------------------
	# We need to obtain B1 from flip angle α and a generic T=3ms duration
	# i.e. we need to resolve the equation above

	T_rf = 3e-3   	# Pulse duration
	Gss = 4.5e-3    # Slice-select gradient

	# With T = 3ms, we need B1 = 8,69e-8 T to produce a flip angle α = 1°
	B_1° = 8.6649e-8
	B1 = α*B_1°
	EX = PulseDesigner.RF_sinc(B1,T_rf,sys;G=[0,0,Gss],Δf=Δf)

	# Bipolar gradients
	# We multiply the venc by 2 since we don't want to exceed ϕ = π
	# and the base phase for static spins is ϕ = π/2
	bipolar = bipolar_gradients(2 * venc,  direction, 1.5e-3, sys) 

	# Acquisition ----------------------------------------------
	# Square acquisition (Nx = Ny = N) 
	# PHASE
	ζ_phase = EX[2].GR[1].rise
	T_phase = EX[2].GR[1].T

	Δk = (1/FOV)
	FOVk = (N-1)*Δk
	Gx = Gy = FOVk/(γ*(T_phase + ζ_phase))
	step = Δk/(γ*(T_phase + ζ_phase))

	# FE and Readout
	G_ro = 0.5 * sys.Gmax
	ζ_ro = G_ro / sys.Smax
	T_ro = (FOVk/(γ * G_ro)) - ζ_ro

	# FE and Readout
	TR_min = 2 * (sys.ADC_Δt*(N-1)/2 + (EX.DUR[1]/2) + EX.DUR[2] + dur(bipolar))
	@assert TR > TR_min "Error: TR must be greater than TR_min = $(TR_min*1e3) ms"

	RO = Sequence([Grad(G_ro,T_ro,ζ_ro,ζ_ro,0); Grad(0,0); Grad(0,0);;])
	RO.ADC[1] = ADC(N, T_ro, ζ_ro)

	bssfp_pos = Sequence()
	bssfp_neg = Sequence()
	bssfp 	= Sequence()

	for i in 0:(N-1)
		EX = PulseDesigner.RF_sinc(B1,T_rf,sys;G=[0,0,Gss],Δf=Δf)
		EX[end].GR[1].A = -Gx/2
		EX[end].GR[2].A = -Gy/2 + i*step
		balance = Sequence(reshape([ EX[end].GR[1], -EX[end].GR[2], EX[end].GR[3]],(3,1)))

		delay_TR = TR - (dur(EX) + 2*dur(bipolar) + dur(RO) + dur(balance))
		bssfp_pos += R * EX + Delay(delay_TR/2) + bipolar 		      + R * RO - bipolar 			   + Delay(delay_TR/2) + R * balance		  
		bssfp_neg += R * EX + Delay(delay_TR/2) - bipolar			  + R * RO + bipolar 			   + Delay(delay_TR/2) + R * balance
		bssfp 	  += R * EX + Delay(delay_TR/2) + Delay(dur(bipolar)) + R * RO + Delay(dur(bipolar))   + Delay(delay_TR/2) + R * balance 
	end
	bssfp_pos.DEF = Dict("Nx"=>N,"Ny"=>N,"Nz"=>1,"Name"=>"bssfp_pos_"*string(N)*"x"*string(N),"FOV"=>[FOV, FOV, 0])
	bssfp_neg.DEF = Dict("Nx"=>N,"Ny"=>N,"Nz"=>1,"Name"=>"bssfp_neg_"*string(N)*"x"*string(N),"FOV"=>[FOV, FOV, 0])
	bssfp.DEF 	  = Dict("Nx"=>N,"Ny"=>N,"Nz"=>1,"Name"=>"bssfp_"*string(N)*"x"*string(N),"FOV"=>[FOV, FOV, 0])
	return bssfp_pos, bssfp_neg, bssfp
end