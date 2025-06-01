include("../sequences/bipolar_gradients.jl")

function PC_GRE(
	venc::Real, 
	direction::Array,
	FOV::Union{Array,Real},
	N::Union{Array,Real},
	TE::Real, TR::Real, 
	flip_angle, sys::Scanner;
	Δf=0,
	R = collect(I(3)),
	pulse_duration = 3e-3,
	adc_duration = 0.0,
	venc_duration_flat = 0.1e-3,
	venc_duration_rise = 1.8e-3,
	balanced = false,
	crusher_duration = 1e-3,
	crusher_area = 0.0,
	z0 = 0.0,
	slice_thickness = 10e-3,
	TBP = 4
)
	# If only fov and N are provided in one dimension: Square acquisition
	FOV = FOV isa Array ? FOV : [FOV,FOV]
	N = N isa Array ? N : [N,N]

	# SINC pulse parameters 
	BW = TBP / pulse_duration

	# B_1° = 2.59947e-7 / (pulse_duration * 1e3)
	# B1 = flip_angle*  B_1°

	flip_angle_rad = flip_angle * π / 180 
	t_rf, unit_wf = generate_unit_sinc_waveform(pulse_duration, TBP, sys; apodization=0.5)
	amplitude = scale_rf_waveform(unit_wf, flip_angle_rad, sys)

	# Choose slice thickness
	Gss = BW / (γ * slice_thickness)
	f0 = γ * z0 * Gss

	# EX = PulseDesigner.RF_sinc(-1im*amplitude, pulse_duration, sys; G=[0; 0; Gss], TBP=TBP, Δf=f0, a=0.5)

	ζ_ss = Gss / sys.Smax
	area_Gss = Gss * (pulse_duration + ζ_ss)
	area_refocus = area_Gss / 2
	G_refocus = sqrt(area_refocus * sys.Smax)
	ζ_refocus = area_refocus / G_refocus

	EX = Sequence([Grad(0., pulse_duration, ζ_ss)  Grad(0.,     	0, ζ_refocus); 
				   Grad(0., pulse_duration, ζ_ss)  Grad(0.,     	0, ζ_refocus); 
				   Grad(Gss, pulse_duration, ζ_ss) Grad(-G_refocus, 0, ζ_refocus);], 
				  [RF(-1im * amplitude .* unit_wf, pulse_duration, f0, ζ_ss) RF(0,0)])

	# Acquisition ----------------------------------------------
	ζ_phase = 0.05e-3
	T_phase = 0.15e-3

	Δkx, Δky = 1 ./ FOV
	FOVkx, FOVky = (N .- 1) .* [Δkx, Δky]

	Gy = FOVky/(γ*(T_phase + ζ_phase))
	Gx = FOVkx/(γ*(T_phase + ζ_phase))

	step = Δky/(γ*(T_phase + ζ_phase))

	# Bipolar gradients
	bipolar = bipolar_gradients(venc,  direction, venc_duration_flat, venc_duration_rise) 

	# FE and Readout
	if adc_duration == 0.0
		TE_min = sys.ADC_Δt*(N[1]-1)/2 + EX.DUR[1]/2 + EX.DUR[2] + dur(bipolar) + T_phase + 2*ζ_phase
		if TE < TE_min
			print("TE too short. Setting it to TE_min = ", TE_min*1e3, " ms\n")
			TE = TE_min
		end
		adc_duration = 2 * (TE - ( EX.DUR[1]/2 + EX.DUR[2] + dur(bipolar) + T_phase + 2*ζ_phase ))
	else
		TE_min = max(sys.ADC_Δt*(N[1]-1), adc_duration)/2 + EX.DUR[1]/2 + EX.DUR[2] + dur(bipolar) + T_phase + 2*ζ_phase
		if TE < TE_min
			print("TE too short. Setting it to TE_min = ", TE_min*1e3, " ms\n")
			TE = TE_min
		end
	end

	G_ro = FOVkx/(γ*adc_duration)
	ζ_ro = G_ro / sys.Smax
	T_ro = adc_duration - ζ_ro
	GR = reshape([Grad(G_ro,T_ro,ζ_ro), Grad(0,T_ro,ζ_ro), Grad(0,T_ro,ζ_ro)],(3,1))
	RO = Sequence(GR)
	RO.ADC[1] = ADC(N[1], T_ro, ζ_ro)

	t_balanced = balanced ? T_phase + 2*ζ_phase : 0.0
	TR_min = dur(EX) + dur(bipolar) + T_phase + 2*ζ_phase + dur(RO) + max(t_balanced, crusher_duration)
	if TR < TR_min
		print("TR too short. Setting it to TR_min = ", TR_min*1e3, " ms\n")
		TR = TR_min
	end
	delay_TR = TR - TR_min

	crusher = Grad(crusher_area/(0.9*crusher_duration), 0.8*crusher_duration, 0.1*crusher_duration) 

	gre_a = Sequence()
	gre_b = Sequence()
	gre   = Sequence()

	for i in 0:(N[2]-1)
		gre_a += R * EX + bipolar
		gre_b += R * EX - bipolar
		gre   += R * EX + Delay(dur(bipolar))

		grad = Grad(1, T_phase, ζ_phase)
		dephase          =            Sequence([(-Gx/2)*grad; (-Gy/2 + i*step)*grad;  0*grad;;])
		balanced_dephase = balanced ? Sequence([0*crusher; 	  (Gy/2 - i*step)*grad;  crusher;;]) : Sequence()

		gre_a += R * (dephase + RO + balanced_dephase + Delay(delay_TR))
		gre_b += R * (dephase + RO + balanced_dephase + Delay(delay_TR))
		gre   += R * (dephase + RO + balanced_dephase + Delay(delay_TR))
	end

	gre_a.DEF = Dict("Nx"=>N[1],"Ny"=>N[2],"Nz"=>1,"Name"=>"gre_a_"*string(N[1])*"x"*string(N[2]),"FOV"=>[FOV[1], FOV[2], 0])
	gre_b.DEF = Dict("Nx"=>N[1],"Ny"=>N[2],"Nz"=>1,"Name"=>"gre_b_"*string(N[1])*"x"*string(N[2]),"FOV"=>[FOV[1], FOV[2], 0])
	gre.DEF   = Dict("Nx"=>N[1],"Ny"=>N[2],"Nz"=>1,"Name"=>"gre_"*string(N[1])*"x"*string(N[2]),"FOV"=>[FOV[1], FOV[2], 0])
	
	return gre_a, gre_b, gre
end

function generate_unit_sinc_waveform(duration, TBP, sys::Scanner; apodization=0.5)
    Δt = sys.RF_Δt  # Raster time [s]
    n_steps = round(Int, duration / Δt)
    t = range(-duration / 2, stop = duration / 2, length = n_steps + 1)
    bw = TBP / duration

    # Ventana tipo Hanning o Hamming
    window = (1 .- apodization) .+ apodization .* cos.(2π .* collect(-n_steps ÷ 2 : n_steps ÷ 2) ./ n_steps)

    wf = sinc.(bw .* t) .* window
    wf .-= wf[1]  # quitar el valor DC si hace falta (como en Python)

    return t, wf
end

function scale_rf_waveform(unit_wf, flip_angle_rad, sys::Scanner)
    Δt = sys.RF_Δt
	gamma_rad = 2 * π * γ

    # Trapecio para integración
    integral = sum((unit_wf[1:end-1] .+ unit_wf[2:end]) ./ 2) * Δt
    unit_flip_angle = gamma_rad * integral 

    # amplitude = unit_wf .* (flip_angle_rad / unit_flip_angle)

    return (flip_angle_rad / unit_flip_angle)
end