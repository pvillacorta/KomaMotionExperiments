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