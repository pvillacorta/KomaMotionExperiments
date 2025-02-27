using LinearAlgebra, StatsBase

"""
	bipolar = bipolar_gradients(venc, direction, delay, sys)

Bipolar gracients for PC MRI
"""
bipolar_gradients(venc::Float64, direction::Array, T_flat, T_rise) = begin
	T = T_flat + 2*T_rise
	A =  1 / (2* γ * venc * T * (T-T_rise)) 
	gr = Grad(A, T_flat, T_rise)
	lobe = Sequence((direction ./ norm(direction)) .* [gr; gr; gr;;])
	return (lobe - lobe)
end