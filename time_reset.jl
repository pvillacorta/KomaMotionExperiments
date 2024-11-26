using KomaMRICore
# using CUDA
using oneAPI

obj = Phantom(x = [0.0, 0.0])
obj.motion = MotionList(FlowPath([0.0 0.0; 0.0 0.0], [0.0 1.0; 0.0 1.0], [0.0 0.0; 0.0 0.0], [false false; false false], TimeRange(0.0, 10.0)))
# obj = Phantom(x = [0.0])
# obj.motion = MotionList(FlowPath([0.0 0.0], [0.0 1.0], [0.0 0.0], [false false], TimeRange(0.0, 10.0)))
seq = PulseDesigner.EPI_example()
sys = Scanner()

sim_params = Dict{String, Any}(
    "sim_method"=>KomaMRICore.BlochSimple(),
    "return_type"=>"mat"
)

sig = simulate(obj, seq, sys; sim_params)  

##
using KomaMRICore, oneAPI

Nadc = 25
M0 = 1.0
T1 = 100e-3
T2 = 10e-3
B1 = 20e-6
Trf = 3e-3
γ = 2π * 42.58e6
φ = π / 4
B1e(t) = B1 * (0 <= t <= Trf)
duration = 2*Trf

Gx = 1e-3
Gy = 1e-3
Gz = 0

# obj = brain_phantom3D(;ss=2)

obj = Phantom(x = [0.0, 0.0])
obj.motion = MotionList(FlowPath([0.0 0.0; 0.0 0.0], [0.0 1.0; 0.0 1.0], [0.0 0.0; 0.0 0.0], [false false; false false], TimeRange(0.0, 10.0)))

Ns = length(obj)
Nt = 3
dx = dy = dz = rand(Ns, Nt)
spin_reset = Float64.(dx .> 0.5)
obj.motion = MotionList(FlowPath(dx, dy, dz, spin_reset, TimeRange(0.0, 1.0)))

sys = Scanner()
seq = Sequence()
seq += RF(cis(φ) .* B1, Trf)
seq.GR[1,1] = Grad(Gx, duration)
seq.GR[2,1] = Grad(Gy, duration)
seq.GR[3,1] = Grad(Gz, duration)
seq.ADC[1] = ADC(Nadc, duration-Trf, Trf)

sim_params = Dict{String, Any}(
    "sim_method"=>KomaMRICore.BlochSimple(),
    "return_type"=>"mat"
)

sig = simulate(obj, seq, sys; sim_params)

##
using CUDA
t = CuArray(Float32.([0.01 0.3 0.32 0.4 0.8 0.85 0.9 0.99 1]))
spin_reset = [0 0 0 0;
              0 0 0 1;
              0 0 1 1;
              0 1 0 1;
              1 0 0 1]

spin_reset = CuArray(Array{Bool}(spin_reset))

T = Float32
# t_nodes = CUDA.rand(T, 4)
# t_nodes .= range(zero(T), oneunit(T), size(spin_reset, 2))


# [1 1 1 1 1 1 1 1 1;
#  1 1 1 1 1 1 1 1 0;
#  1 1 1 1 0 0 0 0 0;
#  1 0 0 0 0 0 0 0 0;
#  0 0 0 0 0 0 0 0 0]


idx = 1
t_reset = CUDA.zeros(T, size(spin_reset, 1))

# for row in eachrow(spin_reset)
#     first = findfirst(row) 
#     first = first === nothing ? size(spin_reset, 2) : first - 1 
#     @view(t_reset[idx]) .= first/(T(size(spin_reset, 2) - 1))
#     idx += 1
# end

# t .< t_reset


# function get_mask(t, spin_reset)
#     T = eltype(t)
#     Ns, Nt = size(spin_reset)
#     idx = 1
#     t_reset = similar(t, Ns) # alloc
#     for row in eachrow(spin_reset)
#         first = findfirst(row) 
#         first = first === nothing ? size(spin_reset, 2) : first - 1 
#         @view(t_reset[idx]) .= first/(T(size(spin_reset, 2) - 1))
#         idx += 1
#     end
#     return t.< t_reset
# end


first = findfirst.(eachrow(spin_reset))
first = replace(first, nothing => size(spin_reset, 2) + 1)
first .-= 1
first = T.(first) ./ (size(spin_reset, 2) - 1)

mask = get_mask(t, spin_reset)

@view(t_reset)



Interpolations.GriddedInterpolation{Bool, 1, CuArray{Bool, 1, CUDA.DeviceMemory}, Interpolations.Gridded{Interpolations.Constant{Interpolations.Previous, Interpolations.Throw{Interpolations.OnGrid}}}, Tuple{CuArray{Float32, 1, CUDA.DeviceMemory}}}
Interpolations.GriddedInterpolation{T, 1, V, Itp, K} where {V<:(AbstractArray{<:Union{Bool, T}}), Itp<:Interpolations.Gridded, K<:Tuple{AbstractVector{T}}}



## Compare allocations
using KomaMRICore

sys = Scanner()
obj = brain_phantom3D()[1:10000]
seq = PulseDesigner.EPI_example()
simParams = KomaMRICore.default_sim_params()
simParams["Nthreads"] = 8
simulate(obj, seq, sys; sim_params=simParams)


# Allocations:
#   before: 2.90M

#   only reset in excitation: 2.90M
#   reset final excitation y precession: 2.90M
#   solo reset matriz precession: 3.16M
#   reset matrix precession (view): 3.07M

#   after: 2.79M
