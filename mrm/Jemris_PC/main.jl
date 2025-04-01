cd(@__DIR__)

using KomaMRI, CUDA, StatsBase, JLD2, JSON3

include("../sequences/GRE.jl")
include("../utils/divide_spins_ranges.jl") 

## ---- Phantom ----
obj = read_phantom("../phantoms/fortin.phantom")
obj.motion = NoMotion()
obj = obj[abs.(obj.x) .<= 0.005]
# sigma = 5e-4
# obj.x .+= sigma .* randn(length(obj))
# obj.x .+= sigma .* randn(length(obj))
# obj.x .+= sigma .* randn(length(obj))
## ---- Scanner ---- 
sys = Scanner(Gmax=80e-3, Smax=200, GR_Δt=1e-5, RF_Δt=1e-5, ADC_Δt=1e-5)

## ---- Sequence ----
# Overload KomaMRI `*` function to avoid the following: ---------
# Rotating an array of gradients (in x, y, and z) with 
# different durations causes all durations to become the same
# https://github.com/JuliaHealth/KomaMRI.jl/issues/545
## Execute this only if the GitHub issue above is not solved:
# import KomaMRIBase: *
# function *(α::Matrix, x::Array{Grad})
#     y = deepcopy(x)
#     A_values = [g.A for g in x]  
#     A_result = α * A_values     
#     for (i, g) in enumerate(y)
#         g.A = A_result[i]     
#     end
#     return y
# end

##
# obj = brain_phantom2D()
# obj.z .= -obj.x
# obj.y .= obj.y
# obj.x .= 0
# ---------------------------------------------------------------

fov      = [0.18, 0.13]
N_matrix = [92, 66]
# N_matrix = [115, 83]
# N_matrix = [461, 333]
res = fov ./ N_matrix

TE = 10e-3
TR = 16e-3
flip_angle = 15.0
slice_thickness = 2.5e-3

R = [ 0.  0. 1. ; 
      0.  1. 0. ; 
     -1.  0. 0. ]

# R = [ 1.  0. 0. ; 
#       0.  1. 0. ; 
#       0.  0. 1. ]

#VENC
vs                  = "vx" 
venc                = 200.0 .* 1e-3
venc_durations_flat = 0.8   .* 1e-3
venc_durations_rise = 0.1   .* 1e-3

##
kspace      = []
magnitude   = []
phase       = [] 
seqs        = Sequence[]
acqs        = []

rg = 1:1

MAX_SPINS_PER_GPU = 200_000
sequential_parts = divide_spins_ranges(length(obj), MAX_SPINS_PER_GPU)

## ---- Sequence ----
seq = read_seq("../sequences/PC_2D_tra_1dVz_92x66_TE10_TR16.seq")
seq_a = R*seq[1:convert(Int, length(seq)/2 -1)]
seq_b = R*seq[convert(Int, length(seq)/2)+1:end-1]

seq_a, seq_b, _ = PC_GRE(
    venc,
    Float64.([vs=="vx", vs=="vy", vs=="vz"]),
    fov,
    N_matrix,
    TE,
    TR,
    flip_angle,
    sys;
    R = R,
    pulse_duration = 1e-3,
    adc_duration = 10e-3,
    venc_duration_flat = venc_durations_flat,
    venc_duration_rise = venc_durations_rise,
    balanced = true,
    crusher_duration = 1e-3,
    crusher_area = 4 * π / (2 * π * γ) / res[1],
    slice_thickness = slice_thickness,
    rf_phase_offset = 0
)

push!(seqs, seq_a)
push!(seqs, seq_b)

## ---- Simulation ----
for seq in seqs
    sim_params = KomaMRICore.default_sim_params()
    sim_params["Nblocks"] = 5000
    sim_params["Δt"]    = 8e-4
    sim_params["Δt_rf"] = 1e-5

    if length(sequential_parts) > 1
        @info "Dividing phantom ($(length(obj)) spins) into $(length(sequential_parts)) parts that will be simulated sequentially"
    end

    raws = []
    for (j, sequential_part) in enumerate(sequential_parts)
        if length(sequential_parts) > 1
            @info "Simulating phantom part $(j)/$(length(sequential_parts))"
        end
        push!(raws, simulate(obj[sequential_part], seq, sys))
    end
    raw = reduce(+, raws)

    ## ---- Reconstruction ----
    recParams = Dict{Symbol,Any}(:reco=>"direct")
    Nx, Ny = N_matrix
    recParams[:reconSize] = (Nx, Ny)
    recParams[:densityWeighting] = false
    push!(acqs, raw)
    acqData = AcquisitionData(raw)
    seq_no_rot = inv(R) * seq
    _, ktraj = get_kspace(seq_no_rot)
    # Kdata
    acqData.kdata[1] = reshape(acqData.kdata[1],(Nx*Ny,1))
    # Traj
    acqData.traj[1].circular = false
    acqData.traj[1].nodes = transpose(ktraj[:, 1:2])
    acqData.traj[1].nodes = acqData.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqData.traj[1].nodes[:]))
    acqData.traj[1].numProfiles = Ny
    # subsampleIndices
    acqData.subsampleIndices[1] = acqData.subsampleIndices[1][1:Nx*Ny]
    # Reconstruction
    aux = @timed reconstruction(acqData, recParams)
    recon = reshape(aux.value.data, Nx, Ny, :)

    push!(magnitude, abs.(recon[:,:,1]))
    push!(phase,     angle.(recon[:,:,1]))
end


## Plot results
magnitude_mean = (magnitude[1] .+ magnitude[2]) ./ 2
phase_diff     = -(mod.(phase[1] .- phase[2] .+ π, 2π) .- π)

magnitudes = [magnitude[1], magnitude[2], magnitude_mean]
phases     = [phase[1], phase[2], phase_diff]

magnitude_titles = ["Magnitude A", "Magnitude B", "Mean Magnitude (A+B)/2"] 
phase_titles     = ["Phase A", "Phase B", "Phase A - Phase B"]

fig = KomaMRIPlots.make_subplots(
    rows=2, cols=3, 
    subplot_titles= hcat(magnitude_titles, phase_titles),
    shared_yaxes=true, 
    shared_xaxes=true,
    vertical_spacing=0.05,
    horizontal_spacing=0.0
)

for i in 1:3
    KomaMRIPlots.add_trace!(fig, plot_image(magnitudes[i], zmin=minimum(vcat(magnitudes...)), zmax=percentile(vec(vcat(magnitudes...)), 99)).plot.data[1], row=1, col=i)
    KomaMRIPlots.add_trace!(fig, plot_image(phases[i], zmin=-π, zmax=π, colorscale="Jet").plot.data[1], row=2, col=i)
end

display(fig)

# savefig(fig, "PC_GRE.png"; height=800, width=1500)

##
data = [acqs[1].profiles[i].data for i in 1:66]
kspace = hcat(data...)
plot_image(abs.(kspace))

