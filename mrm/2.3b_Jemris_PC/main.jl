cd(@__DIR__)

using KomaMRI, CUDA, StatsBase, JLD2, JSON3

include("../sequences/bipolar_gradients.jl")
include("PC_GRE.jl")
include("../utils/divide_spins_ranges.jl") 
include("rotate_phantom.jl")

## ---- Phantom ----
Nt = 400
obj = load_phantom("fortin_2M_spins.phantom")
obj.T1 .= 850e-3
obj.T2 .= 5e-3 
obj = rotate_phantom(obj)
## ---- Scanner ---- 
sys = Scanner(Gmax=80e-3, Smax=200, GR_Δt=1e-5, RF_Δt=1e-5, ADC_Δt=1e-5)

## ---- Sequence ----
fov      = [0.18, 0.13]
# N_matrix = [92, 66]
# N_matrix = [115, 83]
# N_matrix = [230, 166]
N_matrix = [461, 333]

res = fov ./ N_matrix

TE = 10e-3
TR = 16e-3
flip_angle = 15.0
slice_thickness = 2.5e-3

Δt = 8e-4
Δt_rf = 1e-5

#VENC
vs                  = "vz" 
venc                = 20.0  .* 1e-2
venc_durations_flat = 0.8   .* 1e-3
venc_durations_rise = 0.1   .* 1e-3

##
# obj = brain_phantom2D()
# obj.T2 .= TR/4 # Manual spoiling

##
# L = 0.1
# dx = 1e-3

# r = -L/2:dx:L/2
# x = repeat(r, outer=(1, length(r)))
# y = repeat(r', outer=(length(r), 1))

# obj = Phantom(x=x[:], y=y[:])
# obj.T2 .= TR/4 # Manual spoiling

## ---- Sequence ----
seqs = Sequence[]
seq = read_seq("../sequences/PC_2D_tra_1dVz_$(N_matrix[1])x$(N_matrix[2])_TE10_TR16_flipangle15.seq") # Made with JEMRIS and exported to Pulseq

# for i in 1:1
for i in findall(is_RF_on.(seq))
    # Make RF real
    seq[i].RF.A[1] .= abs.(seq[i].RF.A[1]) .* (-1).^(abs.(angle.(seq[i].RF.A[1])) .> 0.05)
end

seq_a = seq[1:convert(Int, length(seq)/2 -1)]
seq_b = seq[convert(Int, length(seq)/2)+1:end-1]

# seq_a = seq_b = seq

seq_a, seq_b, _ = PC_GRE(
    venc,
    Float64.([vs=="vx", vs=="vy", vs=="vz"]),
    fov,
    N_matrix,
    TE,
    TR,
    flip_angle,
    sys;
    pulse_duration = 1e-3,
    adc_duration = 2e-3,
    venc_duration_flat = venc_durations_flat,
    venc_duration_rise = venc_durations_rise,
    balanced = false,
    spoiled = false,
    flow_compensation = true,
    slice_thickness = slice_thickness,
    rf_phase_offset = π/2,
    dummy_pulses = 0
)

push!(seqs, seq_a)
push!(seqs, seq_b)

## ---- Simulation ----
MAX_SPINS_PER_GPU = 200_000
raw_data    = []
@time for seq in seqs
    sim_params = KomaMRICore.default_sim_params()
    sim_params["Nblocks"] = 5000
    sim_params["Δt"]    = Δt
    sim_params["Δt_rf"] = Δt_rf

    sequential_parts = divide_spins_ranges(length(obj), MAX_SPINS_PER_GPU)

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
    push!(raw_data, raw)
end

## ---- Save ----
save(
    "result_$(N_matrix[1])x$(N_matrix[2])_Nt$(Nt)_Deltat$(Δt)_Deltat_RF$(Δt_rf).jld2", 
    Dict(
        # "magnitude" => magnitude, "phase" => phase, "magnitude_mean" => magnitude_mean, "phase_diff" => phase_diff,
        "raw_data" => raw_data
    )
)

## Test slice thickness
rf1 = seq_a[1]
z = range(-5., 5., 200) * 1e-3; # -5 to 5 mm
M1 = simulate_slice_profile(rf1; z)

using KomaMRIPlots.PlotlyJS
plot(
    scatter(x=z*1e3, y=abs.(M1.xy), name="Slice 1"),
    Layout(xaxis=attr(title="z [mm]"), height=300,margin=attr(t=40,l=0,r=0), title="Slice profiles for the slice-selective sequence")
)






## ------------------------------------------------------------
cd(@__DIR__)
using KomaMRI, StatsBase, JLD2, ProgressMeter

Nt = 400
fov      = [0.18, 0.13]
N_matrix = [461, 333]
res = fov ./ N_matrix
Δt = 8e-4
Δt_rf = 1e-5
venc = 200.0 .* 1e-3

## Masks
x = -fov[1]/2:res[1]:(fov[1]/2-res[1])
y = -fov[2]/2:res[2]:(fov[2]/2-res[2])

xx = reshape(x, (length(x),1)) 
yy = reshape(y, (1,length(y))) 

x = 1*xx .+ 0*yy
y = 0*xx .+ 1*yy

⚪(R, x0, y0) =  ((x .- x0).^2 .+ (y .- y0).^2 .<= R^2) # circle of radius R and center (x0, y0)

R_internal = 9.5f-3  # Original: 10mm
x0_internal = 2f-2
y0_internal = 1f-4

R_external = 10.6f-3 # Original: 11mm
x0_external = -6.5f-2
y0_external = 2f-2

mask_internal = Bool.(⚪(R_internal, x0_internal, y0_internal))
mask_external = Bool.(⚪(R_external, x0_external, y0_external))

mask = mask_internal .| mask_external

## ---- Load ----
data = load("result_$(N_matrix[1])x$(N_matrix[2])_Nt$(Nt)_Deltat$(Δt)_Deltat_RF$(Δt_rf).jld2")
raw_data = data["raw_data"]

## ---- Reconstruction ----
N_samples = 100
σ = 7

means_internal = []
maxs_internal  = []
means_external = []
maxs_external  = []

Vs = []
Vs_masked = []

@showprogress desc="Computing..." for j in 1:N_samples
    magnitude   = []
    phase       = [] 
    for r in raw_data
        raw = deepcopy(r)
        # Add gaussian noise
        for i in 1:N_matrix[2]
            noise = randn(ComplexF32, (N_matrix[1], 1)) .* σ
            raw.profiles[i].data .+= noise
        end
        acqData = AcquisitionData(raw)
        recParams = Dict{Symbol,Any}(:reco=>"direct")
        Nx, Ny = N_matrix
        recParams[:reconSize] = (Nx, Ny)
        recParams[:densityWeighting] = false
        # Kdata
        acqData.kdata[1] = reshape(acqData.kdata[1],(Nx*Ny,1))
        # Traj
        acqData.traj[1].circular = false
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

    magnitude_mean = (magnitude[1] .+ magnitude[2]) ./ 2
    phase_diff     = -(mod.(phase[1] .- phase[2] .+ π, 2π) .- π)
    
    ## Get velocity
    V = venc .* phase_diff ./ π .* 1e3 # mm/s
    push!(Vs, V)

    V_masked = V .* mask 
    push!(Vs_masked, V_masked)

    V_internal = V[mask_internal] 
    V_external = V[mask_external] 

    push!(means_internal, mean(V_internal))
    push!(maxs_internal,  minimum(V_internal))
    push!(means_external, mean(V_external))
    push!(maxs_external,  maximum(V_external))
end

##
println("σ: $σ")
println("Nt: $Nt")
println("N_samples: $N_samples")
println("Mean velocity internal: $(round(mean(means_internal), digits=1)) +- $(round(std(means_internal), digits=1))")
println("Max velocity internal: $(round(mean(maxs_internal), digits=1)) +- $(round(std(maxs_internal), digits=1))")
println("Mean velocity external: $(round(mean(means_external), digits=1)) +- $(round(std(means_external), digits=1))")
println("Max velocity external: $(round(mean(maxs_external), digits=1)) +- $(round(std(maxs_external), digits=1))")

##
p1 = plot_image(Vs[1], title="Velocity map (mm/s)", colorscale="Jet");
p2 = plot_image(Vs_masked[1], title="Velocity map (mm/s) masked", colorscale="Jet");

KomaMRIPlots.savefig(p1, "velocity_map.svg", width=415, height=280)
KomaMRIPlots.savefig(p2, "velocity_map_masked.svg", width=415, height=280)