cd(@__DIR__)

using KomaMRI, CUDA, StatsBase, JLD2, JSON3

include("../sequences/GRE.jl")
include("../utils/divide_spins_ranges.jl")
include("rotate_sUbend.jl")

## ---- Phantom ----
obj = read_phantom("../phantoms/sUbend.phantom") # This file must be downloaded from Zenodo: https://shorturl.at/G8Dsc
obj = rotate_sUbend(obj)
obj.z .-= (maximum(obj.z) + minimum(obj.z))/2
obj.y .-= (maximum(obj.y) + minimum(obj.y))/2
# obj.motion = NoMotion()
## ---- Scanner ---- 
sys = Scanner(Gmax=80e-3, Smax=200, GR_Δt=1e-5, RF_Δt=1e-5, ADC_Δt=1e-5)

## ---- Sequence ----
# Overload KomaMRI `*` function to avoid the following: ---------
# Rotating an array of gradients (in x, y, and z) with 
# different durations causes all durations to become the same
# https://github.com/JuliaHealth/KomaMRI.jl/issues/545
## Execute this only if the GitHub issue above is not solved:
import KomaMRIBase: *
function *(α::Matrix, x::Array{Grad})
    y = deepcopy(x)
    A_values = [g.A for g in x]  
    A_result = α * A_values     
    for (i, g) in enumerate(y)
        g.A = A_result[i]     
    end
    return y
end

# ---------------------------------------------------------------

fov      = [0.345, 0.142]
N_matrix = [171, 71]
res = fov ./ N_matrix

TE = 5e-3
TR =  8.621e-3
flip_angle = 15.0

R = [0. 0. 1.; 0. -1. 0.; 1. 0. 0.]

#VENC
vs                  = ["vz", "vz",  "vz", "vy",  "vy", "vx",  "vx" ]
vencs               = [ Inf,  550.0, 50.0, 250.0, 50.0, 100.0, 50.0] .* 1e-2
venc_durations_flat = [ 0.8,  0.8,   0.0,  0.8,   0.0,  0.8,   0.0 ] .* 1e-3
venc_durations_rise = [ 0.1,  0.1,   0.5,  0.1,   0.5,  0.1,   0.5 ] .* 1e-3

##
kspace      = []
magnitude   = []
phase       = []
seqs        = Sequence[]

rg = 1:7

MAX_SPINS_PER_GPU = 200_000
sequential_parts = divide_spins_ranges(length(obj), MAX_SPINS_PER_GPU)

@time for (i, v) in enumerate(vs[rg])
    @info "----------- Simulating PC sequence with VENC = $(vencs[i] * 1e2) cm/s \n Direction: $v -----------"

    ## ---- Sequence ----
    seq, _, _ = PC_GRE(
        vencs[i],
        Float64.([v=="vx", v=="vy", v=="vz"]),
        fov,
        N_matrix,
        TE,
        TR,
        flip_angle,
        sys;
        R = R,
        pulse_duration = 1e-3,
        adc_duration = 4e-3,
        venc_duration_flat = venc_durations_flat[i],
        venc_duration_rise = venc_durations_rise[i],
        balanced = true,
        crusher_duration = 1e-3,
        crusher_area = 4 * π / (2 * π * γ) / res[1]
    )
    push!(seqs, seq)

    ## ---- Simulation ----
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
    acqData = AcquisitionData(raw)
    seq_no_rot = inv(R) * seqs[i]
    _, ktraj = get_kspace(seq_no_rot)
    # Kdata
    acqData.kdata[1] = reshape(acqData.kdata[1],(Nx*Ny,1))
    # Traj
    acqData.traj[1].circular = false
    acqData.traj[1].nodes = transpose(ktraj[:, 1:2])
    acqData.traj[1].nodes = acqData.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqData.traj[1].nodes[:]))
    acqData.traj[1].numProfiles = Ny
    acqData.traj[1].times = acqData.traj[1].times
    # Reconstruction
    aux = @timed reconstruction(acqData, recParams)
    recon = reshape(aux.value.data, Nx, Ny, :)

    m = abs.(recon[end-110:end,:,1])
    percen = percentile(vec(m), 99.9)
    m[m .> percen] .= percen

    push!(magnitude, m)
    push!(phase,     angle.(recon[end-110:end,:,1]))
end

## Plot results
magnitude_ref = copy(magnitude[1])
phase_ref = copy(phase[1])

RdBu_matplotlib = [
    [0.0, "rgb(103,0,31)"],
    [0.125, "rgb(178,24,43)"],
    [0.25, "rgb(214,96,77)"],
    [0.375, "rgb(244,165,130)"],
    [0.5, "rgb(235,235,235)"],
    [0.625, "rgb(146,197,222)"],
    [0.75, "rgb(67,147,195)"],
    [0.875, "rgb(33,102,172)"],
    [1.0, "rgb(5,48,97)"]
]

mask = load("sUbend_image_mask_koma.jld2")["mask"]

magnitudes = []
phases = []
phase_diffs = []
turb_estimations = []

for (i, v) in enumerate(vs[rg])
    mg = magnitude[i]
    ph = -phase[i]
    ph_diff_masked = map((x, m) -> m ? x : missing, -(mod.(phase[i] .- phase_ref .+ π, 2π) .- π), mask)
    tke_masked = (log.(abs.(magnitude_ref) ./ abs.(magnitude[i]))) .* mask .+ .!(mask)

    push!(magnitudes, plot_image(magnitude[i]; title="Magnitude $v, VENC = $(vencs[i]*1e2) cm/s", zmin=minimum(magnitude_ref), zmax=percentile(vec(magnitude_ref), 99)))
    push!(phases, plot_image(ph; colorscale=RdBu_matplotlib, title="ϕ $v, VENC = $(vencs[i]*1e2) cm/s", zmin=-π, zmax=π))
    push!(phase_diffs, plot_image(ph_diff_masked; colorscale=RdBu_matplotlib, title="ϕ-ϕREF $v, VENC = $(vencs[i]*1e2) cm/s", zmin=-π, zmax=π))
    push!(turb_estimations, plot_image(tke_masked; colorscale="Hot", title="TKE $v, VENC = $(vencs[i]*1e2) cm/s", zmin=0, zmax=1))
end

fig = KomaMRIPlots.make_subplots(
    rows=4, cols=length(magnitudes), 
    subplot_titles= hcat(
        vcat([trace.plot.layout.title for trace in magnitudes]...),
        vcat([trace.plot.layout.title for trace in phases]...),
        vcat([trace.plot.layout.title for trace in phase_diffs]...),
        vcat([trace.plot.layout.title for trace in turb_estimations]...)
        ),
    shared_yaxes=true, 
    shared_xaxes=true,
    vertical_spacing=0.05,
    horizontal_spacing=0.0
)

for (i, v) in enumerate(vs[rg])
    phase_diffs[i].plot.data[1].xaxis_showgrid = false
    KomaMRIPlots.add_trace!(fig, magnitudes[i].plot.data[1], row=1, col=i)
    KomaMRIPlots.add_trace!(fig, phases[i].plot.data[1], row=2, col=i)
    KomaMRIPlots.add_trace!(fig, phase_diffs[i].plot.data[1], row=3, col=i)
    KomaMRIPlots.add_trace!(fig, turb_estimations[i].plot.data[1], row=4, col=i)
end

display(fig)

## Results directory
results_dirname = "results/"
if !isdir(results_dirname)
    mkdir(results_dirname)
end

## Save fig
using Dates
KomaMRIPlots.savefig(fig, results_dirname*"sUbend_result_koma_$(filename)_$(now()).svg", width=2200, height=900)

## Save results as julia structs
using JLD2, Dates
fname = results_dirname*"sUbend_result.jld2"
save(fname, Dict("magnitude" => magnitude, "phase" => phase))



## Boxplots ----------------------------------------
using StatsBase, JSON3, JLD2
## Read Koma data
koma_data = load(fname)

magnitude_koma = koma_data["magnitude"]
phase_koma = koma_data["phase"]

mask = load("sUbend_image_mask_koma.jld2")["mask"]

phase_koma_z = (mod.(phase_koma[2] .- phase_koma[1] .+ π, 2π) .- π)[mask]
phase_koma_y = (mod.(phase_koma[4] .- phase_koma[1] .+ π, 2π) .- π)[mask]
phase_koma_x = (mod.(phase_koma[6] .- phase_koma[1] .+ π, 2π) .- π)[mask]

vz_koma = abs(vencs[2]) .* phase_koma_z ./ π
vy_koma = abs(vencs[4]) .* phase_koma_y ./ π
vx_koma = abs(vencs[6]) .* phase_koma_x ./ π

## Read CMRsim data
using NPZ
cmrsim_data = npzread("turb_images.npy")
magnitude_cmrsim = []
phase_cmrsim = []
for i in 1:size(cmrsim_data, 1)
    m = abs.(cmrsim_data[i,:,:])
    percen = percentile(vec(m), 99.9)
    m[m .> percen] .= percen

    push!(magnitude_cmrsim, rotr90(m))
    push!(phase_cmrsim,     rotr90(-angle.(cmrsim_data[i,:,:])))
end

mask = (reduce(.+, magnitude_cmrsim) ./ 7) .> 850

phase_cmrsim_z = (mod.(phase_cmrsim[2] .- phase_cmrsim[1] .+ π, 2π) .- π)[mask]
phase_cmrsim_y = (mod.(phase_cmrsim[4] .- phase_cmrsim[1] .+ π, 2π) .- π)[mask]
phase_cmrsim_x = (mod.(phase_cmrsim[6] .- phase_cmrsim[1] .+ π, 2π) .- π)[mask]

vz_cmrsim = abs(vencs[2]) .* phase_cmrsim_z ./ π
vy_cmrsim = abs(vencs[4]) .* phase_cmrsim_y ./ π
vx_cmrsim = abs(vencs[6]) .* phase_cmrsim_x ./ π

# Read Groundtruth velocity data (JSON)
fid = open("sUbend_velocity.json", "r") # File extracted from sUbend_rotated.vtu 
U = JSON3.read(fid)

vx_gt = Float64[]
vy_gt = Float64[]
vz_gt = Float64[]

ss = 50 # Get 1 of ss elements of velocity field
for (i, v) in enumerate(U)
    if i%50 == 0
        append!(vx_gt, v[3])
        append!(vy_gt, v[2])
        append!(vz_gt, v[1])
    end
end


## BoxPlots (StatsPlots)
using StatsPlots, CategoricalArrays

# Outlier filtering function
function remove_outliers(data)
    Q1 = quantile(data, 0.25)
    Q3 = quantile(data, 0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    return filter(x -> lower_limit <= x <= upper_limit, data)
end

# Filter outliers
data_filtered = remove_outliers.([vx_gt, vx_cmrsim, vx_koma, 
                                  vy_gt, vy_cmrsim, vy_koma,
                                  vz_gt, vz_cmrsim, vz_koma,] .* 1e2)

# Create labels and concatenate data
all_data = vcat(data_filtered...)
groups = vcat(fill("vx", length(data_filtered[1]) + length(data_filtered[2]) + length(data_filtered[3])),
              fill("vy", length(data_filtered[4]) + length(data_filtered[5]) + length(data_filtered[6])),
              fill("vz", length(data_filtered[7]) + length(data_filtered[8]) + length(data_filtered[9])))

subgroups = vcat(fill("Groundtruth", length(data_filtered[1])),
                 fill("CMRsim",      length(data_filtered[2])),
                 fill("KomaMRI",     length(data_filtered[3])),
                 fill("Groundtruth", length(data_filtered[4])),
                 fill("CMRsim",      length(data_filtered[5])),
                 fill("KomaMRI",     length(data_filtered[6])),
                 fill("Groundtruth", length(data_filtered[7])),
                 fill("CMRsim",      length(data_filtered[8])),
                 fill("KomaMRI",     length(data_filtered[9])))
subgroups = categorical(subgroups, levels=["Groundtruth", "CMRsim", "KomaMRI"])

## Create the grouped boxplot
bplot = groupedboxplot(groups, all_data, group=subgroups, bar_width=0.8, legend=:topleft, notch=true, ylabel="velocity (cm/s)")

x_rg = findall(x->x=="vx",groups)
y_rg = findall(x->x=="vy",groups)
z_rg = findall(x->x=="vz",groups)

bplot_x = groupedboxplot(groups[x_rg], all_data[x_rg], group=subgroups[x_rg], bar_width=0.6, legend=:topleft, notch=true, ylabel="velocity (cm/s)")
bplot_y = groupedboxplot(groups[y_rg], all_data[y_rg], group=subgroups[y_rg], bar_width=0.6, legend=:topleft, notch=true, ylabel="velocity (cm/s)")
bplot_z = groupedboxplot(groups[z_rg], all_data[z_rg], group=subgroups[z_rg], bar_width=0.6, legend=:topleft, notch=true, ylabel="velocity (cm/s)")

StatsPlots.savefig(bplot, results_dirname*"boxplots.svg")
StatsPlots.savefig(bplot_x, results_dirname*"boxplot_x.svg")
StatsPlots.savefig(bplot_y, results_dirname*"boxplot_y.svg")
StatsPlots.savefig(bplot_z, results_dirname*"boxplot_z.svg")