cd(@__DIR__)

import Pkg
Pkg.activate(".")
Pkg.instantiate()

using KomaMRI, CUDA, StatsBase, JLD2, NPZ, Dates, StatsPlots, CategoricalArrays, HypothesisTests

include("../utils/divide_spins_ranges.jl")
include("../utils/file_loader.jl")
include("rotate_sUbend.jl")
include("PC_GRE.jl")

## Results directory
results_dirname = "results/"
if !isdir(results_dirname)
    mkdir(results_dirname)
end

## ---- Phantom ----
obj = load_phantom("sUbend_1p5M_spins_turb.phantom") # This function downloads the phantom from Zenodo if it does not exist in the ../phantoms directory
obj = rotate_sUbend(obj)
obj.z .-= (maximum(obj.z) + minimum(obj.z))/2 # Center phantom
obj.y .-= (maximum(obj.y) + minimum(obj.y))/2 # Center phantom
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

fov      = [0.222, 0.142]
N_matrix = [111, 71]
res = fov ./ N_matrix

TE = 5e-3
TR =  8.621e-3

adc_duration = 4e-3

# RF
flip_angle = 15.0
pulse_duration = 1e-3
slice_thickness = 5e-3
TBP = 4

R = [0. 0. 1.; 0. -1. 0.; 1. 0. 0.]

#VENC
vs                  = ["vz", "vz",  "vz", "vy",  "vy", "vx",  "vx" ]
vencs               = [ Inf,  550.0, 50.0, 250.0, 50.0, 100.0, 50.0] .* 1e-2
venc_durations_flat = [ 0.8,  0.8,   0.0,  0.8,   0.0,  0.8,   0.0 ] .* 1e-3
venc_durations_rise = [ 0.1,  0.1,   0.5,  0.1,   0.5,  0.1,   0.5 ] .* 1e-3

##
kspace           = []
magnitude_koma   = []
phase_koma       = []
seqs             = Sequence[]

rg = 1:7

MAX_SPINS_PER_GPU = 200_000
sequential_parts = divide_spins_ranges(length(obj), MAX_SPINS_PER_GPU)

total_sims   = length(vs) * length(sequential_parts)
global counter = 0

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
        pulse_duration = pulse_duration,
        adc_duration = adc_duration,
        venc_duration_flat = venc_durations_flat[i],
        venc_duration_rise = venc_durations_rise[i],
        balanced = true,
        crusher_duration = 1e-3,
        crusher_area = 4 * π / (2 * π * γ) / res[1],
        slice_thickness = slice_thickness,
        TBP = TBP
    )
    push!(seqs, seq)

    # ---- Simulation ----
    sim_params = KomaMRICore.default_sim_params()
    sim_params["Nblocks"] = 5000
    sim_params["Δt"]    = 8e-4
    sim_params["Δt_rf"] = 1e-6

    if length(sequential_parts) > 1
        @info "Dividing phantom ($(length(obj)) spins) into $(length(sequential_parts)) parts that will be simulated sequentially"
    end
    
    raws = []
    for (n, sequential_part) in enumerate(sequential_parts)
        global counter += 1
        @info "Simulation sagittal $(counter)/$(total_sims):" Venc_direction = v Venc = "$(vencs[i] * 1e2) cm/s" Phantom_part = "$n/$(length(sequential_parts))"
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

    m = abs.(recon[:,:,1])
    percen = percentile(vec(m), 99.9)
    m[m .> percen] .= percen

    push!(magnitude_koma, m)
    push!(phase_koma, angle.(recon[:,:,1]))
end

## Read CMRsim data
cmrsim_data = npzread("cmrsim_images.npy")
magnitude_cmrsim = []
phase_cmrsim = []
for i in 1:size(cmrsim_data, 1)
    m = abs.(cmrsim_data[i,:,:])
    percen = percentile(vec(m), 99.9)
    m[m .> percen] .= percen

    push!(magnitude_cmrsim, rotr90(m))
    push!(phase_cmrsim,     rotr90(-angle.(cmrsim_data[i,:,:])))
end

## Plot results
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

mg_ref_koma   = copy(magnitude_koma[1])
ph_ref_koma   = copy(phase_koma[1])
mg_ref_cmrsim = copy(magnitude_cmrsim[1])
ph_ref_cmrsim = copy(phase_cmrsim[1])

mask_koma   = load("mask_sagittal_koma.jld2")["mask"]
mask_cmrsim = load("mask_sagittal_cmrsim.jld2")["mask"]

magnitudes_koma         = []
phases_koma             = []
phase_diffs_koma        = []
turb_estimations_koma   = []
magnitudes_cmrsim       = []
phases_cmrsim           = []
phase_diffs_cmrsim      = []
turb_estimations_cmrsim = []


for (i, v) in enumerate(vs[rg])
    ph_diff_masked_cmrsim = map((x, m) -> m ? x : missing, -(mod.(phase_cmrsim[i] .- ph_ref_cmrsim .+ π, 2π) .- π), mask_cmrsim)
    ph_diff_masked_koma   = map((x, m) -> m ? x : missing, -(mod.(phase_koma[i]   .- ph_ref_koma   .+ π, 2π) .- π), mask_koma)
    tke_masked_cmrsim = (log.(abs.(mg_ref_cmrsim) ./ abs.(magnitude_cmrsim[i]))) .* mask_cmrsim .+ .!(mask_cmrsim)
    tke_masked_koma   = (log.(abs.(mg_ref_koma)   ./ abs.(magnitude_koma[i])))   .* mask_koma   .+ .!(mask_koma)

    push!(phases_cmrsim, plot_image(-phase_cmrsim[i]; colorscale=RdBu_matplotlib, zmin=-π, zmax=π))
    push!(phases_koma,   plot_image(-phase_koma[i];  colorscale=RdBu_matplotlib, zmin=-π, zmax=π))
    
    push!(magnitudes_cmrsim, plot_image(magnitude_cmrsim[i];zmin=minimum(mg_ref_cmrsim), zmax=percentile(vec(mg_ref_cmrsim), 99.5)))
    push!(magnitudes_koma,   plot_image(magnitude_koma[i];zmin=minimum(mg_ref_koma), zmax=percentile(vec(mg_ref_koma),   99.5)))
    
    push!(phase_diffs_cmrsim, plot_image(ph_diff_masked_cmrsim; colorscale=RdBu_matplotlib, zmin=-π, zmax=π))
    push!(phase_diffs_koma,   plot_image(ph_diff_masked_koma;   colorscale=RdBu_matplotlib, zmin=-π, zmax=π))
    
    push!(turb_estimations_cmrsim, plot_image(tke_masked_cmrsim; colorscale="Hot", zmin=0, zmax=1))
    push!(turb_estimations_koma,   plot_image(tke_masked_koma;   colorscale="Hot", zmin=0, zmax=1))
end

fig = KomaMRIPlots.make_subplots(
    rows=8, cols=length(rg), 
    subplot_titles= hcat(
        vcat([trace.plot.layout.title for trace in phases_cmrsim]...),
        vcat(["" for i in rg]...),
        vcat([trace.plot.layout.title for trace in phase_diffs_cmrsim]...),
        vcat(["" for i in rg]...),
        vcat([trace.plot.layout.title for trace in magnitudes_cmrsim]...),
        vcat(["" for i in rg]...),
        vcat([trace.plot.layout.title for trace in turb_estimations_cmrsim]...),
        vcat(["" for i in rg]...)
        ),
    shared_yaxes=true, 
    shared_xaxes=true,
    vertical_spacing=0.01,
    horizontal_spacing=0.0,
    row_titles = ["ϕ CMRsim", "ϕ KomaMRI", "ϕ-ϕREF CMRsim", "ϕ-ϕREF KomaMRI", "Magn CMRsim", "Magn KomaMRI", "TKE CMRsim", "TKE KomaMRI"],
    column_titles = ["VENC = $(vencs[i]*1e2) cm/s" for i in rg]
)

for (i, v) in enumerate(vs[rg])
    phase_diffs_koma[i].plot.data[1].xaxis_showgrid = false

    KomaMRIPlots.add_trace!(fig, phases_cmrsim[i].plot.data[1], row=1, col=i)
    KomaMRIPlots.add_trace!(fig, phases_koma[i].plot.data[1], row=2, col=i)
    KomaMRIPlots.add_trace!(fig, phase_diffs_cmrsim[i].plot.data[1], row=3, col=i)
    KomaMRIPlots.add_trace!(fig, phase_diffs_koma[i].plot.data[1], row=4, col=i)
    KomaMRIPlots.add_trace!(fig, magnitudes_cmrsim[i].plot.data[1], row=5, col=i)
    KomaMRIPlots.add_trace!(fig, magnitudes_koma[i].plot.data[1], row=6, col=i)
    KomaMRIPlots.add_trace!(fig, turb_estimations_cmrsim[i].plot.data[1], row=7, col=i)
    KomaMRIPlots.add_trace!(fig, turb_estimations_koma[i].plot.data[1], row=8, col=i)
end
KomaMRIPlots.relayout!(fig, width=1800, height=1300)

display(fig)
KomaMRIPlots.savefig(fig, results_dirname*"result_sagittal_$(now()).pdf", width=1800, height=1300)

## Save results as julia structs
fname = results_dirname*"result_koma_sagittal_$(now()).jld2"
save(fname, Dict("magnitude" => magnitude_koma, "phase" => phase_koma))

## Velocity comparison between CRMsim and KomaMRI ----------------------------------------
# Calculate KomaMRI velocities
phase_koma_z = (mod.(phase_koma[2] .- phase_koma[1] .+ π, 2π) .- π)[mask_koma]
phase_koma_y = (mod.(phase_koma[4] .- phase_koma[1] .+ π, 2π) .- π)[mask_koma]
phase_koma_x = (mod.(phase_koma[6] .- phase_koma[1] .+ π, 2π) .- π)[mask_koma]

vz_koma = abs(vencs[2]) .* phase_koma_z ./ π
vy_koma = abs(vencs[4]) .* phase_koma_y ./ π
vx_koma = abs(vencs[6]) .* phase_koma_x ./ π

# Calculate CMRsim velocities
phase_cmrsim_z = (mod.(phase_cmrsim[2] .- phase_cmrsim[1] .+ π, 2π) .- π)[mask_cmrsim]
phase_cmrsim_y = (mod.(phase_cmrsim[4] .- phase_cmrsim[1] .+ π, 2π) .- π)[mask_cmrsim]
phase_cmrsim_x = (mod.(phase_cmrsim[6] .- phase_cmrsim[1] .+ π, 2π) .- π)[mask_cmrsim]

vz_cmrsim = abs(vencs[2]) .* phase_cmrsim_z ./ π
vy_cmrsim = abs(vencs[4]) .* phase_cmrsim_y ./ π
vx_cmrsim = abs(vencs[6]) .* phase_cmrsim_x ./ π

## Mann-Whitney U test
MannWhitneyUTest(vx_cmrsim, vx_koma) # See REPL output
MannWhitneyUTest(vy_cmrsim, vy_koma)
MannWhitneyUTest(vz_cmrsim, vz_koma)

## BoxPlots (StatsPlots)
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
data_filtered = remove_outliers.([vx_cmrsim, vx_koma, 
                                  vy_cmrsim, vy_koma,
                                  vz_cmrsim, vz_koma,] .* 1e2)

# Create labels and concatenate data
all_data = vcat(data_filtered...)
groups = vcat(fill("vx", length(data_filtered[1]) + length(data_filtered[2])),
              fill("vy", length(data_filtered[3]) + length(data_filtered[4])),
              fill("vz", length(data_filtered[5]) + length(data_filtered[6])))

subgroups = vcat(fill("CMRsim",      length(data_filtered[1])),
                 fill("KomaMRI",     length(data_filtered[2])),
                 fill("CMRsim",      length(data_filtered[3])),
                 fill("KomaMRI",     length(data_filtered[4])),
                 fill("CMRsim",      length(data_filtered[5])),
                 fill("KomaMRI",     length(data_filtered[6])))
subgroups = categorical(subgroups, levels=["CMRsim", "KomaMRI"])

# Create the grouped boxplot
bplot = groupedboxplot(groups, all_data, group=subgroups, ylims=(-150, 100), bar_width=0.8, legend=:topleft, notch=true, ylabel="velocity (cm/s)", size=(450, 300))
StatsPlots.savefig(bplot,   results_dirname*"boxplots_sagittal.svg")

# Boxplots for each velocity component
x_rg = findall(x->x=="vx",groups)
y_rg = findall(x->x=="vy",groups)
z_rg = findall(x->x=="vz",groups)

bplot_x = groupedboxplot(groups[x_rg], all_data[x_rg], group=subgroups[x_rg], bar_width=0.6, legend=:topleft, notch=true, ylabel="velocity (cm/s)", size=(300, 200))
bplot_y = groupedboxplot(groups[y_rg], all_data[y_rg], group=subgroups[y_rg], bar_width=0.6, legend=:topleft, notch=true, ylabel="velocity (cm/s)", size=(300, 200))
bplot_z = groupedboxplot(groups[z_rg], all_data[z_rg], group=subgroups[z_rg], bar_width=0.6, legend=:topleft, notch=true, ylabel="velocity (cm/s)", size=(300, 200))

StatsPlots.savefig(bplot_x, results_dirname*"boxplot_x_sagittal.svg")
StatsPlots.savefig(bplot_y, results_dirname*"boxplot_y_sagittal.svg")
StatsPlots.savefig(bplot_z, results_dirname*"boxplot_z_sagittal.svg")


