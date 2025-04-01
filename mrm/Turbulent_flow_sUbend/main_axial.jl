cd(@__DIR__)

using KomaMRI, CUDA, StatsBase, JLD2, JSON3

include("../sequences/GRE.jl")
include("../utils/divide_spins_ranges.jl")

## ---- Phantom ----
obj = read_phantom("../phantoms/sUbend_no_turb_5M_particles.phantom") # This file must be downloaded from Zenodo: https://shorturl.at/G8Dsc
# obj = obj[1:800_000]
dx1 = (maximum(obj.x) + minimum(obj.x)) / 2
dy1 = (maximum(obj.y) + minimum(obj.y)) / 2
dx2 = 10e-2 # 10 cm to center the slice position in x=0
obj.x .-= dx1 + dx2
obj.y .-= dy1
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

fov      = [0.07, 0.14]
N_matrix = [40, 80]
res = fov ./ N_matrix

TE = 5e-3
TR =  8.621e-3
flip_angle = 15.0
z0 = 0.0
slice_thickness = 5e-3

R = [0. 0. 1.; 0. 1. 0.; -1. 0. 0.] # Rotate sequence 90º around y following right-hand rule

#VENC
vs                  = ["vz", "vx",   "vz",  "vy"   ]
vencs               = [ Inf,  550.0,  100.0, 100.0 ] .* 1e-2
venc_durations_flat = [ 0.8,  0.8,    0.8,   0.8   ] .* 1e-3
venc_durations_rise = [ 0.1,  0.1,    0.1,   0.1   ] .* 1e-3

##
kspace      = []
magnitude   = []
phase       = []
seqs        = Sequence[]

rg = 1:2

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
        z0 = z0,
        R = R,
        pulse_duration = 1e-3,
        adc_duration = 4e-3,
        venc_duration_flat = venc_durations_flat[i],
        venc_duration_rise = venc_durations_rise[i],
        balanced = true,
        crusher_duration = 1e-3,
        crusher_area = 4 * π / (2 * π * γ) / res[1],
        slice_thickness = slice_thickness
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
    # subsampleIndices
    acqData.subsampleIndices[1] = acqData.subsampleIndices[1][1:Nx*Ny]
    # Reconstruction
    aux = @timed reconstruction(acqData, recParams)
    recon = reshape(aux.value.data, Nx, Ny, :)

    push!(magnitude, abs.(recon[:,:,1]))
    push!(phase,     angle.(recon[:,:,1]))
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

# mask = magnitude_ref .> 5.5
# save("sUbend_image_mask_koma_axial_$(N_matrix[1])x$(N_matrix[2]).jld2", Dict("mask" => mask))
mask =  load("sUbend_image_mask_koma_axial_$(N_matrix[1])x$(N_matrix[2]).jld2")["mask"]

magnitudes = []
phases = []
phase_diffs = []
turb_estimations = []

for (i, v) in enumerate(vs[rg])
    mg = magnitude[i]
    ph = -phase[i]
    ph_diff_masked = map((x, m) -> m ? x : missing, -(mod.(phase[i] .- phase_ref .+ π, 2π) .- π), mask)
    tke_masked = (log.(abs.(magnitude_ref) ./ abs.(magnitude[i]))) .* mask .+ .!(mask)

    push!(magnitudes, plot_image(magnitude[i]'; title="Magnitude $v, VENC = $(vencs[i]*1e2) cm/s", zmin=minimum(magnitude_ref), zmax=percentile(vec(magnitude_ref), 99)))
    push!(phases, plot_image(ph'; colorscale=RdBu_matplotlib, title="ϕ $v, VENC = $(vencs[i]*1e2) cm/s", zmin=-π, zmax=π))
    push!(phase_diffs, plot_image(ph_diff_masked'; colorscale=RdBu_matplotlib, title="ϕ-ϕREF $v, VENC = $(vencs[i]*1e2) cm/s", zmin=-π, zmax=π))
    push!(turb_estimations, plot_image(tke_masked'; colorscale="Hot", title="TKE $v, VENC = $(vencs[i]*1e2) cm/s", zmin=0, zmax=1))
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
KomaMRIPlots.savefig(fig, results_dirname*"sUbend_result_koma_axial__$(length(obj))_spins_no_turb_$(now()).svg", width=850, height=900)

## Save results as julia structs
using JLD2, Dates
fname_no_turb = results_dirname*"sUbend_result_axial_$(length(obj))_spins_no_turb_$(N_matrix[1])x$(N_matrix[2]).jld2"
# fname_turb = results_dirname*"sUbend_result_axial_$(length(obj))_spins_turb_$(N_matrix[1])x$(N_matrix[2]).jld2"

save(fname_no_turb, Dict("magnitude" => magnitude, "phase" => phase))



## Retreive velocity data ----------------------------------------
using StatsBase, JSON3, JLD2, NPZ, PlotlyJS

## Read Koma data
koma_data_no_turb = load(fname_no_turb)
# koma_data_turb = load(fname_turb)

magnitude_koma_no_turb = koma_data_no_turb["magnitude"]
phase_koma_no_turb = koma_data_no_turb["phase"]

# magnitude_koma_turb = koma_data_turb["magnitude"]
# phase_koma_turb = koma_data_turb["phase"]

mask = load("sUbend_image_mask_koma_axial_$(N_matrix[1])x$(N_matrix[2]).jld2")["mask"]

idx = findall(mask)

phase_koma_x_no_turb = (mod.(phase_koma_no_turb[2] .- phase_koma_no_turb[1] .+ π, 2π) .- π)[idx]
vx_koma_no_turb = abs(vencs[2]) .* phase_koma_x_no_turb ./ π

# phase_koma_x_turb = (mod.(phase_koma_turb[2] .- phase_koma_turb[1] .+ π, 2π) .- π)[idx]
# vx_koma_turb = abs(vencs[2]) .* phase_koma_x_turb ./ π

# phase_koma_x = map((x, m) -> m ? x : 0.0, (mod.(phase_koma[2] .- phase_koma[1] .+ π, 2π) .- π), mask)

# Ver perfil de velocidad media en x
# plot(reduce(+, vx_koma, dims=1)' ./ size(vx_koma, 1))

V_GT = npzread(results_dirname*"velocity_data_axial_$(N_matrix[1])x$(N_matrix[2]).npz")

vx_gt = V_GT["vx"]
vy_gt = V_GT["vy"]
vz_gt = V_GT["vz"]

## Plot velocity data
using PlotlyJS

p_gt = PlotlyJS.scatter(y=vx_gt, name="Ground truth (No turb)", opacity=0.5)
p_koma_no_turb = PlotlyJS.scatter(y=vx_koma_no_turb, name="Koma No turb", opacity=0.8)
p_koma_turb = PlotlyJS.scatter(y=vx_koma_turb, name="Koma Turb", opacity=0.8)
layout = Layout(yaxis_title="Velocity [m/s]", xaxis_title="Index", title="Velocity comparison")
p1 = PlotlyJS.plot([p_gt, p_koma_no_turb], layout)

##
z = range(-5e-2,5e-2,length=100)
M = simulate_slice_profile(inv(R)*seqs[1]; z=z)
# M = simulate_slice_profile(seq; z=z)
using PlotlyJS # hide
pa = scatter(x=z*1e2, y=abs.(M.xy), name="Slice 1") # hide
pd = plot([pa], Layout(xaxis=attr(title="z [cm]"), height=300,margin=attr(t=40,l=0,r=0), title="Slice profiles for the slice-selective sequence")) # hide

## Get and store pixel positions (mask)
dy = dy1
dx = dx1 + dx2
dz = 0.0
z, y, x = Float64[], Float64[], Float64[]    
for index in idx
    push!(z, -fov[1]/2 + (2 * index[1] - 1) * res[1] / 2 + dz)
    push!(y, -fov[2]/2 + (2 * index[2] - 1) * res[2] / 2 + dy)
    push!(x, + dx)
end
npzwrite(results_dirname*"pixel_positions_axial_mask_$(N_matrix[1])x$(N_matrix[2]).npz", Dict("x" => x, "y" => y, "z" => z, "pixel_size" => res[1]))
PlotlyJS.plot(PlotlyJS.scatter(x=y,y=z,mode="markers"))

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
# data_filtered = remove_outliers.([vx_gt, vx_cmrsim, vx_koma, 
#                                   vy_gt, vy_cmrsim, vy_koma,
#                                   vz_gt, vz_cmrsim, vz_koma,] .* 1e2)

data_filtered = [vx_gt, vx_koma] .* 1e2

# Create labels and concatenate data
all_data = vcat(data_filtered...)
groups = vcat(fill("vx", length(data_filtered[1]) + length(data_filtered[2])))



subgroups = vcat(fill("Groundtruth", length(data_filtered[1])),
                 fill("KomaMRI",     length(data_filtered[2])))

subgroups = categorical(subgroups, levels=["Groundtruth", "KomaMRI"])

## Create the grouped boxplot
bplot = groupedboxplot(groups, all_data, group=subgroups, bar_width=0.8, legend=:topleft, notch=true, ylabel="velocity (cm/s)")

x_rg = findall(x->x=="vx",groups)

bplot_x = groupedboxplot(groups[x_rg], all_data[x_rg], group=subgroups[x_rg], bar_width=0.6, legend=:topleft, notch=true, ylabel="velocity (cm/s)")

## PlotlyJS
trace_gt = PlotlyJS.box(;y=vx_gt .* 1e2, name="Ground truth (No turb)", boxmean=true)
trace_koma_no_turb = PlotlyJS.box(;y=vx_koma_no_turb .* 1e2, name="KomaMRI No Turb", boxmean=true)
trace_koma_turb = PlotlyJS.box(;y=vx_koma_turb .* 1e2, name="KomaMRI Turb", boxmean=true)
data = [trace_gt, trace_koma_no_turb, trace_koma_turb]
layout = PlotlyJS.Layout(;title="Velocity comparison Vx (axial)", yaxis_title="Velocity [cm/s]")
p2 = PlotlyJS.plot(data, layout)