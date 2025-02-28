cd(@__DIR__)

using KomaMRI, JSON3, StatsBase, JLD2
using CUDA

## ---- Phantom ---- 
filename = "sUbend_2M_stepsize_5e-4_numSteps_5"
obj = read_phantom(filename * "_rotated.phantom")
obj.z .-= (maximum(obj.z) + minimum(obj.z))/2
obj.y .-= (maximum(obj.y) + minimum(obj.y))/2

## ---- Scanner ---- 
sys = Scanner(Gmax=80e-3, Smax=200, GR_Δt=1e-5, RF_Δt=1e-5, ADC_Δt=1e-5)

## ---- Sequence ----
include("../sequences/GRE.jl")

fov      = [0.345, 0.142]
N_matrix = [171, 71]
res = fov ./ N_matrix

TE = 5e-3
TR =  8.621e-3
flip_angle = 15.0

R = [0. 0. 1.; 0. -1. 0.; 1. 0. 0.]

#VENC
vs                  = ["vz", "vz",  "vy",   "vx" ]
vencs               = [ Inf, -500.0, -250.0, -100.0] .* 1e-2
venc_durations_flat = [ 0.0,  0.8,    0.0,    0.0  ] .* 1e-3
venc_durations_rise = [ 0.5,  0.1,    0.5,    0.5  ] .* 1e-3

seqs        = Sequence[]

rg = 1:4

for i in rg
    seq, _, _ = PC_GRE(
        vencs[i],
        Float64.([vs[i]=="vx", vs[i]=="vy", vs[i]=="vz"]),
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
end

## ---- Simulation ----
kspace      = []
magnitude   = []
phase       = []

@time for i in rg
    @info "Simulating PC sequence with VENC = $(vencs[i] * 1e2) cm/s \n Direction: $(vs[i])"

    seq = seqs[i - rg[1] + 1]

    ## ---- Simulation ----
    sim_params = KomaMRICore.default_sim_params()
    sim_params["Nblocks"] = 2000
    sim_params["Δt"]    = 8e-4
    sim_params["Δt_rf"] = 1e-5
    raw = simulate(obj[1:1_300_000], seq, sys; sim_params=sim_params)
    
    ## ---- Reconstruction ----
    recParams = Dict{Symbol,Any}(:reco=>"direct")
    Nx, Ny = N_matrix
    recParams[:reconSize] = (Nx, Ny)
    recParams[:densityWeighting] = false
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
    acqData.traj[1].times = acqData.traj[1].times
    # subsampleIndices
    acqData.subsampleIndices[1] = acqData.subsampleIndices[1][1:Nx*Ny]
    # Reconstruction
    aux = @timed reconstruction(acqData, recParams)
    recon = reshape(aux.value.data, Nx, Ny, :)

    m = abs.(recon[end-111:end,:,1])
    percen = percentile(vec(m), 99.9)
    m[m .> percen] .= percen

    push!(magnitude, m)
    push!(phase,     angle.(recon[end-111:end,:,1]))
end

## Plot
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

magnitudes = []
phases = []
phase_diffs = []

mask = load("sUbend_image_mask.jld2")["mask"]

for (i, v) in enumerate(vs[rg])
    mg = magnitude[i]
    ph = -phase[i]
    ph_diff_masked = map((x, m) -> m ? x : missing, (phase[i] .- phase_ref), mask)

    push!(magnitudes, plot_image(magnitude[i]; title="Magnitude $v, VENC = $(-vencs[i]*1e2) cm/s", zmin=minimum(magnitude_ref), zmax=maximum(magnitude_ref)))
    push!(phases, plot_image(ph; colorscale=RdBu_matplotlib, title="ϕ $v, VENC = $(-vencs[i]*1e2) cm/s", zmin=-π, zmax=π))
    push!(phase_diffs, plot_image(ph_diff_masked; colorscale=RdBu_matplotlib, title="ϕ-ϕREF $v, VENC = $(-vencs[i]*1e2) cm/s", zmin=-π, zmax=π))
end

fig = KomaMRIPlots.make_subplots(
    rows=3, cols=length(magnitudes), 
    subplot_titles= hcat(
        vcat([trace.plot.layout.title for trace in magnitudes]...),
        vcat([trace.plot.layout.title for trace in phases]...),
        vcat([trace.plot.layout.title for trace in phase_diffs]...)
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
end

display(fig)

## Save fig
using Dates
KomaMRIPlots.savefig(fig, "sUbend_validation_$(filename)_$(now()).svg", width=1500, height=900)


## Read velocity data (JSON)
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

phase_z = -(phase[2] .- phase_ref)[mask]
phase_y = -(phase[3] .- phase_ref)[mask]
phase_x = -(phase[4] .- phase_ref)[mask]

vz_sim = abs(vencs[2]) .* phase_z ./ π
vy_sim = abs(vencs[3]) .* phase_y ./ π
vx_sim = abs(vencs[4]) .* phase_x ./ π

## Save results as julia structs
save("simulated_velocity.jld2", Dict("vx_sim" => vx_sim, "vy_sim" => vy_sim, "vz_sim" => vz_sim))



## BoxPlots (StatsPlots)
using StatsPlots

# Función para filtrar outliers
function remove_outliers(data)
    Q1 = quantile(data, 0.25)
    Q3 = quantile(data, 0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    return filter(x -> lower_limit <= x <= upper_limit, data)
end

# Filtrar los outliers
data_filtered = remove_outliers.([vx_gt, vx_sim, vy_gt, vy_sim, vz_gt, vz_sim] .* 1e2)

# Crear etiquetas y concatenar datos
all_data = vcat(data_filtered...)  # Todos los valores en un solo vector
groups = vcat(fill("vx", length(data_filtered[1]) + length(data_filtered[2])),
              fill("vy", length(data_filtered[3]) + length(data_filtered[4])),
              fill("vz", length(data_filtered[5]) + length(data_filtered[6])))  # Grupo principal (ejes)

subgroups = vcat(fill("Groundtruth", length(data_filtered[1])),
                 fill("Simulated", length(data_filtered[2])),
                 fill("Groundtruth", length(data_filtered[3])),
                 fill("Simulated", length(data_filtered[4])),
                 fill("Groundtruth", length(data_filtered[5])),
                 fill("Simulated", length(data_filtered[6])))  # Subgrupo dentro de cada eje

# Crear el boxplot agrupado
groupedboxplot(groups, all_data, group=subgroups, bar_width=0.8, legend=:topleft, title="VENCs=$([vencs[4], vencs[3], vencs[2]] .* -1e2) cm/s")








## Plot box plot (PlotlyJS)
using KomaMRIPlots.PlotlyJS

x_sim = vcat(repeat(["vx"], length(vx_sim)), repeat(["vy"], length(vx_sim)), repeat(["vz"], length(vx_sim)))
x_gt  = vcat(repeat(["vx"], length(vx_gt)),  repeat(["vy"], length(vx_gt)),  repeat(["vz"], length(vx_gt)))

trace_sim = box(; y = vcat(vx_sim, vy_sim, vz_sim),
                  x = x_sim,
                  name = "Simulated")

trace_gt  = box(; y = vcat(vx_gt, vy_gt, vz_gt),
                  x = x_gt,
                  name = "Groundtruth")

data = [trace_gt, trace_sim]
layout = Layout(; boxmode="group", title="VENCs=$([vencs[4], vencs[3], vencs[2]] .* 1e2) cm/s")
plot(data, layout)

# open("./boxplots.html", "w") do io
#     PlotlyBase.to_html(io, p.plot)
# end