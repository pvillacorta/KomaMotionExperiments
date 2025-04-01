#Read CMRsim data
using NPZ
cmrsim_data = npzread("turb_images.npy")
magnitude = []
phase = []
for i in 1:size(cmrsim_data, 1)
    m = abs.(cmrsim_data[i,:,:])
    percen = percentile(vec(m), 99.9)
    m[m .> percen] .= percen

    push!(magnitude, rotr90(m))
    push!(phase,     rotr90(-angle.(cmrsim_data[i,:,:])))
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

mask = load("sUbend_image_mask_cmrsim.jld2")["mask"]

magnitudes = []
phases = []
phase_diffs = []
turb_estimations = []

for (i, v) in enumerate(vs[rg])
    mg = magnitude[i]
    ph = -phase[i]
    ph_diff_masked = map((x, m) -> m ? x : missing, -(mod.(phase[i] .- phase_ref .+ π, 2π) .- π), mask)
    tke_masked = (log.(abs.(magnitude_ref) ./ abs.(magnitude[i]))) .* mask .+ .!(mask)

    push!(magnitudes, plot_image(magnitude[i]; title="Magnitude $v, VENC = $(vencs[i]*1e2) cm/s", zmin=minimum(magnitude_ref), zmax=maximum(magnitude_ref)))
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

## Save fig
using Dates
KomaMRIPlots.savefig(fig, results_dirname*"sUbend_result_cmrsim_$(filename)_$(now()).svg", width=2200, height=900)