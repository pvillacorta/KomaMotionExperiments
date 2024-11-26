using Plots
using Printf

function plot_cine(frames, fps; Δt=1/fps, filename="cine_recon.gif")

	x = 0:size(frames[1])[2]-1
	y = 1:size(frames[1])[1]

	t = 0

	anim = @animate for image in frames
		t += Δt
		Plots.plot!(Plots.heatmap(x,y,image',color=:greys; aspect_ratio=:equal, colorbar=false),
					title="Reconstruction (t="*Printf.@sprintf("%.2f", t)*"s)", 
					xlims=(minimum(x), maximum(x)), 
					ylims=(minimum(y), maximum(y)))
	end

	gif(anim, filename, fps = fps)
end