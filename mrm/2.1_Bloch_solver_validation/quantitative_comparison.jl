cd(@__DIR__)

import Pkg
Pkg.activate(".")
Pkg.instantiate()

using KomaMRICore, Plots

include("yuan_sequence.jl")

## Results directory
results_dirname = "results/"
if !isdir(results_dirname)
    mkdir(results_dirname)
end

NRMSE(x, x_true) = sqrt.( sum(abs.(x .- x_true).^2) ./ sum(abs.(x_true).^2) ) * 100.



# Sequence 
seq = yuan_sequence()

# Phantom
Nspins = 50
Lt = 30e-3 # 30mm

vs = [0, 80, 120, 160, 200] .* 1e-2			# velocity profiling
Δts = [1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7]  # Δt profiling
nrmse_vs_mx = Vector{Float64}[]
nrmse_vs_my = Vector{Float64}[]
nrmse_vs_mz = Vector{Float64}[]
for (i, vz) in enumerate(vs) # m/s <--------- VELOCITY
	dz = vz * dur(seq)
	obj = Phantom(x=zeros(Nspins), z=collect(range(-Lt/2-dz, Lt/2-dz, Nspins)))
	obj.motion = translate(0.0, 0.0, dz, TimeRange(0.0, dur(seq)), AllSpins())

	## Solving using DiffEquations ---------------------------------------------------
	using OrdinaryDiffEqTsit5

	M0 = obj.ρ[1]
	T1 = obj.T1[1]
	T2 = obj.T2[1]

	function bloch!(dm, m, p, t)
		mx, my, mz = m
		# Dynamic part
		x, y, z = get_spin_coords(obj.motion, obj[p].x, obj[p].y, obj[p].z, t)
		B1e, _ = KomaMRIBase.get_rfs(seq, [t])
		gx, gy, gz = KomaMRIBase.get_grads(seq, [t])
		# Effective field
		bx = real(B1e[1])
		by = imag(B1e[1])
		bz = x[1] * gx[1] + 
			y[1] * gy[1] + 
			z[1] * gz[1]
		# Bloch equations
		γ = 2π * KomaMRIBase.γ
		dm[1] = -mx / T2 + γ * bz * my - γ * by * mz
		dm[2] = -γ * bz * mx - my / T2 + γ * bx * mz
		dm[3] =  γ * by * mx - γ * bx * my - mz / T1 + M0 / T1
		return nothing
	end

	m0 = [0.0, 0.0, 1.0]
	tspan = (0.0, dur(seq))
	mx_diffeq = Float64[]; my_diffeq = Float64[]; mz_diffeq = Float64[];
	@time for j in 1:length(obj)
		prob = ODEProblem(bloch!, m0, tspan, j)
		tadc= range(0, dur(seq), 1000)
		sol = solve(prob, Tsit5(), saveat = tadc, abstol = 1e-9, reltol = 1e-9)
		sol_diffeq = hcat(sol.u...)'
		append!(mx_diffeq, sol_diffeq[:, 1][end])
		append!(my_diffeq, sol_diffeq[:, 2][end])
		append!(mz_diffeq, sol_diffeq[:, 3][end])
	end

	## Solving using KomaMRICore ---------------------------------------------------
	# Scanner
	sys = Scanner()
	# Simulation
	sim_params = KomaMRICore.default_sim_params()
	sim_params["return_type"] = "state"
	nrmse_mx = Float64[]
	nrmse_my = Float64[]
	nrmse_mz = Float64[]

	for (j, Δt) in enumerate(Δts)
		sim_params["Δt_rf"] = Δt
		sim_params["Δt"] 	= Δt
		M = simulate(obj, seq, sys; sim_params)

		push!(nrmse_mx, NRMSE(real.(M.xy), mx_diffeq))
		push!(nrmse_my, NRMSE(imag.(M.xy), my_diffeq))
		push!(nrmse_mz, NRMSE(M.z,         mz_diffeq))
	end
	
	push!(nrmse_vs_mx, nrmse_mx)
	push!(nrmse_vs_my, nrmse_my)
	push!(nrmse_vs_mz, nrmse_mz)
end


## Plot
fig_size = (350, 300)

xticks_values = [0.1, 1, 10] 
xticks_labels = ["0.1", "1", "10"]

px = Plots.plot(;
	xscale = :log10,
	xflip = true,
	xlabel = "Time Step Size [μs]", 
	ylabel = "NRMSE [%]",
	title = "Mx",
	grid = true,     
	minorgrid = true,
	legend=:topright,
	size=fig_size,
	xticks = (xticks_values, xticks_labels),
	ylims = (0, 14)
);

py = Plots.plot(;
	xscale = :log10,
	xflip = true,
	xlabel = "Time Step Size [μs]", 
	ylabel = "NRMSE [%]",
	title = "My",
	grid = true,
	minorgrid = true,
	legend=:topright,
	size=fig_size,
	xticks = (xticks_values, xticks_labels),
	ylims = (0, 14)
)

pz = Plots.plot(;
	xscale = :log10,
	xflip = true,
	xlabel = "Time Step Size [μs]", 
	ylabel = "NRMSE [%]",
	title = "Mz",
	grid = true,
	minorgrid = true,
	legend=:topright,
	size=fig_size,
	xticks = (xticks_values, xticks_labels),
	ylims = (0, 14)
)

for (i, vz) in enumerate(vs) # m/s <--------- VELOCITY
	plot!(px, reverse(Δts) .* 1e6, reverse(nrmse_vs_mx[i]), label="$(vz * 1e2) cm/s", marker=:circle, linewidth=2)
	plot!(py, reverse(Δts) .* 1e6, reverse(nrmse_vs_my[i]), label="$(vz * 1e2) cm/s", marker=:circle, linewidth=2)
	plot!(pz, reverse(Δts) .* 1e6, reverse(nrmse_vs_mz[i]), label="$(vz * 1e2) cm/s", marker=:circle, linewidth=2)	
end

Plots.savefig(px, results_dirname*"nrmse_x.svg")
Plots.savefig(py, results_dirname*"nrmse_y.svg")
Plots.savefig(pz, results_dirname*"nrmse_z.svg")