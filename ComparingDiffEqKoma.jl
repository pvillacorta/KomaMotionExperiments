using OrdinaryDiffEqTsit5

using KomaMRICore 
using KomaMRIPlots
using KomaMRIPlots.PlotlyJS # for plotting
using CUDA

Nadc = 25
M0 = 1.0
T1 = 100e-3
T2 = 10e-3
B1 = 20e-6
Trf = 3e-3
γ = 2π * 42.58e6
# ΔBz = (2π * 100.0) / γ
φ = π / 4
B1e(t) = B1 * (0 <= t <= Trf)
# dt = 1 / (γ * sqrt(abs(B1)^2 + ΔBz^2)) * 1e6
duration = 2*Trf

Gx = 1e-3
Gy = 1e-3
Gz = 0

motions = [
	Translate(0.0, 0.1, 0.0, TimeRange(0.0, 1.0)),
	Rotate(0.0, 0.0, 45.0, TimeRange(0.0, 1.0)),
	HeartBeat(-0.6, 0.0, 0.0, Periodic(1.0)),
	Path([0.0 0.0], [0.0 1.0], [0.0 0.0], TimeRange(0.0, 10.0)),
	FlowPath([0.0 0.0], [0.0 1.0], [0.0 0.0], [0.0 0.0], TimeRange(0.0, 10.0))
]

x0 = [0.1]
y0 = [0.1]
z0 = [0.0]

# for m in motions
	motion = MotionList(motions[5])

	coords(t) = get_spin_coords(motion, x0, y0, z0, t)
	x(t) = (coords(t)[1])[1]
	y(t) = (coords(t)[2])[1]
	z(t) = (coords(t)[3])[1]

	# Utilizar get_spin_coords, que dependa de t y de las posiciones iniciales, en vez de funciones internas
	# get_spin_coords debe admitir escalares como argumentos de entrada
	# ΔBz = KomaMRIBase.interpolate(path.dy .* gy, KomaMRIBase.Gridded(KomaMRIBase.Linear()), Val(size(path.dx,1)))
	# Hacer un for y en cada iteracion probar con un movimiento distinto (y con combinaciones de movimientos)
	# Para flow, testear con el paper de numerical methods for 3d phase contrast (útil para el paper)
	# Reutilizar ejemplo de Belén (el que está en el paper) para un ejemplo de la documentación

	println("==========")

	# ## Solving using DiffEquations.jl
	function bloch!(dm, m, p, t)
		mx, my, mz = m
		# bx, by, bz = [B1e(t) * cos(φ), B1e(t) * sin(φ), ΔBz(t/t_end)]
		bx, by, bz = [B1e(t) * cos(φ), B1e(t) * sin(φ), (x(t) * Gx + y(t) * Gy + z(t) * Gz)]
		dm[1] = -mx / T2 + γ * bz * my - γ * by * mz
		dm[2] = -γ * bz * mx - my / T2 + γ * bx * mz
		dm[3] =  γ * by * mx - γ * bx * my - mz / T1 + M0 / T1
		return nothing
	end
	m0 = [0.0, 0.0, 1.0]
	tspan = (0.0, duration)
	prob = ODEProblem(bloch!, m0, tspan)
	# Only at ADC points
	tadc = range(Trf, duration, Nadc)
	sol = @time solve(prob, Tsit5(), saveat = tadc, abstol = 1e-9, reltol = 1e-9)
	sol_diffeq = hcat(sol.u...)'
	mxy_diffeq = sol_diffeq[:, 1] + im * sol_diffeq[:, 2]
	# Fine solution
	tadc_fine = range(0, duration, 10 * Nadc)
	sol_fine = solve(prob, Tsit5(), saveat = tadc_fine, abstol = 1e-9, reltol = 1e-9)
	sol_diffeq_fine = hcat(sol_fine.u...)'
	mxy_diffeq_fine = sol_diffeq_fine[:, 1] + im * sol_diffeq_fine[:, 2]

	## Solving using KomaMRICore.jl
	# Creating Sequence
	seq = Sequence()
	seq += RF(cis(φ) .* B1, Trf)
	seq.GR[1,1] = Grad(Gx, duration)
	seq.GR[2,1] = Grad(Gy, duration)
	seq.GR[3,1] = Grad(Gz, duration)
	seq.ADC[1] = ADC(Nadc, duration-Trf, Trf)
	# Creating object
	obj = Phantom(x = x0, y = y0, z = z0, ρ = [M0], T1 = [T1], T2 = [T2], motion = motion)
	# Scanner
	sys = Scanner()
	# Simulation
	# for sim_method in [KomaMRICore.Bloch(), KomaMRICore.BlochSimple()]
		sim_params = Dict{String, Any}(
			"sim_method"=>KomaMRICore.Bloch(),
			"return_type"=>"mat"
		)
		raw_aux = simulate(obj, seq, sys; sim_params)
		raw = raw_aux[:, 1, 1]

		NMRSE(x, x_true) = sqrt.( sum(abs.(x .- x_true).^2) ./ sum(abs.(x_true).^2) ) * 100.
		print(KomaMRICore.Bloch(), " "); @show nmrse = NMRSE(raw, mxy_diffeq)
	# end
# end

# Plot
p1 = plot(scatter(x = sol.t * 1e3, y = B1e.(tadc_fine) / B1, name = "B1"), Layout(xaxis_title = "Time [us]", yaxis_title = "B1 [T]"))
p2 = plot(
	[
		scatter(x = sol_fine.t * 1e3, y = real.(mxy_diffeq_fine), name = "Mx_DiffEq_5th", line_width = 1, marker = attr(color = "red", symbol = "circle-open")),
		scatter(x = sol_fine.t * 1e3, y = imag.(mxy_diffeq_fine), name = "My_DiffEq_5th", line_width = 1, marker = attr(color = "blue", symbol = "circle-open")),
		scatter(x = get_adc_sampling_times(seq) * 1e3, y = real.(raw), name = "Mx_Koma_1st", mode = "markers", marker = attr(color = "red", symbol = :x)),
		scatter(x = get_adc_sampling_times(seq) * 1e3, y = imag.(raw), name = "My_Koma_1st", mode = "markers", marker = attr(color = "blue", symbol = :x)),
	],
	Layout(
		# title = "e1 = $(round(error; digits=4)) %; e2 = $(round(error2; digits=4)) %",
		xaxis_title = "Time [us]",
		yaxis_title = "Signal [a.u.]",
	)
)
display(p2)