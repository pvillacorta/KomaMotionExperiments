include("../sequences/bSSFP.jl")

function bSSFP_cine(
	FOV::Float64, 
	N_matrix::Int, 
	TR::Float64, 
	flip_angle, 
	hr,
	N_phases,
	sys::Scanner; 
	G=[0,0,0], 
	Δf=0,
	N_dummy_cycles=0
)
	seq = Sequence()
	base_seq =  bSSFP(FOV, N_matrix, TR, flip_angle, sys; Δf=Δf)

	for i in 0:N_matrix-1 # 1 vps (Views Per Segment)
		line = base_seq[6*i .+ (1:6)]

		if i==0
			d = (60/hr) - (N_dummy_cycles*dur(line))
			seq += Delay(d)
		end
		
		for j in 0:(N_dummy_cycles + N_phases)-1
			l = copy(line)

			if j in 0:N_dummy_cycles-1
				l.ADC = [ADC(0,0) for i in 1:length(l)]
			end

			seq += l
		end

		d = ceil(((N_dummy_cycles + N_phases) * dur(line))/(60/hr)) * (60/hr) - ((N_dummy_cycles + N_phases) * dur(line))
		seq += Delay(d)
	end

	return seq
end