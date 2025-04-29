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
	Δf=0
)
    seq = Sequence()
    base_seq =  bSSFP(FOV, N_matrix, TR, flip_angle, sys; Δf=Δf)

    for i in 0:N_matrix-1 # 1 vps (Views Per Segment)
        line = base_seq[6*i .+ (1:6)]
        
        for j in 1:N_phases
            l = copy(line)

            l[1].RF[1].A  *= (-1)^(j)         # Sign of the RF pulse is alteranted every TR
            l[4].ADC[1].ϕ  = j%2==0 ? 0 : π   # so, phase of the ADC is consecuently alterned between 0 and π

            seq += l
        end

        d = ceil((N_phases * dur(line))/(60/hr)) * (60/hr) - (N_phases)* dur(line)
        seq += Delay(d)
    end

    return seq
end