cd("@__DIR__")

using Distributed
using CUDA

include("../utils/plot_cine.jl")
include("ring3D_phantom.jl")

# Generate phantom and store it in a .phantom file
phantom = ring3D_phantom()
write_phantom(phantom, "../phantoms/ring3D.phantom")

DEVICES = [3,2,1,0] # IDs of the GPUs that we want to use. 
                    # This is limited to the number of GPUs of your system

addprocs(length(DEVICES))

## Small simulation in all GPUs to initialize them -------------->
println("----------- Small simulation to initialize all GPUs --------------")
@everywhere begin
    using KomaMRI, CUDA
    MAX_SPINS_PER_GPU = 1_000_000
    ## --------------------------- PHANTOM ----------------------------
    obj = read_phantom("../phantoms/ring3D.phantom")
    
    ## --------------------------- SYSTEM ---------------------------- 
    sys = Scanner(B0 = 1.5, seq_Δt = 10e-6)

    ## --------------------------- SEQUENCE ---------------------------- 
    include("../sequences/bSSFP.jl")
    include("../utils/divide_spins_ranges.jl")

    hr = 75        # [bpm]
    N_matrix = 32 # image size 
    N_phases = 1 # Number of cardiac phases
    FOV = 15e-2
    TR = 8e-3   #s
    flip_angle = 40 

    Δf = 0
    N_dummy_cycles = 0

    # Tagging --------------------------------------
    T_rf = 3e-3   	 # Pulse duration
    B_1° = 8.6649e-8 # With T = 3ms, we need B1 = 8.6649e-8 T to produce a flip angle α = 1°
    α = 45
    B1 = α*B_1°

    EX_45 = PulseDesigner.RF_sinc(B1, T_rf, sys; G=[0.0, 0.0, 0.0])[1]

    A = 4e-3 	# 4 mT/m
    T = 0.6e-3 	# 0.6 ms
    ζ = A / sys.Smax

    GR_x = Sequence(reshape([Grad(A,T,ζ);
                            Grad(0,0);
                            Grad(0,0)],(3,1)))

    GR_y = Sequence(reshape([Grad(0,0);
                            Grad(A,T,ζ);
                            Grad(0,0)],(3,1)))

    crusher = Sequence(reshape([Grad(5*A,2*T,ζ);
                                Grad(5*A,2*T,ζ);
                                Grad(5*A,2*T,ζ)],(3,1)))

    spamm_x =   EX_45 +
                GR_x +
                EX_45  + crusher 

    spamm_y =   EX_45 +
                GR_y +
                EX_45  + crusher 

    tag = spamm_x 
    # -------------------------------------- Tagging

    local seq_aux = Sequence()
    base_seq =  bSSFP(FOV, N_matrix, TR, flip_angle, sys; Δf=Δf)

    for i in 0:N_matrix-1 # 1 vps (Views Per Segment)
        line = base_seq[6*i .+ (1:6)]

        if i==0
            del = (60/hr) - (N_dummy_cycles*dur(line) + dur(tag))
            seq_aux += Delay(del)
        end

        seq_aux += tag
        
        for j in 0:(N_dummy_cycles + N_phases)-1
            l = copy(line)

            if j in 0:N_dummy_cycles-1
                l.ADC = [ADC(0,0) for i in 1:length(l)]
            end

            l[1].RF[1].A  *= (-1)^(j)         # Sign of the RF pulse is alteranted every TR
            l[4].ADC[1].ϕ  = j%2==0 ? 0 : π   # so, phase of the ADC is consecuently alterned between 0 and π

            seq_aux += l
        end

        del = ceil(((N_dummy_cycles + N_phases) * dur(line) + dur(tag))/(60/hr)) * (60/hr) - ((N_dummy_cycles + N_phases) * dur(line) + dur(tag))
        seq_aux += Delay(del)
    end

    seq = copy(seq_aux)
end

cine = []
@everywhere begin
    #Divide phantom in N_GPU parts
    gpu_parts = kfoldperm(length(obj), nworkers())

    sequential_parts = []
    for (i, gpu_part) in enumerate(gpu_parts)
        push!(sequential_parts, divide_spins_ranges(length(gpu_part), MAX_SPINS_PER_GPU))
    end
end

@time begin # Global time
    if length(sequential_parts[1]) > 1
        @info "Dividing phantom ($(length(obj)) spins) into $(length(sequential_parts[1])) parts that will be simulated sequentially"
    end

    #Distribute simulation across workers
    raw_signal = Distributed.@distributed (+) for i=1:nworkers()
        KomaMRICore.set_device!(DEVICES[i]) #Sets device for this worker
        raw_gpu = []
        for (j, sequential_part) in enumerate(sequential_parts[i])
            if length(sequential_parts[i]) > 1
                @info "Simulating phantom part $(j)/$(length(sequential_parts[i]))"

            end
            push!(raw_gpu, simulate(obj[gpu_parts[i][sequential_part]], seq, sys))
        end
        reduce(+, raw_gpu)
    end

    ## ------------------------- RECONSTRUCTION -------------------------- 
    @info "Running reconstruction"
    @time begin
        recParams = Dict{Symbol,Any}(:reco=>"direct")
        Nx = Ny = N_matrix
        recParams[:reconSize] = (Nx, Ny)
        recParams[:densityWeighting] = false

        acqData = AcquisitionData(raw_signal)

        _, ktraj = get_kspace(seq)
        
        frames = []
        for i in 1:N_phases
            acqAux = copy(acqData)
            range = reduce(vcat,[j*(N_matrix*N_phases).+((i-1)*N_matrix.+(1:N_matrix)) for j in 0:N_matrix-1])

            # Kdata
            acqAux.kdata[1] = reshape(acqAux.kdata[1][range],(N_matrix^2,1))

            # Traj
            acqAux.traj[1].circular = false

            acqAux.traj[1].nodes = transpose(ktraj[:, 1:2])[:,range]
            acqAux.traj[1].nodes = acqAux.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqAux.traj[1].nodes[:]))

            acqAux.traj[1].numProfiles = N_matrix
            acqAux.traj[1].times = acqAux.traj[1].times[range]

            # subsampleIndices
            acqAux.subsampleIndices[1] = acqAux.subsampleIndices[1][1:N_matrix^2]

            # Reconstruction
            aux = @timed reconstruction(acqAux, recParams)
            image  = reshape(aux.value.data,Nx,Ny,:)
            image_aux = abs.(image[:,:,1])

            push!(frames,image_aux)
            cine = frames
        end
    end
end # Global time
## <--------------------------- Small simulation 


## REAL SIMULATION -----------------------> 
println("\n----------- Starting simulations --------------")
@everywhere begin
    using KomaMRI, CUDA

    MAX_SPINS_PER_GPU = 1_000_000

    ## --------------------------- PHANTOM ---------------------------- 
    obj = read_phantom("../phantoms/ring3D.phantom")
    
    ## --------------------------- SYSTEM ---------------------------- 
    sys = Scanner(B0 = 1.5, seq_Δt = 10e-6)

    ## --------------------------- SEQUENCE ---------------------------- 
    include("../sequences/bSSFP.jl")
    include("../utils/divide_spins_ranges.jl")

    hr = 75        # [bpm]
    N_matrix = 128 # image size 
    N_phases = 100 # Number of cardiac phases
    FOV = 15e-2
    TR = 8e-3   #s
    flip_angle = 40 

    Δf = 0
    N_dummy_cycles = 0

    # Tagging --------------------------------------
    T_rf = 3e-3   	 # Pulse duration
    B_1° = 8.6649e-8 # With T = 3ms, we need B1 = 8.6649e-8 T to produce a flip angle α = 1°
    α = 45
    B1 = α*B_1°

    EX_45 = PulseDesigner.RF_sinc(B1, T_rf, sys; G=[0.0, 0.0, 0.0])[1]

    A = 4e-3 	# 4 mT/m
    T = 0.6e-3 	# 0.6 ms
    ζ = A / sys.Smax

    GR_x = Sequence(reshape([Grad(A,T,ζ);
                            Grad(0,0);
                            Grad(0,0)],(3,1)))

    GR_y = Sequence(reshape([Grad(0,0);
                            Grad(A,T,ζ);
                            Grad(0,0)],(3,1)))

    crusher = Sequence(reshape([Grad(5*A,2*T,ζ);
                                Grad(5*A,2*T,ζ);
                                Grad(5*A,2*T,ζ)],(3,1)))

    spamm_x =   EX_45 +
                GR_x +
                EX_45  + crusher 

    spamm_y =   EX_45 +
                GR_y +
                EX_45  + crusher 

    tag = spamm_x 
    # -------------------------------------- Tagging

    local seq_aux = Sequence()
    base_seq =  bSSFP(FOV, N_matrix, TR, flip_angle, sys; Δf=Δf)

    for i in 0:N_matrix-1 # 1 vps (Views Per Segment)
        line = base_seq[6*i .+ (1:6)]

        if i==0
            del = (60/hr) - (N_dummy_cycles*dur(line) + dur(tag))
            seq_aux += Delay(del)
        end

        seq_aux += tag
        
        for j in 0:(N_dummy_cycles + N_phases)-1
            l = copy(line)

            if j in 0:N_dummy_cycles-1
                l.ADC = [ADC(0,0) for i in 1:length(l)]
            end

            l[1].RF[1].A  *= (-1)^(j)         # Sign of the RF pulse is alteranted every TR
            l[4].ADC[1].ϕ  = j%2==0 ? 0 : π   # so, phase of the ADC is consecuently alterned between 0 and π

            seq_aux += l
        end

        del = ceil(((N_dummy_cycles + N_phases) * dur(line) + dur(tag))/(60/hr)) * (60/hr) - ((N_dummy_cycles + N_phases) * dur(line) + dur(tag))
        seq_aux += Delay(del)
    end

    seq = copy(seq_aux)
end

cine = []
for p in procs()[end]:-1:procs()[2]
    println("\n----------- NUMBER of GPUs: $(nworkers()) --------------")
    @everywhere begin
        
        #Divide phantom in N_GPU parts
        gpu_parts = kfoldperm(length(obj), nworkers())
    
        sequential_parts = []
        for (i, gpu_part) in enumerate(gpu_parts)
            push!(sequential_parts, divide_spins_ranges(length(gpu_part), MAX_SPINS_PER_GPU))
        end
    end
    
    @time begin # Global time
        if length(sequential_parts[1]) > 1
            @info "Dividing phantom ($(length(obj)) spins) into $(length(sequential_parts[1])) parts that will be simulated sequentially"
        end
    
        #Distribute simulation across workers
        raw_signal = Distributed.@distributed (+) for i=1:nworkers()
            KomaMRICore.set_device!(DEVICES[i]) #Sets device for this worker
            raw_gpu = []
            for (j, sequential_part) in enumerate(sequential_parts[i])
                if length(sequential_parts[i]) > 1
                    @info "Simulating phantom part $(j)/$(length(sequential_parts[i]))"
    
                end
                push!(raw_gpu, simulate(obj[gpu_parts[i][sequential_part]], seq, sys))
            end
            reduce(+, raw_gpu)
        end
    
        ## ------------------------- RECONSTRUCTION -------------------------- 
        @info "Running reconstruction"
        @time begin
            recParams = Dict{Symbol,Any}(:reco=>"direct")
            Nx = Ny = N_matrix
            recParams[:reconSize] = (Nx, Ny)
            recParams[:densityWeighting] = false
    
            acqData = AcquisitionData(raw_signal)
    
            _, ktraj = get_kspace(seq)
            
            frames = []
            for i in 1:N_phases
                acqAux = copy(acqData)
                range = reduce(vcat,[j*(N_matrix*N_phases).+((i-1)*N_matrix.+(1:N_matrix)) for j in 0:N_matrix-1])
    
                # Kdata
                acqAux.kdata[1] = reshape(acqAux.kdata[1][range],(N_matrix^2,1))
    
                # Traj
                acqAux.traj[1].circular = false
    
                acqAux.traj[1].nodes = transpose(ktraj[:, 1:2])[:,range]
                acqAux.traj[1].nodes = acqAux.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqAux.traj[1].nodes[:]))
    
                acqAux.traj[1].numProfiles = N_matrix
                acqAux.traj[1].times = acqAux.traj[1].times[range]
    
                # subsampleIndices
                acqAux.subsampleIndices[1] = acqAux.subsampleIndices[1][1:N_matrix^2]
    
                # Reconstruction
                aux = @timed reconstruction(acqAux, recParams)
                image  = reshape(aux.value.data,Nx,Ny,:)
                image_aux = abs.(image[:,:,1])
    
                push!(frames,image_aux)
                cine = frames
            end
        end
    end # Global time
    rmprocs(p)
end
println("\nAll simulations finished")
## <------------------- REAL SIMULATION

## Create the /results folder if it does not exist:
results_dirname = "results/"
if isdir(results_dirname)
    rm(results_dirname; recursive=true)
end
mkdir(results_dirname)

## Plot cine
plot_cine(cine, 10; Δt=75/(60*100), filename="frames.gif")

## Save resulting frames
frames_dirname = results_dirname*"frames/"
if isdir(frames_dirname)
    rm(frames_dirname; recursive=true)
end
mkdir(frames_dirname)
for (i, frame) in enumerate(frames)
    p = plot_image(frame)
    KomaMRIPlots.PlotlyJS.savefig(p, frames_dirname*"$(i).png")
end