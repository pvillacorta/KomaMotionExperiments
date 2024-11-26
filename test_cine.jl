using KomaMRI
using StatsBase
using LinearAlgebra

include("read_phantom_MAT.jl")
include("plot_cine.jl")
include("CardiacCine.jl")

hr = 60        # [bpm]
N_matrix = 64 # image size = N x N
N_phases = 10  # Number of cardiac phases

# ------------ CHOOSE ONE OF THIS PHANTOMS: -----------

## Artery
include("phantoms/artery_phantom.jl")
path = "artery.phantom" 
phantom = read_phantom(path)

## Heart 



##

# Motion heart
# phantom = read_phantom_MAT("/datos/work/phantomXCAT_1mm_5cardPhases/"; ss=1, Δx=1)
# phantom = phantom[abs.(phantom.z) .< 0.008]
# phantom = phantom[abs.(phantom.x) .< 0.06]
# phantom = phantom[abs.(phantom.y) .< 0.06]
# # ------------------------------------------------------

# FOV = 2.3*maximum([maximum(abs.(phantom.x)),
#                  maximum(abs.(phantom.y)),
#                  maximum(abs.(phantom.z))])

##
sys = Scanner()

FOV = 0.05
TR  = 8e-3

# ROTATE
θ = 0
R = roty(θ / 180 * π)

## Simulate
using CUDA

frames,seq = cardiac_cine(
    FOV, hr, N_phases, N_matrix, phantom, sys;
    Δf=0,
    flip_angle=90, 
    TR=TR,
    dummy_cycles = 0,
    tagging=false,
    rotation_matrix=R,
    max_spins=350_000
);

## Post-processing
post = []
post2 = []
for frame in frames
    aux = copy(frame)
    perc = percentile(frame[:],96)
    aux[frame.>=perc].= perc;
    push!(post,aux)
    aux = round.(UInt8,255*(aux./maximum(aux)))
    push!(post2,aux)
end

## Plot
plot_cine(frames, 1; Δt=TR, filename="frames.gif")
plot_cine(post,   10; Δt=TR, filename="post.gif")
plot_cine(post2,  10; Δt=TR, filename="post2.gif")


plot_cine(frames,N_phases) 

## Frame intensity profiling
tissue_profile = [frame[16,10]/1.6 for frame in post]
blood_profile  = [frame[16,16] for frame in post]
x = TR*(1:10)*1e3
blood = KomaMRIPlots.PlotlyJS.scatter(;x=x,y=blood_profile,name="Flow pixel intensity")
tissue = KomaMRIPlots.PlotlyJS.scatter(;x=x,y=tissue_profile,name="Wall tissue pixel intensity")


KomaMRIPlots.PlotlyJS.plot([tissue,blood])


## -------------------------------------------
using KomaMRI, CUDA
using StatsBase
using LinearAlgebra

include("read_phantom_MAT.jl")
include("plot_cine.jl")
include("CardiacCine.jl")

hr = 60        # [bpm]
N_matrix = 64 # image size = N x N
N_phases = 10  # Number of cardiac phases
FOV = 0.05

sys = Scanner()

phantom = read_phantom("artery.phantom")

for v in 12:2:20                      # Fluid velocity from 2 to 20 cm/s
    period = 4 / v # L=4
    phantom.motion.motions[1].time = Periodic(period, 1.0)
    for TR in 10e-3:10e-3:100e-3     # TR from 10 to 100 ms
        for flip_angle in 10.0:10:90.0  # Flip angle from 10º to 90º
            @info "v = $(v)cm/s      TR = $(Int(TR*1e3)) ms      FlipAngle = $(Int(flip_angle))º"
            frames,seq = cardiac_cine(
                FOV, hr, N_phases, N_matrix, phantom, sys;
                flip_angle=flip_angle, 
                TR=TR,
                max_spins=350_000
            );
            plot_cine(frames, 1; Δt=TR, filename="tof_cines_64x64/v$(v)cms_TR$(Int(TR*1e3))_FlipAngle$(Int(flip_angle)).gif")
        end
    end
end

