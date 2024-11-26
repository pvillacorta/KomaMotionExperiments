using KomaMRI, CUDA

include("Sequences.jl")

## ---- Phantom ---- 
path = "aorta_1Mspins_duration=1s_dt=10ms.phantom"
path = "aorta_100kspins_duration=30s_dt=100ms.phantom"
phantom = read_phantom(path)

##
phantom = brain_phantom2D()
phantom.Δw .= 0.0

## ---- Scanner ---- 
sys = Scanner()


## ---- Sequence ---- 
fov = 2.1*maximum([maximum(abs.(phantom.x)),
                   maximum(abs.(phantom.y)),
                   maximum(abs.(phantom.z))])
N_matrix = 64
TE = 10e-3
TR = 300e-3
flip_angle = 3
delta_f = -0
# Bipolar gradiens
venc = 60e-2
direction = [0, 0, 1.0]

seq = PC_GRE(
    venc,
	direction,
    fov,
    N_matrix,
    TE,
    TR,
    flip_angle,
    sys;
    Δf = delta_f
)

seq = GRE(
    fov,
    N_matrix,
    TE,
    TR,
    flip_angle,
    sys;
    Δf = delta_f
)

seq = bSSFP(
    fov,
    N_matrix,
    TR,
    flip_angle,
    sys;
    Δf = delta_f
)


## PC-EPI
fov = 0.11
N_matrix = 80
TE = 30e-3
TR = 300e-3
flip_angle = 3
delta_f = -3e3
# Bipolar gradiens
venc = 50e-2
direction = [0.0, 0.0, 1.0]

# ROTATE
θ = 0
R = rotx(θ / 180 * π) * roty(θ / 180 * π)
 
ex = selective_rf(flip_angle, sys; Δf=delta_f)
bipolar = -bipolar_gradients(venc, direction, sys)
epi = EPI(fov, N_matrix, sys)
TE_delay = TE - ( ex.DUR[1]/2 + ex.DUR[2] + dur(bipolar) + dur(epi)/2 )
if TE_delay < 0
    TE_delay = 0.0
    println("TE too short, setting it to TE_min = " * string(( ex.DUR[1]/2 + ex.DUR[2] + dur(bipolar) + dur(epi)/2 )*1e3) * " ms")
end
delay = Delay(TE_delay)
seq = ex + delay + bipolar + epi
seq_rot = R*ex + delay + bipolar + R*epi
seq_rot_noPC = R*ex + delay + Delay(dur(bipolar)) + R*epi

## Simulation
raw_signal = simulate(phantom, seq_rot, sys)
raw_signal_noPC = simulate(phantom, seq_rot_noPC, sys)
## Reconstruction
_, ktraj = get_kspace(seq)
global recParams = Dict{Symbol,Any}(:reco=>"direct")

acqData = AcquisitionData(raw_signal)
acqData.traj[1].circular = false #Removing circular window
acqData.traj[1].nodes = transpose(ktraj[:, 1:2]) #<----------------CAMBIO
acqData.traj[1].nodes = acqData.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqData.traj[1].nodes[:])) #Normalize k-space to -.5 to .5 for NUFFT

acqData_noPC = AcquisitionData(raw_signal_noPC)
acqData_noPC.traj[1].circular = false #Removing circular window
acqData_noPC.traj[1].nodes = transpose(ktraj[:, 1:2]) #<----------------CAMBIO
acqData_noPC.traj[1].nodes = acqData_noPC.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqData_noPC.traj[1].nodes[:])) #Normalize k-space to -.5 to .5 for NUFFT

Nx, Ny = raw_signal.params["reconSize"][1:2]
recParams[:reconSize] = (Nx, Ny)
recParams[:densityWeighting] = true

aux      = @timed reconstruction(acqData, recParams)
aux_noPC = @timed reconstruction(acqData_noPC, recParams)

global image       = reshape(aux.value.data,Nx,Ny,:)
global image_noPC  = reshape(aux_noPC.value.data,Nx,Ny,:)

## plot magnitude
plot_image(abs.(image[:,:,1]))

## plot phase
plot_image(angle.(image[:,:,1]))
plot_image(angle.(image_noPC[:,:,1]))

img_diff = angle.(image_noPC[:,:,1]) .- angle.(image[:,:,1])
plot_image(img_diff)


## Plot displacements
init_time = cumsum(seq.DUR)[3]
end_time = cumsum(seq.DUR)[5]
sample_times = collect(init_time:1e-4:end_time)
# sample_times = collect(0:0.01:1)
n = 119010
final_coords = get_spin_coords(phantom.motion[n:n], phantom.x[n:n], phantom.y[n:n], phantom.z[n:n], sample_times')
init_coords = (phantom.x[n], phantom.y[n], phantom.z[n])
displacements = vcat([final_coords[i] .- init_coords[i] for i=1:3]...)'

displacements

p3 = KomaMRIPlots.plot( # hide
    sample_times, # hide
    displacements .* 1e2, # hide
    KomaMRIPlots.Layout( # hide
        title = "Blood displacement in x, y and z", # hide
        xaxis_title = "time (s)", # hide
        yaxis_title = "Displacement (cm)" # hide
    )) # hide
KomaMRIPlots.restyle!(p3,1:3, name=["ux(t)", "uy(t)", "uz(t)"]) # hide


## Maximum velocity
init_time = cumsum(seq.DUR)[3]
end_time = cumsum(seq.DUR)[5]
sample_times = [init_time, end_time]
dt = end_time - init_time

x, y, z = get_spin_coords(phantom.motion, phantom.x, phantom.y, phantom.z, sample_times')
dx, dy, dz = x[:,2] .- x[:,1], y[:,2] .- y[:,1], z[:,2] .- z[:,1]
vx_max, idx_max = maximum(abs.(dx)) / dt, argmax(abs.(dx))
vy_max, idy_max = maximum(abs.(dy)) / dt, argmax(abs.(dy))
vz_max, idz_max = maximum(abs.(dz)) / dt, argmax(abs.(dz))

maximum(abs.(dz))


