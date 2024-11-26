# # Patient's Motion During Acquisition

using KomaMRI # hide
sys = Scanner() # hide

obj = brain_phantom2D()
obj.Δw .= 0 # hide

## Motion: translation
dx = 0.02
dy = 0.02
dz = 0.0
t_start = 0e-3
t_end = 200e-3
obj.motion = SimpleMotion([
    Translation(dx, dy, dz, t_start, t_end)
    ])

p1 = plot_phantom_map(obj, :T2 ; height=450, intermediate_time_samples=4)
seq_file1 = joinpath(dirname(pathof(KomaMRI)), "../examples/5.koma_paper/comparison_accuracy/sequences/EPI/epi_100x100_TE100_FOV230.seq") # hide
seq1 = read_seq(seq_file1)[1:end-1] # hide

## Simulate # hide
raw1 = simulate(obj, seq1, sys) # hide

## Recon # hide
acq1 = AcquisitionData(raw1) # hide
acq1.traj[1].circular = false # hide
Nx, Ny = raw1.params["reconSize"][1:2] # hide
reconParams = Dict{Symbol,Any}(:reco=>"direct", :reconSize=>(Nx, Ny)) # hide
image1 = reconstruction(acq1, reconParams) # hide

## Plotting the recon # hide
p3 = plot_image(abs.(image1[:, :, 1]); height=400) # hide
 










## Motion Corrected Reconstruction
# Start times of each sequence block
# start_times = cumsum([0.0;seq1[1:end-1].DUR])

# Take only start times of blocks where ADC is on
# ADC_on_times = start_times[is_ADC_on.(seq1)]

# Get time instant where each k-space point is sampled
# sample_times = reduce(vcat, map(.+, collect.(times.(seq1.ADC[is_ADC_on.(seq1)])), ADC_on_times))
sample_times = get_adc_sampling_times(seq1)

# Get translation in each direction for each sample
# Δx = reduce(vcat, KomaMRIBase.displacement_x.(obj.motion.types,Ref([0.0]),Ref([0.0]),Ref([0.0]),Ref(sample_times)))
# Δy = reduce(vcat, KomaMRIBase.displacement_y.(obj.motion.types,Ref([0.0]),Ref([0.0]),Ref([0.0]),Ref(sample_times)))

displacements = get_spin_coords(obj.motion, [0.0], [0.0], [0.0], sample_times)
displacement = hcat(displacements...)

## Get k-space coordinates of each sample
# FOVx, FOVy, __ = acq1.fov .* 1e-3  #[m]
# Δkx = 1/FOVx;               
# Δky = 1/FOVy
# Nx, Ny = acq1.encodingSize
# kxFOV = (Nx-1)*Δkx;         
# kyFOV = (Ny-1)*Δky;

# nodes = acq1.traj[1].nodes # This gives us the (kx, ky) coordinates, normalized to the kFOV, of each point
# kx = nodes[1,:] .* kxFOV
# ky = nodes[2,:] .* kyFOV

## Get k-space
_, kspace = get_kspace(seq1)

# Phase correction: ΔΦcor = 2π(kx*Δx + ky*Δy)
ΔΦ = 2π*sum(kspace .* displacement, dims=2)

# Apply phase correction
acq2 = copy(acq1)
acq2.kdata[1] .*= exp.(im*ΔΦ)

image2 = reconstruction(acq2, reconParams) # hide
p4 = plot_image(abs.(image2[:, :, 1]); height=400) # hide


# Mostrar desplazamientos como señal 1D -> get_displacement