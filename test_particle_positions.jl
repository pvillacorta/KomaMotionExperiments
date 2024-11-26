using KomaMRI
using ProgressMeter, StatsBase

include("CardiacCine.jl")


n_particles = 1_000_000
"""
--------------particle_positions.txt information -----------------
n_particles = 1_000_000
start_time = np.float64(0.000)
end_time = np.float64(60.000)
delta_t = np.float64(0.01)
time_steps = int((end_time - start_time) / delta_t)
"""

n_desired_particles = 1_000_000
t_start = 0.0
t_end = 1.0
delta_t = 0.01
t_ss = 1 
duration = t_end - t_start
time_steps = Int(duration / delta_t)

begin 
    # Initial positions
    x = Float64[]
    y = Float64[]
    z = Float64[]  

    # Displacements
    dx = Array{Float64}[]
    dy = Array{Float64}[]
    dz = Array{Float64}[]

    open("/home/export/personal/pvilayl/VTK probe points/particle_positions.txt") do file 
        progress_bar = Progress(n_particles; desc="Reading initial positions...")
        for i in 1:n_particles
            line = readline(file)
            particle_id, xi, yi, zi, flag = split(line, " ")
            if i in 1:n_desired_particles
                push!(x, parse(Float64, xi))
                push!(y, parse(Float64, yi))
                push!(z, parse(Float64, zi))
            end
            next!(progress_bar)
        end

        # Displacements
        progress_bar = Progress(time_steps; desc="Reading displacements...")
        for i in 1:time_steps - 1
            aux_x = Float64[]
            aux_y = Float64[]
            aux_z = Float64[]
            for j in 1:n_particles
                line = readline(file)
                particle_id, xd, yd, zd, flag = split(line, " ")
                if i in 1:t_ss:time_steps && j in 1:n_desired_particles 
                    push!(aux_x, parse(Float64, xd) - x[j])
                    push!(aux_y, parse(Float64, yd) - y[j])
                    push!(aux_z, parse(Float64, zd) - z[j])
                end
            end
            if i in 1:t_ss:time_steps
                push!(dx, aux_x)
                push!(dy, aux_y)
                push!(dz, aux_z)
            end
            next!(progress_bar)
        end

    end
end

##

aorta = Phantom(
    x = x .- mean(x),
    y = y .- mean(y),
    z = z .- mean(z),
    T1 = ones(length(x)) .* 0.8,
    T2 = ones(length(x)) .* 0.18
)

dx = reduce(hcat, dx)
dy = reduce(hcat, dy)
dz = reduce(hcat, dz)

aorta.motion = ArbitraryMotion(t_start, t_end, dx, dy, dz)

##
write_phantom(aorta, "/home/export/personal/pvilayl/koma_tests/aorta_1Mspins_duration=1s_dt=10ms.phantom")

##
# phantom = read_phantom("/home/export/personal/pvilayl/koma_tests/aorta_10000_spins.phantom")
# sys = Scanner()
# seq = GRE(0.12, 100, 30e-3, 80e-3, 5, sys)

# #ROTATE
# θ = 0
# seq_rot = rotx(θ / 180 * π) * roty(θ / 180 * π) * seq #<--------------CAMBIO

# # Simulation
# global simParams = Dict{String,Any}()
# raw_signal = simulate(phantom, seq_rot, sys; sim_params=simParams, w=nothing)

# # Reconstruction
# _, ktraj = get_kspace(seq)
# global recParams = Dict{Symbol,Any}(:reco=>"direct")
# acqData = AcquisitionData(raw_signal)
# acqData.traj[1].circular = false #Removing circular window
# acqData.traj[1].nodes = transpose(ktraj[:, 1:2]) #<----------------CAMBIO
# acqData.traj[1].nodes = acqData.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqData.traj[1].nodes[:])) #Normalize k-space to -.5 to .5 for NUFFT
# Nx, Ny = raw_signal.params["reconSize"][1:2]
# recParams[:reconSize] = (Nx, Ny)
# recParams[:densityWeighting] = true

# aux = @timed reconstruction(acqData, recParams)
# global image  = reshape(aux.value.data,Nx,Ny,:)
# # global kspace = KomaMRI.fftc(reshape(aux.value.data,Nx,Ny,:))

# plot_image(abs.(image)[:,:,1])

