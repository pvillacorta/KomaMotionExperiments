cd(@__DIR__)

using KomaMRI

## Static spins
path = "_FlowStat - 419109 spins.dat"

global x, y, z = Float32[], Float32[], Float32[]

open(path, "r") do file
    spins = 0
    for (i, linea) in enumerate(eachline(file))
        if startswith(linea, "0")
            data = split(linea, " ")
            if i==1 
                println(data)
            end
            push!(x, parse(Float32, data[4]))
            push!(y, parse(Float32, data[5]))
            push!(z, parse(Float32, data[6]))
            spins += 1
        end
    end
    println("spins: ", spins)
end

static = Phantom(x=x .* 1f-3, y=y .* 1f-3, z=z .* 1f-3)

## Dynamic spins
using Interpolations, ProgressMeter

path = "_FlowDynamic - 1650834 spins.dat"

function leer_bloques(path)
    bloques = []
    bloque_actual = String[]
    dentro_de_bloque = false

    open(path, "r") do file
        for linea in eachline(file)
            if isempty(linea)
                continue  # Ignorar líneas en blanco entre bloques
            elseif startswith(linea, "0")  # Inicio de bloque
                if dentro_de_bloque
                    push!(bloques, bloque_actual)
                end
                bloque_actual = [linea]
                dentro_de_bloque = true
            elseif occursin("-999999", linea)  # Fin de bloque
                push!(bloques, bloque_actual)
                dentro_de_bloque = false
                bloque_actual = String[]
            elseif dentro_de_bloque
                if !occursin("-111", linea)
                    push!(bloque_actual, linea)
                end
            end
        end
    end
    return bloques
end

bloques = leer_bloques(path)

Nt = 400
tf = 7210f-3
tq = range(0, tf, length=Nt)

global x, y, z = Float32[], Float32[], Float32[]
global dx, dy, dz = zeros(Float32, 1650834, Nt), zeros(Float32, 1650834, Nt), zeros(Float32, 1650834, Nt)

@showprogress desc="Computing..." for (i,bloque) in enumerate(bloques)
    xs, ys, zs, ts = Float32[], Float32[], Float32[], Float32[]
    for linea in bloque
        data = split(linea, " ")
        if startswith(linea, "0")
            push!(x, parse(Float32, data[4]) * 1f-3)
            push!(y, parse(Float32, data[5]) * 1f-3)
            push!(z, parse(Float32, data[6]) * 1f-3)
        end
        push!(ts, parse(Float32, data[1]) * 1f-3)
        push!(xs, parse(Float32, data[4]) * 1f-3) 
        push!(ys, parse(Float32, data[5]) * 1f-3)
        push!(zs, parse(Float32, data[6]) * 1f-3)
    end
    itpx = interpolate((ts,), xs, Gridded(Linear()))
    # itpy = interpolate((ts,), ys, Gridded(Linear())) #Activate this if there is motion not only in x
    # itpz = interpolate((ts,), zs, Gridded(Linear()))
    dx[i, :] .= itpx.(tq) 
    # dy[i, :] .= itpy.(tq)
    # dz[i, :] .= itpz.(tq)
end

dx .-= x
# dy .-= y
# dz .-= z

dynamic = Phantom(
    x=x, 
    y=y, 
    z=z, 
    T1= 850f-3 .* zeros(Float32, length(x),),
    T2= 5f-3   .* zeros(Float32, length(x),),
    motion=Path(dx, dy, dz, TimeRange(0f0, tf))
)

write_phantom(static + dynamic, "fortin_float32_Nt$(Nt).phantom")