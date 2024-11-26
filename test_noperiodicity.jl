# Opción 1 - sin opción de periodicidad

struct SimpleMotion{S <: SimpleMotionType} <: MotionModel
    types::AbstractVector{S}
end

struct Translation{T<:Real} <: SimpleMotionType{T}
    offset::Vector{T} 
    velocity::Vector{T}
end

struct Rotation{T<:Real} <: SimpleMotionType{T}
    initial_angle::T 
    rotation_axis::Vector{T}
    rotation_center::Vector{T}
    angular_velocity::T 
end

struct Expansion{T<:Real} <: SimpleMotionType{T}
    offset::T
    normal_velocity::T
    contraction_center::Vector{T}
end

# Quitar offset
# movimientos no periodicos: tiempo de inicio, tiempo final y desplazamiento (siempre empieza en 0)
#  displacement_x(t<t_inicial) = 0
#  displacement_x(t>t_final) = displacement_x(t_final)
# movimientos periodicos: periodo y desplazamiento
#   en vez de periodo se podrían definir dos atributos: rise_time (cuanto tarda en llegar al desplazamiento final)
#   y fall_time (cuanto tarda en volver a cero)

# Rotación: sustituir rotation_axis por x_angle, y_angle, z_angle
# Quitar movimiento circular uniforme (no tiene sentido a nivel anatómico)
# PeriodicExpansion/Contraction === HeartBeating
# Para este movimiento heartbeating se puede definir una contracción 

# Summary
# Tipos de mov. simple:
# No periodicos: deben tener un rango de tiempo [t_inicial, t_final] y cúanto se desplazan.
#   --> Translation: ti, tf, dx, dy, dz
#     - Rotation: ti, tf, yaw, pitch, roll; centerx, centery, centerz  -  Buscar en papers de MRI cómo se llama a estos atributos
#     - RadialStrain: ti, tf, strain; centrerx, centery, centerz

# Periodicos:
#     - PeriodicTranslation: period, assimetry, dx, dy, dz
#     - PeriodicRotation: period, assimetry, pitch, roll; centerx, centery, centerz
#     - PeriodicRadialStrain: period, assimetry, strain; centrerx, centery, centerz

# Todos los atributos de los simplemotiontypes deben ser 0 por defecto

# Comprobar si es posible devolver los simplemotiontypes a gpu

# TODO: Todos los SimpleMotions deberían tener una función get_range, que devuelva [ti, tf]

# TODO: plot_phantom_map con slider del tiempo (lo ideal sería que el slider estuviese dentro de la función). 
# El rango del slider debería ir acorde con los tiempos de los movimientos del phantom
# Debe funcionar para todos los tipos de movimiento (Simple, Arbitrary, NoMotion...)
# Wrapper a plot_phantom_map, que tenga un slider y que cambie los datos del plot cuando éste cambie
# p = plot_phantom_map(...)
# p.datos.x = (...)

# TODO: documentación (Literate.jl) generar el plot con el slider. Preguntar a Boris
# https://plotly.com/javascript/sliders/
# Poner en el slider la mínima cantidad de puntos temporales que permitan ver cómo es el movimiento de manera correcta

# TODO: phantom con coordenadas cilindricas (útil para corazón)

# motion = HeartBeat(
#     amplitude
#     rr = [t1,t2,t3,t4...]
# )