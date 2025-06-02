cd(@__DIR__)

using KomaMRI

function rotate_phantom(obj)
    # First rotation: 90° around Y (right-hand rule)
    dx1 = -obj.motion.action.dz
    dy1 =  obj.motion.action.dy
    dz1 =  obj.motion.action.dx

    x1 = -obj.z
    y1 =  obj.y
    z1 =  obj.x

    obj_rot = copy(obj)
    obj_rot.x = x1
    obj_rot.y = y1
    obj_rot.z = z1

    obj_rot.motion = Path(
        dx1,
        dy1,
        dz1,
        obj.motion.time,
        obj.motion.spins
    )

    return obj_rot
end