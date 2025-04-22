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

    # Second rotation: 90° around Z (left-hand rule)
    # dx2 = -dy1
    # dy2 = dx1
    # dz2 = dz1

    # x2 = -y1
    # y2 = x1
    # z2 = z1

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