cd(@__DIR__)

using KomaMRI

function rotate_sUbend(obj)
    ## Rotate -90º in y (Right-hand rule)

    dx_rot = obj.motion.action.dz
    dy_rot = obj.motion.action.dy
    dz_rot = obj.motion.action.dx

    obj_rot = copy(obj)

    obj_rot.x = obj.z
    obj_rot.y = obj.y
    obj_rot.z = obj.x

    obj_rot.motion = FlowPath(
        dx_rot,
        dy_rot,
        dz_rot,
        obj.motion.action.spin_reset,
        obj.motion.time,
        obj.motion.spins
    )

    return obj_rot
end