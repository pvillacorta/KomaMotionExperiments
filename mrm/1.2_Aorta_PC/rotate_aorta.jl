cd(@__DIR__)

using KomaMRI

function rotate_aorta(obj)
    ## Rotate 90º in z (Right-hand rule)

    dx_rot =  -obj.motion.action.dy
    dy_rot =  obj.motion.action.dx
    dz_rot =  obj.motion.action.dz

    obj_rot = copy(obj)

    obj_rot.x =  -obj.y
    obj_rot.y =  obj.x
    obj_rot.z =  obj.z

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