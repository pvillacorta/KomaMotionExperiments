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

    if obj.motion.action isa FlowPath
        obj_rot.motion.action = FlowPath(
            dx_rot,
            dy_rot,
            dz_rot,
            obj.motion.action.spin_reset
        )
    elseif obj.motion.action isa Path
        obj_rot.motion.action = Path(
            dx_rot,
            dy_rot,
            dz_rot
        )
    end
    return obj_rot
end