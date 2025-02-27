cd("/home/export/personal/pvilayl/koma_tests")

using KomaMRI

obj = read_phantom("sUbend_2M_stepsize_4e-4_numSteps_5.phantom")

# Rotate -90º in y (Right-hand rule)

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

write_phantom(obj_rot, "sUbend_2M_stepsize_4e-4_numSteps_5_rotated.phantom")