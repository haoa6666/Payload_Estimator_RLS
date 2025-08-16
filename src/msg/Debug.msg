std_msgs/Header header

# 运动相关
float64[] pos
float64[] vel 
float64[] acc
float64[] jerk

# 力矩相关
float64[] feedforward_torque
float64[] gravity_torque
float64[] coriolis_torque
float64[] friction_torque
float64[] inertia_torque
float64[] payload_comp_toque
float64[] payload_comp_gravity_torque
float64[] payload_comp_inertia_torque

# 规划时间相关
float64 T_traj_final
float64 Tv
float64 Ta
float64 Tj
float64 max_joint_delta
