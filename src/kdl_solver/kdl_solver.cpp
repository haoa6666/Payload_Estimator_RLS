#include "kdl_solver.h"

#include <cmath>

#include "common/log.h"

namespace imeta {
namespace controller {

int sign(double x) {
  if (x > 0) return 1;
  if (x < 0) return -1;
  return 0;  // x == 0
}

KdlSolver::KdlSolver(const std::string& urdf_file, int arm_end_type)
    : urdf_file_(urdf_file), arm_end_type_(arm_end_type) {}

bool KdlSolver::Init() {
  // 加载 URDF 模型
  if (!model_.initFile(urdf_file_)) {
    AERROR << "Failed to parse URDF";
    return false;
  }

  // 从 URDF 构建 KDL Tree
  if (!kdl_parser::treeFromUrdfModel(model_, kdl_tree_)) {
    AERROR << "Failed to construct KDL tree";
    return false;
  }

  // 从 KDL Tree 提取 KDL Chain
  // TODO:1. 前两个参数需根据urdf实际名称修改 2.如果末端有夹爪或示教器如何修改？
  if (!kdl_tree_.getChain("base_link", "Link6", kdl_chain_)) {
    AERROR << "Failed to extract chain from base_link to J6_Link";
    return false;
  }

  // 获取关节数量
  joint_min_.resize(kdl_chain_.getNrOfJoints());
  joint_max_.resize(kdl_chain_.getNrOfJoints());

  AINFO << "kdl_chain_segments_size: " << kdl_chain_.segments.size();
  for (int i = 0; i < kdl_chain_.segments.size(); ++i) {
    auto segment = kdl_chain_.segments.at(i);
    const KDL::Joint& joint = segment.getJoint();
    if (joint.getType() == KDL::Joint::None) {
      continue;
    }

    const std::string& joint_name = joint.getName();
    auto it = model_.joints_.find(joint_name);
    if (it == model_.joints_.end()) {
      AERROR << "Joint " << joint_name << " not found in URDF!";
      return false;
    }

    const auto& urdf_joint = it->second;

    if ((urdf_joint->type == urdf::Joint::REVOLUTE ||
         urdf_joint->type == urdf::Joint::PRISMATIC) &&
        urdf_joint->limits) {
      joint_min_(i) = urdf_joint->limits->lower;
      joint_max_(i) = urdf_joint->limits->upper;
      AINFO << "Joint '" << joint_name
            << "': lower=" << urdf_joint->limits->lower
            << ", upper=" << urdf_joint->limits->upper;
    }
  }

  KDL::Vector gravity(0.0, 0.0, -9.81);  // 默认重力方向
  fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(kdl_chain_);
  ik_solver_ = std::make_unique<KDL::ChainIkSolverPos_LMA>(kdl_chain_);
  dyn_solver_ = std::make_unique<KDL::ChainDynParam>(kdl_chain_, gravity);
  // 初始化 TracIK
  tracik_solver_.reset(new TRAC_IK::TRAC_IK(kdl_chain_, joint_min_, joint_max_,
                                            100, 1e-5, TRAC_IK::Speed));

  // 新增负载估计器
  payload_estimator_ = std::make_unique<PayloadEstimator>(kdl_chain_);

  return true;
}

bool KdlSolver::FkSolver(const std::array<double, 6>& joint_positions,
                         std::array<double, 6>& end_pose) {
  KDL::Frame out_pose;

  KDL::JntArray q(joint_positions.size());

  for (size_t i = 0; i < joint_positions.size(); ++i) {
    q(i) = joint_positions[i];
  }

  if (fk_solver_->JntToCart(q, out_pose) < 0) {
    AERROR << "fk_solver Failed !";
    return false;
  }

  end_pose[0] = out_pose.p.x();
  end_pose[1] = out_pose.p.y();
  end_pose[2] = out_pose.p.z();
  double roll, pitch, yaw;
  out_pose.M.GetRPY(roll, pitch, yaw);
  end_pose[3] = roll;
  end_pose[4] = pitch;
  end_pose[5] = yaw;

  return true;
}

bool KdlSolver::IkSolver(const std::array<double, 6>& arm_end_pose,
                         std::vector<double>& joint_positions) {
  // 进行数据类型转换
  KDL::Frame desired_pose;
  // 设置位置 (x, y, z)
  desired_pose.p =
      KDL::Vector(arm_end_pose[0], arm_end_pose[1], arm_end_pose[2]);
  // 设置旋转 (roll, pitch, yaw)
  desired_pose.M =
      KDL::Rotation::RPY(arm_end_pose[3], arm_end_pose[4], arm_end_pose[5]);

  KDL::JntArray q_init(kdl_chain_.getNrOfJoints());
  for (unsigned i = 0; i < q_init.rows(); ++i) {
    q_init(i) = 0.0;
  }

  KDL::JntArray q_result(kdl_chain_.getNrOfJoints());
  int ret = ik_solver_->CartToJnt(q_init, desired_pose, q_result);
  if (ret < 0) {
    AERROR << "IkSolver Failed !";
    return false;
  }

  joint_positions.resize(q_result.rows());
  for (unsigned i = 0; i < q_result.rows(); ++i) {
    joint_positions[i] = q_result(i);
  }

  return true;
}

bool KdlSolver::IkSolverWithTracIK(const std::array<double, 6>& arm_end_pose,
                                   const std::array<double, 6>& init_joint_positions,
                                   std::array<double, 6>& joint_positions) {
  // 构造目标末端位姿
  KDL::Frame desired_pose(
      KDL::Rotation::RPY(arm_end_pose[3], arm_end_pose[4], arm_end_pose[5]),
      KDL::Vector(arm_end_pose[0], arm_end_pose[1], arm_end_pose[2]));

  KDL::JntArray q_init(kdl_chain_.getNrOfJoints());
  for (int i = 0; i < 6; ++i) {
    q_init(i) = init_joint_positions[i];
  }

  KDL::JntArray q_result(kdl_chain_.getNrOfJoints());

  int rc = tracik_solver_->CartToJnt(q_init, desired_pose, q_result);
  if (rc < 0) {
    AERROR << "[TracIK] Failed to solve IK.";
    return false;
  }

  for (int i = 0; i < q_result.rows(); ++i) {
    joint_positions[i] = q_result(i);
  }

  return true;
}

bool KdlSolver::FeedforwardTorqueCompensation(
    const std::vector<double>& q_actual, const std::vector<double>& dq_actual,
    const std::vector<double>& q_desired, const std::vector<double>& dq_desired,
    const std::vector<double>& ddq_desired,
    std::vector<double>& torque_output) {
  /*动力学前馈控制，前馈力矩计算：
  T = M(q)*ddq_d + C(q,dq)*dq_d + G(q)

  单独补偿重力,电机实际输出的力矩：
  T_motor = T_pid + G(q)

  作为前馈控制的其中一部分：
  // TODO: 我们需不需要在这里计算T_pid？
  T_motor = T_pid + M(q)*ddq_d + C(q,dq)*dq_d + G(q)*/
  std::vector<double> tau_g, tau_c, tau_m;

  // 使用反馈的关节位置用于重力、科氏力、惯性项计算
  // TODO:均使用期望关节值

  // TODO:20250515
  /*
  if(主臂){
    重力补偿
  }else if(从臂){
  完整前馈补偿（已包括重力补偿）
  }
  */
  if (!GravityCompensation(q_desired, tau_g) ||
      !CoriolisTorque(q_desired, dq_desired, tau_c) ||
      !InertiaTorque(q_desired, ddq_desired, tau_m)) {
    AERROR << "Failed to calculate forward orque.";
    return false;
  }

  size_t dof = q_actual.size();
  torque_output.resize(dof);
  for (size_t i = 0; i < dof; ++i) {
    torque_output[i] = tau_m[i] + tau_c[i] + tau_g[i];
  }
  return true;
}

// TODO: fix param type to std::array<double, 6>
bool KdlSolver::FeedforwardTorqueCompensation(
    const std::vector<double>& current_joint_position,
    const std::vector<double>& current_joint_velocity,
    std::vector<double>& feed_forward_torque) {
  // 先清空
  feed_forward_torque.clear();

  auto joint_position = current_joint_position;
  auto joint_velocity = current_joint_velocity;

  if (current_joint_position.size() == 7) {
    joint_position.pop_back();
    joint_velocity.pop_back();
  }

  std::vector<double> gravity_torque_coe({1.1, 1.2, 1.2, 1.1, 1.0, 1.0, 1.0});
  std::vector<double> coriolis_torque_coe({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  std::vector<double> friction_torque_coe(
      {0.25, 0.45, 0.5, 0.16, 0.1, 0.16, 0.16});
  // std::vector<double> friction_torque_coe(7, 0);
  // compute gravity torque
  std::vector<double> gravity_torque;
  if (!GravityCompensation(joint_position, gravity_torque)) {
    return false;
  }

  // compute coriolis torque
  std::vector<double> coriolis_torque;
  if (!CoriolisTorque(joint_position, joint_velocity,
                      coriolis_torque)) {
    return false;
  }

  // compute friction torque
  std::vector<double> friction_torque;
  ComputeFriction(joint_velocity, friction_torque);

  // static friction torque
  std::vector<double> static_friction_torque({0, 0, 0, 0, 0, 0, 0});

  // for (int i = 0; i < gravity_torque.size(); i++) {
  //   AINFO << "J" << i + 1 << " gravity: " << gravity_torque.at(i)
  //         << " , coriolis: " << coriolis_torque.at(i)
  //         << " ,friction: " << friction_torque.at(i) << " , total t: "
  //         << (gravity_torque.at(i) + coriolis_torque.at(i)) +
  //                friction_torque.at(i);
  // }

  for (int i = 0; i < joint_position.size(); i++) {
    double temp_ff_torque = gravity_torque_coe[i] * gravity_torque[i] +
                     coriolis_torque_coe[i] * coriolis_torque[i] +
                     friction_torque_coe[i] * friction_torque[i] +
                     static_friction_torque[i] * sign(gravity_torque[i]);
    feed_forward_torque.push_back(temp_ff_torque);
  }

  return true;
}

bool KdlSolver::InertiaTorque(const std::vector<double>& joint_positions,
                              const std::vector<double>& joint_acc,
                              std::vector<double>& inertia_torque) {
  size_t dof = joint_positions.size();
  KDL::JntArray q(dof), q_ddot(dof);
  for (size_t i = 0; i < dof; ++i) {
    q(i) = joint_positions[i];
    q_ddot(i) = joint_acc[i];
  }

  KDL::JntSpaceInertiaMatrix M(dof);
  if (dyn_solver_->JntToMass(q, M) < 0) {
    AERROR << "Failed to compute Mass Matrix";
    return false;
  }

  KDL::JntArray torque(dof);
  torque.data = M.data * q_ddot.data;

  inertia_torque.resize(dof);
  for (size_t i = 0; i < dof; ++i) {
    inertia_torque[i] = torque(i);
  }

  return true;
}

bool KdlSolver::CoriolisTorque(const std::vector<double>& joint_positions,
                               const std::vector<double>& joint_vel,
                               std::vector<double>& coriolis_torque) {
  KDL::JntArray q(joint_positions.size());
  KDL::JntArray q_dot(joint_vel.size());
  for (size_t i = 0; i < joint_positions.size(); ++i) {
    q(i) = joint_positions[i];
    q_dot(i) = joint_vel[i];
  }
  // TODO: 大模型说kdl的JntToCoriolis函数内部已经做了和q_ddot相乘的计算
  // 输出的coriolis就是科氏力和向心力矩项
  // 这里需要确认一下
  KDL::JntArray coriolis(kdl_chain_.getNrOfJoints());
  if (dyn_solver_->JntToCoriolis(q, q_dot, coriolis) < 0) {
    AERROR << "Failed to compute Coriolis torque";
    return false;
  }

  coriolis_torque.resize(kdl_chain_.getNrOfJoints());
  for (int i = 0; i < kdl_chain_.getNrOfJoints(); i++) {
    coriolis_torque[i] = coriolis(i);
  }

  return true;
}

bool KdlSolver::GravityCompensation(
    const std::array<double, 6>& joint_positions,
    std::array<double, 6>& gravity_torque) {
  size_t dof = joint_positions.size();
  KDL::JntArray q(dof);
  for (size_t i = 0; i < dof; ++i) {
    q(i) = joint_positions[i];
  }

  KDL::JntArray gravity(dof);
  if (dyn_solver_->JntToGravity(q, gravity) < 0) {
    AERROR << "Failed to compute gravity torque";
    return false;
  }

  // gravity_torque.resize(dof);
  for (size_t i = 0; i < dof; ++i) {
    gravity_torque[i] = gravity(i);
  }
  return true;
}

bool KdlSolver::GravityCompensation(const std::vector<double>& joint_positions,
                                    std::vector<double>& gravity_torque) {
  size_t dof = joint_positions.size();
  KDL::JntArray q(dof);
  for (size_t i = 0; i < dof; ++i) {
    q(i) = joint_positions[i];
  }

  KDL::JntArray gravity(dof);
  if (dyn_solver_->JntToGravity(q, gravity) < 0) {
    AERROR << "Failed to compute gravity torque";
    return false;
  }

  gravity_torque.resize(dof);
  gravity_torque_kdl_.resize(dof);
  for (size_t i = 0; i < dof; ++i) {
    gravity_torque[i] = gravity(i);
    gravity_torque_kdl_[i] = gravity(i);
  }
  return true;
}

void KdlSolver::ComputeFriction(const std::vector<double>& joint_velocity,
                                std::vector<double>& friction_torque) {
  // static const double Dn_L[] = {0.6, 0.9, 0.9, 0.9, 0.10, 0.160, 0.160,
  // 0.07};

  // if (vel[0] < -0.006) f[0] = 0.06 * vel[0];
  // if (vel[0] > 0.006) f[0] = 0.06 * vel[0];

  // if (vel[1] < -0.006) f[1] = -0.1 + 0.05 * vel[1];
  // if (vel[1] > 0.006) f[1] = 0.1 + 0.05 * vel[1];

  // if (vel[2] < -0.006) f[2] = -0.2 + 0.1 * vel[2];
  // if (vel[2] > 0.006) f[2] = 0.2 + 0.1 * vel[2];

  // if (vel[3] < -0.006) f[3] = -0.30 + 0.2 * vel[3];
  // if (vel[3] > 0.006) f[3] = 0.30 + 0.2 * vel[3];

  // if (vel[4] < -0.01) f[4] = 0.03 - 0.05 * vel[4];
  // if (vel[4] > 0.01) f[4] = -0.03 - 0.05 * vel[4];

  friction_torque.resize(joint_velocity.size());
  std::vector<double> viscous_friction_coefficient(
      {0.3, 0.45, 0.85, 0.05, 0.15, 0.03, 0.03});
  std::vector<double> coulomb_friction_coefficient({0, 0.1, 0.2, 0.3, 0.03, 0, 0});

  for (size_t i = 0; i < joint_velocity.size(); ++i) {
    double viscous_friction = 0;

    // 粘滞摩擦力（Viscous Friction）
    if (i == 2) {
      double vel = std::max(-2.2, std::min(2.2, joint_velocity.at(i)));
      viscous_friction = vel * viscous_friction_coefficient.at(i);

    } else {
      viscous_friction =
          joint_velocity.at(i) * viscous_friction_coefficient.at(i);
    }

    // 库伦摩擦力
    // double coulomb_friction =
    //     sign(joint_velocity.at(i)) * coulomb_friction_coefficient.at(i);

    friction_torque.at(i) = viscous_friction;
  }
}

// bool KdlSolver::ComputePayloadCompensation(
//     std::vector<double> current_joint_position,
//     std::vector<double> current_joint_tau,
//     std::vector<double>& tau_ext_torque) {
//   if (!payload_estimator_->ComputePayloadCompensation(
//           current_joint_position, current_joint_tau, gravity_torque_kdl_,
//           tau_ext_torque)) {
//     AERROR << "payload compensation failed";
//     return false;
//   }
//   return true;
// }

bool KdlSolver::ComputePayloadCompWithRLS(
    const std::vector<double> &q, const std::vector<double> &dq,
    const std::vector<std::array<double, 3>> &target_traj_state,
    const std::vector<double> &tau_measured, std::vector<double> &tau_robot,
    std::vector<double> &tau_comp, double &mass_rls) {

  if (!payload_estimator_->UpdateMassRLS(q, dq, target_traj_state,
                                             tau_measured, tau_robot, tau_comp,
                                             mass_rls)) {
    AERROR << "payload compensation failed";
    return false;
  }
  return true;
}

bool KdlSolver::ComputePayloadCompWithPID(
    const std::vector<double> &q, const std::vector<double> &dq,
    const std::vector<std::array<double, 3>> &target_traj_state,
    const std::vector<double> &tau_measured, std::vector<double> &tau_robot,
    const std::vector<double> &tau_rls, std::vector<double> &tau_comp, PIDMode mode) {

  if (!payload_estimator_->ComputePayloadTorqueWithPID(
          q, dq, target_traj_state, tau_measured, tau_robot, tau_rls, tau_comp, mode)) {
    AERROR << "payload compensation failed";
    return false;
  }
  return true;
}

}  // namespace controller
}  // namespace imeta