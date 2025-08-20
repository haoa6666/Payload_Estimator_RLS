#pragma once

#include <ros/ros.h>
#include <urdf/model.h>

#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/jntspaceinertiamatrix.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <memory>
#include <trac_ik/trac_ik.hpp>

#include "payload_estimator/payload_estimator.h"

namespace imeta {
namespace controller {
class KdlSolver {
 public:
  KdlSolver() = delete;
  explicit KdlSolver(const std::string& urdf_file, int arm_end_type);

  ~KdlSolver() = default;

 public:
  bool Init();

  bool FkSolver(const std::array<double, 6>& joint_positions,
                std::array<double, 6>& end_pose);

  bool IkSolver(const std::array<double, 6>& arm_end_pose,
                std::vector<double>& joint_positions);

  bool IkSolverWithTracIK(const std::array<double, 6>& arm_end_pose,
                          const std::array<double, 6>& init_joint_positions,
                          std::array<double, 6>& joint_positions);

  bool FeedforwardTorqueCompensation(const std::vector<double>& q_actual,
                                     const std::vector<double>& dq_actual,
                                     const std::vector<double>& q_desired,
                                     const std::vector<double>& dq_desired,
                                     const std::vector<double>& ddq_desired,
                                     std::vector<double>& torque_output);

  bool FeedforwardTorqueCompensation(const std::vector<double>& current_joint_position,
                                     const std::vector<double>& current_joint_velocity,
                                     std::vector<double>& feed_forward_torque);

  bool CoriolisTorque(const std::vector<double>& joint_positions,
                      const std::vector<double>& joint_vel,
                      std::vector<double>& coriolis_torque);

  bool InertiaTorque(const std::vector<double>& joint_positions,
                     const std::vector<double>& joint_acc,
                     std::vector<double>& inertia_torque);

  bool GravityCompensation(const std::array<double, 6>& joint_positions,
                           std::array<double, 6>& gravity_torque);

  bool GravityCompensation(const std::vector<double>& joint_positions,
                           std::vector<double>& gravity_torque);

  void ComputeFriction(const std::vector<double>& joint_velocity,
                       std::vector<double>& friction_torque);

  bool ComputePayloadCompensation(std::vector<double> current_joint_position,
                                  std::vector<double> current_joint_tau,
                                  std::vector<double>& tau_ext_torque);


  bool ComputePayloadCompWithRLS(
    const std::vector<double> &q, const std::vector<double> &dq,
    const std::vector<std::array<double, 3>> &ddq,
    const std::vector<double> &tau_measured, std::vector<double> &tau_robot,
    std::vector<double> &tau_comp, double &mass_rls);

  bool ComputePayloadCompWithPID(const std::vector<double> &q,
                                 const std::vector<double> &dq,
                                 const std::vector<std::array<double, 3>> &ddq,
                                 const std::vector<double> &tau_measured,
                                 std::vector<double> &tau_robot,
                                 const std::vector<double> &tau_rls,
                                 std::vector<double> &tau_comp,
                                 PIDMode mode);

private:
  urdf::Model model_;
  KDL::Tree kdl_tree_;
  KDL::Chain kdl_chain_;
  std::shared_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
  std::shared_ptr<KDL::ChainIkSolverPos_LMA> ik_solver_;
  std::shared_ptr<KDL::ChainDynParam> dyn_solver_;
  std::shared_ptr<KDL::ChainIkSolverVel_pinv> ik_vel_solver_;
  std::shared_ptr<TRAC_IK::TRAC_IK> tracik_solver_;

  // joint limit
  KDL::JntArray joint_min_, joint_max_;
  KDL::Vector gravity_;

  // 负载识别
  std::unique_ptr<PayloadEstimator> payload_estimator_;

  std::string urdf_file_;
  int arm_end_type_;
  // 用于估计负载质量时传入
  std::vector<double> gravity_torque_kdl_;
};

}  // namespace controller
}  // namespace imeta