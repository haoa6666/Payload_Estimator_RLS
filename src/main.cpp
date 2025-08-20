#include "common/log.h"
#include "kdl_solver/kdl_solver.h"
#include "payload_estimator/ArmJointState.h"
#include "payload_estimator/Debug.h"
#include "payload_estimator/PayloadDebug.h"
#include "payload_estimator/payload_estimator.h"
#include "ros/forwards.h"
#include "ros/init.h"
#include "ros/node_handle.h"
#include "ros/publisher.h"
#include "ros/subscriber.h"

#include <iostream>
#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <ros/ros.h>
#include <urdf/model.h>

std::vector<double> q_;
std::vector<double> dq_;
std::vector<std::array<double, 3>> full_state_;
std::vector<double> tau_measured_;
std::vector<double> feed_forward_torque_ = {0.1,-4,3,2,0.1,0.1};

void JointStateCallback(
    const payload_estimator::ArmJointStateConstPtr &joint_states) {
  tau_measured_ = joint_states->joint_effort;
};

void InterpInfoCallback(const payload_estimator::DebugConstPtr &debug_info) {
  full_state_.clear();
  // 插值位置、速度、加速度
  std::vector<double> q;
  std::vector<double> dq;
  std::vector<std::array<double, 3>> ddq;
  
  // feed_forward_torque_ = debug_info->feedforward_torque;

  q_ = debug_info->pos;
  dq_ = debug_info->vel;
  // AINFO << "1111111111111debug_acc_size: " << debug_info->acc.size();

  for (int i = 0; i < debug_info->acc.size(); i++) {
    full_state_.push_back({0, 0, debug_info->acc[i]});
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "payload_estimator");
  // 1) 加载 URDF
  std::string urdf_file =
      "/home/ahao/WXY/IMETA_LAB/TEST_FUNCTION/Payload_Estimator_RLS/src/urdf/"
      "y10804.urdf";
  // std::ifstream ifs(urdf_file);
  // std::string urdf_xml((std::istreambuf_iterator<char>(ifs)),
  // std::istreambuf_iterator<char>());

  ros::NodeHandle nh;
  ros::Subscriber joints_states_sub =
      nh.subscribe<payload_estimator::ArmJointState>("/y1/arm_joint_state", 1,
                                                     &JointStateCallback);

  ros::Subscriber interp_info_sub = nh.subscribe<payload_estimator::Debug>(
      "/y1/debug_msg", 1, &InterpInfoCallback);

  ros::Publisher payload_pub =
      nh.advertise<payload_estimator::PayloadDebug>("/y1/payload_debug", 1);

  // urdf::Model model;
  // if (!model.initFile(urdf_file)) {
  //   std::cerr << "Failed to parse URDF" << std::endl;
  //   return -1;
  // }

  // // 2) 构建 KDL 链
  // KDL::Tree tree;
  // if (!kdl_parser::treeFromUrdfModel(model, tree)) {
  //   std::cerr << "Failed to construct KDL tree" << std::endl;
  //   return -1;
  // }

  // KDL::Chain chain;
  // tree.getChain("base_link", "Link6", chain);

  imeta::controller::KdlSolver kdl_solver(urdf_file, 0);

  // imeta::controller::PayloadEstimator estimator(chain);

  std::vector<double> q = {0, -3, 3, 0, 0, 0};
  std::vector<double> dq = {0, 0, 0, 0, 0, 0};
  std::vector<std::array<double, 3>> ddq(6, {0, 0, 0});
  std::vector<double> tau_measured = {0.0, 20.0, 5.0, 1.0, 0.0, 0.0};
  std::vector<double> tau_rls_comp;
  std::vector<double> tau_pid_comp;
  // std::vector<double> tau_robot;
  double mass_rls;

  payload_estimator::PayloadDebug payload_debug_msg;

  while (ros::ok()) {

    ros::spinOnce();

    ros::Rate rate(50);

    AINFO << "feed_forward_torque_size: " << feed_forward_torque_.size();
    AINFO << "full_state_size: " << full_state_.size();
    // 添加对输入向量大小的检查，确保它们不为空且大小一致
    if (q_.size() == dq_.size() && full_state_.size() == q_.size() &&
        tau_measured_.size() == q_.size() && q_.size() > 0 &&
        feed_forward_torque_.size() == q_.size() &&
        full_state_.size() == q_.size()) {
      if (kdl_solver.ComputePayloadCompWithRLS(
              q_, dq_, full_state_, tau_measured_, feed_forward_torque_,
              tau_rls_comp, mass_rls)) {
        payload_debug_msg.header.stamp = ros::Time::now();
        payload_debug_msg.joints_pos = q_;
        payload_debug_msg.tau_measured = tau_measured_;
        payload_debug_msg.mass_rls = mass_rls;
        payload_debug_msg.tau_rls_comp = tau_rls_comp;
        payload_debug_msg.tau_feedforward = feed_forward_torque_;
      } else {
        AWARN << "UpdateMassRLS 执行失败";
      }

      if (kdl_solver.ComputePayloadCompWithPID(
              q_, dq_, full_state_, tau_measured_, feed_forward_torque_,
              tau_rls_comp, tau_pid_comp,
              imeta::controller::PIDMode::TORQUE_TRACKING)) {
        payload_debug_msg.tau_pid_comp = tau_pid_comp;
      } else {
        AWARN << "UpdateMassRLS 执行失败";
      }
    } else {
      // 等待回调函数填充数据
      // AINFO << "等待数据填充: q_size=" << q_.size()
      //       << ", dq_size=" << dq_.size() << ", ddq_size=" << ddq_.size()
      //       << ", tau_measured_size=" << tau_measured_.size();
    }
    payload_pub.publish(payload_debug_msg);

    rate.sleep();
  }

  return 0;
}
