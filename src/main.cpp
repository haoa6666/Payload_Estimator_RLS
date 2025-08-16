#include "common/log.h"
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
std::vector<std::array<double, 3>> ddq_;
std::vector<double> tau_measured_;

void JointStateCallback(
    const payload_estimator::ArmJointStateConstPtr &joint_states) {
  tau_measured_ = joint_states->joint_effort;
};

void InterpInfoCallback(const payload_estimator::DebugConstPtr &debug_info) {
  ddq_.clear();
  // 插值位置、速度、加速度
  std::vector<double> q;
  std::vector<double> dq;
  std::vector<std::array<double, 3>> ddq;
  // auto acc = debug_info->acc;

  q_ = debug_info->pos;
  dq_ = debug_info->vel;
  // AINFO << "1111111111111debug_acc_size: " << debug_info->acc.size();

  for (int i = 0; i < debug_info->acc.size(); i++) {
    ddq_.push_back({0, 0, debug_info->acc[i]});
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "payload_estimator");
  // 1) 加载 URDF
  std::string urdf_file = "/home/ubuntu/WXY/TEST_FUNC/Payload_Estimator_RLS/"
                          "payload_comp_test/src/urdf/y10804.urdf";
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

  urdf::Model model;
  if (!model.initFile(urdf_file)) {
    std::cerr << "Failed to parse URDF" << std::endl;
    return -1;
  }

  // 2) 构建 KDL 链
  KDL::Tree tree;
  if (!kdl_parser::treeFromUrdfModel(model, tree)) {
    std::cerr << "Failed to construct KDL tree" << std::endl;
    return -1;
  }

  KDL::Chain chain;
  tree.getChain("base_link", "Link6", chain);

  PayloadEstimator estimator(chain);

  std::vector<double> q = {0, -3, 3, 0, 0, 0};
  std::vector<double> dq = {0, 0, 0, 0, 0, 0};
  std::vector<std::array<double, 3>> ddq(6, {0, 0, 0});
  std::vector<double> tau_measured = {0.0, 20.0, 5.0, 1.0, 0.0, 0.0};
  std::vector<double> tau_comp;
  std::vector<double> tau_robot;
  double mass_rls;

  payload_estimator::PayloadDebug payload_debug_msg;

  while (ros::ok()) {

    ros::spinOnce();

    ros::Rate rate(50);

    // 添加对输入向量大小的检查，确保它们不为空且大小一致
    if (q_.size() == dq_.size() && ddq_.size() == q_.size() &&
        tau_measured_.size() == q_.size() && q_.size() > 0) {
      if (estimator.UpdateMassRLS(q_, dq_, ddq_, tau_measured_, tau_robot,
                                  tau_comp, mass_rls)) {
                    payload_debug_msg.header.stamp = ros::Time::now();
        // AINFO << "tau_comp: " << tau_comp[0] << ", " << tau_comp[1] << ", "
        //       << tau_comp[2] << ", " << tau_comp[3] << ", " << tau_comp[4]
        //       << ", " << tau_comp[5];
        // AINFO << "tau_measured: " << tau_measured_[0] << ", "
        //       << tau_measured_[1] << ", " << tau_measured_[2] << ", "
        //       << tau_measured_[3] << ", " << tau_measured_[4] << ", "
        //       << tau_measured_[5];
        // AINFO << "tau_robot: " << tau_robot[0] << ", "
        //       << tau_robot[1] << ", " << tau_robot[2] << ", "
        //       << tau_robot[3] << ", " << tau_robot[4] << ", "
        //       << tau_robot[5];
        payload_debug_msg.joints_pos = q_;
        payload_debug_msg.tau_measured = tau_measured_;
        payload_debug_msg.mass_rls = mass_rls;
        payload_debug_msg.tau_comp = tau_comp;
        payload_debug_msg.tau_gravity = tau_robot;
      } else {
        std::cerr << "UpdateMassRLS 执行失败" << std::endl;
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
