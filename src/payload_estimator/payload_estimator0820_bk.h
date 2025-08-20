#ifndef PAYLOAD_ESTIMATOR_H
#define PAYLOAD_ESTIMATOR_H

#include <Eigen/Dense>
#include <array>
#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/jntspaceinertiamatrix.hpp>
#include <kdl/tree.hpp>
#include <memory>
#include <vector>

class PayloadEstimator {
public:
  PayloadEstimator(KDL::Chain chain, double lambda = 0.99);

  bool UpdateMassRLS(const std::vector<double> &q,
                     const std::vector<double> &dq,
                     const std::vector<std::array<double, 3>> &ddq,
                     const std::vector<double> &tau_measured,
                     std::vector<double> &tau_robot,
                     std::vector<double> &tau_comp, double &mass_rls);
  bool UpdateMassRLS0816(const std::vector<double> &q,
                     const std::vector<double> &dq,
                     const std::vector<std::array<double, 3>> &ddq,
                     const std::vector<double> &tau_measured,
                     std::vector<double> &tau_robot,
                     std::vector<double> &tau_comp, double &mass_rls);

 private:
  // 工具函数
  bool isValidNumber(double x);
  double saturate(double x, double limit);
  double lowPassFilter(double prev, double current, double alpha);

  // KDL 动力学和雅可比求解器
  KDL::Chain chain_;
  // std::shared_ptr<KDL::ChainIdSolver_RNE> dyn_solver_;
  std::shared_ptr<KDL::ChainDynParam> dyn_solver_;
  std::shared_ptr<KDL::ChainJntToJacSolver> jac_solver_;

  // RLS 参数
  double lambda_;         // 遗忘因子
  Eigen::Matrix4d P_;     // 协方差矩阵
  Eigen::Vector4d theta_; // 质量 + 质心估计

  // 重力方向
  Eigen::Vector3d gravity_;

  // 滤波缓存
  std::vector<double> tau_comp_prev_;
};

#endif // PAYLOAD_ESTIMATOR_H
