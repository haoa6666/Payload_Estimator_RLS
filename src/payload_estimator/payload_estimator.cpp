#include "payload_estimator.h"
#include "common/log.h"
#include <cmath>
#include <iostream>

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

PayloadEstimator::PayloadEstimator(KDL::Chain chain, double lambda)
    : chain_(chain), lambda_(lambda) {
  jac_solver_ = std::make_shared<KDL::ChainJntToJacSolver>(chain_);
  dyn_solver_ =
      std::make_shared<KDL::ChainDynParam>(chain_, KDL::Vector(0, 0, -9.81));
  P_ = MatrixXd::Identity(4, 4) * 1000.0;
  theta_.setZero();
  gravity_ << 0, 0, -9.81;
}

bool PayloadEstimator::isValidNumber(double x) {
  return std::isfinite(x) && !std::isnan(x);
}

double PayloadEstimator::saturate(double x, double limit) {
  if (x > limit)
    return limit;
  if (x < -limit)
    return -limit;
  return x;
}

double PayloadEstimator::lowPassFilter(double prev, double current,
                                       double alpha) {
  return alpha * prev + (1 - alpha) * current;
}

bool PayloadEstimator::UpdateMassRLS(
    const std::vector<double> &q, const std::vector<double> &dq,
    const std::vector<std::array<double, 3>> &ddq,
    const std::vector<double> &tau_measured, std::vector<double> &tau_robot,
    std::vector<double> &tau_comp, double &mass_rls) {
  const unsigned int N = q.size();
  if (dq.size() != N || ddq.size() != N || tau_measured.size() != N) {
    std::cerr << "UpdateMassRLS: input size mismatch" << std::endl;
    return false;
  }

  // 初始化滤波缓存
  if (tau_comp_prev_.size() != N) {
    tau_comp_prev_.assign(N, 0.0);
  }
  
  // 调整tau_robot大小以匹配当前关节数量
  tau_robot.resize(N);

  // KDL 转换
  KDL::JntArray q_kdl(N), dq_kdl(N), ddq_kdl(N);
  for (unsigned int i = 0; i < N; ++i) {
    q_kdl(i) = q[i];
    dq_kdl(i) = dq[i];
    ddq_kdl(i) = ddq[i][2];
  }

  // 动力学计算
  KDL::JntArray coriolis(N), gravity(N);
  KDL::JntSpaceInertiaMatrix inertia(N);
  dyn_solver_->JntToCoriolis(q_kdl, dq_kdl, coriolis);
  dyn_solver_->JntToGravity(q_kdl, gravity);

  dyn_solver_->JntToMass(q_kdl, inertia);

  VectorXd tau_model(N);
  for (unsigned int i = 0; i < N; ++i) {
    tau_model(i) = gravity(i); // 简化版
    tau_robot[i] = gravity(i);   // 修复：使用索引赋值而不是push_back
  }

  AINFO << "tau_model: " << tau_model(0) << ", " << tau_model(1) << ", "
        << tau_model(2) << ", " << tau_model(3) << ", " << tau_model(4) << ", "
        << tau_model(5);
        

  // 残差
  VectorXd y = VectorXd::Zero(N);
  for (unsigned int i = 0; i < N; ++i) {
    y(i) = tau_measured[i] - tau_model(i);
  }

  // --- 计算雅可比 ---
  // KDL::Jacobian jac_kdl(N);
  // jac_solver_->JntToJac(q_kdl, jac_kdl);
  // MatrixXd J = jac_kdl.data;
  // MatrixXd Jv = J.block(0, 0, 3, N);
  // MatrixXd Jw = J.block(3, 0, 3, N);

  // Vector3d e1(1.0, 0.0, 0.0), e2(0.0, 1.0, 0.0), e3(0.0, 0.0, 1.0);
  // Vector3d s1 = e1.cross(gravity_);
  // Vector3d s2 = e2.cross(gravity_);
  // Vector3d s3 = e3.cross(gravity_);

  // MatrixXd Phi(N, 4);
  // Phi.col(0) = Jv.transpose() * gravity_;
  // Phi.col(1) = Jw.transpose() * s1;
  // Phi.col(2) = Jw.transpose() * s2;
  // Phi.col(3) = Jw.transpose() * s3;

  // --- 计算雅可比 ---
  KDL::Jacobian jac_kdl(N);
  jac_solver_->JntToJac(q_kdl, jac_kdl);
  Eigen::MatrixXd J = jac_kdl.data;

  // KDL Jacobian: rows 0..2 = angular (Jw), rows 3..5 = linear (Jv)
  Eigen::MatrixXd Jw = J.block(0, 0, 3, N); // 正确：角速度部分
  Eigen::MatrixXd Jv = J.block(3, 0, 3, N); // 正确：线速度部分

  // --- 注意：与 KDL 的重力约定对齐，使用 g_eff = -g_base ---
  Eigen::Vector3d g_base = gravity_; // 例如 [0,0,-9.81]
  Eigen::Vector3d g_eff = -g_base;   // 关键：翻一次符号

  // 单位基向量
  const Eigen::Vector3d e1(1, 0, 0), e2(0, 1, 0), e3(0, 0, 1);
  // 分解 rc×g = rcx*(e1×g) + rcy*(e2×g) + rcz*(e3×g)
  Eigen::Vector3d s1 = e1.cross(g_eff);
  Eigen::Vector3d s2 = e2.cross(g_eff);
  Eigen::Vector3d s3 = e3.cross(g_eff);

  // --- 构造回归矩阵 Phi (N x 4) ---
  Eigen::MatrixXd Phi(N, 4);
  Phi.col(0) = Jv.transpose() * g_eff; // 质量项列（对应 m）
  Phi.col(1) = Jw.transpose() * s1;    // m*rcx
  Phi.col(2) = Jw.transpose() * s2;    // m*rcy
  Phi.col(3) = Jw.transpose() * s3;    // m*rcz

  // AINFO << "[DEBUG] y: " << y.transpose();
  // AINFO << "[DEBUG] Phi_col0: " << Phi.col(0).transpose();

  // RLS 更新
  for (unsigned int i = 0; i < N; ++i) {
    Eigen::RowVector4d phi_i = Phi.row(i);
    double yi = y(i);

    double denom = lambda_ + (phi_i * P_ * phi_i.transpose())(0, 0);
    if (denom < 1e-12)
      denom = 1e-12;

    Eigen::Vector4d K = (P_ * phi_i.transpose()) / denom;
    double innovation = yi - (phi_i * theta_)(0, 0);

    theta_ += K * innovation;
    P_ = (P_ - K * phi_i * P_) / lambda_;

    // 数值安全检查
    for (int k = 0; k < 4; ++k) {
      if (!isValidNumber(theta_(k)))
        theta_(k) = 0.0;
    }
    if (theta_(0) < 0.0)
      theta_(0) = 0.0;
  }

  // 重力补偿
  VectorXd tau_g_payload = Phi * theta_;

  mass_rls = theta_(0);

  // AINFO << "mass_rls = " << theta_(0) << ", rc = " << theta_(1)/theta_(0) <<
  // ", "
  //       << theta_(2)/theta_(0) << ", " << theta_(3)/theta_(0);

  // 惯性补偿
  Vector3d rc(0, 0, 0);
  if (theta_(0) > 1e-6) {
    rc << theta_(1) / theta_(0), theta_(2) / theta_(0), theta_(3) / theta_(0);
  }

  auto Skew = [](const Vector3d &v) {
    Matrix3d S;
    S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return S;
  };

  Matrix3d Src = Skew(rc);
  MatrixXd Jc = Jv + Src * Jw;

  MatrixXd Mp = MatrixXd::Zero(N, N);
  if (theta_(0) > 1e-9) {
    Mp = theta_(0) * (Jc.transpose() * Jc);
  }

  VectorXd qdd(N);
  for (unsigned int i = 0; i < N; ++i)
    qdd(i) = ddq[i][2];
  VectorXd tau_inertia = Mp * qdd;

  // 总补偿 + 限幅 + 滤波
  tau_comp.resize(N);
  for (unsigned int i = 0; i < N; ++i) {
    double raw_val = tau_g_payload(i) + tau_inertia(i);
    double limited_val = saturate(raw_val, 200.0);                    // 限幅
    tau_comp[i] = lowPassFilter(tau_comp_prev_[i], limited_val, 0.9); // 滤波
    tau_comp_prev_[i] = tau_comp[i];
  }

  return true;
}
