#include "payload_estimator.h"

#include <common/log.h>

#include <Eigen/Dense>
#include <iostream>
#include <kdl_parser/kdl_parser.hpp>

namespace imeta {
namespace controller {

// === 工具函数 ===
namespace {
inline Eigen::Vector3d ToEigen(const KDL::Vector &v) {
  return Eigen::Vector3d(v.x(), v.y(), v.z());
}

inline Eigen::VectorXd MakeGravityWrench(const KDL::Vector &g) {
  Eigen::VectorXd wrench(6);
  wrench.setZero();
  wrench.segment<3>(0) = ToEigen(g);  // 只赋值前三维力，后三维力矩为 0
  return wrench;
}
}  // namespace

// === 构造与初始化 ===
PayloadEstimator::PayloadEstimator(const KDL::Chain &chain)
    : chain_(chain),
      gravity_(0.0, 0.0, -9.81),
      theta_(Eigen::Vector4d::Zero()),
      lambda_(0.98), P_(Eigen::Matrix4d::Identity() * 1000.0){
      jac_solver_ = std::make_shared<KDL::ChainJntToJacSolver>(chain_);
      dyn_solver_ = std::make_shared<KDL::ChainDynParam>(chain_, KDL::Vector(0, 0, -9.81));
      P_ = Eigen::MatrixXd::Identity(4, 4) * 1000.0;
      theta_.setZero();
      // gravity_ << 0, 0, -9.81;
}

void PayloadEstimator::Init(double lambda, double init_mass) {
  lambda_ = lambda;
  mass_ = init_mass;
  // P_ = 1000.0;
}

bool PayloadEstimator::ComputePayloadTorqueWithPID(
    const std::vector<double> &q, const std::vector<double> &dq,
    const std::vector<std::array<double, 3>> &target_traj_state,
    const std::vector<double> &tau_measured, std::vector<double> &tau_robot,
    const std::vector<double> &tau_rls, std::vector<double> &tau_comp,
    PIDMode mode) // 👈 模式选择
{
  const unsigned int N = q.size();
  if (dq.size() != N || target_traj_state.size() != N ||
      tau_measured.size() != N) {
    std::cerr << "ComputePayloadTorqueWithPID: input size mismatch" << std::endl;
    return false;
  }

  // --- 初始化缓存 ---
  if (pid_tau_comp_prev_.size() != N) pid_tau_comp_prev_.assign(N, 0.0);
  if (pid_tau_meas_med_.size() != N)  pid_tau_meas_med_.assign(N, 0.0);
  if (pid_tau_meas_filt_.size() != N) pid_tau_meas_filt_.assign(N, 0.0);
  if (pid_med_buf_.size() != N)       pid_med_buf_.assign(N, {0.0,0.0,0.0,0.0,0.0});

  // 两种模式分开存放积分和微分缓存
  if (pid_integral_torque_.size() != N) pid_integral_torque_.assign(N, 0.0);
  if (pid_prev_error_torque_.size() != N) pid_prev_error_torque_.assign(N, 0.0);
  if (pid_integral_state_.size() != N) pid_integral_state_.assign(N, 0.0);
  if (pid_prev_error_state_.size() != N) pid_prev_error_state_.assign(N, 0.0);

  // --- Step 1: 滤波 tau_measured ---
  const double alpha = ema_alpha_from_fc(lp_fc_, dt_);
  std::vector<double> tau_measured_filtered(N);
  for (unsigned int i = 0; i < N; ++i) {
    auto &buf = pid_med_buf_[i];
    buf[0]=buf[1]; buf[1]=buf[2]; buf[2]=buf[3]; buf[3]=buf[4];
    buf[4]=tau_measured[i];
    double med = median5(buf[0], buf[1], buf[2], buf[3], buf[4]);
    pid_tau_meas_med_[i] = med;
    pid_tau_meas_filt_[i] = (1.0 - alpha) * pid_tau_meas_filt_[i] + alpha * med;
    tau_measured_filtered[i] = pid_tau_meas_filt_[i];
  }

  tau_comp.resize(N);

  // --- Step 2: PID 参数 ---
  // TODO: 力矩PID调参
  // Torque tracking PID
  std::vector<double> Kp_tau = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double> Ki_tau = {0.05, 0.05, 0.05, 0.02, 0.0, 0.0};
  std::vector<double> Kd_tau = {0.5, 0.5, 0.3, 0.2, 0.1, 0.1};
  
  // TODO: 位置、速度PID调参
  // State tracking PID
  std::vector<double> Kp_state = {30.0, 30.0, 25.0, 20.0, 10.0, 10.0};
  std::vector<double> Ki_state = {0.1, 0.1, 0.1, 0.05, 0.0, 0.0};
  std::vector<double> Kd_state = {2.0, 2.0, 1.5, 1.0, 0.5, 0.5};

  const double tau_limit = 10.0;      // 力矩限幅
  const double tau_rate_limit = 0.1;  // 限速

  auto slewRateLimit = [&](double prev, double target, double max_delta) {
    double delta = std::clamp(target - prev, -max_delta, +max_delta);
    return prev + delta;
  };

  // --- Step 3: 根据模式计算 ---
  for (unsigned int i = 0; i < N; ++i) {
    double u = 0.0;

    if (mode == PIDMode::TORQUE_TRACKING) {
      // --- Torque Tracking PID ---
      // double e_tau = tau_measured_filtered[i] - tau_robot[i] - tau_rls[i];
      double e_tau = tau_measured_filtered[i] - tau_robot[i];
      pid_integral_torque_[i] += e_tau * dt_;
      double derivative = (e_tau - pid_prev_error_torque_[i]) / dt_;
      u = Kp_tau[i] * e_tau + Ki_tau[i] * pid_integral_torque_[i] + Kd_tau[i] * derivative;
      pid_prev_error_torque_[i] = e_tau;
    }
    else if (mode == PIDMode::STATE_TRACKING) {
      // --- State Tracking PID ---
      double q_des = target_traj_state[i][0];
      double dq_des = target_traj_state[i][1];

      double e_q = q_des - q[i];
      double e_dq = dq_des - dq[i];

      pid_integral_state_[i] += e_q * dt_;
      double derivative = (e_q - pid_prev_error_state_[i]) / dt_;
      u = Kp_state[i] * e_q + Ki_state[i] * pid_integral_state_[i] + Kd_state[i] * e_dq;
      pid_prev_error_state_[i] = e_q;
    }

    // 限幅 + 限速
    double limited_val = std::clamp(u, -tau_limit, tau_limit);
    tau_comp[i] = slewRateLimit(pid_tau_comp_prev_[i], limited_val, tau_rate_limit);
    pid_tau_comp_prev_[i] = tau_comp[i];

    // 末端关节屏蔽补偿
    if (i == 4 || i == 5) {
      tau_comp[i] = 0.0;
      pid_tau_comp_prev_[i] = 0.0;
    }
  }

  // mass_rls = 0.0;
  return true;
}

bool PayloadEstimator::UpdateMassRLS(
    const std::vector<double> &q, const std::vector<double> &dq,
    const std::vector<std::array<double, 3>> &target_traj_state,
    const std::vector<double> &tau_measured, std::vector<double> &tau_robot,
    std::vector<double> &tau_comp, double &mass_rls) {
    const unsigned int N = q.size();
    if (dq.size() != N || target_traj_state.size() != N ||
        tau_measured.size() != N || tau_robot.size() != N) {
      std::cerr << "UpdateMassRLS: input size mismatch" << std::endl;
      AINFO << "dq_size = " << dq.size()
            << ", target_traj_state_size = " << target_traj_state.size()
            << ", tau_measured_size = " << tau_measured.size() << std::endl;
      return false;
    }
    // 初始化滤波缓存
    if (rls_tau_comp_prev_.size() != N)
      rls_tau_comp_prev_.assign(N, 0.0);
    if (rls_tau_meas_med_.size() != N)
      rls_tau_meas_med_.assign(N, 0.0);
    if (rls_tau_meas_filt_.size() != N)
      rls_tau_meas_filt_.assign(N, 0.0);
    if (rls_med_buf_.size() != N)
      rls_med_buf_.assign(N, {0.0, 0.0, 0.0, 0.0, 0.0});

    // 2.5) 对 tau_measured 进行两级滤波：median(5) -> EMA 低通
    const double alpha = ema_alpha_from_fc(lp_fc_, dt_);
    std::vector<double> tau_measured_filtered(N);

    for (unsigned int i = 0; i < N; ++i) {
      // --- Median 部分 ---
      // 注意这里是auto &的形式，是可以修改med_buf_的原始值的
      auto &buf = rls_med_buf_[i]; // 每个关节一个长度为 5 的环形/滑窗 buffer
      buf[0] = buf[1];
      buf[1] = buf[2];
      buf[2] = buf[3];
      buf[3] = buf[4];
      buf[4] = tau_measured[i]; // 新输入值放到末尾

      double med = median5(buf[0], buf[1], buf[2], buf[3], buf[4]);
      rls_tau_meas_med_[i] = med; // 中值输出（去尖峰）

      // --- EMA 部分 ---
      rls_tau_meas_filt_[i] = (1.0 - alpha) * rls_tau_meas_filt_[i] + alpha * med;

      // 最终滤波结果
      tau_measured_filtered[i] = rls_tau_meas_filt_[i];
    }
    // tau_robot.resize(N);
    tau_comp.resize(N);

    // 1) KDL 转换（使用测量值）
    KDL::JntArray q_kdl(N), dq_kdl(N), ddq_kdl(N);
    for (unsigned int i = 0; i < N; ++i) {
      // TODO: 尝试使用期望的位置、速度、加速度
      // 这里主要是计算雅可比矩阵会用到位置 q
      // q_kdl(i) = target_traj_state[i][0];
      // dq_kdl(i) = target_traj_state[i][1];
      // ddq_kdl(i) = target_traj_state[i][2]; // ddq[i][2] 对应关节加速度
      q_kdl(i) = q[i];
      dq_kdl(i) = dq[i];
      ddq_kdl(i) = target_traj_state[i][2]; // ddq[i][2] 对应关节加速度
    }

    // 2) 计算机器人标称力矩（不含负载）
    Eigen::VectorXd qdd(N);
    for (unsigned int i = 0; i < N; ++i)
      qdd(i) = target_traj_state[i][2];

    Eigen::VectorXd tau_nominal(N);
    for (unsigned int i = 0; i < N; ++i) {
      // tau_nominal(i) = G(i);
      tau_nominal(i) = tau_robot[i];

      // tau_nominal(i) = H(i, i) * qdd(i) + C(i) + G(i);  // 简化近似
    }

    // 3) 残差（负载引起的部分）
    Eigen::VectorXd y(N);
    for (unsigned int i = 0; i < N; ++i) {
      // y(i) = tau_measured[i] + tau_comp_prev_[i] - tau_nominal(i);
      y(i) = tau_measured[i] - tau_nominal(i);
    }

    // 4) 构造 Phi
    KDL::Jacobian jac_kdl(N);
    jac_solver_->JntToJac(q_kdl, jac_kdl);
    Eigen::MatrixXd J = jac_kdl.data;
    Eigen::MatrixXd Jv = J.block(0, 0, 3, N); // linear
    Eigen::MatrixXd Jw = J.block(3, 0, 3, N); // angular

    Eigen::Vector3d g_eff = -gravity_; // 已经校准好的重力方向
    const Eigen::Vector3d e1(1, 0, 0), e2(0, 1, 0), e3(0, 0, 1);
    Eigen::Vector3d s1 = e1.cross(g_eff);
    Eigen::Vector3d s2 = e2.cross(g_eff);
    Eigen::Vector3d s3 = e3.cross(g_eff);

    Eigen::MatrixXd Phi(N, 4);
    Phi.col(0) = Jv.transpose() * g_eff; // 质量项
    Phi.col(1) = Jw.transpose() * s1;    // m*rcx
    Phi.col(2) = Jw.transpose() * s2;    // m*rcy
    Phi.col(3) = Jw.transpose() * s3;    // m*rcz

    // 5) RLS 更新（仅近静态关节更新）
    const double dq_th = 2.0;  // rad/s
    const double ddq_th = 5.0; // rad/s^2
    // const double R_n = 1e-3;    // 小量防除零
    double R_n = 1e-6;
    double eps = 1e-6;

    for (unsigned int i = 0; i < N; ++i) {
      Eigen::RowVector4d phi_i = Phi.row(i);
      double y_i = y(i);
      RLSUpdate(phi_i, y_i, theta_, P_, lambda_, R_n, eps);
    }

    // 6) 边界约束
    theta_(0) = std::clamp(theta_(0), 0.0, 10.0); // m_max=10kg，可按实际改
    for (int k = 1; k < 4; ++k)
      theta_(k) =
          std::clamp(theta_(k), -0.3 * theta_(0), 0.3 * theta_(0)); // rc限制

    mass_rls = theta_(0);

    AINFO << "mass_rls: " << mass_rls;

    // 7) 重力补偿
    Eigen::VectorXd tau_g_payload = Phi * theta_;

    // 8) 惯性补偿
    Eigen::Vector3d rc(0, 0, 0);
    if (theta_(0) > 1e-6)
      rc << theta_(1) / theta_(0), theta_(2) / theta_(0), theta_(3) / theta_(0);

    // AINFO << "rc: " << rc(0) << " " << rc(1) << " " << rc(2);

    auto Skew = [](const Eigen::Vector3d &v) {
      Eigen::Matrix3d S;
      S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
      return S;
    };

    Eigen::MatrixXd Jc = Jv + Skew(rc) * Jw;
    Eigen::MatrixXd Mp = theta_(0) * (Jc.transpose() * Jc);
    Eigen::VectorXd tau_inertia = Mp * qdd;

    // 9) 输出总补偿 + 限幅 + 限斜率
    const double tau_limit = 10.0;
    const double tau_rate_limit = 0.1; // 每周期最大变化量
    auto slewRateLimit = [&](double prev, double target, double max_delta) {
      double delta = std::clamp(target - prev, -max_delta, +max_delta);
      return prev + delta;
    };

    // 单个关节补偿力矩调节参数
    // std::vector<double> k_tau_comp = {0.5, 0.5, 0.9, 0.9, 1.0, 1.0};
    for (unsigned int i = 0; i < N; ++i) {
      // double raw_val = tau_g_payload(i) + tau_inertia(i);
      double raw_val = tau_g_payload(i);
      double limited_val = std::clamp(raw_val, -tau_limit, tau_limit);

      tau_comp[i] = slewRateLimit(rls_tau_comp_prev_[i],
                                                  limited_val, tau_rate_limit);

      rls_tau_comp_prev_[i] = tau_comp[i];

      if (i == 4 || i == 5) {
        tau_comp[i] = 0.0001;
        rls_tau_comp_prev_[i] = tau_comp[i];
      }

      AINFO << "tau_comp[" << i << "]: " << tau_comp[i];
    }

    return true;
  }


void PayloadEstimator::RLSUpdate(const Eigen::RowVector4d &phi,  // 回归向量
                                 double y,                       // 观测输出
                                 Eigen::Vector4d &theta,         // 参数估计
                                 Eigen::Matrix4d &P,             // 协方差矩阵
                                 double lambda,                  // 遗忘因子
                                 double Rn,                      // 观测噪声协方差
                                 double eps                      // 数值阈值
) {
  lambda = 0.99;
  Rn = 1e-6;
  eps = 1e-6;

  // ---------- 1. 检查 phi 是否有效 ----------
  double norm_phi = phi.norm();
  if (norm_phi < eps) {
    // phi 过小，跳过本次更新
    return;
  }

  // ---------- 2. 计算 denom ----------
  double denom = lambda + (phi * P * phi.transpose())(0, 0) + Rn;
  if (!(denom > eps) || std::isnan(denom)) {
    // denom 太小或 NaN，跳过更新
    return;
  }

  // ---------- 3. 计算增益 ----------
  Eigen::VectorXd K = (P * phi.transpose()) / denom;

  // 对 K 范数加上限，避免爆增益
  double K_max = 100.0;  // 可以根据实际调小，比如 10~100
  double K_norm = K.norm();
  if (K_norm > K_max) {
    K *= (K_max / K_norm);
  }

  // ---------- 4. 计算创新 ----------
  double innovation = y - (phi * theta)(0, 0);

  // 可选：Huber 抑制异常创新
  double c = 3.0;  // Huber 阈值，可调
  double scale = std::sqrt(denom);
  if (std::abs(innovation) > c * scale) {
    innovation = c * scale * (innovation > 0 ? 1.0 : -1.0);
  }

  // ---------- 5. 参数更新 ----------
  theta += K * innovation;

  // 可选：强制质量非负
  if (theta(0) < 0.0) theta(0) = 0.0;

  // 可选：冻结 rz，避免不可观测方向发散
  // theta(3) = 0.0;

  // ---------- 6. 协方差更新 ----------
  P = (P - K * phi * P) / lambda;

  // ---------- 7. 强制 P 对称正定 ----------
  P = 0.5 * (P + P.transpose());  // 强制对称
  for (int i = 0; i < P.rows(); ++i) {
    if (P(i, i) < eps) P(i, i) = eps;  // 对角线小于阈值则修正
  }

  // 限制 P 的最大值，避免数值爆炸
  double maxP = 1e6;
  P = P.cwiseMin(maxP);
}


// === 静态雅可比质量估计（静止情况下）===
// double
// PayloadEstimator::EstimateMassJacobian(const KDL::JntArray &q,
//                                        const KDL::JntArray &tau,
//                                        const KDL::JntArray &gravity_tau) const {
//   // KDL::JntArray gravity(q.rows());
//   // dyn_solver_->JntToGravity(q, gravity);

//   KDL::Jacobian jacobian(q.rows());
//   jac_solver_->JntToJac(q, jacobian);

//   Eigen::VectorXd tau_ext(q.rows());
//   for (unsigned int i = 0; i < q.rows(); ++i) {
//     tau_ext(i) = tau(i) - gravity_tau(i);
//   }

//   Eigen::VectorXd phi = jacobian.data.transpose() * MakeGravityWrench(gravity_);
//   double numerator = tau_ext.transpose() * phi;

//   double denominator = gravity_.x() * gravity_.x() +
//                        gravity_.y() * gravity_.y() +
//                        gravity_.z() * gravity_.z();

//   return numerator / denominator;
// }

// === 获取当前估计质量 ===
double PayloadEstimator::GetEstimatedMass() const { return mass_; }

// // === 计算补偿力矩（用于前馈）===
// bool PayloadEstimator::ComputePayloadCompensation(
//     const KDL::JntArray &q, KDL::JntArray &tau_comp) const {
//   if (q.rows() != chain_.getNrOfJoints()) {
//     return false;
//   }

//   KDL::Jacobian jacobian(q.rows());
//   if (jac_solver_->JntToJac(q, jacobian) < 0) {
//     return false;
//   }

//   // 先试用基于雅克比矩阵的负载估算方法
//   //   double mass_jac = EstimateMassJacobian(q, tau_);
//   Eigen::VectorXd Fg = MakeGravityWrench(gravity_) * mass_;
//   Eigen::VectorXd tau_eigen = jacobian.data.transpose() * Fg;

//   tau_comp.resize(q.rows());
//   for (unsigned int i = 0; i < q.rows(); ++i) {
//     tau_comp(i) = tau_eigen(i);
//   }

//   return true;
// }

// 适配当前算法的函数接口
// bool PayloadEstimator::ComputePayloadCompensation(
//     const std::vector<double> &q, const std::vector<double> &cur_joint_tau,
//     const std::vector<double> &gravity_torque_kdl_,
//     std::vector<double> &tau_comp) const {
//   // 检查输入维度
//   if (q.size() != chain_.getNrOfJoints()) {
//     return false;
//   }

//   // 将 std::vector 转换为 KDL::JntArray
//   KDL::JntArray q_kdl(q.size());
//   for (size_t i = 0; i < q.size(); ++i) {
//     q_kdl(i) = q[i];
//   }

//   // 计算雅可比矩阵
//   KDL::Jacobian jacobian(q.size());
//   if (jac_solver_->JntToJac(q_kdl, jacobian) < 0) {
//     return false;
//   }

//   // TODO:先使用雅可比矩阵估算负载的方法
//   // 将 std::vector 转换为 KDL::JntArray
//   KDL::JntArray tau_kdl(cur_joint_tau.size());
//   for (size_t i = 0; i < cur_joint_tau.size(); ++i) {
//     tau_kdl(i) = cur_joint_tau[i];
//   }
//   KDL::JntArray gravity_kdl(gravity_torque_kdl_.size());
//   for (size_t i = 0; i < gravity_torque_kdl_.size(); ++i) {
//     gravity_kdl(i) = gravity_torque_kdl_[i];
//   }

//   double mass_jac = EstimateMassJacobian(q_kdl, tau_kdl, gravity_kdl);
//   // debug
//   AINFO << "mass_jac: " << mass_jac;

//   // 构造重力力矩
//   Eigen::VectorXd Fg = MakeGravityWrench(gravity_) * mass_jac;
//   Eigen::VectorXd tau_eigen = jacobian.data.transpose() * Fg;

//   // 写入输出
//   tau_comp.resize(q.size());
//   for (size_t i = 0; i < q.size(); ++i) {
//     tau_comp[i] = tau_eigen(i);
//   }

//   return true;
// }

}  // namespace controller
}  // namespace imeta
