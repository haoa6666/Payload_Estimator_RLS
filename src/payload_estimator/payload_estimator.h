#pragma once

#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/jntarray.hpp>
#include <memory>

namespace imeta {
namespace controller {

enum class PIDMode { TORQUE_TRACKING, STATE_TRACKING };

class PayloadEstimator {
public:
  explicit PayloadEstimator(const KDL::Chain &chain);

  // 初始化估计器（可调遗忘因子与初始质量）
  void Init(double lambda = 0.98, double init_mass = 1.0);

  bool ComputePayloadTorqueWithPID(
      const std::vector<double> &q, const std::vector<double> &dq,
      const std::vector<std::array<double, 3>> &target_traj_state,
      const std::vector<double> &tau_measured, std::vector<double> &tau_robot,
      const std::vector<double> &tau_rls, std::vector<double> &tau_comp,
      PIDMode mode);

  // 使用反演动力学 + RLS 更新估计的质量
  bool
  UpdateMassRLS(const std::vector<double> &q, const std::vector<double> &dq,
                const std::vector<std::array<double, 3>> &target_traj_state,
                const std::vector<double> &tau_measured,
                 std::vector<double> &tau_robot, std::vector<double> &tau_comp,
                double &mass_rls);

  void RLSUpdate(const Eigen::RowVector4d &phi, // 回归向量
                 double y,                      // 观测输出
                 Eigen::Vector4d &theta,        // 参数估计
                 Eigen::Matrix4d &P,            // 协方差矩阵
                 double lambda,                 // 遗忘因子
                 double Rn,                     // 观测噪声协方差
                 double eps                     // 数值阈值
  );

  // 使用雅可比矩阵估计质量（静止工况）
  double EstimateMassJacobian(const KDL::JntArray &q, const KDL::JntArray &tau,
                              const KDL::JntArray &gravity_tau) const;

  // 获取当前估计质量（单位：kg）
  double GetEstimatedMass() const;

  // 返回末端负载质量引起的补偿力矩
  bool ComputePayloadCompensation(const KDL::JntArray &q,
                                  KDL::JntArray &tau_comp) const;
  //   适配参数接口
  bool
  ComputePayloadCompensation(const std::vector<double> &q,
                             const std::vector<double> &cur_joint_tau,
                             const std::vector<double> &gravity_torque_kdl_,
                             std::vector<double> &tau_comp) const;

private:
  KDL::Chain chain_;
  double mass_; // 当前估计的质量值

  std::shared_ptr<KDL::ChainDynParam> dyn_solver_;
  std::shared_ptr<KDL::ChainJntToJacSolver> jac_solver_;

  // RLS 参数
  double lambda_;         // RLS遗忘因子，例如 0.995
  Eigen::Matrix4d P_;     // 参数协方差矩阵
  Eigen::Vector4d theta_; // 质量 + 质心估计  [ m, m*rx, m*ry, m*rz ]

  // 重力方向
  Eigen::Vector3d gravity_;

  // 滤波缓存
  std::vector<double> tau_comp_prev_;

  KDL::Vector gravity_kdl; // 重力方向向量，默认 (0, 0, -9.81)

  // 过滤器状态
  // RLS
  std::vector<double> tau_meas_med_;  // 中值滤波输出
  std::vector<double> tau_meas_filt_; // 低通后输出
  // 位置、速度PID
  std::vector<double> pid_integral_torque_;   // 积分缓存
  std::vector<double> pid_prev_error_torque_; // 微分缓存
                                              // 力矩PID 积分、微分缓存
  std::vector<double> pid_integral_state_;    // 中值滤波输出
  std::vector<double> pid_prev_error_state_;  // 低通后输出
  std::vector<std::array<double, 5>> med_buf_; // 每关节5点窗口
  double lp_fc_ = 1.0; // 低通截止频率 [Hz]，按采样率调
  double dt_ = 0.005;  // 采样周期 [s]，100Hz示例
};

inline double ema_alpha_from_fc(double fc, double dt) {
  // alpha = 1 - exp(-2πfc dt)
  double a = 1.0 - std::exp(-2.0 * M_PI * fc * dt);
  // 数值保护
  if (a < 1e-6)
    a = 1e-6;
  if (a > 1.0)
    a = 1.0;
  return a;
}

inline double median5(double a, double b, double c, double d, double e) {
  // 无分配、分支有限的 5 点中位数（小实现亦可用 std::array + nth_element）
  double x[5] = {a, b, c, d, e};
  std::nth_element(x, x + 2, x + 5);
  return x[2];
}

} // namespace controller
} // namespace imeta
