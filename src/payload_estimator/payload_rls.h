#pragma once
#include <Eigen/Dense>
#include <KDL/chain.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/chainidsolver_vereshchagin.hpp>
#include <kdl/chaindynparam.hpp>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

// ==============================================
//  载荷（质量/质心）RLS + 补偿 力矩 计算器（稳健版）
//  参数向量: theta = [ m, m*rc_x, m*rc_y, m*rc_z ]^T
//  回归矩阵: Phi (N x 4), 行对应各关节，由雅可比与重力向量构造
//  更新：Joseph 形式 + 遗忘因子 + 参数投影 + 离群抑制 + 条件数守卫
//  输出：tau_comp = tau_g_payload + tau_inertia
// ==============================================

struct PayloadRLSOptions {
  // 遗忘因子 λ ∈ (0,1]；越小越快忘，越大越稳。实机建议 0.995~0.999
  double lambda = 0.997;

  // 协方差初值 P0 = p0 * I，越大代表先验越不确定；建议 1e4~1e6
  double p0 = 1e5;

  // 观测噪声方差 R，作为 Joseph 形式里的标量（各关节相同），建议 1e-2~1e0
  double meas_var_R = 1e-1;

  // 质量与质心范围（物理约束/投影）
  double mass_min = 0.0;      // 质量非负
  double mass_max = 200.0;    // 结合实际末端最大可能载荷设置
  double rc_abs_max = 0.5;    // 质心坐标最大绝对值（m），按末端安装结构设定

  // 离群点抑制（Huber）阈值（标准化残差）
  double huber_k = 2.5;

  // 条件数与数值守卫
  double cond_max = 1e8;      // P 条件数上限，超过则触发重置
  double denom_floor = 1e-9;  // RLS 增益分母下限
  double eps_safe = 1e-12;    // 一般数值保护 eps

  // 输出补偿的限幅与一阶滤波（避免突变）
  double tau_limit = 200.0;   // 每个关节补偿力矩绝对值上限
  double tau_alpha = 0.2;     // 一阶 IIR 滤波系数（0~1，越小越平滑）

  // 异常自动重置：theta 或 P 进入 nan/inf 时，是否重置
  bool auto_reset_on_nan = true;
};

class PayloadRLS {
public:
  // 构造：传入 KDL 动力学/雅可比求解器（外部持有/管理生命周期）
  PayloadRLS(KDL::ChainDynParam* dyn_solver,
             KDL::ChainJntToJacSolver* jac_solver,
             const Eigen::Vector3d& gravity_vec_in_base,
             const PayloadRLSOptions& opt = PayloadRLSOptions())
  : dyn_solver_(dyn_solver),
    jac_solver_(jac_solver),
    g_base_(gravity_vec_in_base),
    opt_(opt) {
    // 参数向量 4x1 初始化为 0（未知载荷）
    theta_.setZero(4);

    // 协方差 P 初始化为 p0 * I
    P_.setIdentity(4,4);
    P_ *= opt_.p0;

    // 噪声方差 R（标量）
    R_ = opt_.meas_var_R;

    // 观测噪声的滑动估计（用于标准化残差），初始化为 R
    resid_var_ = opt_.meas_var_R;

    // 滤波输出初始化为空
    last_tau_comp_.resize(0);
  }

  // 主入口：输入 q, dq, ddq(用第三个分量或直接给标量)，测量力矩 tau_measured
  // 输出：稳健的补偿力矩 tau_comp
  // 返回 false 表示输入维度错误或出现致命异常且自动重置失败
  bool Update(const std::vector<double>& q,
              const std::vector<double>& dq,
              const std::vector<std::array<double,3>>& ddq, // 使用 ddq[i][2]
              const std::vector<double>& tau_measured,
              std::vector<double>& tau_comp) {
    // ------- 尺寸检查 -------
    const unsigned int N = q.size();
    if (dq.size() != N || ddq.size() != N || tau_measured.size() != N) {
      // 尺寸不匹配直接返回
      return false;
    }

    // ------- 转 KDL 容器 -------
    KDL::JntArray q_kdl(N), dq_kdl(N), ddq_kdl(N);
    for (unsigned int i = 0; i < N; ++i) {
      q_kdl(i)   = q[i];
      dq_kdl(i)  = dq[i];
      ddq_kdl(i) = ddq[i][2];  // 你当前数据源定义如此
    }

    // ------- 动力学项（重力/科氏/质量矩阵） -------
    KDL::JntArray coriolis(N), gravity(N);
    KDL::JntSpaceInertiaMatrix H(N);
    dyn_solver_->JntToCoriolis(q_kdl, dq_kdl, coriolis);
    dyn_solver_->JntToGravity(q_kdl, gravity);
    dyn_solver_->JntToMass(q_kdl, H);

    // ------- 模型项 tau_model = H * ddq + C + G -------
    Eigen::VectorXd tau_model(N); tau_model.setZero();
    for (unsigned int i = 0; i < N; ++i) {
      double inertia_term = 0.0;
      for (unsigned int j = 0; j < N; ++j) {
        inertia_term += H(i,j) * ddq_kdl(j);
      }
      // 注意：你原代码遗漏了 H*ddq 与 coriolis
      tau_model(i) = inertia_term + coriolis(i) + gravity(i);
    }

    // ------- 观测残差 y = tau_meas - tau_model -------
    Eigen::VectorXd y(N);
    for (unsigned int i = 0; i < N; ++i) y(i) = tau_measured[i] - tau_model(i);

    // ------- 雅可比一次性计算 -------
    KDL::Jacobian J_kdl(N); J_kdl.resize(N);
    jac_solver_->JntToJac(q_kdl, J_kdl);
    Eigen::MatrixXd J = J_kdl.data;      // 6 x N
    Eigen::MatrixXd Jv = J.block(0,0,3,N); // 线速度部分
    Eigen::MatrixXd Jw = J.block(3,0,3,N); // 角速度部分

    // ------- 构造 Phi (N x 4) -------
    // g_base_ 必须为 {0,0,-|g|} 或与机器人基坐标一致的重力方向
    Eigen::Vector3d g = g_base_;
    // s1,s2,s3 = e1×g, e2×g, e3×g
    const Eigen::Vector3d e1(1,0,0), e2(0,1,0), e3(0,0,1);
    const Eigen::Vector3d s1 = e1.cross(g);
    const Eigen::Vector3d s2 = e2.cross(g);
    const Eigen::Vector3d s3 = e3.cross(g);

    Eigen::MatrixXd Phi(N,4);
    // Phi.col(0) = Jv^T * g  （质量项对应重力力）
    Phi.col(0) = Jv.transpose() * g;
    // Phi.col(1:3) = Jw^T * s{1,2,3} （质心力矩项）
    Phi.col(1) = Jw.transpose() * s1;
    Phi.col(2) = Jw.transpose() * s2;
    Phi.col(3) = Jw.transpose() * s3;

    // ------- RLS（对每个关节观测逐行更新） -------
    // Joseph 形式：P = (I - Kφ) P (I - Kφ)^T + K R K^T
    // Huber 损失：对离群残差降低增益（相当于增大 R）
    Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();

    // 在更新前做一次健康检查与必要时重置
    if (!checkAndMaybeReset_()) return false;

    for (unsigned int i = 0; i < N; ++i) {
      Eigen::RowVector4d phi = Phi.row(i);                    // 1x4
      double yi = y(i);                                       // 标量

      // 预测输出
      double yhat = (phi * theta_)(0);

      // 残差与标准化
      double innov = yi - yhat;
      double rinv  = std::max(opt_.eps_safe, resid_var_);
      double z     = innov / std::sqrt(rinv);                 // 标准化残差

      // Huber 权：|z|<=k -> 1；否则 ~ k/|z|
      double w = 1.0;
      if (std::abs(z) > opt_.huber_k) {
        w = opt_.huber_k / (std::abs(z) + opt_.eps_safe);
      }

      // 有效测量方差 R_eff = R / w^2（权越小，相当于噪声越大）
      double R_eff = R_ / (w*w);

      // 增益分母 denom = λ + φ P φ^T + R_eff
      double denom = opt_.lambda + (phi * P_ * phi.transpose())(0,0) + R_eff;
      if (!(denom > opt_.denom_floor)) denom = opt_.denom_floor;

      // 卡尔曼增益 K = P φ^T / denom
      Eigen::Vector4d K = (P_ * phi.transpose()) / denom;

      // 参数更新
      theta_ = theta_ + K * innov;

      // 协方差 Joseph 形式（数值更稳）
      Eigen::Matrix4d I_Kphi = (I4 - K * phi);
      P_ = I_Kphi * P_ * I_Kphi.transpose() + K * R_eff * K.transpose();

      // 遗忘因子（等效于协方差放大）
      P_ /= opt_.lambda;

      // 残差方差更新（简单 EWMA）
      resid_var_ = 0.98 * resid_var_ + 0.02 * (innov*innov + R_eff);

      // 参数投影（物理可行域）
      projectTheta_();
    }

    // ------- 负载重力补偿 tau_g_payload = Phi * theta -------
    Eigen::VectorXd tau_g_payload = Phi * theta_;

    // ------- 惯性补偿 tau_inertia = Mp * qdd -------
    // rc = theta(1:3)/m（若 m ~ 0 则置 0）
    Eigen::Vector3d rc(0,0,0);
    double m = std::max(opt_.eps_safe, theta_(0));
    if (m > 1e-6) {
      rc << theta_(1)/m, theta_(2)/m, theta_(3)/m;
    }

    // S(rc) 反对称矩阵
    auto Skew = [](const Eigen::Vector3d& v){
      Eigen::Matrix3d S;
      S << 0.0, -v.z(), v.y(),
           v.z(), 0.0, -v.x(),
          -v.y(), v.x(), 0.0;
      return S;
    };
    Eigen::Matrix3d Src = Skew(rc);

    // Jc = Jv + S(rc) * Jw
    Eigen::MatrixXd Jc = Jv + Src * Jw;

    // Mp = m * (Jc^T Jc)
    Eigen::MatrixXd Mp = Eigen::MatrixXd::Zero(N,N);
    if (m > 1e-9) {
      Mp = m * (Jc.transpose() * Jc);
    }

    // qdd
    Eigen::VectorXd qdd(N);
    for (unsigned int i = 0; i < N; ++i) qdd(i) = ddq[i][2];

    // 惯性项
    Eigen::VectorXd tau_inertia = Mp * qdd;

    // ------- 总补偿（限幅 + 一阶滤波）-------
    Eigen::VectorXd tau_total = tau_g_payload + tau_inertia;

    // 限幅
    for (unsigned int i = 0; i < N; ++i) {
      double v = tau_total(i);
      if (v >  opt_.tau_limit) v =  opt_.tau_limit;
      if (v < -opt_.tau_limit) v = -opt_.tau_limit;
      tau_total(i) = v;
    }

    // 一阶滤波避免突变
    if (last_tau_comp_.size() != N) {
      last_tau_comp_.assign(N, 0.0);
    }
    for (unsigned int i = 0; i < N; ++i) {
      double filtered = opt_.tau_alpha * tau_total(i) + (1.0 - opt_.tau_alpha) * last_tau_comp_[i];
      last_tau_comp_[i] = filtered;
    }

    // 输出
    tau_comp.resize(N);
    for (unsigned int i = 0; i < N; ++i) tau_comp[i] = last_tau_comp_[i];

    // 最终健康检查
    if (!checkAndMaybeReset_()) {
      return false;
    }
    return true;
  }

  // 读取估计的质量与质心（以 m 与 rc 输出）
  void GetMassAndCOM(double& m_out, Eigen::Vector3d& rc_out) const {
    double m = std::max(opt_.eps_safe, theta_(0));
    m_out = m;
    if (m > 1e-6) {
      rc_out = Eigen::Vector3d(theta_(1)/m, theta_(2)/m, theta_(3)/m);
    } else {
      rc_out.setZero();
    }
  }

  // 手动重置估计
  void Reset() {
    theta_.setZero(4);
    P_.setIdentity(4,4);
    P_ *= opt_.p0;
    resid_var_ = opt_.meas_var_R;
    last_tau_comp_.clear();
  }

  // 调参接口
  void SetOptions(const PayloadRLSOptions& opt) { opt_ = opt; }

private:
  // 将参数投影到可行域（m ∈ [mass_min, mass_max]；|rc_i| ≤ rc_abs_max）
  void projectTheta_() {
    // 质量投影
    if (std::isnan(theta_(0)) || std::isinf(theta_(0))) theta_(0) = opt_.mass_min;
    theta_(0) = std::clamp(theta_(0), opt_.mass_min, opt_.mass_max);

    // 质心限幅：通过限制 m*rc_i 的幅值来实现
    double m = std::max(opt_.eps_safe, theta_(0));
    double max_mrc = m * opt_.rc_abs_max;
    for (int i = 1; i < 4; ++i) {
      if (std::isnan(theta_(i)) || std::isinf(theta_(i))) theta_(i) = 0.0;
      theta_(i) = std::clamp(theta_(i), -max_mrc, max_mrc);
    }
  }

  // 健康检查与必要的自动重置（NaN/Inf 或条件数过大）
  bool checkAndMaybeReset_() {
    // NaN/Inf 检查
    auto invalid = [](double v){ return !(v==v) || std::isinf(v); };
    bool bad = false;
    for (int r=0; r<4; ++r) {
      if (invalid(theta_(r))) { bad = true; break; }
      for (int c=0; c<4; ++c) {
        if (invalid(P_(r,c))) { bad = true; break; }
      }
    }
    if (bad && opt_.auto_reset_on_nan) {
      Reset();
      return true; // 重置后继续
    } else if (bad) {
      return false; // 禁止自动重置时直接失败
    }

    // 条件数估计（对称正定用特征值比）
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(P_);
    if (es.info() != Eigen::Success) {
      if (opt_.auto_reset_on_nan) { Reset(); return true; }
      return false;
    }
    double lam_min = std::max(opt_.eps_safe, es.eigenvalues()(0));
    double lam_max = std::max(opt_.eps_safe, es.eigenvalues()(3));
    double cond = lam_max / lam_min;
    if (cond > opt_.cond_max) {
      // 轻度“降火”：P <- P + δI
      P_ += 1e-6 * Eigen::Matrix4d::Identity();
      // 仍过大则重置
      if (opt_.auto_reset_on_nan) {
        Reset();
        return true;
      }
      return false;
    }
    return true;
  }

private:
  // 外部提供的 KDL 求解器
  KDL::ChainDynParam* dyn_solver_ = nullptr;
  KDL::ChainJntToJacSolver* jac_solver_ = nullptr;

  // 基坐标系下的重力向量（务必保证方向与数值正确，如 [0,0,-9.81]）
  Eigen::Vector3d g_base_;

  // 估计量
  Eigen::Vector4d theta_;     // [m, m*rcx, m*rcy, m*rcz]
  Eigen::Matrix4d P_;         // 协方差
  double R_ = 1e-1;           // 观测噪声方差（标量）
  double resid_var_ = 1e-1;   // 残差方差滑动估计

  // 输出滤波状态
  std::vector<double> last_tau_comp_;

  // 选项
  PayloadRLSOptions opt_;
};

