"""第三章实验：保险破产概率"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
from scipy.stats import expon
from surplus_model import (
    SurplusProcess, find_adjustment_R, exp_claim_mgf_factory,
    lundberg_psi_exact, psi_exact_exponential,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def run_R_experiment(lam=1.0, mu=1.0, theta_values=(0.1, 0.2, 0.5)):
    """计算不同安全负荷下的调整系数 R。"""
    claim_dist = expon(scale=mu)
    mgf = exp_claim_mgf_factory('exponential', rate=1/mu)

    rows = []
    for theta in theta_values:
        expected_claim = lam * mu  # λ E[Y]
        c = expected_claim * (1 + theta)  # 安全负荷
        R = find_adjustment_R(lam, c, mgf)
        rows.append({'theta': theta, 'c': c, 'R': R, 'lam': lam, 'mu': mu})

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / 'exp3_R.csv', index=False)
    return df


def run_ruin_prob_experiment(lam=1.0, mu=1.0, theta=0.2,
                              u_values=np.arange(0, 21),
                              n_paths=10000, T=100.0, seed=42):
    """Monte Carlo 估计不同初始资本下的破产概率。"""
    expected_claim = lam * mu
    c = expected_claim * (1 + theta)

    claim_dist = expon(scale=mu)
    mgf = exp_claim_mgf_factory('exponential', rate=1/mu)
    R = find_adjustment_R(lam, c, mgf)

    rng = np.random.default_rng(seed)
    rows = []
    for u in u_values:
        proc = SurplusProcess(u, c, lam, claim_dist)
        ruins = 0
        for _ in range(n_paths):
            ruined, _, _, _ = proc.simulate_until_ruin_or_T(T)
            if ruined:
                ruins += 1
        psi_mc = ruins / n_paths
        psi_se = np.sqrt(psi_mc * (1 - psi_mc) / n_paths)
        psi_lundberg = lundberg_psi_exact(u, R)
        psi_exact = psi_exact_exponential(u, lam, c, mu)
        rows.append({
            'u': u, 'psi_mc': psi_mc, 'psi_se': psi_se,
            'psi_lundberg': psi_lundberg, 'psi_exact': psi_exact,
            'R': R, 'theta': theta,
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / 'exp3_ruin_prob.csv', index=False)
    return df


def run_example_traj(u=5.0, lam=1.0, mu=1.0, theta=0.2,
                     n_paths=15, T=50.0, seed=42):
    """生成盈余轨迹样本（用于图 3.1 和 3.3）。"""
    c = lam * mu * (1 + theta)
    claim_dist = expon(scale=mu)
    rng = np.random.default_rng(seed)
    trajectories = []
    for i in range(n_paths):
        proc = SurplusProcess(u, c, lam, claim_dist)
        times, values = proc.simulate_path(T)
        ruined = values[-1] < 0
        trajectories.append((times, values, ruined))
    return trajectories


if __name__ == '__main__':
    print("=== 第三章实验：保险破产模型 ===")
    print("实验1: 调整系数 R ...")
    df_R = run_R_experiment()
    for _, row in df_R.iterrows():
        print(f"  θ={row['theta']:.1f}, c={row['c']:.2f}, R={row['R']:.6f}")

    print("实验2: 破产概率 vs 初始资本 (theta=0.2, 20000 paths per u) ...")
    df = run_ruin_prob_experiment()
    for _, row in df[df['u'] % 5 == 0].iterrows():
        print(f"  u={row['u']:.0f}, ψ_MC={row['psi_mc']:.4f}±{1.96*row['psi_se']:.4f}, "
              f"ψ_exact={row['psi_exact']:.4f}, Lundberg={row['psi_lundberg']:.4f}")
    print(f"  保存到 output/data/exp3_ruin_prob.csv")
    print("第三章实验完成。")
