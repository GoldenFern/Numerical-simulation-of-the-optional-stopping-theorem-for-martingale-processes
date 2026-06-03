"""Cramér-Lundberg 保险盈余过程 + 指数鞅"""
import numpy as np
from scipy.optimize import brentq


class SurplusProcess:
    """Cramér-Lundberg 盈余过程。

    U_t = u + c·t − Σ_{i=1}^{N_t} Y_i

    u : 初始资本
    c : 保费速率（单位时间）
    λ : 泊松索赔强度
    Y ~ claim_dist: 索赔额分布（需实现 rvs 方法）
    """

    def __init__(self, u: float, c: float, lam: float, claim_dist):
        self.u = u
        self.c = c
        self.lam = lam
        self.claim = claim_dist
        self._t = 0.0
        self._U = u

    def reset(self, u: float = None):
        if u is not None:
            self.u = u
        self._t = 0.0
        self._U = self.u

    @property
    def t(self) -> float:
        return self._t

    @property
    def state(self) -> float:
        return self._U

    def step_until_next_claim(self):
        """跳到下一次索赔时刻，更新盈余。返回 (t_new, U_new)。"""
        inter_arrival = np.random.exponential(1 / self.lam)
        self._t += inter_arrival
        self._U += self.c * inter_arrival
        claim_amount = self.claim.rvs()
        self._U -= claim_amount
        return self._t, self._U

    def simulate_path(self, T: float) -> tuple[np.ndarray, np.ndarray]:
        """生成 [0, T] 上的盈余轨迹（事件驱动）。

        返回 (times, U_values)，times[0]=0, U[0]=u。
        """
        self.reset()
        times = [0.0]
        values = [self._U]
        while self._t < T:
            # 纯保费增长直到下次索赔
            inter_arrival = np.random.exponential(1 / self.lam)
            claim_amount = self.claim.rvs()
            # 在索赔时刻增加一个点
            self._t += inter_arrival
            self._U += self.c * inter_arrival
            times.append(self._t)
            values.append(self._U)
            # 索赔后
            self._U -= claim_amount
            times.append(self._t)
            values.append(self._U)
            if self._U < 0:
                break
        return np.array(times), np.array(values)

    def simulate_until_ruin_or_T(self, T: float):
        """模拟直到破产或 T，返回 (ruined: bool, stopping_time: float, traj)。"""
        self.reset()
        traj_t = [0.0]
        traj_U = [self._U]
        while self._t < T and self._U >= 0:
            inter_arrival = np.random.exponential(1 / self.lam)
            claim_amount = self.claim.rvs()
            self._t += inter_arrival
            self._U += self.c * inter_arrival
            traj_t.append(self._t)
            traj_U.append(self._U)
            self._U -= claim_amount
            traj_t.append(self._t)
            traj_U.append(self._U)
        ruined = self._U < 0
        return ruined, self._t, np.array(traj_t), np.array(traj_U)


def find_adjustment_R(lam: float, c: float, claim_mgf,
                       r_min: float = 1e-12, r_max: float = 10.0) -> float:
    """求解调整系数 R > 0，满足 λ(M_Y(R) − 1) = Rc。

    claim_mgf(r) = E[e^{rY}] 索赔额的矩母函数。
    使用 brentq 求根。
    """
    def f(r):
        return lam * (claim_mgf(r) - 1) - r * c

    # 安全负荷条件：c > λ E[Y]
    # 此时 f'(0) = λ E[Y] - c < 0，且 f(0) = 0
    # f(r) 先负后正 → 存在唯一正根（对大多数索赔分布）
    try:
        return brentq(f, r_min, r_max, xtol=1e-12)
    except ValueError:
        # 搜索更宽的区间
        for ub in [20, 50, 100]:
            try:
                return brentq(f, r_min, ub, xtol=1e-12)
            except ValueError:
                continue
        raise ValueError(f"无法在 (0, {r_max}) 内找到调整系数 R")


def exp_claim_mgf_factory(dist_name: str, **params):
    """索赔额矩母函数工厂。

    'exponential': M_Y(r) = 1 / (1 - r/rate)  (r < rate)
    'gamma': M_Y(r) = (1 - r/rate)^{-shape}  (r < rate)
    """
    if dist_name == 'exponential':
        rate = params.get('rate', 1.0)
        def mgf(r):
            if r >= rate:
                return np.inf
            return 1.0 / (1.0 - r / rate)
        return mgf
    else:
        raise ValueError(f"未知分布: {dist_name}")


def lundberg_psi_exact(u: float, R: float) -> float:
    """Lundberg 指数上界：ψ(u) ≤ e^{−Ru}。这里返回的上界。"""
    return np.exp(-R * u)


def psi_exact_exponential(u: float, lam: float, c: float, mu: float) -> float:
    """指数索赔下破产概率的精确解。

    ψ(u) = (1 / (1 + θ)) * exp(−θ u / (μ (1 + θ)))
    其中 θ = c/(λμ) − 1 = 安全负荷。
    """
    theta = c / (lam * mu) - 1
    if theta <= 0:
        return 1.0  # 安全负荷为负，必然破产
    coeff = 1.0 / (1.0 + theta)
    exponent = -theta * u / (mu * (1.0 + theta))
    return coeff * np.exp(exponent)
