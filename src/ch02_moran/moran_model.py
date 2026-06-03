"""Moran 模型 —— 有限种群遗传漂变"""
import numpy as np
from scipy.linalg import solve


class MoranProcess:
    """haploid Moran 模型。

    N 个个体，每个携带 A 或 a 等位基因。
    每代：随机选一个个体繁殖 → 随机选一个个体被替换。
    X_n = 第 n 代 A 等位基因的数量，{0,...,N}，0 和 N 为吸收态。
    X_n 本身是鞅：E[X_{n+1} | X_n] = X_n。
    """

    def __init__(self, N: int):
        self.N = N
        self._x = 0

    def reset(self, x0: int):
        self._x = x0

    @property
    def state(self) -> int:
        return self._x

    def step(self) -> int:
        """执行一代，返回新状态。"""
        i = self._x
        N = self.N
        reproduce_A = np.random.random() < i / N
        kill_A = np.random.random() < i / N
        if reproduce_A and not kill_A:
            self._x = i + 1
        elif not reproduce_A and kill_A:
            self._x = i - 1
        # else: 不变
        return self._x

    def simulate_path(self, max_gen: int = 100000) -> np.ndarray:
        """模拟直到吸收或 max_gen 代。返回包含初始值的路径。"""
        path = [self._x]
        for _ in range(max_gen):
            if self._x == 0 or self._x == self.N:
                break
            path.append(self.step())
        return np.array(path)


def expected_tau_moran(N: int) -> np.ndarray:
    """利用出生-死亡链递推关系精确求解 E[τ | X_0 = i]，返回形状 (N+1,)。"""
    # 边界: E[τ_0] = 0, E[τ_N] = 0
    # 内部: E[τ_i] = 1 + p_up*E[τ_{i+1}] + p_down*E[τ_{i-1}] + p_stay*E[τ_i]
    # 其中 p_up = p_down = i(N-i)/N^2, p_stay = 1 - 2i(N-i)/N^2
    # 化简为三对角线性方程组
    n_int = N - 1  # 内部状态数
    A = np.zeros((n_int, n_int))
    b = -np.ones(n_int)

    for k in range(n_int):
        i = k + 1  # 实际状态
        p = i * (N - i) / (N * N)
        if i > 1:
            A[k, k - 1] = p          # i-1
        A[k, k] = -2 * p              # i (stay term cancels out)
        if i < N - 1:
            A[k, k + 1] = p          # i+1

    # 处理 i=1 和 i=N-1 的边界贡献（来自吸收态 τ_0=τ_N=0）
    # i=1: -2p*τ_1 + p*τ_2 = -1 (无 τ_0 项)
    # i=N-1: p*τ_{N-2} - 2p*τ_{N-1} = -1 (无 τ_N 项)
    # 已经自动处理正确

    tau_interior = solve(A, b)
    tau = np.zeros(N + 1)
    tau[1:N] = tau_interior
    return tau
