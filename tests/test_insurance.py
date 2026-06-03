"""测试 ch03_insurance/surplus_model.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import pytest
from scipy.stats import expon
from ch03_insurance.surplus_model import (
    SurplusProcess, find_adjustment_R, exp_claim_mgf_factory,
    lundberg_psi_exact, psi_exact_exponential,
)


class TestSurplusProcess:
    def test_initial_state(self):
        claim = expon(scale=1.0)
        proc = SurplusProcess(u=10.0, c=1.5, lam=1.0, claim_dist=claim)
        assert proc.state == 10.0
        assert proc.t == 0.0

    def test_reset(self):
        claim = expon(scale=1.0)
        proc = SurplusProcess(u=10.0, c=1.5, lam=1.0, claim_dist=claim)
        proc.step_until_next_claim()
        proc.reset(20.0)
        assert proc.state == 20.0
        assert proc.t == 0.0

    def test_step_advances_time(self):
        claim = expon(scale=1.0)
        proc = SurplusProcess(u=10.0, c=1.5, lam=1.0, claim_dist=claim)
        t_before = proc.t
        proc.step_until_next_claim()
        assert proc.t > t_before

    def test_simulate_path_shape(self):
        claim = expon(scale=1.0)
        proc = SurplusProcess(u=10.0, c=1.5, lam=1.0, claim_dist=claim)
        times, values = proc.simulate_path(T=20.0)
        assert len(times) >= 2
        assert len(times) == len(values)
        assert times[0] == 0.0
        assert values[0] == 10.0

    def test_simulate_until_ruin_or_T(self):
        claim = expon(scale=0.1)  # 小索赔 → 不容易破产
        proc = SurplusProcess(u=100.0, c=2.0, lam=1.0, claim_dist=claim)
        ruined, tau, traj_t, traj_U = proc.simulate_until_ruin_or_T(T=50.0)
        # 高初始资本 + 小索赔 → 大概率不破产
        assert tau > 0


class TestAdjustmentR:
    def test_known_exponential(self):
        """指数索赔 M_Y(r)=1/(1-μr)，调整系数 R = θ/(μ(1+θ))。"""
        lam, mu, theta = 1.0, 0.5, 0.3
        c = lam * mu * (1 + theta)
        mgf = exp_claim_mgf_factory('exponential', rate=1/mu)
        R = find_adjustment_R(lam, c, mgf)
        R_theory = theta / (mu * (1 + theta))
        assert R == pytest.approx(R_theory, rel=1e-3)

    def test_R_positive_with_safety_loading(self):
        """安全负荷正时 R > 0。"""
        lam, mu, theta = 1.0, 1.0, 0.5
        c = lam * mu * (1 + theta)
        mgf = exp_claim_mgf_factory('exponential', rate=1/mu)
        R = find_adjustment_R(lam, c, mgf)
        assert R > 0

    def test_R_larger_for_larger_theta(self):
        """安全负荷越大，调整系数越大。"""
        mu = 1.0
        mgf = exp_claim_mgf_factory('exponential', rate=1/mu)
        R1 = find_adjustment_R(1.0, 1.0 * mu * 1.1, mgf)
        R2 = find_adjustment_R(1.0, 1.0 * mu * 1.5, mgf)
        assert R2 > R1


class TestLundberg:
    def test_psi_at_zero(self):
        assert lundberg_psi_exact(0.0, 0.5) == 1.0

    def test_psi_decreasing(self):
        assert lundberg_psi_exact(10.0, 0.5) < lundberg_psi_exact(5.0, 0.5)


class TestPsiExponential:
    def test_psi_at_zero(self):
        psi = psi_exact_exponential(0.0, lam=1.0, c=1.5, mu=1.0)
        assert pytest.approx(psi, abs=1e-4) == 1.0 / (1.0 + 0.5)  # 1/(1+θ)

    def test_psi_decreasing_in_u(self):
        psi1 = psi_exact_exponential(5.0, lam=1.0, c=1.5, mu=1.0)
        psi2 = psi_exact_exponential(10.0, lam=1.0, c=1.5, mu=1.0)
        assert psi2 < psi1

    def test_ruin_certain_without_safety_loading(self):
        psi = psi_exact_exponential(10.0, lam=1.0, c=0.9, mu=1.0)
        assert psi == 1.0  # c <= λμ → 必然破产


class TestMGF:
    def test_exponential_mgf_at_zero(self):
        mgf = exp_claim_mgf_factory('exponential', rate=0.5)
        assert mgf(0.0) == 1.0

    def test_exponential_mgf_blows_up(self):
        mgf = exp_claim_mgf_factory('exponential', rate=2.0)
        assert mgf(3.0) == np.inf
