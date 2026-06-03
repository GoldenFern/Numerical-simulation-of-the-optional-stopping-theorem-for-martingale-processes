"""第四章绘图：赌徒破产 —— OST 条件失效"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.visualization import (
    set_style, new_figure, save_figure,
    COLOR_BLUE, COLOR_RED, COLOR_GRAY,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def fig4_1_convergence():
    """图 4.1：双边 vs 单边收敛对比。"""
    set_style()
    df = pd.read_csv(DATA_DIR / 'exp4_convergence.csv')

    fig, ax = new_figure()
    for stype, color, label, y_theory in [
        ('two_sided', COLOR_BLUE, 'two-sided $\\tau = \\inf\\{n: S_n=-a \\;\\text{or}\\; S_n=b\\}$', 0),
        ('one_sided', COLOR_RED, 'one-sided $\\tau = \\inf\\{n: S_n=b\\}$', 10.0),
    ]:
        sub = df[df['stop_type'] == stype]
        ax.errorbar(sub['n_paths'], sub['mean'], yerr=1.96 * sub['se'],
                    fmt='o-', color=color, capsize=2, markersize=3.5, lw=0.8,
                    label=label)
        ax.axhline(y_theory, color=color, lw=0.6, ls='--', alpha=0.5)

    ax.set_xscale('log')
    ax.set_xlabel('Monte Carlo 路径数 $M$')
    ax.set_ylabel('$\\mathbb{E}[S_\\tau]$ 估计值')
    ax.set_title('OST 收敛对比：双边成立 vs 单边失效')
    ax.legend(loc='best', frameon=False, fontsize=7)
    save_figure(fig, 'ch04_convergence.pdf')


def fig4_2_truncation():
    """图 4.2：截断偏误。"""
    set_style()
    df = pd.read_csv(DATA_DIR / 'exp4_truncation.csv')

    fig, ax = new_figure()
    ax.errorbar(df['N'], df['mean'], yerr=1.96 * df['se'],
                fmt='o', color=COLOR_BLUE, capsize=1.5, markersize=2.5,
                lw=0, alpha=0.6)
    ax.axhline(10.0, color=COLOR_RED, lw=0.6, ls='--', label='理论极限 $b=10$')
    ax.axhline(0.0, color=COLOR_GRAY, lw=0.4, ls=':', label='$\\mathbb{E}[S_0]=0$')

    ax.set_xscale('log')
    ax.set_xlabel('截断值 $N$')
    ax.set_ylabel('$\\mathbb{E}[S_{\\tau \\wedge N}]$')
    ax.set_title('截断停时偏误：$\\mathbb{E}[S_{\\tau\\wedge N}]$ vs $N$')
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    save_figure(fig, 'ch04_truncation.pdf')


def fig4_3_tail():
    """图 4.3：停时尾部对比（双对数）。"""
    set_style()
    data = np.load(DATA_DIR / 'exp4_tail.npz')

    fig, ax = new_figure()
    for t, surv, color, label in [
        (data['t_two'], data['surv_two'], COLOR_BLUE, '双边障碍'),
        (data['t_one'], data['surv_one'], COLOR_RED, '单边障碍'),
    ]:
        mask = surv > 1e-4
        ax.loglog(t[mask], surv[mask], color=color, lw=0.8, label=label)

    # 参考线：1/sqrt(pi*t) 重尾
    t_ref = np.logspace(1, 4, 100)
    ax.loglog(t_ref, 1.0 / np.sqrt(np.pi * t_ref), '--', color=COLOR_GRAY,
              lw=0.6, alpha=0.6, label='$\\sim 1/\\sqrt{\\pi t}$')

    ax.set_xlabel('$t$')
    ax.set_ylabel('$P(\\tau > t)$')
    ax.set_title('停时尾部对比 (双对数)')
    ax.legend(loc='lower left', frameon=False, fontsize=8)
    save_figure(fig, 'ch04_tail.pdf')


if __name__ == '__main__':
    set_style()
    print("绘制图 4.1: 收敛对比 ...")
    fig4_1_convergence()
    print("绘制图 4.2: 截断偏误 ...")
    fig4_2_truncation()
    print("绘制图 4.3: 停时尾部 ...")
    fig4_3_tail()
    print("第四章三张图绘制完成。")
