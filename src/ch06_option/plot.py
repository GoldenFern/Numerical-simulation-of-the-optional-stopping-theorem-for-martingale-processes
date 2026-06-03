"""第六章绘图：美式期权 LSM"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.visualization import (
    set_style, new_figure, save_figure,
    COLOR_BLUE, COLOR_RED, COLOR_GREEN, COLOR_GRAY, COLOR_ORANGE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def fig6_1_paths():
    """图 6.1: 15 条股价路径 + LSM 行权位置标记。"""
    set_style()
    data = np.load(DATA_DIR / 'exp6_paths.npz', allow_pickle=True)
    S = data['S']
    tau = data['tau']
    exercised = data['exercised']
    dt = 1.0 / 50

    fig, ax = new_figure()
    for i in range(len(S)):
        ax.plot(np.arange(len(S[i])) * dt, S[i], lw=0.4, color=COLOR_GRAY, alpha=0.5)
        t_i = tau[i] * dt
        ax.scatter(t_i, S[i, tau[i]], color=COLOR_RED, s=12, zorder=5, alpha=0.8)

    ax.axhline(40, color=COLOR_BLUE, lw=0.5, ls='--', alpha=0.5, label='strike $K$')
    ax.set_xlabel('time $t$')
    ax.set_ylabel('stock price $S_t$')
    ax.set_title('American put: stock paths with exercise times')
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    save_figure(fig, 'ch06_paths.pdf')


def fig6_2_boundary():
    """图 6.2: 行权边界热力图。"""
    set_style()
    data = np.load(DATA_DIR / 'exp6_boundary.npz')
    t_grid, s_grid, prob = data['t_grid'], data['s_grid'], data['prob']

    fig, ax = new_figure()
    im = ax.contourf(t_grid, s_grid, prob.T, levels=20,
                     cmap='RdYlBu_r', vmin=0, vmax=1, extend='both')
    cbar = fig.colorbar(im, ax=ax, label='exercise probability', shrink=0.8)
    ax.set_xlabel('time $t$')
    ax.set_ylabel('stock price $S$')
    ax.set_title('American put exercise boundary')
    save_figure(fig, 'ch06_boundary.pdf')


def fig6_3_price():
    """图 6.3: 三线价格对比。"""
    set_style()
    df = pd.read_csv(DATA_DIR / 'exp6_price.csv')

    fig, ax = new_figure()
    ax.errorbar(df['S0'], df['american'], yerr=1.96 * df['am_se'],
                fmt='o', color=COLOR_BLUE, capsize=2, markersize=3,
                lw=0, label='American (LSM)')
    ax.plot(df['S0'], df['european'], '-', color=COLOR_RED, lw=1.0,
            label='European (BS)')
    ax.plot(df['S0'], df['payoff'], '--', color=COLOR_GRAY, lw=0.8,
            alpha=0.6, label='payoff $\\max(K-S,0)$')

    ax.set_xlabel('initial stock price $S_0$')
    ax.set_ylabel('option price')
    ax.set_title('American vs European put option')
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    save_figure(fig, 'ch06_price.pdf')


if __name__ == '__main__':
    set_style()
    print("绘制图 6.1: 路径 + 行权位置...")
    fig6_1_paths()
    print("绘制图 6.2: 行权边界热力图...")
    fig6_2_boundary()
    print("绘制图 6.3: 三线价格对比...")
    fig6_3_price()
    print("第六章三张图绘制完成。")
