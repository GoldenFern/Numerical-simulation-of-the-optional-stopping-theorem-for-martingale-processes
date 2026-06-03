"""第五章绘图：秘书问题与 Snell 包络"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.visualization import (
    set_style, new_figure, save_figure,
    COLOR_BLUE, COLOR_RED, COLOR_ORANGE, COLOR_GRAY,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def fig5_1_success():
    """图 5.1: 成功率 vs 观察比例 r/N。"""
    set_style()
    df = pd.read_csv(DATA_DIR / 'exp5_success.csv')

    fig, ax = new_figure()
    colors = [COLOR_BLUE, COLOR_RED, COLOR_ORANGE, 'green']
    for color, (N, grp) in zip(colors, df.groupby('N')):
        ax.errorbar(grp['r_frac'], grp['success_mc'],
                     yerr=1.96 * grp['se'],
                     fmt='o', color=color, capsize=1, markersize=2,
                     lw=0, alpha=0.4, label=f'N={N}')
        # 理论光滑曲线
        idx = np.argsort(grp['r_frac'])
        ax.plot(grp['r_frac'].iloc[idx], grp['theory'].iloc[idx],
                '-', color=color, lw=1.0)

    ax.axvline(1 / np.e, color=COLOR_GRAY, lw=0.6, ls='--', alpha=0.6,
               label='$r^*/N = 1/e$')
    ax.axhline(1 / np.e, color=COLOR_GRAY, lw=0.6, ls=':', alpha=0.6)

    ax.set_xlabel('proportion observed $r/N$')
    ax.set_ylabel('probability of selecting best')
    ax.set_title('Secretary problem: success rate vs $r/N$')
    ax.legend(loc='upper right', frameon=False, fontsize=7)
    save_figure(fig, 'ch05_success.pdf')


def fig5_2_snell():
    """图 5.2: Snell 包络 V_n 与即时 payoff g(n)。"""
    set_style()
    data = np.load(DATA_DIR / 'exp5_snell.npz')
    V, g = data['V'], data['g']
    N = len(V)
    n = np.arange(N)

    fig, ax = new_figure()
    ax.plot(n, V, '-', color=COLOR_BLUE, lw=1.0, label='Snell envelope $V_n$')
    ax.plot(n, g, '--', color=COLOR_RED, lw=0.8, label='immediate payoff $g(n)$')

    # 标注相交区域：V_n == g(n) 时应该停止
    cross = np.where(np.isclose(V, g))[0]
    if len(cross) > 0:
        ax.axvline(cross[0], color=COLOR_GRAY, lw=0.5, ls=':', alpha=0.5)
        ax.annotate(f'stop region starts\nat n={cross[0]}',
                    xy=(cross[0], V[cross[0]]), fontsize=7, color=COLOR_GRAY)

    ax.set_xlabel('$n$ (candidates seen)')
    ax.set_ylabel('value')
    ax.set_title(f'Snell envelope (N={N})')
    ax.legend(loc='lower left', frameon=False, fontsize=8)
    save_figure(fig, 'ch05_snell.pdf')


def fig5_3_control():
    """图 5.3: 鞅对照组 vs 秘书问题。"""
    set_style()
    df_sec = pd.read_csv(DATA_DIR / 'exp5_success.csv')
    df_ctrl = pd.read_csv(DATA_DIR / 'exp5_control.csv')

    fig, ax = new_figure()

    # 秘书问题 (N=100)
    sub_sec = df_sec[df_sec['N'] == 100]
    ax.errorbar(sub_sec['r_frac'], sub_sec['success_mc'],
                 yerr=1.96 * sub_sec['se'],
                 fmt='o', color=COLOR_BLUE, capsize=1, markersize=2,
                 lw=0, alpha=0.4, label='secretary (N=100)')
    idx = np.argsort(sub_sec['r_frac'])
    ax.plot(sub_sec['r_frac'].iloc[idx], sub_sec['theory'].iloc[idx],
            '-', color=COLOR_BLUE, lw=1.0)

    # 对照组 (N=100)
    sub_ctrl = df_ctrl[df_ctrl['N'] == 100]
    ax.errorbar(sub_ctrl['r_frac'], sub_ctrl['success_mc'],
                 yerr=1.96 * sub_ctrl['se'],
                 fmt='s', color=COLOR_RED, capsize=1, markersize=2,
                 lw=0, alpha=0.4, label='martingale noise')
    ax.axhline(1.0 / 100, color=COLOR_RED, lw=0.6, ls=':', alpha=0.5)

    ax.set_xlabel('proportion observed $r/N$')
    ax.set_ylabel('success probability')
    ax.set_title('Secretary problem vs martingale control')
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    save_figure(fig, 'ch05_control.pdf')


if __name__ == '__main__':
    set_style()
    print("绘制图 5.1: 成功率 vs r/N...")
    fig5_1_success()
    print("绘制图 5.2: Snell 包络...")
    fig5_2_snell()
    print("绘制图 5.3: 鞅对照组...")
    fig5_3_control()
    print("第五章三张图绘制完成。")
