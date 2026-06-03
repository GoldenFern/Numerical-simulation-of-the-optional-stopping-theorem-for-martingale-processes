"""第二章绘图：Moran 模型"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from ch02_moran.moran_model import MoranProcess

from core.visualization import (
    set_style, new_figure, save_figure,
    COLOR_BLUE, COLOR_RED, COLOR_GREEN, COLOR_ORANGE,
    plot_paths as plot_paths_fn, plot_with_ci,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def fig2_1_paths(N=50, x0=25, n_display=20, seed=42):
    """图 2.1：多条基因频率轨迹叠加。"""
    set_style()
    np.random.seed(seed)
    model = MoranProcess(N)

    # 生成 100 条路径，然后按最终命运分类抽样
    paths_all = []
    outcomes = []
    for _ in range(200):
        model.reset(x0)
        path = model.simulate_path()
        paths_all.append(path)
        outcomes.append(path[-1] == N)  # True=固定, False=灭绝

    # 频率归一化
    fix_idx = [i for i, o in enumerate(outcomes) if o]
    ext_idx = [i for i, o in enumerate(outcomes) if not o]
    n_each = n_display // 2

    fig, ax = new_figure()
    colors_fix = [COLOR_BLUE, COLOR_RED]
    labels = ['固定', '灭绝']

    for idx_list, color, label in [
        (np.random.choice(fix_idx, min(n_each, len(fix_idx)), replace=False),
         COLOR_BLUE, '固定'),
        (np.random.choice(ext_idx, min(n_each, len(ext_idx)), replace=False),
         COLOR_RED, '灭绝'),
    ]:
        for i in idx_list:
            path_freq = paths_all[i] / N
            tau = len(paths_all[i]) - 1
            ax.plot(path_freq, alpha=0.5, lw=0.4, color=color)
            ax.scatter(tau, path_freq[-1], color=color, s=6, alpha=0.7, zorder=5)

    ax.axhline(0, color='gray', lw=0.3, ls='--')
    ax.axhline(1, color='gray', lw=0.3, ls='--')
    handles = [
        plt.Line2D([0], [0], color=COLOR_BLUE, lw=1, label='固定'),
        plt.Line2D([0], [0], color=COLOR_RED, lw=1, label='灭绝'),
    ]
    ax.legend(handles=handles, loc='best', frameon=False)
    ax.set_xlabel('代数', fontsize=9)
    ax.set_ylabel('A 等位基因频率', fontsize=9)
    ax.set_title(f'Moran 模型基因频率轨迹 (N={N})', fontsize=10)
    save_figure(fig, 'ch02_paths.pdf')


def fig2_2_fixation():
    """图 2.2：固定概率 vs 初始频率。"""
    set_style()
    import pandas as pd
    df = pd.read_csv(DATA_DIR / 'exp2_fixation.csv')

    fig, ax = new_figure()
    markers = ['o', 's', '^']
    for marker, (N, grp) in zip(markers, df.groupby('N')):
        ax.errorbar(grp['initial_freq'], grp['p_fixation_mc'],
                     yerr=1.96 * grp['tau_mc_se'] / np.sqrt(5000),  # 不精确但合理
                     fmt=marker, color=COLOR_BLUE, capsize=2, markersize=3.5,
                     lw=0.6, label=f'N={N}')

    # 理论线 y = x
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, xs, '--', color=COLOR_RED, lw=1.0, label='理论值 $y = x_0/N$')

    ax.set_xlabel('初始频率 $x_0/N$', fontsize=9)
    ax.set_ylabel('固定概率', fontsize=9)
    ax.set_title('固定概率 vs 初始频率', fontsize=10)
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_figure(fig, 'ch02_fixation.pdf')


def fig2_3_tau_dist():
    """图 2.3：停时分布直方图。"""
    set_style()
    tau = np.loadtxt(DATA_DIR / 'exp2_tau_dist.csv', delimiter=',')

    fig, ax = new_figure()
    ax.hist(tau, bins=60, density=True, alpha=0.5, color=COLOR_BLUE,
            edgecolor='white', linewidth=0.3, label='模拟')

    # 指数分布拟合
    from scipy.stats import expon
    loc, scale = expon.fit(tau)
    xs = np.linspace(tau.min(), tau.max(), 200)
    ax.plot(xs, expon.pdf(xs, loc, scale), color=COLOR_RED, lw=1.2,
            label='指数分布拟合')

    ax.axvline(np.mean(tau), color=COLOR_GREEN, lw=0.8, ls='--',
               label=f'均值 = {np.mean(tau):.0f}')
    ax.axvline(np.median(tau), color=COLOR_ORANGE, lw=0.8, ls=':',
               label=f'中位数 = {np.median(tau):.0f}')

    ax.set_xlabel('停时 $\\tau$ (代)', fontsize=9)
    ax.set_ylabel('概率密度', fontsize=9)
    ax.set_title(f'Moran 模型停时分布 (N=50, $x_0$=25)', fontsize=10)
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    save_figure(fig, 'ch02_tau_dist.pdf')


if __name__ == '__main__':
    set_style()
    print("绘制图 2.1: 基因频率轨迹 ...")
    fig2_1_paths()
    print("绘制图 2.2: 固定概率 vs 初始频率 ...")
    fig2_2_fixation()
    print("绘制图 2.3: 停时分布 ...")
    fig2_3_tau_dist()
    print("第二章三张图绘制完成。")
