# %%
# 导入所需的库
import gc
import hashlib
import json
import math
import os
import shutil
import uuid
import warnings
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import matplotlib.cm as cm
import lightgbm as lgb
from lightgbm import LGBMRegressor
from scipy import *
import random
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from typing import Any, Literal, NamedTuple
import shap
import torch
from xrfm import xRFM
import optuna

# ==========================================
# 🎨 全局图表风格（匹配教程 PDF：英文 Times New Roman + 中文衬线回退）
# ==========================================
from matplotlib import font_manager as _fm

# 字体探测：避免硬编码字体名导致的缺字方块
def _has_font(family):
    """检查 matplotlib 是否已注册该字体族名（稳健，不依赖 findfont 版本差异）。"""
    try:
        names = {f.name for f in _fm.fontManager.ttflist}
        return family in names
    except Exception:
        return False

# 中文衬线优先级（Ubuntu 首选 Noto Serif CJK，Windows 首选 SimSun）
_cjk_priority = ['Noto Serif CJK SC', 'SimSun', 'Source Han Serif SC',
                 'STSong', 'Microsoft YaHei', 'SimHei']
_cjk_found = next((f for f in _cjk_priority if _has_font(f)), 'DejaVu Serif')
# 双字体回退栈：英文走 Times New Roman，中文走衬线 CJK
_font_stack = ['Times New Roman', _cjk_found, 'DejaVu Serif']

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': _font_stack,
    'axes.unicode_minus': False,      # 解决负号显示问题
    'axes.titlesize': 18, 'axes.titleweight': 'bold',
    'axes.labelsize': 15, 'axes.labelweight': 'bold',
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'xtick.color': '#333333', 'ytick.color': '#333333',
    'axes.edgecolor': '#333333', 'axes.linewidth': 1.0,
    'legend.frameon': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})
# 统一矢量级出图 dpi（与教程一致）
_SAVE_DPI = 1200
# PNG 预览 dpi（文件小、便于快速查看，不用于出版）
_PREVIEW_DPI = 300
print(f"🎨 图表字体栈: {_font_stack} | PDF dpi: {_SAVE_DPI} | PNG dpi: {_PREVIEW_DPI}")


# ==========================================
# 双格式保存：同时输出 PDF（出版）+ PNG（预览）
# path 可带任意扩展名（.pdf/.png/无后缀），统一按 basename 生成 .pdf 和 .png
# ==========================================
def _save_both(path, show=False, pdf_dpi=_SAVE_DPI, png_dpi=_PREVIEW_DPI):
    """
    同时保存当前 figure 为 PDF 和 PNG，并打印两条路径。
    PDF 用矢量级高 dpi（出版），PNG 用较低 dpi（预览，文件小）。
    """
    import os as _os
    import matplotlib.pyplot as _plt
    base, _ = _os.path.splitext(str(path))
    pdf_path = base + '.pdf'
    png_path = base + '.png'
    _plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=pdf_dpi)
    _plt.savefig(png_path, format='png', bbox_inches='tight', dpi=png_dpi)
    print(f"  💾 已保存 PDF(出版): {pdf_path}")
    print(f"  🖼️  已保存 PNG(预览): {png_path}")
    if show:
        _plt.show()
    _plt.close()


# ==========================================
# 🎨 Nature Communications 风格 SHAP 三联复合图
#   左：按类别着色的水平条形图 + LOWESS 趋势线
#   右上：极坐标百分比环形图
#   右下：类别汇总水平条形小图
# 参考：s41467-025-68246-7 Figure 2a
# ==========================================

# ── 特征 → 物理机制类别映射 ──
SHAP_CATEGORY_MAP = {
    # Urban Morphology（城市形态）
    'BC': 'Urban Morphology', 'MBH': 'Urban Morphology',
    'CAR': 'Urban Morphology', 'HBC': 'Urban Morphology',
    # Blue-Green Space（蓝绿空间）
    'NDVI': 'Blue-Green Space', 'water': 'Blue-Green Space',
    # Human Activity（人类活动）
    'Pop2025': 'Human Activity', 'NPP_VIIRS': 'Human Activity',
    # Landscape Pattern（景观格局）
    'PLAND': 'Landscape Pattern', 'ED': 'Landscape Pattern',
    'PD': 'Landscape Pattern', 'COHESION': 'Landscape Pattern',
    'CONTAG': 'Landscape Pattern',
    # Transportation（交通）
    'Road_Dens': 'Transportation', 'Road_width': 'Transportation',
    # Background（背景变量——非城市形态机制）
    'Hour_cos': 'Background', 'Temp_2m': 'Background',
    'longitude': 'Background', 'latitude': 'Background',
    'LU_Type': 'Background', 'LCZ_Function': 'Background',
    'Season_code': 'Background',
}

# ── 类别配色（Nature 风格柔和学术色）──
SHAP_CATEGORY_COLORS = {
    'Urban Morphology':  '#8FD2C9',   # 薄荷青绿
    'Blue-Green Space':  '#5B9BD5',   # 柔蓝
    'Human Activity':    '#E68084',   # 柔粉红
    'Landscape Pattern': '#F0D676',   # 暖鹅黄
    'Transportation':    '#A5A5A5',   # 中灰
    'Background':        '#D9D9D9',   # 浅灰
}

# 横轴单位仅使用项目代码或数据方法中能够确认的定义。未知单位保持空值，
# 绘图时不会猜测；正式投稿前需由数据字典补齐。
SHAP_FEATURE_UNITS = {
    'Hour_cos': 'dimensionless',
    'Temp_2m': '°C',
    'longitude': 'degrees east',
    'latitude': 'degrees north',
    'MBH': 'm',
    # Authoritative Fishnet stores BC as a 0-1 building-coverage fraction.
    'BC': 'fraction (0-1)',
    'CAR': 'dimensionless',
    'NPP_VIIRS': None,
    'NDVI': 'dimensionless',
    'Road_width': 'm',
    'Road_Dens': 'm m$^{-2}$',
    'Pop2025': None,
    'water': None,
    'ED': 'm ha$^{-1}$',
    'CONTAG': '%',
    'PLAND': '%',
    'PD': 'patches per 100 ha',
    'COHESION': '%',
    'HBC': '%',
}

# ── statsmodels 可用性检测（LOWESS 趋势线依赖）──
try:
    import statsmodels.api as _sm_lowess
    _LOWESS_AVAILABLE = True
except ImportError:
    _LOWESS_AVAILABLE = False


def plot_shap_nature_style(X_df, shap_values_matrix, feature_cols,
                           category_map, colors_map,
                           r2=None, n_samples=None,
                           save_path=None, dpi=1200,
                           lowess_frac=0.3, figsize=(18, 9)):
    """
    Nature Communications 风格 SHAP 左右组合图。

    原始布局：
      - 左侧：水平条形 + LOWESS 趋势线
      - 左侧右下空白区：极坐标环形图 + 类别汇总图
      - 右侧：SHAP 蜂群图（按特征值高低着色）
    """
    import matplotlib.pyplot as _plt
    from matplotlib.colors import LinearSegmentedColormap as _LSCmap
    import numpy as _np
    import os as _os

    # ── 准备数据 ──
    if isinstance(shap_values_matrix, _np.ndarray):
        shap_df = __import__('pandas').DataFrame(shap_values_matrix, columns=X_df.columns)
    else:
        shap_df = shap_values_matrix.copy()

    mean_shap = _np.abs(shap_df).mean(axis=0).values
    total_shap = mean_shap.sum()

    df = __import__('pandas').DataFrame({
        'Feature': feature_cols,
        'SHAP': mean_shap,
        'Category': [category_map.get(f, 'Background') for f in feature_cols],
    })
    df['Color'] = df['Category'].map(colors_map)
    df['Percent'] = df['SHAP'] / total_shap * 100 if total_shap > 0 else 0.0
    df = df.sort_values('SHAP', ascending=True).reset_index(drop=True)

    # Sized for reduction from the 18-inch source canvas to an approximately
    # 183 mm journal double-column figure: primary text remains about 7-8 pt,
    # while secondary numeric annotations remain above about 5.5 pt.
    _font_feature = 18
    _font_axis = 19
    _font_tick = 17
    _font_metric = 18
    _font_category = 17
    # The 18-inch source canvas is reduced to approximately 183 mm at
    # submission. A 15 pt source minimum therefore renders at about 6 pt.
    _font_polar = 15
    _font_bar_value = 15

    # ── 字体设置（Arial 优先，覆盖全局 serif）──
    _plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': _font_tick,
    })

    # ── 蜂群图连续色谱：从柔蓝到柔粉红（与 Nature 风格协调）──
    _nature_cmap = _LSCmap.from_list(
        'nature',
        ['#5B9BD5', '#8FD2C9', '#F0D676', '#E68084'],
        N=256,
    )

    fig = _plt.figure(figsize=figsize, facecolor='white')

    # ════════════════════════════════════════════
    # 1. 左侧主图：水平条形 + LOWESS 趋势线
    # ════════════════════════════════════════════
    ax_main = fig.add_axes([0.16, 0.1, 0.25, 0.8])
    y_pos = _np.arange(len(df))
    bar_height = 0.7

    bars = ax_main.barh(
        y_pos, df['SHAP'], height=bar_height,
        color=df['Color'], edgecolor='none', alpha=0.85,
    )
    # 底部微小变量看不清的修复：在每根条形末端标注数值
    for _bi, (_bar, _sv) in enumerate(zip(bars, df['SHAP'])):
        if _sv > 0:
            ax_main.text(_sv + df['SHAP'].max() * 0.01, _bi,
                         f'{_sv:.4f}', va='center', ha='left',
                         fontsize=_font_bar_value, color='#555555')

    # LOWESS 趋势线（每条内部叠加：特征值低→高的 SHAP 变化趋势）
    if _LOWESS_AVAILABLE and total_shap > 0:
        print("  [Nature图] 计算 LOWESS 趋势线...")
        _lowess_subsample = 5000   # LOWESS 子采样量（20万行全量太慢，5000足够稳定）
        for idx, row in df.iterrows():
            feat = row['Feature']
            bar_length = row['SHAP']
            y_center = idx
            if feat not in X_df.columns or bar_length <= 0:
                continue
            raw_x = X_df[feat].values
            raw_shap = shap_df[feat].values
            mask = ~( _np.isnan(raw_x) | _np.isnan(raw_shap))
            if mask.sum() < 10:
                continue
            raw_x = raw_x[mask]
            raw_shap = raw_shap[mask]
            # 子采样加速 LOWESS（5000 点足够拟合稳定趋势线）
            if len(raw_x) > _lowess_subsample:
                _lrng = _np.random.RandomState(idx)
                _lidx = _lrng.choice(len(raw_x), size=_lowess_subsample, replace=False)
                raw_x = raw_x[_lidx]
                raw_shap = raw_shap[_lidx]
            try:
                lowess_result = _sm_lowess.nonparametric.lowess(
                    raw_shap, raw_x, frac=lowess_frac, it=3, return_sorted=True)
            except Exception:
                continue
            sorted_x = lowess_result[:, 0]
            fitted_shap = lowess_result[:, 1]
            x_min, x_max = sorted_x.min(), sorted_x.max()
            if x_max - x_min == 0:
                continue
            x_norm = (sorted_x - x_min) / (x_max - x_min)
            plot_x = x_norm * bar_length
            fs_max = _np.max(_np.abs(fitted_shap))
            if fs_max == 0:
                continue
            scale_y = (bar_height / 2) * 0.8
            plot_y = y_center + (fitted_shap / fs_max) * scale_y
            ax_main.plot(plot_x, plot_y, color='#2b2b2b', linewidth=1.2, alpha=0.9)
    else:
        print("  [Nature图] statsmodels 不可用，跳过 LOWESS 趋势线")

    ax_main.set_yticks(y_pos)
    # 特征名带百分比
    _main_labels = [f"{r['Feature']} ({r['Percent']:.2f}%)" for _, r in df.iterrows()]
    ax_main.set_yticklabels(_main_labels, fontsize=_font_feature)
    ax_main.set_xlabel('Mean |SHAP value|', fontsize=_font_axis)
    ax_main.set_xlim(0, df['SHAP'].max() * 1.15 if df['SHAP'].max() > 0 else 1.0)
    ax_main.set_ylim(-0.8, len(df) - 0.2)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['left'].set_linewidth(1.5)
    ax_main.spines['bottom'].set_linewidth(1.5)
    ax_main.tick_params(axis='both', labelsize=_font_tick, length=5, width=1.5)

    # R² 注释
    if r2 is not None:
        _ns_text = f' (N = {n_samples:,})' if n_samples else ''
        ax_main.text(0.5, 1.02, rf'$R^2 = {r2}${_ns_text}',
                     transform=ax_main.transAxes, ha='center', fontsize=_font_metric)

    # ════════════════════════════════════════════
    # 2. 右下空白区：百分比环形图
    # ════════════════════════════════════════════
    df_polar = df.iloc[::-1].reset_index(drop=True)
    N_polar = len(df_polar)
    # 相对于条形图定位：右侧区域只对应低重要性短条形的空白部分。
    ax_polar = ax_main.inset_axes(
        # Move the ring left and trim its right extent so the outer percentage
        # labels do not collide with the beeswarm feature labels.
        [0.18, 0.32, 0.76, 0.48], projection='polar',
        transform=ax_main.transAxes,
    )
    ax_polar.patch.set_alpha(0)
    if N_polar > 0 and df_polar['Percent'].max() > 0:
        scale = 10 / df_polar['Percent'].max()
        plot_heights = df_polar['Percent'] * scale
        angles = _np.linspace(0, 2 * _np.pi, N_polar, endpoint=False)
        angles = (angles + _np.pi / 3) % (2 * _np.pi)


        light_gray = '#E8E8E8'
        off_white = '#F9F9F9'
        for i, (h, a) in enumerate(zip(plot_heights, angles)):
            ax_polar.bar(a, h * 0.75, width=0.45, bottom=2,
                         color=off_white if i % 2 == 0 else light_gray,
                         edgecolor='none')
        for h, a, c in zip(plot_heights, angles, df_polar['Color']):
            ax_polar.bar(a, h * 0.25, width=0.45, bottom=2 + h * 0.75,
                         color=c, edgecolor='none')
        for i, (h, a, pct) in enumerate(zip(plot_heights, angles, df_polar['Percent'])):
            if i < 3:
                # Stagger adjacent top labels radially so they remain distinct
                # after the full figure is reduced to journal column width.
                _label_offset = (1.35, 0.55, 1.05)[i]
                _label_angle = a + (-0.10, 0.0, 0.0)[i]
                ax_polar.text(_label_angle, 2 + h + _label_offset, f'{pct:.1f}%',
                              ha='center', va='center',
                              fontsize=_font_polar, fontweight='bold')
        ax_polar.set_xticks([])
        ax_polar.set_yticklabels([])
        ax_polar.spines['polar'].set_visible(False)
        ax_polar.grid(False)

    # ════════════════════════════════════════════
    # 3. 右下类别汇总：每类平均 SHAP
    # ════════════════════════════════════════════
    ax_sum = fig.add_axes([0.27, 0.14, 0.10, 0.22])
    df_cat = df.groupby('Category')['SHAP'].mean().sort_values(ascending=True)
    cats = df_cat.index.tolist()
    vals = df_cat.values
    cols = [colors_map.get(c, '#333333') for c in cats]
    y_cat = _np.arange(len(cats))

    ax_sum.barh(y_cat, vals, height=0.5, color=cols)
    x_text = vals.max() * 0.03 if len(vals) > 0 and vals.max() > 0 else 0.001
    for i, c in enumerate(cats):
        ax_sum.text(x_text, i, c, va='center', fontsize=_font_category)
    if len(vals) > 0 and vals.max() > 0:
        ax_sum.set_xlim(0, vals.max() * 1.25)
        ax_sum.set_xticks(_np.round(_np.linspace(0, vals.max(), 3), 3))
    ax_sum.set_yticks([])
    ax_sum.tick_params(axis='x', labelsize=_font_tick, length=4, width=1.2)
    ax_sum.spines['right'].set_visible(False)
    ax_sum.spines['top'].set_visible(False)
    ax_sum.spines['left'].set_linewidth(1.5)
    ax_sum.spines['bottom'].set_linewidth(1.5)

    # ════════════════════════════════════════════
    # 4. 右半：SHAP 蜂群图（手动绘制，颜色匹配整体风格）
    # ════════════════════════════════════════════
    ax_bs = fig.add_axes([0.55, 0.1, 0.42, 0.8])
    # 蜂群图与左侧 ax_main 用完全相同的特征顺序：
    #   df 已 sort_values('SHAP', ascending=True) → 索引 0=最小 SHAP 在底部，
    #   最后一个=最大 SHAP 在顶部。蜂群图 y_idx 直接对应 ax_main 的 y_pos。
    feature_order = df['Feature'].tolist()   # 从小到大（与 ax_main y_pos 一致）

    print("  [Nature图] 绘制蜂群散点...")
    _bs_x_all = []
    _bs_y_all = []
    _bs_c_all = []
    for y_idx, feat in enumerate(feature_order):
        if feat not in X_df.columns:
            continue
        sv = shap_df[feat].values
        fv = X_df[feat].values
        mask = ~( _np.isnan(sv) | _np.isnan(fv))
        if mask.sum() == 0:
            continue
        sv = sv[mask]
        fv = fv[mask]
        # 特征值归一化到 0-1 用于颜色映射
        fv_min, fv_max = fv.min(), fv.max()
        if fv_max - fv_min == 0:
            fv_norm = _np.zeros_like(fv)
        else:
            fv_norm = (fv - fv_min) / (fv_max - fv_min)
        # 子采样控制绘图点数
        rng_bs = _np.random.RandomState(y_idx)
        n_pts = len(sv)
        if n_pts > 5000:
            _si = rng_bs.choice(n_pts, size=5000, replace=False)
            sv = sv[_si]
            fv_norm = fv_norm[_si]
            n_pts = 5000
        # ── 密度驱动蜂群抖动（视觉等价于 SHAP summary_plot）──────────────
        # 蜂群图的本质：SHAP 值越密集的区域，点在 y 方向展开越宽；稀疏区域
        # 点贴近基线 y=0。实现三步：
        #   1. 在 SHAP 值上做细粒度直方图 → 每点局部密度
        #   2. 把局部密度归一化 → 决定该点允许的 y 展开半径
        #   3. 在 [-半径, +半径] 内均匀抖动 + 轻微随机化打破规则性
        # 旧实现的两个 bug 已修复：
        #   • 等距整数偏移(arange) →「梳齿」 → 改为随机均匀分布
        #   • 系数 0.12×0.35 太小 → 抖动不可见 → 改为密度自适应，可达 ±0.4
        sv_range = sv.max() - sv.min()
        if sv_range < 1e-12:
            y_jitter_full = _np.zeros(n_pts)
        else:
            n_bins = min(120, max(20, n_pts // 30))
            hist, bin_edges = _np.histogram(sv, bins=n_bins)
            bin_idx = _np.clip(_np.digitize(sv, bin_edges) - 1, 0, n_bins - 1)
            local_density = hist[bin_idx].astype(float)
            max_density = max(hist.max(), 1)
            # 归一化密度 → y 展开半径（0~0.42 行高）
            radius = (local_density / max_density) * 0.42
            # 均匀抖动 + 微扰，避免规则条纹
            rng_local = _np.random.RandomState(y_idx * 7 + 13)
            y_jitter_full = (rng_local.random_sample(n_pts) - 0.5) * 2 * radius
            # 轻微额外噪声打破边界（防止边缘整齐）
            y_jitter_full += rng_local.normal(0, 0.01, n_pts)
            y_jitter_full = _np.clip(y_jitter_full, -0.42, 0.42)
        # 追加（保持 SHAP 值、特征值、抖动三者的索引一致）
        _bs_x_all.extend(sv.tolist())
        _bs_y_all.extend((y_idx + y_jitter_full).tolist())
        _bs_c_all.extend(fv_norm.tolist())

    ax_bs.scatter(_bs_x_all, _bs_y_all, c=_np.asarray(_bs_c_all), cmap=_nature_cmap,
                  s=8, alpha=0.45, edgecolors='none')

    ax_bs.set_yticks(_np.arange(len(feature_order)))
    _bs_labels = []
    for feat in feature_order:
        pct = df[df['Feature'] == feat]['Percent'].values[0]
        _bs_labels.append(f'{feat} ({pct:.2f}%)')
    ax_bs.set_yticklabels(_bs_labels, fontsize=_font_feature)
    ax_bs.set_xlabel('SHAP value (impact on model output)', fontsize=_font_axis)
    ax_bs.axvline(0, color='#999999', linewidth=0.8, linestyle='--', alpha=0.6)
    # Display-only clipping; all SHAP values still contribute to importance statistics.
    ax_bs.set_xlim(-2, 2)
    # 关键：显式锁定 ylim 与 ax_main 完全一致
    # 不设的话，scatter 的自动 ylim 会因抖动偏移而与 barh 错位，
    # 导致蜂群图与左侧条形图特征顺序视觉上「颠倒」。
    ax_bs.set_ylim(-0.8, len(feature_order) - 0.2)
    ax_bs.spines['right'].set_visible(False)
    ax_bs.spines['top'].set_visible(False)
    ax_bs.spines['left'].set_linewidth(1.5)
    ax_bs.spines['bottom'].set_linewidth(1.5)
    ax_bs.tick_params(axis='both', labelsize=_font_tick, length=5, width=1.5)

    # 色条（Feature value 低→高）
    cbar = fig.colorbar(
        _plt.cm.ScalarMappable(cmap=_nature_cmap), ax=ax_bs,
        fraction=0.03, pad=0.02, aspect=30,
    )
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.set_label('Feature value', fontsize=_font_axis)
    cbar.ax.tick_params(labelsize=_font_tick)

    # ── 保存原始左右组合图 ──
    if save_path:
        base = _os.path.splitext(str(save_path))[0]
        pdf_path = base + '.pdf'
        png_path = base + '.png'
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=dpi)
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
        print(f"  💾 已保存 PDF(出版): {pdf_path}")
        print(f"  🖼️  已保存 PNG(预览): {png_path}")
    _plt.show()
    _plt.close()


# ==========================================
# 🎨 SHAP 多项式拟合依赖图（完全复刻参考教程 SHAP_3~6.pdf）
# 当前做法：三次 B-spline + 聚类稳健 95% CI + 50 箱直方图
# ==========================================
def _shap_random_subsample_indices(n_rows, max_rows, random_state=42):
    """Reproduce the original SHAP workflow's row-wise random sample."""
    if max_rows is None or n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)

    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError("max_rows must be a positive integer or None")
    rng = np.random.RandomState(random_state)
    return np.sort(
        rng.choice(n_rows, size=max_rows, replace=False)
    ).astype(np.int64, copy=False)


def _validate_shap_additivity(shap_values, expected_value, predictions,
                              atol=1e-5, rtol=1e-5):
    """Validate prediction = expected value + sum(SHAP) for regression."""
    values = np.asarray(shap_values, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    base = np.asarray(expected_value, dtype=np.float64).reshape(-1)
    if values.ndim != 2:
        raise ValueError("Regression SHAP values must be a 2-D matrix")
    if len(predictions) != values.shape[0]:
        raise ValueError("predictions must align with SHAP rows")
    if base.size != 1:
        raise ValueError("Regression TreeExplainer must have one expected value")

    reconstructed = float(base[0]) + values.sum(axis=1)
    absolute_error = np.abs(reconstructed - predictions)
    tolerance = atol + rtol * np.maximum(np.abs(predictions), np.abs(reconstructed))
    passed = bool(np.all(absolute_error <= tolerance))
    return {
        'passed': passed,
        'n_rows': int(len(predictions)),
        'expected_value': float(base[0]),
        'max_abs_error': float(absolute_error.max(initial=0.0)),
        'mean_abs_error': float(absolute_error.mean()) if len(absolute_error) else 0.0,
        'max_allowed_error': float(tolerance.max(initial=atol)),
        'atol': float(atol),
        'rtol': float(rtol),
    }


def _fit_shap_spline_with_ci(x_values, y_values, cluster_ids=None,
                             spline_df=6, support_quantiles=(0.01, 0.99)):
    """Fit a descriptive cubic spline with cluster-robust mean-response CI."""
    from sklearn.preprocessing import SplineTransformer

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    q_low, q_high = np.quantile(x_values, support_quantiles)
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
        raise ValueError("Supported x range is empty or constant")
    support_mask = (x_values >= q_low) & (x_values <= q_high)
    x_fit = x_values[support_mask]
    y_fit = y_values[support_mask]

    if np.unique(x_fit).size < spline_df + 1:
        raise ValueError("Too few unique supported x values for spline fitting")

    transformer = SplineTransformer(
        n_knots=max(4, spline_df - 1), degree=3,
        knots='quantile', extrapolation='constant', include_bias=False,
    )
    spline_basis = transformer.fit_transform(x_fit.reshape(-1, 1))
    design = np.column_stack([np.ones(len(x_fit)), spline_basis])
    params, _, _, _ = np.linalg.lstsq(design, y_fit, rcond=None)
    residuals = y_fit - design @ params
    bread = np.linalg.pinv(design.T @ design)
    n_obs, n_params = design.shape

    ci_method = 'HC3 heteroskedasticity-robust'
    n_clusters = 0
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        if len(cluster_ids) != len(x_values):
            raise ValueError("cluster_ids must align with SHAP rows")
        fit_clusters = cluster_ids[support_mask]
        unique_clusters, cluster_inverse = np.unique(
            fit_clusters, return_inverse=True
        )
        n_clusters = int(unique_clusters.size)
        if n_clusters >= 2:
            cluster_scores = np.zeros((n_clusters, n_params), dtype=float)
            np.add.at(
                cluster_scores, cluster_inverse,
                design * residuals[:, np.newaxis],
            )
            meat = cluster_scores.T @ cluster_scores
            correction = (
                (n_clusters / (n_clusters - 1))
                * ((n_obs - 1) / max(n_obs - n_params, 1))
            )
            covariance = correction * bread @ meat @ bread
            ci_method = 'Grid_ID cluster-robust'
        else:
            cluster_ids = None

    if cluster_ids is None or n_clusters < 2:
        leverage = np.einsum('ij,jk,ik->i', design, bread, design)
        adjusted = residuals / np.clip(1.0 - leverage, 1e-8, None)
        meat = design.T @ ((adjusted ** 2)[:, np.newaxis] * design)
        covariance = bread @ meat @ bread

    x_grid = np.linspace(q_low, q_high, 500)
    grid_basis = transformer.transform(x_grid.reshape(-1, 1))
    grid_design = np.column_stack([np.ones(len(x_grid)), grid_basis])
    fitted = grid_design @ params
    variance = np.einsum('ij,jk,ik->i', grid_design, covariance, grid_design)
    standard_error = np.sqrt(np.clip(variance, 0.0, None))

    return {
        'x_grid': x_grid,
        'fitted': fitted,
        'ci_lower': fitted - 1.96 * standard_error,
        'ci_upper': fitted + 1.96 * standard_error,
        'r_squared': float(
            1.0 - np.sum(residuals ** 2)
            / max(np.sum((y_fit - y_fit.mean()) ** 2), np.finfo(float).eps)
        ),
        'support_low': float(q_low),
        'support_high': float(q_high),
        'n_fit': int(support_mask.sum()),
        'n_clusters': n_clusters,
        'ci_method': ci_method,
    }


def plot_shap_dependence(feature_name, X_data, shap_values_df,
                         cluster_ids=None, spline_df=6, save_pdf_name=None,
                         legend_location='lower right', target_col='UHII',
                         feature_unit=None,
                         scatter_sample_size=8000, scatter_random_state=42):
    """Plot SHAP dependence on the inverse-transformed original feature scale."""
    import matplotlib.patches as mpatches

    x_values = X_data[feature_name].to_numpy(dtype=float)
    y_values = shap_values_df[feature_name].to_numpy(dtype=float)
    valid_indices = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid_indices]
    y_values = y_values[valid_indices]
    valid_clusters = (
        None if cluster_ids is None
        else np.asarray(cluster_ids)[valid_indices]
    )

    if len(x_values) < spline_df + 2:
        print(
            f"Warning: Not enough valid rows ({len(x_values)}) to fit a "
            f"cubic spline. Skipping plot for {feature_name}."
        )
        return None

    spline = _fit_shap_spline_with_ci(
        x_values, y_values, cluster_ids=valid_clusters,
        spline_df=spline_df, support_quantiles=(0.01, 0.99),
    )

    sns.set_theme(
        style="whitegrid",
        rc={"axes.facecolor": "white", "grid.color": ".9", "grid.linestyle": "-"},
    )

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.1, 0.7, 0.2])
    ax_scatter = fig.add_subplot(gs[1, 0])

    if scatter_sample_size is not None and len(x_values) > scatter_sample_size:
        scatter_rng = np.random.RandomState(scatter_random_state)
        scatter_indices = scatter_rng.choice(
            len(x_values), size=scatter_sample_size, replace=False,
        )
        x_scatter = x_values[scatter_indices]
        y_scatter = y_values[scatter_indices]
    else:
        x_scatter = x_values
        y_scatter = y_values

    ax_scatter.scatter(
        x_scatter, y_scatter, s=20, alpha=0.1, color='darkblue',
        edgecolors='none', rasterized=True,
    )
    ax_scatter.fill_between(
        spline['x_grid'], spline['ci_lower'], spline['ci_upper'],
        color='orange', alpha=0.2, linewidth=0,
    )
    ax_scatter.plot(
        spline['x_grid'], spline['fitted'],
        color='orange', linestyle='--', linewidth=2,
    )

    y_label = (
        "SHAP value for predicted UHII (°C)"
        if target_col == 'UHII'
        else f"SHAP value for predicted {target_col}"
    )
    ax_scatter.set_ylabel(y_label, fontsize=20, fontweight='bold')
    # The dependence x-axis is now in the original physical/data unit. Robust
    # 1st-99th percentile limits prevent rare extremes from flattening the
    # supported spline region without changing the values used for fitting.
    display_low = spline['support_low']
    display_high = spline['support_high']
    display_padding_fraction = 0.03
    display_span = display_high - display_low
    if display_span > 0:
        display_padding = display_padding_fraction * display_span
        ax_scatter.set_xlim(
            display_low - display_padding,
            display_high + display_padding,
        )
    ax_scatter.axhline(0, color='gray', linestyle='-', linewidth=0.8)
    plt.setp(ax_scatter.get_xticklabels(), fontsize=18, fontweight='bold')
    plt.setp(ax_scatter.get_yticklabels(), fontsize=18, fontweight='bold')

    ci_short = (
        'Grid_ID cluster-robust'
        if spline['ci_method'] == 'Grid_ID cluster-robust'
        else 'HC3 robust'
    )
    legend_elements = [
        plt.Line2D(
            [0], [0], marker='o', color='w', label='Sample',
            markerfacecolor='darkblue', markersize=6, alpha=0.5,
        ),
        mpatches.Patch(
            color='orange', alpha=0.2, label=f'95% CI ({ci_short})',
        ),
        plt.Line2D(
            [0], [0], color='orange', linestyle='--', linewidth=2,
            label='Cubic spline',
        ),
    ]
    ax_scatter.legend(
        handles=legend_elements, loc=legend_location, frameon=True,
        prop={'size': 9, 'weight': 'normal'}, framealpha=0.82,
        borderpad=0.45, labelspacing=0.35, handlelength=1.8,
        handletextpad=0.55,
    )
    plt.setp(ax_scatter.get_xticklabels(), visible=False)

    ax_hist = fig.add_subplot(gs[2, 0], sharex=ax_scatter)
    histogram_edges = np.linspace(display_low, display_high, 51)
    sns.histplot(
        x_values, ax=ax_hist, kde=False, color='black',
        bins=histogram_edges,
    )
    ax_hist.set_ylabel("Dist.", fontsize=20, fontweight='bold')
    x_label = (
        f"{feature_name} ({feature_unit})"
        if feature_unit else feature_name
    )
    ax_hist.set_xlabel(x_label, fontsize=20, fontweight='bold')
    ax_hist.tick_params(axis='x', labelbottom=True)
    plt.setp(ax_hist.get_xticklabels(), visible=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_pdf_name:
        _save_both(save_pdf_name, show=True)
    else:
        plt.show()
    return spline

# ==========================================
# 🔧 配置区域：选择数据格式和预测模式
# ==========================================
# 数据格式：'wide' 或 'long'
DATA_FORMAT = 'long'  # 'wide': 宽格式（原始），'long': 长格式（添加时间特征）

# 预测模式：'single' 或 'all'（仅在wide格式下有效）
PREDICT_MODE = 'single'  # 'single': 预测单个小时, 'all': 预测24小时

# 如果是单小时模式，选择要预测的小时（0-23）
TARGET_HOUR = 22  # 例如：12 表示预测 Hour_12

# 训练模式：'all', 'by_climate' 或 'by_season'
# 'all': 全部数据(夏季和冬季一起混合训练，绝对不拆分), 'by_climate': 按气候带, 'by_season': 夏冬单独训练
TRAIN_MODE = 'all'

# 如果是按气候带训练，选择要训练的气候带（可多选）
# 选项：[1, 2, 3, 4, 5] 或 'all'
# 1=严寒, 2=寒冷, 3=夏热冬冷, 4=夏热冬暖, 5=温和
SELECTED_CLIMATES = [1, 2, 3, 4, 5]  # 例如：[3, 4] 只训练夏热冬冷和夏热冬暖

# ==========================================
# 📂 路径配置：改为 Ubuntu 主机路径并自动分类
# ==========================================
# 1. 输入数据目录（注意：这里是目录，不是CSV文件）
DATA_BASE_PATH = "/home/ubuntu"
# 根据 DATA_FORMAT 动态选择文件名在下方定义

# 2. 输出总目录（版本化：每次正式运行用独立目录，不覆盖旧结果）
_RUN_VERSION = "v20260820_BC_noFAR"
_SHAP_RESULT_VERSION = "v20260820_BC_noFAR_spline_cluster_ci"
FEATURE_SCHEMA_VERSION = "BC_noFAR_v1"
BASE_OUTPUT_DIR = f"/home/ubuntu/UHI_Output_Results_{_RUN_VERSION}"
_RUN_INSTANCE_ID = (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    + "_" + uuid.uuid4().hex[:12]
)

# 3. 自动创建子文件夹分类
METRICS_DIR = os.path.join(BASE_OUTPUT_DIR, "1_Metrics_CSVs")        # 评估指标
PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "2_Performance_Plots")     # 散点图
CORR_DIR = os.path.join(BASE_OUTPUT_DIR, "3_Correlations_VIF")       # 相关性与VIF
SHAP_DIR = os.path.join(
    BASE_OUTPUT_DIR, f"4_Interpretability_SHAP_{_SHAP_RESULT_VERSION}"
)  # 修正后的可解释性分析；复用 BASE_OUTPUT_DIR 中已保存的模型

# 一键创建所有目录
# Pearson 鍒嗗潡澶у皬锛氶粯璁?2,000,000 琛岄€傚悎 128G 鍐呭瓨鏈嶅姟鍣ㄣ€?
# 鍧楀お澶э細鍗曞潡 numpy 鏁扮粍鍐呭瓨宄板€兼洿楂橈紱鍧楀お灏忥細寰幆娆℃暟澧炲姞銆佽绠楅€熷害鏇存參銆?
PEARSON_CHUNK_SIZE = 2_000_000

for folder in [METRICS_DIR, PLOTS_DIR, CORR_DIR, SHAP_DIR]:
    os.makedirs(folder, exist_ok=True)
    print(f"📁 已就绪子目录: {folder}")

# %%


def evaluate_regress(y_pre, y_true, y_name=''):

    print('*****************************************************')

    # 1. MAE (平均绝对误差)
    MAE = np.sum(np.abs(y_pre - y_true)) / len(y_true)
    print(y_name + ' MAE为: ', str(MAE))

    # 2. MCE/MAPE (你原始代码中的定义：归一化平均绝对误差)
    # 注意：通常 MAPE 分母是 y_true，这里保留你原有的 max(y_true) 逻辑
    MCE = np.sum(np.abs((y_pre - y_true) / np.max(y_true))) / len(y_true)
    print(y_name + ' MAPE(MCE)为: ', str(MCE))

    # 3. MSE (均方误差)
    MSE = np.sum((y_pre - y_true) ** 2) / len(y_true)
    print(y_name + ' MSE为: ', str(MSE))

    # 4. RMSE (均方根误差)
    RMSE = np.sqrt(MSE)
    print(y_name + ' RMSE为: ', str(RMSE))

    # 5. R2 (决定系数)
    R2 = r2_score(y_true, y_pre)
    print(y_name + ' R2为: ', str(R2))

    # --- 新增指标 ---

    # 6. Bias (平均偏差 Mean Bias Error)
    # 正值表示高估，负值表示低估
    Bias = np.mean(y_pre - y_true)
    print(y_name + ' Bias为: ', str(Bias))

    # 7. CVRMSE (变异系数均方根误差)
    # 公式：RMSE / mean(y_true) * 100%
    y_mean = np.mean(y_true)
    if y_mean != 0:
        CVRMSE = (RMSE / y_mean) * 100
    else:
        CVRMSE = float('inf')  # 防止除以零
    print(y_name + ' CVRMSE(%)为: ', str(CVRMSE))

    print('*****************************************************')

    # 记得在调用这个函数的地方，也要增加接收这两个新返回值
    return MAE, MCE, MSE, RMSE, R2, Bias, CVRMSE

# %%


def _build_pearson_chunk_array(chunk_df, feature_cols, target_name, derived_target_cols=None):
    """构建 Pearson 分块分析矩阵，避免整表复制。"""
    if chunk_df.empty:
        return np.empty((0, len(feature_cols) + 1), dtype=np.float64)

    # 按块提取特征矩阵，并统一提升到 float64 做精确累计。
    feature_array = chunk_df.loc[:, feature_cols].to_numpy(dtype=np.float64, copy=False)

    if derived_target_cols:
        # 宽表 24 小时场景下，按块即时计算目标均值，避免新增整列副本。
        target_array = chunk_df.loc[:, derived_target_cols].to_numpy(
            dtype=np.float64, copy=False
        ).mean(axis=1, dtype=np.float64)
    else:
        target_array = chunk_df.loc[:, target_name].to_numpy(dtype=np.float64, copy=False)

    analysis_array = np.empty((feature_array.shape[0], feature_array.shape[1] + 1), dtype=np.float64)
    analysis_array[:, :-1] = feature_array
    analysis_array[:, -1] = target_array
    return analysis_array


def compute_pearson_corr_chunked_exact(
    data_source,
    feature_cols,
    target_name,
    chunk_size=PEARSON_CHUNK_SIZE,
    row_filter=None,
    derived_target_cols=None,
    progress_label='Pearson'
):
    """对全量数据做分块、精确、无抽样的 Pearson 相关性计算。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须是正整数。")

    analysis_cols = list(feature_cols) + [target_name]
    required_cols = list(feature_cols)

    if derived_target_cols:
        required_cols.extend(list(derived_target_cols))
    else:
        required_cols.append(target_name)

    if row_filter is not None:
        required_cols.append(row_filter[0])

    # 去重后保留原顺序，避免重复读取列。
    required_cols = list(dict.fromkeys(required_cols))

    total_count = 0
    sum_vector = np.zeros(len(analysis_cols), dtype=np.float64)
    sum_square_vector = np.zeros(len(analysis_cols), dtype=np.float64)
    cross_product_sum = np.zeros((len(analysis_cols), len(analysis_cols)), dtype=np.float64)

    if isinstance(data_source, pd.DataFrame):
        total_rows = len(data_source)
        total_chunks = max(1, math.ceil(total_rows / chunk_size)) if total_rows else 0
        chunk_iter = (
            data_source.iloc[start:start + chunk_size][required_cols]
            for start in range(0, total_rows, chunk_size)
        )
    elif isinstance(data_source, (str, os.PathLike)):
        try:
            import pyarrow.dataset as ds
        except ImportError as exc:
            raise ImportError("读取 parquet 路径做分块 Pearson 需要安装 pyarrow。") from exc

        dataset = ds.dataset(os.fspath(data_source), format='parquet')
        total_rows = dataset.count_rows()
        total_chunks = max(1, math.ceil(total_rows / chunk_size)) if total_rows else 0
        scanner = dataset.scanner(columns=required_cols, batch_size=chunk_size)
        chunk_iter = (batch.to_pandas() for batch in scanner.to_batches())
    else:
        raise TypeError("data_source 仅支持 pandas.DataFrame 或 parquet 路径。")

    if total_chunks == 0:
        raise ValueError("Pearson 分析数据为空，无法计算相关矩阵。")

    for chunk_idx, chunk_df in enumerate(chunk_iter, start=1):
        if row_filter is not None:
            filter_col, filter_value = row_filter
            chunk_df = chunk_df.loc[chunk_df[filter_col] == filter_value]

        if chunk_df.empty:
            print(f"    [{chunk_idx}/{total_chunks}] {progress_label} 当前块无有效行，已跳过")
            continue

        analysis_array = _build_pearson_chunk_array(
            chunk_df=chunk_df,
            feature_cols=feature_cols,
            target_name=target_name,
            derived_target_cols=derived_target_cols,
        )

        chunk_count = analysis_array.shape[0]
        total_count += chunk_count
        sum_vector += analysis_array.sum(axis=0, dtype=np.float64)
        sum_square_vector += np.square(analysis_array, dtype=np.float64).sum(axis=0, dtype=np.float64)
        cross_product_sum += analysis_array.T @ analysis_array

        print(
            f"    [{chunk_idx}/{total_chunks}] {progress_label} 分块累计完成，"
            f"当前块有效行数: {chunk_count:,}，累计有效行数: {total_count:,}"
        )

        del chunk_df, analysis_array
        if chunk_idx % 5 == 0:
            gc.collect()

    if total_count == 0:
        raise ValueError("Pearson 分析在当前筛选条件下没有有效样本。")

    numerator = total_count * cross_product_sum - np.outer(sum_vector, sum_vector)
    variance_term = total_count * sum_square_vector - np.square(sum_vector)
    variance_term = np.maximum(variance_term, 0.0)
    denominator = np.sqrt(np.outer(variance_term, variance_term))

    corr_values = np.zeros_like(numerator, dtype=np.float64)
    valid_mask = denominator > 0
    corr_values[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    corr_values = np.clip(corr_values, -1.0, 1.0)

    # 对角线按需求固定为 1，常量列的非对角项保持为 0，避免绘图和排序阶段崩溃。
    np.fill_diagonal(corr_values, 1.0)

    return pd.DataFrame(corr_values, index=analysis_cols, columns=analysis_cols)


def train_muti_model(X_train, y_train, X_test, y_test, method_choose,
                     X_val=None, y_val=None):
    """
    训练多种机器学习模型或 xRFM 模型。
    对于 xRFM，自动检测并使用 GPU 进行训练和预测。
    X_val/y_val: 可选验证集，仅用于评估监控，不参与训练。
    """

    best_model = None  # 初始化，防止报错

    print(f"当前选择的模型: {method_choose}")

    # =======================================================
    # 1. 传统机器学习模型 (使用 GridSearchCV)
    # =======================================================

    if method_choose == 'LightGBM':
        from lightgbm import LGBMRegressor
        from sklearn.model_selection import RandomizedSearchCV
        estimator = LGBMRegressor(
            objective='regression', force_col_wise=True, verbose=-1,
            random_state=42, n_jobs=12)
        param_grid = {
            'n_estimators': [150, 200],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [63, 127],
            'max_depth': [7, 10],
            'subsample': [0.8]
        }
        grid = RandomizedSearchCV(estimator, param_distributions=param_grid,
                                  n_iter=3, cv=5, scoring='r2', n_jobs=5, random_state=42)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

    elif method_choose == 'XGBoost':
        from xgboost import XGBRegressor
        from sklearn.model_selection import RandomizedSearchCV
        # XGBoost 内置 n_jobs=-1，外层搜索保持 1
        estimator = XGBRegressor(
            random_state=42, n_jobs=12, tree_method='hist')
        param_grid = {
            'n_estimators': [150, 200],
            'max_depth': [6, 8],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        grid = RandomizedSearchCV(estimator, param_distributions=param_grid,
                                  n_iter=3, cv=5, scoring='r2', n_jobs=5, random_state=42)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

    elif method_choose == 'CatBoost':
        from catboost import CatBoostRegressor
        from sklearn.model_selection import RandomizedSearchCV
        estimator = CatBoostRegressor(
            random_seed=42, verbose=0, allow_writing_files=False,
            thread_count=12,
            subsample=0.5,
            border_count=128
        )
        param_grid = {
            'iterations': [150, 200],
            'learning_rate': [0.05, 0.1],
            'depth': [6, 8]
        }
        grid = RandomizedSearchCV(estimator, param_distributions=param_grid,
                                n_iter=3, cv=5, scoring='r2', n_jobs=3, random_state=42)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

    elif method_choose == 'RF':
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import RandomizedSearchCV
        # 大样本RF只保留一层并行：外层CV串行，内部16线程并行建树。
        # Bootstrap仍为全量抽样(max_samples=None)，不改变原实验口径。
        estimator = RandomForestRegressor(
            random_state=42, n_jobs=16, max_features='sqrt')
        param_grid = {
            'n_estimators': [60, 100],
            'max_depth': [15, 20, 25],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [2, 5, 10],
            'max_leaf_nodes': [1000, 2000, 5000],
            'max_features': ['sqrt', 0.5]
        }
        grid = RandomizedSearchCV(estimator, param_distributions=param_grid,
                                  n_iter=1, cv=5, scoring='r2', n_jobs=1, random_state=42)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

    elif method_choose == 'MLP':
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import RandomizedSearchCV
        # MLP本身无n_jobs；外层CV限制为4路并行，在速度与训练折复制内存之间折中。
        estimator = MLPRegressor(random_state=42, max_iter=200)
        param_grid = {
            'hidden_layer_sizes': [(64,)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.001, 0.01]
        }
        grid = RandomizedSearchCV(estimator, param_distributions=param_grid,
                                  n_iter=2, cv=5, scoring='r2', n_jobs=12, random_state=42)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

    elif method_choose == 'DT':
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import RandomizedSearchCV
        # DT 底层单线程，无 n_jobs 参数
        estimator = DecisionTreeRegressor(random_state=42)
        param_grid = {
            'max_depth': [8, 12],
            'min_samples_split': [30, 50],
            'min_samples_leaf': [15, 20]
        }
        grid = RandomizedSearchCV(estimator, param_distributions=param_grid,
                                  n_iter=2, cv=5, scoring='r2', n_jobs=8, random_state=42)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

    elif method_choose == 'Bay':
        from sklearn.linear_model import BayesianRidge
        # 1.17 亿行极大数据集下，贝叶斯线性回归能自行通过数据推断出 alpha 和 lambda 参数，
        # 所以超参数搜索 (CV拆解) 不仅几乎零收益，它的分折切片反而会引发高达几十 GB 的额外内存开销！
        # 直接使用全量数据进行默认拟合，算法底层自带 X^T X 矩阵运算，零内存外溢风险。
        best_model = BayesianRidge()
        best_model.fit(X_train, y_train)
        print("      - BayesianRidge 训练完成 (防 OOM 极速直出)")

    elif method_choose == 'KNN':
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.model_selection import RandomizedSearchCV

        # 【核心策略调整】内层固定单线程，防线程爆炸；外层暴力塞满 28 核心搜索
        estimator = KNeighborsRegressor(algorithm='auto', n_jobs=12)

        param_dist = {
            'n_neighbors': [50, 100],
            'weights': ['uniform', 'distance'],
            'p': [2]
        }

        random_search = RandomizedSearchCV(
            estimator, param_distributions=param_dist,
            n_iter=2, cv=5, scoring='r2',
            n_jobs=5,
            random_state=42
        )
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        print(f"      - KNN 最优参数: {random_search.best_params_}")

    elif method_choose == 'GBR':
        # 从低效的串行 GBR 升级为带直方图加速的超音速版本
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import RandomizedSearchCV
        # HistGBR 内置 n_jobs 支持，解放全部 64 线程
        estimator = HistGradientBoostingRegressor(random_state=42)
        param_grid = {
            'max_iter': [100, 150],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 4]
        }
        grid = RandomizedSearchCV(estimator, param_distributions=param_grid,
                                  n_iter=3, cv=5, scoring='r2', n_jobs=2, random_state=42)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

    elif method_choose == 'LinearRegression':
        import numpy as np
        # ⚠️ sklearn 的 LinearRegression 底层依赖 scipy.linalg.lstsq (LAPACK gelsd)，
        # 它会在 1.17 亿行的数据上不仅内部强制转为 float64 (内存翻倍至 24GB)，还会在 SVD 分治过程中申请相同大小的工作区缓存，
        # 直接瞬间干爆即使是 128GB 的可用内存。
        #
        # 【救火方案】：对于标准线性回归，解方程组的解析解就是 (X^T X)^-1 X^T Y。
        # 我们直接使用内存占用几乎为零 (仅 27x27 矩阵尺寸) 的法线方程求解器：

        class MemSafeLinearRegression:
            def fit(self, X, y):
                # 均值极速计算，不改变原矩阵
                X_mean = np.mean(X, axis=0)
                y_mean = np.mean(y)
                N = X.shape[0]

                # O(N) 复杂度的矩阵乘法，完美利用当前 AMD EPYC 的高达 64 线程的 BLAS 向量指令集加速，且 0 额外内存
                # X^T X_c = X^T X - N * X_m X_m^T
                XTX = np.dot(X.T, X) - N * np.outer(X_mean, X_mean)
                XTy = np.dot(X.T, y) - N * y_mean * X_mean

                # 求解 27 维的系数 (极简、瞬时)
                self.coef_ = np.linalg.solve(XTX, XTy)
                self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
                return self

            def predict(self, X):
                return np.dot(X, self.coef_) + self.intercept_

        best_model = MemSafeLinearRegression()
        best_model.fit(X_train, y_train)
        print("      - LinearRegression 训练完成 (防 OOM 定制法线方程解析器)")

    # =======================================================
    # 2. xRFM 模型 (GPU/CUDA 加速版)
    # =======================================================
    elif method_choose == 'xRFM':
        from sklearn.model_selection import train_test_split

        # 0. 重新激活 GPU (配合下截断使用)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("=" * 40)
            print(f"✅ xRFM: CUDA 检测成功，将重返 GPU 极速训练。")
            print("=" * 40)
        else:
            device = torch.device("cpu")

        # 1. 训练数据量限制
        # GPU: RTX 4060 8GB 显存，限制 25 万条
        # CPU: 无显存但核矩阵计算 O(n²)，限制 50 万条防内存爆炸
        from sklearn.utils import resample
        if device.type == 'cuda':
            max_samples = 250000
        else:
            max_samples = 500000

        if X_train.shape[0] > max_samples:
            X_tmp, y_tmp = resample(
                X_train, y_train, n_samples=max_samples, random_state=42)
            _dev = device.type.upper()
            print(f"      - xRFM {_dev} 自动抽取 {max_samples} 条数据训练")
        else:
            X_tmp, y_tmp = X_train, y_train

        # 2. 从处理后的数据中划分出验证集
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.2, random_state=42
        )

        # 2. 准备数据 (转 Tensor 并移动到 device)
        X_train_t = torch.from_numpy(X_train_sub).float().to(device)
        y_train_t = torch.from_numpy(
            y_train_sub).float().view(-1, 1).to(device)
        X_val_t = torch.from_numpy(X_val).float().to(device)
        y_val_t = torch.from_numpy(y_val).float().view(-1, 1).to(device)

        # 3. 定义固定参数
        rfm_params = {
            "model": {
                "kernel": "l1",
                "exponent": 1.3,
                "bandwidth": 6.0,
                "diag": True,
                "bandwidth_mode": "adaptive",
            },
            "fit": {
                "reg": 3e-2,
                "iters": 5,
                "early_stop_rfm": 10,
            },
        }

        # 4. 初始化模型 (确保传入 device)
        best_model = xRFM(
            rfm_params=rfm_params,
            device=device,
            min_subset_size=10000,
            task_type="regression",
            tuning_metric="mse",
            split_method="top_vector_agop_on_subset",
            n_threads=64,
        )

        # 5. 训练（使用验证集）
        best_model.fit(X_train_t, y_train_t, X_val_t, y_val_t)

    # =======================================================
    # 3. 统一预测部分
    # =======================================================

    # 检查模型是否成功初始化
    if best_model is None:
        raise ValueError(f"未找到名为 {method_choose} 的模型或初始化失败")

    if method_choose == 'xRFM':
        # xRFM 特殊处理（恢复动态设备分配，让推断也走 GPU 算力）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 转换为 Tensor 并移动到设备（如果是 GPU 极快）
        X_test_t = torch.from_numpy(X_test).float().to(device)
        X_train_t = torch.from_numpy(X_train).float().to(device)

        # 2. 分批预测以防 OOM 内存爆炸 (切块大小为 20000 避免瞬间开辟 1M*N 的矩阵)
        def batch_predict(model, X_tensor, batch_sz=20000):
            preds = []
            for i in range(0, X_tensor.shape[0], batch_sz):
                batch_X = X_tensor[i:i+batch_sz]
                with torch.no_grad():
                    pred = model.predict(batch_X)
                preds.append(pred)
            # 兼容底层预测结果可能是 Numpy 返回也可能是 Tensor 返回
            if isinstance(preds[0], torch.Tensor):
                return torch.cat(preds, dim=0)
            else:
                return np.concatenate(preds, axis=0)

        y_test_pred_raw = batch_predict(best_model, X_test_t)
        y_train_pred_raw = batch_predict(best_model, X_train_t)

        # 3. 处理输出 (统一拉平)
        if isinstance(y_test_pred_raw, torch.Tensor):
            y_test_predict = y_test_pred_raw.cpu().numpy().flatten()
            y_train_predict = y_train_pred_raw.cpu().numpy().flatten()
        else:
            y_test_predict = np.array(y_test_pred_raw).flatten()
            y_train_predict = np.array(y_train_pred_raw).flatten()

    else:
        # Sklearn 风格模型通用预测
        y_test_predict = best_model.predict(X_test)
        y_train_predict = best_model.predict(X_train)

    # 验证集预测（仅评估监控，不参与训练）
    y_val_predict = None
    if X_val is not None and y_val is not None:
        if method_choose == 'xRFM':
            X_val_t = torch.from_numpy(X_val).float().to(device)
            y_val_pred_raw = batch_predict(best_model, X_val_t)
            y_val_predict = y_val_pred_raw.cpu().numpy().flatten() if isinstance(
                y_val_pred_raw, torch.Tensor) else np.array(y_val_pred_raw).flatten()
        else:
            y_val_predict = best_model.predict(X_val)

    return y_train_predict, y_test_predict, best_model, y_val_predict


# %%


def test_method(y_train_predict, y_test_predict,
                train_y, test_y,
                method_label,
                val_y_predict=None, val_y=None):

    train_ev = []
    test_ev = []
    val_ev = []

    # 定义输出时的标签名称
    method_label_train = method_label + '_训练集_输出 '
    method_label_test = method_label + '_测试集_输出 '
    method_label_val = method_label + '_验证集_输出 '

    # =================== 数据降维 ===================
    y_train_pred_flat = np.array(y_train_predict).flatten()
    y_train_true_flat = np.array(train_y).flatten()

    y_test_pred_flat = np.array(y_test_predict).flatten()
    y_test_true_flat = np.array(test_y).flatten()
    # ===============================================

    # --- 1. 评估训练集 ---
    MAE_train, MCE_train, MSE_train, RMSE_train, R2_train, Bias_train, CVRMSE_train = evaluate_regress(
        y_train_pred_flat, y_train_true_flat, y_name=method_label_train
    )
    train_ev.append([MAE_train, MCE_train, MSE_train,
                    RMSE_train, R2_train, Bias_train, CVRMSE_train])

    # --- 2. 评估验证集（如有）---
    if val_y_predict is not None and val_y is not None:
        y_val_pred_flat = np.array(val_y_predict).flatten()
        y_val_true_flat = np.array(val_y).flatten()
        MAE_val, MCE_val, MSE_val, RMSE_val, R2_val, Bias_val, CVRMSE_val = evaluate_regress(
            y_val_pred_flat, y_val_true_flat, y_name=method_label_val
        )
        val_ev.append([MAE_val, MCE_val, MSE_val, RMSE_val,
                       R2_val, Bias_val, CVRMSE_val])

    # --- 3. 评估测试集 ---
    MAE_test, MCE_test, MSE_test, RMSE_test, R2_test, Bias_test, CVRMSE_test = evaluate_regress(
        y_test_pred_flat, y_test_true_flat, y_name=method_label_test
    )
    test_ev.append([MAE_test, MCE_test, MSE_test, RMSE_test,
                   R2_test, Bias_test, CVRMSE_test])

    return train_ev, test_ev, val_ev


# %%
# ==========================================
# 0_1. 缓存指纹与配置项准备
# ==========================================
def _file_version_fingerprint(path, edge_bytes=8 * 1024 * 1024):
    """Return a cheap content-aware fingerprint for very large source files.

    The digest covers the file size plus the first and last ``edge_bytes``.
    External static mappings are fingerprinted separately and included in the
    cache key whenever the production CSV does not contain BC.
    """
    path = os.path.abspath(path)
    stat = os.stat(path)
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with open(path, "rb") as handle:
        digest.update(handle.read(edge_bytes))
        if stat.st_size > edge_bytes:
            handle.seek(max(0, stat.st_size - edge_bytes))
            digest.update(handle.read(edge_bytes))
    return {
        "path": path,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "head_tail_sha256": digest.hexdigest(),
        "head_tail_bytes_each": int(edge_bytes),
    }


def _sha256_file(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _static_source_fingerprint(path):
    """Fingerprint a standalone table or every material component of a shapefile."""
    source_path = os.path.abspath(os.fspath(path))
    suffix = os.path.splitext(source_path)[1].lower()
    if suffix in {'.shp', '.dbf', '.shx', '.prj', '.cpg'}:
        base = os.path.splitext(source_path)[0]
        component_paths = [
            base + ext for ext in ('.shp', '.dbf', '.shx', '.prj', '.cpg')
            if os.path.exists(base + ext)
        ]
    else:
        component_paths = [source_path]
    if not component_paths:
        raise FileNotFoundError(f"BC 静态源不存在: {source_path}")
    components = {
        os.path.basename(component): _file_version_fingerprint(component)
        for component in component_paths
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(
        components, sort_keys=True, ensure_ascii=False
    ).encode('utf-8'))
    return {
        'path': source_path,
        'components': components,
        'combined_sha256': digest.hexdigest(),
    }


def _read_dbf_columns(dbf_path, requested_columns):
    """Read selected DBF attributes without requiring geopandas/GDAL."""
    import struct

    requested_columns = list(requested_columns)
    with open(dbf_path, 'rb') as handle:
        header = handle.read(32)
        if len(header) != 32:
            raise RuntimeError(f"无效 DBF 文件头: {dbf_path}")
        n_records = struct.unpack('<I', header[4:8])[0]
        header_length = struct.unpack('<H', header[8:10])[0]
        record_length = struct.unpack('<H', header[10:12])[0]
        fields = []
        while True:
            descriptor = handle.read(32)
            if not descriptor:
                raise RuntimeError(f"DBF 字段描述意外结束: {dbf_path}")
            if descriptor[0] == 0x0D:
                break
            name = descriptor[:11].split(b'\x00', 1)[0].decode('ascii')
            fields.append({
                'name': name,
                'type': chr(descriptor[11]),
                'length': int(descriptor[16]),
            })
        field_names = [field['name'] for field in fields]
        missing = [name for name in requested_columns if name not in field_names]
        if missing:
            raise RuntimeError(f"BC 权威 DBF 缺少字段: {missing}")
        offsets = {}
        offset = 1
        for field in fields:
            offsets[field['name']] = (offset, field['length'], field['type'])
            offset += field['length']
        handle.seek(header_length)
        records = {name: [] for name in requested_columns}
        for _ in range(n_records):
            record = handle.read(record_length)
            if len(record) != record_length:
                raise RuntimeError(f"DBF 记录长度异常: {dbf_path}")
            if record[:1] == b'*':
                continue
            for name in requested_columns:
                start, width, field_type = offsets[name]
                raw = record[start:start + width].strip()
                if not raw:
                    value = np.nan if field_type in {'N', 'F'} else None
                elif field_type in {'N', 'F'}:
                    value = float(raw)
                else:
                    value = raw.decode('utf-8', errors='replace').strip()
                records[name].append(value)
    return pd.DataFrame(records)


def _locate_authoritative_bc_source():
    """Locate the user-confirmed Fishnet source; fail closed if unavailable."""
    configured = (
        os.environ.get('UHI_BC_STATIC_MAPPING')
        or os.environ.get('BC_STATIC_MAPPING_PATH')
    )
    candidates = [configured] if configured else []
    candidates.extend([
        os.path.join(DATA_BASE_PATH, 'Fishnet', 'Fishnet.shp'),
        os.path.join(DATA_BASE_PATH, 'Fishnet.shp'),
        os.path.join(DATA_BASE_PATH, 'Fishnet', 'Fishnet.dbf'),
        os.path.join(DATA_BASE_PATH, 'BC_Grid_Mapping.csv'),
    ])
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "生产 CSV 不含 BC，且未找到权威 Fishnet。请把 Fishnet 上传到 "
        "/home/ubuntu/Fishnet/，或设置 UHI_BC_STATIC_MAPPING；已停止训练。"
    )


def _load_authoritative_bc_mapping(source_path):
    """Load and validate Grid_ID-level BC from the authoritative Fishnet."""
    suffix = os.path.splitext(source_path)[1].lower()
    if suffix in {'.shp', '.dbf'}:
        dbf_path = os.path.splitext(source_path)[0] + '.dbf'
        mapping = _read_dbf_columns(dbf_path, ['Grid_ID', 'City_Name', 'BC'])
    elif suffix == '.csv':
        mapping = pd.read_csv(
            source_path, usecols=lambda col: col in {'Grid_ID', 'City_Name', 'BC'},
            encoding='utf-8-sig',
        )
    elif suffix in {'.parquet', '.pq'}:
        mapping = pd.read_parquet(source_path)
        mapping = mapping[[
            col for col in ['Grid_ID', 'City_Name', 'BC'] if col in mapping.columns
        ]]
    else:
        raise RuntimeError(f"不支持的 BC 静态源格式: {source_path}")

    required = {'Grid_ID', 'BC'}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise RuntimeError(f"BC 权威源缺少字段: {missing}")
    mapping = mapping.copy()
    mapping['Grid_ID'] = pd.to_numeric(mapping['Grid_ID'], errors='raise').astype(np.int64)
    mapping['BC'] = pd.to_numeric(mapping['BC'], errors='raise')
    if mapping['BC'].isna().any():
        raise RuntimeError("BC 权威源存在缺失值，已停止训练")
    bc_min = float(mapping['BC'].min())
    bc_max = float(mapping['BC'].max())
    if bc_min < 0.0 or bc_max > 1.0:
        raise RuntimeError(
            f"BC 权威源范围为 [{bc_min}, {bc_max}]，不符合已核验的 0-1 coverage fraction；"
            "已停止训练，禁止把 BC 与 HBC 或百分数混用。"
        )

    if mapping['Grid_ID'].duplicated().any():
        if 'City_Name' not in mapping.columns:
            raise RuntimeError(
                "BC 权威源 Grid_ID 非全局唯一且缺少 City_Name，无法安全映射"
            )
        mapping['City_Name'] = mapping['City_Name'].astype(str).str.strip()
        key_cols = ['City_Name', 'Grid_ID']
        if mapping.duplicated(key_cols).any():
            raise RuntimeError("BC 权威源 [City_Name, Grid_ID] 仍不唯一")
    else:
        key_cols = ['Grid_ID']
    mapping = mapping[key_cols + ['BC']].copy()
    return mapping, key_cols, {'min': bc_min, 'max': bc_max, 'unit': 'fraction (0-1)'}


def _attach_authoritative_bc(frame, mapping, key_cols):
    """Attach BC without modifying the source CSV and require complete coverage."""
    if key_cols == ['Grid_ID']:
        bc_lookup = mapping.set_index('Grid_ID')['BC']
        frame['BC'] = frame['Grid_ID'].map(bc_lookup).astype(np.float32)
    else:
        if 'City_Name' not in frame.columns:
            raise RuntimeError("复合 BC 映射需要 long data 中存在 City_Name")
        frame = frame.merge(
            mapping, on=key_cols, how='left', validate='many_to_one', sort=False,
        )
        frame['BC'] = frame['BC'].astype(np.float32)
    missing_rows = int(frame['BC'].isna().sum())
    if missing_rows:
        missing_grids = int(frame.loc[frame['BC'].isna(), 'Grid_ID'].nunique())
        raise RuntimeError(
            f"BC 映射不完整: {missing_rows:,} 行、{missing_grids:,} 个 Grid_ID 缺失；"
            "已停止训练"
        )
    return frame


def _validate_bc_is_grid_static(frame, key_cols):
    """Require one and only one BC value for each physical grid."""
    counts = frame.groupby(key_cols, observed=True, sort=False)['BC'].nunique(dropna=False)
    bad = counts[counts != 1]
    if not bad.empty:
        raise RuntimeError(
            f"发现 {len(bad):,} 个物理网格的 BC 随小时变化；已停止训练"
        )
    return int(len(counts))


def build_cache_key(data_format, train_mode, use_cols, feature_cols,
                     target_col=None, source_fingerprint=None,
                     feature_schema_version=None, bc_source_fingerprint=None):
    cache_config = {
        "data_format": data_format,
        "train_mode": train_mode,
        "use_cols": list(use_cols) if use_cols is not None else None,
        "feature_cols": list(feature_cols) if feature_cols is not None else None,
        "target_col": target_col,
        "feature_schema_version": feature_schema_version,
        "source_head_tail_sha256": (
            source_fingerprint.get("head_tail_sha256")
            if source_fingerprint else None
        ),
        "source_size_bytes": (
            source_fingerprint.get("size_bytes")
            if source_fingerprint else None
        ),
        "bc_source_combined_sha256": (
            bc_source_fingerprint.get('combined_sha256')
            if bc_source_fingerprint else None
        ),
    }
    return hashlib.md5(
        json.dumps(cache_config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:10]

if DATA_FORMAT == 'long':
    # ── 配置读取列：FAR 仅保留作数据血缘 QA，不进入任何建模矩阵 ──
    use_cols_long = [
        'Grid_ID', 'Season', 'Temperature',
        'Hour_cos', 'Temp_2m',
        'longitude', 'latitude', 'MBH', 'BC', 'FAR', 'CAR',
        'NPP_VIIRS', 'NDVI', 'Road_width', 'Road_Dens',
        'Pop2025', 'water', 'ED', 'CONTAG', 'PLAND', 'PD', 'COHESION', 'HBC',
        'Class', 'LCZ_Function', 'City_Name'
    ]

    # ── 配置特征列与目标列 ──
    # Hour 相关只保留 Hour_cos（删除 Hour、Hour_sin），复现 v20260820 版本。
    #       Class 在数据读取后已 rename 为 LU_Type，此处用 LU_Type。
    feature_cols = [
        'Hour_cos', 'Temp_2m',
        'longitude', 'latitude', 'MBH', 'BC', 'CAR',
        'NPP_VIIRS', 'NDVI', 'Road_width', 'Road_Dens',
        'Pop2025', 'water', 'ED', 'CONTAG', 'PLAND', 'PD', 'COHESION', 'HBC',
        'LU_Type', 'LCZ_Function', 'Season_code'
    ]
    if len(feature_cols) != 22:
        raise RuntimeError(f"BC-noFAR ML schema 必须恰好包含 22 个变量，实际={len(feature_cols)}")
    if 'Hour_sin' in feature_cols or 'Hour_cos' not in feature_cols:
        raise RuntimeError("v20260820 配置必须删除 Hour_sin 且保留 Hour_cos")
    if 'BC' not in feature_cols:
        raise RuntimeError("正式 ML schema 缺少 BC")
    if 'FAR' in feature_cols:
        raise RuntimeError("FAR 不得进入正式 ML predictors")
    if 'Pop2025' not in feature_cols:
        raise RuntimeError("v20260820 BC-noFAR 配置必须保留 Pop2025")
    target_col = 'UHII'            # 数据列名（读取后已 rename: Temperature→UHII）
    _TARGET_LABEL = 'UHII'         # 统一标签，所有输出（CSV/DAG/图/日志）都用此名

    csv_path = os.path.join(DATA_BASE_PATH, "Sixth_export_Cleaned_full_static_fixed.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"正式输入数据不存在: {csv_path}")
    _SOURCE_DATA_FINGERPRINT = _file_version_fingerprint(csv_path)

    # 只读表头审计：生产 CSV 有 BC 时直接使用；否则从权威 Fishnet 映射。
    _source_header = pd.read_csv(
        csv_path, nrows=0, encoding='utf-8-sig'
    ).columns.tolist()
    _SOURCE_HAS_BC = 'BC' in _source_header
    _BC_MAPPING = None
    _BC_MAPPING_KEYS = ['Grid_ID']
    _BC_SOURCE_FINGERPRINT = None
    if _SOURCE_HAS_BC:
        _BC_SOURCE_PATH = csv_path
        _BC_SOURCE_STATS = {'unit': 'fraction (0-1)'}
        source_use_cols_long = use_cols_long
        print("✅ 生产 long CSV 已包含 BC，将直接使用并执行静态一致性 QA")
    else:
        _BC_SOURCE_PATH = _locate_authoritative_bc_source()
        _BC_SOURCE_FINGERPRINT = _static_source_fingerprint(_BC_SOURCE_PATH)
        _BC_MAPPING, _BC_MAPPING_KEYS, _BC_SOURCE_STATS = \
            _load_authoritative_bc_mapping(_BC_SOURCE_PATH)
        source_use_cols_long = [col for col in use_cols_long if col != 'BC']
        print(f"✅ 生产 long CSV 不含 BC；权威映射源: {_BC_SOURCE_PATH}")
        print(f"   映射键: {_BC_MAPPING_KEYS}; BC 范围: "
              f"[{_BC_SOURCE_STATS['min']:.8f}, {_BC_SOURCE_STATS['max']:.8f}]")

    # 配置、22变量 schema、源 CSV 与 BC 权威源共同决定缓存名。
    cache_key = build_cache_key(
        DATA_FORMAT, TRAIN_MODE, use_cols_long, feature_cols, target_col,
        source_fingerprint=_SOURCE_DATA_FINGERPRINT,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        bc_source_fingerprint=_BC_SOURCE_FINGERPRINT,
    )

    parquet_path = os.path.join(DATA_BASE_PATH, f"Sixth_export_Cleaned_{cache_key}.parquet")
    clean_parquet_path = os.path.join(DATA_BASE_PATH, f"Sixth_export_Cleaned_ready_{cache_key}.parquet")
    meta_path = os.path.join(DATA_BASE_PATH, f"Sixth_export_Cleaned_ready_{cache_key}.json")
    
    print(f"\n====== ⚡ 启用特征指纹缓存机制 (Cache Key: {cache_key}) ======")
    
    # 写入元数据方便排查
    meta_info = {
        "cache_key": cache_key,
        "DATA_FORMAT": DATA_FORMAT,
        "TRAIN_MODE": TRAIN_MODE,
        "use_cols_long": use_cols_long,
        "source_use_cols_long": source_use_cols_long,
        "feature_cols": feature_cols,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target_col": target_col,
        "source_data_fingerprint": _SOURCE_DATA_FINGERPRINT,
        "source_csv_contains_BC": _SOURCE_HAS_BC,
        "BC_source_path": _BC_SOURCE_PATH,
        "BC_source_fingerprint": _BC_SOURCE_FINGERPRINT,
        "BC_mapping_keys": _BC_MAPPING_KEYS,
        "BC_unit": _BC_SOURCE_STATS['unit'],
        "run_instance_id": _RUN_INSTANCE_ID,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

else:
    # 宽格式特征
    feature_cols = ['MBH', 'BC', 'CAR', 'NPP_VIIRS',
                    'Road_width', 'Road_Dens', 'NDVI',
                    'Pop2025', 'water', 'ED', 'CONTAG',
                    'PLAND', 'PD', 'COHESION', 'HBC',
                    'LU_Type', 'LCZ_Function',
                    'longitude', 'latitude']
    target_cols = [f'Hour_{i:02d}' for i in range(24)]

# ==========================================
# 1. 原始数据读取
# ==========================================
if DATA_FORMAT == 'long':
    dtype_dict = {'Season': 'category', 'Grid_ID': 'int32'}
    if 'City_Name' in use_cols_long:
        dtype_dict['City_Name'] = 'category'

    for c in source_use_cols_long:
        if c not in dtype_dict:
            dtype_dict[c] = 'float32'

    if os.path.exists(parquet_path):
        print(f"🚀 发现原始数据 Parquet 缓存，精准读取: {parquet_path}")
        df = pd.read_parquet(parquet_path, columns=use_cols_long)
    else:
        print(f"📄 无受指纹匹配的原始缓存，从 CSV 加载...")
        df = pd.read_csv(
            csv_path,
            usecols=source_use_cols_long,
            dtype=dtype_dict,
            encoding='utf-8-sig',
            engine='pyarrow'
        )
        if not _SOURCE_HAS_BC:
            df = _attach_authoritative_bc(
                df, _BC_MAPPING, _BC_MAPPING_KEYS,
            )
        for c in df.columns:
            if c == 'Grid_ID':
                df[c] = df[c].astype('int32')
            elif c == 'Season':
                df[c] = df[c].astype('category')
            elif c == 'City_Name':
                df[c] = df[c].astype('category')
            else:
                if str(df[c].dtype) != 'category':
                    df[c] = df[c].astype('float32')
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
        print(f"✅ 新原始 Parquet 缓存已生成: {parquet_path}")
else:
    data_path = os.path.join(DATA_BASE_PATH, "third_export_enhanced.csv")
    print("\n====== 正在读取并清洗数据 ======")
    df = pd.read_csv(data_path, encoding='utf-8-sig')

print(f"数据加载成功: {df.shape}")
print(f"数据格式: {DATA_FORMAT}")

if DATA_FORMAT == 'long':
    # Grid_ID is the physical split key. Report and stop if it is reused by
    # different cities instead of silently changing the established design.
    # One grouped pass performs both split-key and time-static BC checks.
    if df['BC'].isna().any():
        raise RuntimeError("加载后的 BC 存在缺失值，已停止训练")
    _grid_static_qa = df.groupby(
        'Grid_ID', observed=True, sort=False
    )[['City_Name', 'BC']].nunique(dropna=False).rename(columns={
        'City_Name': 'city_count', 'BC': 'bc_count',
    })
    _cross_city_grid_ids = _grid_static_qa.loc[
        _grid_static_qa['city_count'] > 1
    ]
    if not _cross_city_grid_ids.empty:
        raise RuntimeError(
            f"发现 {_cross_city_grid_ids.size:,} 个 Grid_ID 跨 City_Name 重复；"
            "按预设要求已停止，未擅自改变 physical Grid_ID split。"
        )
    _non_static_bc_grids = _grid_static_qa.loc[
        _grid_static_qa['bc_count'] != 1
    ]
    if not _non_static_bc_grids.empty:
        raise RuntimeError(
            f"发现 {len(_non_static_bc_grids):,} 个物理网格的 BC 随小时变化；"
            "已停止训练"
        )
    _n_bc_grids = int(len(_grid_static_qa))
    _bc_min_loaded = float(pd.to_numeric(df['BC'], errors='raise').min())
    _bc_max_loaded = float(pd.to_numeric(df['BC'], errors='raise').max())
    if _bc_min_loaded < 0.0 or _bc_max_loaded > 1.0:
        raise RuntimeError(
            f"加载后的 BC 范围 [{_bc_min_loaded}, {_bc_max_loaded}] 超出 0-1；"
            "BC/HBC 或 fraction/% 可能混用，已停止训练。"
        )
    _bc_audit = {
        'run_version': _RUN_VERSION,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'source_csv_contains_BC': bool(_SOURCE_HAS_BC),
        'BC_source_path': _BC_SOURCE_PATH,
        'BC_mapping_key': '+'.join(_BC_MAPPING_KEYS),
        'physical_grid_count': _n_bc_grids,
        'cross_city_duplicate_Grid_ID_count': int(_cross_city_grid_ids.size),
        'BC_missing_rows': int(df['BC'].isna().sum()),
        'BC_min': _bc_min_loaded,
        'BC_max': _bc_max_loaded,
        'BC_unit': 'fraction (0-1)',
        'BC_static_within_grid': True,
        'original_csv_modified': False,
    }
    _bc_audit_path = os.path.join(CORR_DIR, 'BC_grid_coverage_audit.csv')
    pd.DataFrame([_bc_audit]).to_csv(
        _bc_audit_path, index=False, encoding='utf-8-sig'
    )
    _feature_schema = {
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'run_version': _RUN_VERSION,
        'number_of_features': len(feature_cols),
        'feature_cols': feature_cols,
        'FAR_in_predictors': 'FAR' in feature_cols,
        'Pop2025_in_predictors': 'Pop2025' in feature_cols,
        'BC_in_predictors': 'BC' in feature_cols,
        'BC_source_path': _BC_SOURCE_PATH,
        'BC_source_fingerprint': _BC_SOURCE_FINGERPRINT,
        'BC_unit': 'fraction (0-1)',
        'source_csv_modified': False,
    }
    _feature_schema_path = os.path.join(
        BASE_OUTPUT_DIR, 'Feature_Schema_BC_noFAR.json'
    )
    with open(_feature_schema_path, 'w', encoding='utf-8') as handle:
        json.dump(_feature_schema, handle, ensure_ascii=False, indent=2)
    print(f"✅ BC Grid-static QA 通过: {_n_bc_grids:,} 个物理网格")
    print(f"✅ BC-noFAR、无Hour_sin的22变量 schema 已保存: {_feature_schema_path}")

# ── 列名规范化：Class → LU_Type（Land Use Type，土地利用类型，论文标准命名）──
# 原始数据列名为 'Class'，此处统一重命名为论文常用英文名 LU_Type。
# 仅改 DataFrame 列名，不改动原始数据文件；下游 feature_cols / 因果排除列表均用 LU_Type。
if 'Class' in df.columns and 'LU_Type' not in df.columns:
    df = df.rename(columns={'Class': 'LU_Type'})
    print(f"✅ 列名规范化: 'Class' → 'LU_Type' (Land Use Type)")

# ── 列名规范化：Temperature → UHII（Urban Heat Island Intensity，城市热岛强度）──
# 原始 CSV 列名为 'Temperature'，此处统一重命名为研究目标变量 UHII。
# 仅改 DataFrame 列名，不改动原始数据文件；下游 target_col / DAG / 因果分析均用 UHII。
if 'Temperature' in df.columns and 'UHII' not in df.columns:
    df = df.rename(columns={'Temperature': 'UHII'})
    print(f"✅ 列名规范化: 'Temperature' → 'UHII' (Urban Heat Island Intensity)")

# ==========================================
# 添加气候区划列
# ==========================================
if TRAIN_MODE == 'by_climate':
    climate_zone_map = {
        'Harbin': '严寒', 'Urumqi': '严寒', 'Changchun': '严寒', 'Shenyang': '严寒', 'Hohhot': '严寒',
        'Beijing': '寒冷', 'Tianjin': '寒冷', 'Langfang': '寒冷', 'Tangshan': '寒冷', 'Dalian': '寒冷',
        'Baoding': '寒冷', 'Yinchuan': '寒冷', 'Cangzhou': '寒冷', 'Shijiazhuang': '寒冷', 'Taiyuan': '寒冷',
        'Yantai': '寒冷', 'Dezhou': '寒冷', 'Weihai': '寒冷', 'Xingtai': '寒冷', 'Zibo': '寒冷',
        'Jinan': '寒冷', 'Weifang': '寒冷', 'Handan': '寒冷', 'Liaocheng': '寒冷', 'Qingdao': '寒冷',
        'Lanzhou': '寒冷', 'Anyang': '寒冷', 'Jining': '寒冷', 'Xinxiang': '寒冷', 'Heze': '寒冷',
        'Linyi': '寒冷', 'Yuncheng': '寒冷', 'Zhengzhou': '寒冷', 'Kaifeng': '寒冷', 'Luoyang': '寒冷',
        'Lianyungang': '寒冷', 'Shangqiu': '寒冷', 'Xian': '寒冷', "Xi'an": '寒冷', 'Xuzhou': '寒冷',
        'Xuchang': '寒冷', 'Su_4_zhou': '寒冷', 'Nanyang': '寒冷', 'Zhumadian': '寒冷', 'Bengbu': '寒冷',
        'Fuyang': '寒冷',
        'Yancheng': '夏热冬冷', 'Tai_1_zhou': '夏热冬冷', 'Yangzhou': '夏热冬冷', 'Zhenjiang': '夏热冬冷',
        'Nanjing': '夏热冬冷', 'Xinyang': '夏热冬冷', 'Nantong': '夏热冬冷', 'Hefei': '夏热冬冷',
        'Suzhou': '夏热冬冷', 'Changzhou': '夏热冬冷', 'Wuxi': '夏热冬冷', 'Mianyang': '夏热冬冷',
        'Shanghai': '夏热冬冷', 'Wuhu': '夏热冬冷', 'Chengdu': '夏热冬冷', 'Huzhou': '夏热冬冷',
        'Jiaxing': '夏热冬冷', 'Yichang': '夏热冬冷', 'Wuhan': '夏热冬冷', 'Anqing': '夏热冬冷',
        'Hangzhou': '夏热冬冷', 'Shaoxing': '夏热冬冷', 'Ningbo': '夏热冬冷', 'Jiujiang': '夏热冬冷',
        'Chongqing': '夏热冬冷', 'Yueyang': '夏热冬冷', 'Jinhua': '夏热冬冷', 'Nanchang': '夏热冬冷',
        'Tai_4_zhou': '夏热冬冷', 'Taizhou': '夏热冬冷', 'Shangrao': '夏热冬冷', 'Changsha': '夏热冬冷',
        'Wenzhou': '夏热冬冷', 'Zhuzhou': '夏热冬冷', 'Yichun': '夏热冬冷', 'Zunyi': '夏热冬冷',
        'Hengyang': '夏热冬冷', 'Ganzhou': '夏热冬冷', 'Guilin': '夏热冬冷',
        'Fuzhou': '夏热冬暖', 'Putian': '夏热冬暖', 'Quanzhou': '夏热冬暖', 'Xiamen': '夏热冬暖',
        'Zhangzhou': '夏热冬暖', 'Liuzhou': '夏热冬暖', 'Chaozhou': '夏热冬暖', 'Shantou': '夏热冬暖',
        'Guangzhou': '夏热冬暖', 'Foshan': '夏热冬暖', 'Huizhou': '夏热冬暖', 'Zhaoqing': '夏热冬暖',
        'Dongguan': '夏热冬暖', 'Nanning': '夏热冬暖', 'Shenzhen': '夏热冬暖', 'Jiangmen': '夏热冬暖',
        'Zhongshan': '夏热冬暖', 'Zhuhai': '夏热冬暖', 'Haikou': '夏热冬暖',
        'Kunming': '温和', 'Guiyang': '温和',
    }
    df['City'] = df['City_Name'].str.replace('_2021', '')
    df['Climate_Zone'] = df['City'].map(climate_zone_map)

# ==========================================
# 添加 Season 数值编码
# ==========================================
if 'Season' in df.columns:
    df['Season_code'] = (df['Season'] == 'summer').astype(np.int8)
    print(f"\n✅ Season 编码完成：summer=1, winter=0")
else:
    if DATA_FORMAT == 'long':
        df['Season_code'] = np.int8(0)   # 兼容旧数据集

# 检查未匹配的城市
if TRAIN_MODE == 'by_climate':
    unmapped = df[df['Climate_Zone'].isna()]['City'].unique()
    if len(unmapped) > 0:
        print(f"\n⚠️ 以下城市未匹配气候区划: {list(unmapped)}")
        df['Climate_Zone'] = df['Climate_Zone'].fillna('未知')
    print(f"\n✅ 气候区划添加完成")
    climate_map = {'严寒': 1, '寒冷': 2, '夏热冬冷': 3, '夏热冬暖': 4, '温和': 5}
    df['Climate_Code'] = df['Climate_Zone'].map(climate_map).fillna(0).astype(np.int8)

# ==========================================
# 2. 特征打印
# ==========================================
print(f"\n特征数量: {len(feature_cols)}")
print(f"目标变量: {target_cols if DATA_FORMAT != 'long' and PREDICT_MODE == 'all' else (target_col if DATA_FORMAT == 'long' else f'Hour_{TARGET_HOUR:02d}')}")

# ==========================================
# 3. 处理缺失值（按 Grid_ID 整体剔除）
# ==========================================
print("\n检查缺失值...")
print(f"原始数据形状: {df.shape}")
original_df_shape = df.shape[0]

# 缓存配置指纹路径已在顶层生成

if DATA_FORMAT == 'long' and os.path.exists(clean_parquet_path):
    print(f"🚀 发现清洗后缓存，直接读取: {clean_parquet_path}")
    df_clean = pd.read_parquet(clean_parquet_path)
    # 防御性 rename：旧缓存文件（target_col 还叫 Temperature 时生成的）
    # 列名可能是 'Temperature'。cache_key 已含 target_col 会触发重建，
    # 但为防手工搬运缓存文件导致 KeyError，这里再兜底 rename 一次。
    if 'Temperature' in df_clean.columns and 'UHII' not in df_clean.columns:
        df_clean = df_clean.rename(columns={'Temperature': 'UHII'})
        print(f"  (防御性 rename: Temperature → UHII)")
    print(f"\n清洗后数据形状: {df_clean.shape}")
else:
    if DATA_FORMAT == 'long':
        # ── Grid_ID 级别清洗：24行中有任何缺失值，整个Grid_ID的24行全部剔除 ──
        total_grids = df['Grid_ID'].nunique()
        total_rows = len(df)

        # 一次性检查所有特征列 + 目标列的缺失情况
        all_check_cols = [c for c in (feature_cols + [target_col]) if c in df.columns]
        grids_with_nan = df[df[all_check_cols].isna().any(axis=1)]['Grid_ID'].unique()
        valid_grids = total_grids - len(grids_with_nan)

        print(f"\n📊 Grid_ID 级别清洗统计:")
        print(f"   总 Grid_ID 数量: {total_grids:,}")
        print(f"   含缺失值的 Grid_ID: {len(grids_with_nan):,}")
        print(f"   保留的完整 Grid_ID: {valid_grids:,}")
        print(f"   Grid 存活率: {valid_grids/total_grids*100:.1f}%")

        # 整体剔除不完整的 Grid_ID 的全部行
        import gc
        mask_keep = ~df['Grid_ID'].isin(set(grids_with_nan))
        df_clean = df[mask_keep]
        del df
        gc.collect()
        dropped_rows = total_rows - len(df_clean)
        print(f"   剔除行数: {dropped_rows:,} (涉及 {len(grids_with_nan):,} 个 Grid_ID)")
        print(f"   清洗后行数: {len(df_clean):,}")

        # 缓存第一道清洗结果
        df_clean.to_parquet(clean_parquet_path, engine='pyarrow',
                            compression='snappy', index=False)
        print(f"✅ 清洗后缓存已生成: {clean_parquet_path}")
    else:
        if PREDICT_MODE == 'single':
            df_clean = df.dropna(subset=feature_cols +
                                 [f'Hour_{TARGET_HOUR:02d}'])
        else:
            df_clean = df.dropna(subset=feature_cols + target_cols)

    print(f"\n清洗后数据形状: {df_clean.shape}")
    print(f"总计删除 {original_df_shape - df_clean.shape[0]:,} 行数据")

# ==========================================
# 4. 提取特征和目标变量
# ==========================================
X = df_clean[feature_cols].values.astype(np.float32)

if DATA_FORMAT == 'long':
    y = df_clean[target_col].values.astype(np.float32)
    print(f"\n预测模式: 长格式（预测所有24小时）")
else:
    if PREDICT_MODE == 'single':
        y = df_clean[f'Hour_{TARGET_HOUR:02d}'].values.astype(np.float32)
        print(f"\n预测模式: 单小时 (Hour_{TARGET_HOUR:02d})")
    else:
        y = df_clean[target_cols[0]].values.astype(np.float32)
        print(f"\n预测模式: 24小时全预测")

# ==========================================
# 5. 按训练模式处理数据
# ==========================================

# ==========================================
# 5. 按训练模式处理数据
# ==========================================

test_group_ids = None

if TRAIN_MODE == 'all':
    # 全部数据一起训练
    print(f"\n训练模式: 全部数据")

    if DATA_FORMAT == 'long':
        # 长格式：按 Grid_ID 随机划分 → Train(64%) / Val(16%) / Test(20%)
        # 每个 Grid_ID 的 24 小时数据保持在一起，防止数据泄漏
        unique_grids = df_clean['Grid_ID'].unique()
        print(f"\nGrid_ID 数量: {len(unique_grids):,}")

        # 第一步：按 Grid_ID 8:2 拆出测试集
        train_val_grids, test_grids = train_test_split(
            unique_grids, test_size=0.2, random_state=42
        )
        # 第二步：对 train_val Grid_ID 再 8:2 拆出验证集
        train_grids, val_grids = train_test_split(
            train_val_grids, test_size=0.2, random_state=42
        )
        print(f"训练 Grid: {len(train_grids):,}, 验证 Grid: {len(val_grids):,}, 测试 Grid: {len(test_grids):,}")

        _split_records = pd.concat([
            pd.DataFrame({'Grid_ID': train_grids, 'Split': 'train'}),
            pd.DataFrame({'Grid_ID': val_grids, 'Split': 'validation'}),
            pd.DataFrame({'Grid_ID': test_grids, 'Split': 'test'}),
        ], ignore_index=True)
        if _split_records['Grid_ID'].duplicated().any():
            raise RuntimeError("Grid split 中出现重复 Grid_ID，已停止训练")
        _split_path = os.path.join(
            BASE_OUTPUT_DIR, 'Grid_Split_BC_noFAR.csv'
        )
        _split_records.to_csv(_split_path, index=False, encoding='utf-8-sig')
        np.save(os.path.join(BASE_OUTPUT_DIR, 'train_grids.npy'), train_grids)
        np.save(os.path.join(BASE_OUTPUT_DIR, 'val_grids.npy'), val_grids)
        np.save(os.path.join(BASE_OUTPUT_DIR, 'test_grids.npy'), test_grids)
        print(f"✅ 可复现 Grid split 已保存: {_split_path}")

        train_mask = df_clean['Grid_ID'].isin(train_grids)
        val_mask = df_clean['Grid_ID'].isin(val_grids)
        test_mask = df_clean['Grid_ID'].isin(test_grids)

        X_train = df_clean.loc[train_mask, feature_cols].values
        X_val = df_clean.loc[val_mask, feature_cols].values
        X_test = df_clean.loc[test_mask, feature_cols].values
        y_train = df_clean.loc[train_mask, target_col].values
        y_val = df_clean.loc[val_mask, target_col].values
        y_test = df_clean.loc[test_mask, target_col].values
        test_group_ids = df_clean.loc[test_mask, 'Grid_ID'].to_numpy()
    else:
        # 宽格式：随机划分 → Train(64%) / Val(16%) / Test(20%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=df_clean['Climate_Code']
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42,
            stratify=df_clean.loc[df_clean.index.isin(
                pd.Index(range(len(y_train_val)))), 'Climate_Code'] if 'Climate_Code' in df_clean.columns else None
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

elif TRAIN_MODE == 'by_climate':
    # 按气候带分别训练
    print(f"\n训练模式: 按气候带分别训练")

    # 确定要训练的气候带
    if SELECTED_CLIMATES == 'all':
        climates_to_train = [1, 2, 3, 4, 5]
    else:
        climates_to_train = SELECTED_CLIMATES

    climate_names = {1: '严寒', 2: '寒冷', 3: '夏热冬冷', 4: '夏热冬暖', 5: '温和'}
    print(f"选择的气候带: {[f'{c}({climate_names[c]})' for c in climates_to_train]}")

    # 存储每个气候带的数据和模型
    climate_data = {}
    climate_models = {}
    climate_results = {}

    for climate_code in climates_to_train:
        print(f"\n{'='*50}")
        print(f"处理气候带 {climate_code} ({climate_names[climate_code]})")
        print(f"{'='*50}")

        # 筛选该气候带数据
        mask = df_clean['Climate_Code'] == climate_code
        X_climate = X[mask]
        y_climate = y[mask]

        print(f"该气候带样本数: {len(X_climate)}")

        if len(X_climate) < 10:
            print(f"⚠️ 样本数过少，跳过该气候带")
            continue

        # 划分训练测试集
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_climate, y_climate, test_size=0.2, random_state=42
        )

        # 归一化并锁定 float32 抢救内存
        scaler_c = StandardScaler()
        X_train_c = scaler_c.fit_transform(X_train_c).astype(np.float32)
        X_test_c = scaler_c.transform(X_test_c).astype(np.float32)

        # 保存数据
        climate_data[climate_code] = {
            'X_train': X_train_c,
            'X_test': X_test_c,
            'y_train': y_train_c,
            'y_test': y_test_c,
            'scaler': scaler_c
        }

        print(f"训练集: {X_train_c.shape}, 测试集: {X_test_c.shape}")

# ==========================================
# by_season：按夏冬季节划分数据
# ==========================================
if TRAIN_MODE == 'by_season':
    print(f"\n训练模式: 夏冬双模型并行")
    season_data = {}
    season_name_map = {'summer': '夏季', 'winter': '冬季'}

    # 动态判断数据中存在哪些季节
    available_seasons = []
    if 1 in df_clean['Season_code'].values:
        available_seasons.append('summer')
    if 0 in df_clean['Season_code'].values:
        available_seasons.append('winter')

    for season_key in available_seasons:
        print(f"\n{'='*50}")
        print(f"处理季节: {season_name_map[season_key]}")
        print(f"{'='*50}")

        s_code = 1 if season_key == 'summer' else 0
        mask = df_clean['Season_code'] == s_code
        df_s = df_clean[mask]

        unique_cities_s = df_s['City_Name'].unique()
        # 第一步：按城市 8:2 拆出测试集
        train_val_cities_s, test_cities_s = train_test_split(
            unique_cities_s, test_size=0.2, random_state=42
        )
        # 第二步：对 train_val 城市再 8:2 拆出验证集
        train_cities_s, val_cities_s = train_test_split(
            train_val_cities_s, test_size=0.2, random_state=42
        )
        print(f"  训练城市: {len(train_cities_s)}, 验证城市: {len(val_cities_s)}, 测试城市: {len(test_cities_s)}")

        X_train_s = df_s.loc[df_s['City_Name'].isin(
            train_cities_s), feature_cols].values
        X_val_s = df_s.loc[df_s['City_Name'].isin(
            val_cities_s), feature_cols].values
        X_test_s = df_s.loc[df_s['City_Name'].isin(
            test_cities_s),  feature_cols].values
        y_train_s = df_s.loc[df_s['City_Name'].isin(
            train_cities_s), target_col].values
        y_val_s = df_s.loc[df_s['City_Name'].isin(
            val_cities_s), target_col].values
        y_test_s = df_s.loc[df_s['City_Name'].isin(
            test_cities_s),  target_col].values
        test_group_ids_s = (
            df_s.loc[df_s['City_Name'].isin(test_cities_s), 'Grid_ID'].to_numpy()
            if 'Grid_ID' in df_s.columns else None
        )

        scaler_s = StandardScaler()
        X_train_s = scaler_s.fit_transform(X_train_s).astype(np.float32)
        X_val_s = scaler_s.transform(X_val_s).astype(np.float32)
        X_test_s = scaler_s.transform(X_test_s).astype(np.float32)

        season_data[season_key] = {
            'X_train': X_train_s, 'X_test': X_test_s,
            'y_train': y_train_s, 'y_test': y_test_s,
            'X_val': X_val_s, 'y_val': y_val_s,
            'scaler': scaler_s,
            'test_group_ids': test_group_ids_s,
        }
        print(
            f"  {season_name_map[season_key]} 训练集: {X_train_s.shape}, 验证集: {X_val_s.shape}, 测试集: {X_test_s.shape}")

# ==========================================
# 兼容性处理：为 Pearson/VIF 提供默认变量
# ==========================================
if TRAIN_MODE == 'by_climate':
    first_climate = climates_to_train[0]
    X_train = climate_data[first_climate]['X_train']
    X_test = climate_data[first_climate]['X_test']
    y_train = climate_data[first_climate]['y_train']
    y_test = climate_data[first_climate]['y_test']
    scaler = climate_data[first_climate]['scaler']
    print(f"\n⚠️ 后续分析使用气候带 {first_climate} 的数据")
elif TRAIN_MODE == 'by_season':
    # 默认选用有的第一个季节做 Pearson/VIF 分析
    first_season = next(iter(season_data)) if season_data else 'summer'
    X_train = season_data[first_season]['X_train']
    X_test = season_data[first_season]['X_test']
    y_train = season_data[first_season]['y_train']
    y_test = season_data[first_season]['y_test']
    scaler = season_data[first_season]['scaler']
    X_val = season_data[first_season].get('X_val')
    y_val = season_data[first_season].get('y_val')
    print(f"\n⚠️ Pearson/VIF 分析默认使用 {first_season} 数据")


print(f"\n✅ 数据归一化完成（StandardScaler）")

# ==========================================
# 5. Pearson 相关性分析（夏冬各做一次 / 全量）
# ==========================================
print("\n====== 📈 正在生成 Pearson 特征相关性分析 ======")

output_dir_pearson = CORR_DIR

# 确定要分析的数据集列表
if TRAIN_MODE == 'by_season':
    _pearson_items = []
    if 'summer' in season_data:
        _pearson_items.append(
            ('summer', 'Summer', ('Season_code', 1)))
    if 'winter' in season_data:
        _pearson_items.append(
            ('winter', 'Winter', ('Season_code', 0)))
else:
    # suite_name 用纯英文，避免 Linux 服务器无中文字体导致图表标题乱码
    _pearson_items = [('all', 'All Data', None)]

target_name = 'UHII'   # Pearson 分析输出用的目标名（已与 target_col 统一）

for _pk, _pname, _row_filter in _pearson_items:
    print(f"\n  ── [{_pname}] Pearson 分析 ──")

    # 使用分块精确累计替代整表 copy + corr，避免 1 亿级数据触发内存峰值。
    if DATA_FORMAT == 'long':
        target_name = target_col
        corr_matrix = compute_pearson_corr_chunked_exact(
            data_source=df_clean,
            feature_cols=feature_cols,
            target_name=target_name,
            chunk_size=PEARSON_CHUNK_SIZE,
            row_filter=_row_filter,
            progress_label=f'{_pname} Pearson'
        )
    else:
        if PREDICT_MODE == 'single':
            target_name = f'Hour_{TARGET_HOUR:02d}'
            corr_matrix = compute_pearson_corr_chunked_exact(
                data_source=df_clean,
                feature_cols=feature_cols,
                target_name=target_name,
                chunk_size=PEARSON_CHUNK_SIZE,
                row_filter=_row_filter,
                progress_label=f'{_pname} Pearson'
            )
        else:
            target_name = 'Target_Mean'
            corr_matrix = compute_pearson_corr_chunked_exact(
                data_source=df_clean,
                feature_cols=feature_cols,
                target_name=target_name,
                chunk_size=PEARSON_CHUNK_SIZE,
                row_filter=_row_filter,
                derived_target_cols=[f'Hour_{i:02d}' for i in range(24)],
                progress_label=f'{_pname} Pearson'
            )

    prefix = f"{_pk}_" if TRAIN_MODE == 'by_season' else ""
    corr_csv_path = os.path.join(
        output_dir_pearson, f'{prefix}Pearson_Correlation_Matrix.csv'
    )
    corr_matrix.to_csv(corr_csv_path, encoding='utf-8-sig')
    _key_pair_rows = []
    for _other in ['MBH', 'CAR', 'HBC', 'NDVI', target_name]:
        if 'BC' in corr_matrix.index and _other in corr_matrix.columns:
            _key_pair_rows.append({
                'Suite': _pname,
                'Variable_1': 'BC',
                'Variable_2': _other,
                'Pearson_r': float(corr_matrix.loc['BC', _other]),
            })
    pd.DataFrame(_key_pair_rows).to_csv(
        os.path.join(output_dir_pearson, f'{prefix}BC_key_Pearson_pairs.csv'),
        index=False, encoding='utf-8-sig',
    )
    print(f"  ✅ [{_pname}] Pearson 数值矩阵已保存: {corr_csv_path}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(26, 11), gridspec_kw={
                             'width_ratios': [1.8, 1]})

    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .75}, ax=axes[0])
    axes[0].set_title(
        f'Pearson Correlation Heatmap [{_pname}]', fontsize=22, fontweight='bold', pad=20)
    plt.setp(axes[0].get_xticklabels(), rotation=45,
             ha='right', rotation_mode='anchor', fontsize=15)
    plt.setp(axes[0].get_yticklabels(), fontsize=15)

    target_corr = corr_matrix[target_name].drop(
        target_name).sort_values(ascending=False)
    colors = ['#d73027' if c > 0 else '#4575b4' for c in target_corr.values]
    sns.barplot(x=target_corr.values, y=target_corr.index,
                hue=target_corr.index, palette=colors, ax=axes[1], legend=False)
    axes[1].set_title(
        f'Pearson with UHI [{_pname}]', fontsize=22, fontweight='bold', pad=20)
    axes[1].set_xlabel('Correlation Coefficient (r)', fontsize=18)
    axes[1].set_ylabel('Features', fontsize=18)
    axes[1].axvline(x=0, color='black', linewidth=1)
    axes[1].grid(axis='x', linestyle='--', alpha=0.6)
    plt.setp(axes[1].get_yticklabels(), fontsize=15)
    plt.setp(axes[1].get_xticklabels(), fontsize=14)

    plt.tight_layout(pad=3.0)

    combined_path = os.path.join(
        output_dir_pearson, f'{prefix}Pearson_Correlation_Combined.jpg')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✅ [{_pname}] Pearson 图已保存: {combined_path}")


# ==========================================
# 6. VIF多重共线性检验 (满载计算优化版)
# ==========================================
print("\n" + "="*60)
print("多重共线性审查报告 (VIF Analysis)")
print("="*60)

# VIF is a training-set predictor diagnostic. The independent test set is not
# used, and no variable is removed automatically in response to this diagnostic.
print(f"\n[VIF分析] training-set predictor VIF: {X_train.shape[0]:,} 行 × "
      f"{X_train.shape[1]} 个预测变量...")

# 极速且绝对等价的数学法则: 标准化矩阵的相关矩阵 R 的逆的对角线即为 VIF
corr_matrix = np.corrcoef(X_train, rowvar=False)

try:
    inv_corr = np.linalg.inv(corr_matrix)
    vifs = np.diag(inv_corr)
except np.linalg.LinAlgError:
    # 万一出现极度完美的共线性导致矩阵奇异，退化使用广义逆矩阵
    inv_corr = np.linalg.pinv(corr_matrix)
    vifs = np.diag(inv_corr)

vif_data = pd.DataFrame()
vif_data["Feature"] = feature_cols
vif_data["VIF"] = vifs
vif_data = vif_data.sort_values('VIF', ascending=False)

print(f"\n{'特征名称':<20} {'VIF值':>12}")
print("-" * 60)
for idx, row in vif_data.iterrows():
    print(f"{row['Feature']:<20} {row['VIF']:>12.2f}")

# 保存VIF结果
vif_path = os.path.join(output_dir_pearson, 'VIF_Analysis.csv')
vif_data.to_csv(vif_path, index=False, encoding='utf-8-sig')
print(f"\n✅ VIF分析结果已保存: {vif_path}")
print("\n[VIF重点变量]")
for _focus_feature in ['BC', 'MBH', 'CAR', 'HBC']:
    _focus_rows = vif_data.loc[vif_data['Feature'] == _focus_feature, 'VIF']
    if not _focus_rows.empty:
        print(f"  {_focus_feature:<8s}: {float(_focus_rows.iloc[0]):.4f}")

# 预设停止门：VIF >= 10 时保留结果并停止，不自动删变量或重算。
high_vif = vif_data[vif_data['VIF'] >= 10.0]
if len(high_vif) > 0:
    _high_vif_text = ', '.join(
        f"{row.Feature}={row.VIF:.4f}" for row in high_vif.itertuples()
    )
    raise RuntimeError(
        f"VIF_STOP_GATE: {len(high_vif)} 个 predictor 的 training-set VIF >= 10: "
        f"{_high_vif_text}。已按预设停止，未自动删变量，也未开始正式模型训练。"
    )
print("\n✅ 所有 training-set predictor VIF < 10，允许继续正式模型训练")
print("="*60)

# ==========================================
# 3. 模型训练与预测
# ==========================================
method_label = 'LinearRegression'  # 可替换为其他模型标签，如 'RandomForest', 'XGBoost', 'LightGBM'

if TRAIN_MODE == 'all':
    # 全部数据一起训练
    print(f"\n🚀 开始训练模型: {method_label} (全部数据)")

    y_train_pred, y_test_pred, best_model, y_val_pred = train_muti_model(
        X_train, y_train, X_test, y_test, method_label,
        X_val=globals().get('X_val'),
        y_val=globals().get('y_val')
    )

    print(f"\n📊 正在评估结果 ...")
    train_metrics, test_metrics, val_metrics = test_method(
        y_train_pred, y_test_pred, y_train, y_test, method_label,
        val_y_predict=y_val_pred,
        val_y=globals().get('y_val')
    )

    val_r2_str = f"  验证集 R2: {val_metrics[0][4]:.4f}" if val_metrics else ""
    print("-" * 50)
    print(f"模型: {method_label}")
    print(f"训练集 R2: {train_metrics[0][4]:.4f}")
    print(val_r2_str)
    print(f"测试集 R2: {test_metrics[0][4]:.4f}")
    print("-" * 50)

elif TRAIN_MODE == 'by_climate':
    # 按气候带分别训练
    print(f"\n🚀 开始按气候带训练模型: {method_label}")

    for climate_code in climates_to_train:
        if climate_code not in climate_data:
            continue

        print(f"\n{'='*60}")
        print(f"训练气候带 {climate_code} ({climate_names[climate_code]})")
        print(f"{'='*60}")

        data = climate_data[climate_code]
        X_train_c = data['X_train']
        X_test_c = data['X_test']
        y_train_c = data['y_train']
        y_test_c = data['y_test']

        y_train_pred_c, y_test_pred_c, model_c, _ = train_muti_model(
            X_train_c, y_train_c, X_test_c, y_test_c, method_label
        )

        train_metrics_c, test_metrics_c, _ = test_method(
            y_train_pred_c, y_test_pred_c, y_train_c, y_test_c,
            f"{method_label}_Climate{climate_code}"
        )

        climate_models[climate_code] = model_c
        climate_results[climate_code] = {
            'train_r2': train_metrics_c[0][4],
            'test_r2': test_metrics_c[0][4],
            'train_rmse': train_metrics_c[0][3],
            'test_rmse': test_metrics_c[0][3]
        }

        print(
            f"气候带{climate_code} - 训练集R2: {train_metrics_c[0][4]:.4f}, 测试集R2: {test_metrics_c[0][4]:.4f}")

    print(f"\n{'='*60}")
    print("各气候带模型性能汇总")
    print(f"{'='*60}")
    for climate_code in climates_to_train:
        if climate_code in climate_results:
            res = climate_results[climate_code]
            print(
                f"气候带{climate_code}({climate_names[climate_code]}): R2={res['test_r2']:.4f}, RMSE={res['test_rmse']:.4f}")

elif TRAIN_MODE == 'by_season':
    # 夏冬双模型单次预览（详细结果见下方多模型循环）
    print(f"\n🚀 按季节训练单模型预览: {method_label}")
    season_name_map_disp = {'summer': '夏季', 'winter': '冬季'}
    preview_cache = {}  # 缓存夏冬结果供绘图使用
    for sk in ['summer', 'winter']:
        _d = season_data[sk]
        print(f"\n  ── {season_name_map_disp[sk]} ──")
        y_tr_p, y_te_p, _m, y_va_p = train_muti_model(
            _d['X_train'], _d['y_train'], _d['X_test'], _d['y_test'], method_label,
            X_val=_d.get('X_val'), y_val=_d.get('y_val')
        )
        tr_ev, te_ev, va_ev = test_method(y_tr_p, y_te_p, _d['y_train'], _d['y_test'],
                                   f"{method_label}_{sk}",
                                   val_y_predict=y_va_p, val_y=_d.get('y_val'))
        val_r2_str = f"  验证R2: {va_ev[0][4]:.4f}" if va_ev else ""
        print(
            f"  [{season_name_map_disp[sk]}] 训练R2: {tr_ev[0][4]:.4f}  测试R2: {te_ev[0][4]:.4f}{val_r2_str}")

        # 存入缓存字典
        preview_cache[sk] = {
            'y_train_pred': y_tr_p, 'y_test_pred': y_te_p,
            'y_val_pred': y_va_p,
            'train_metrics': tr_ev, 'test_metrics': te_ev,
            'val_metrics': va_ev,
            'model': _m
        }

        if sk == 'summer':
            # 保存夏季结果作为默认变量（向下兼容部分全局变量）
            y_train_pred, y_test_pred, best_model = y_tr_p, y_te_p, _m
            y_val_pred = y_va_p
            train_metrics, test_metrics = tr_ev, te_ev
            val_metrics = va_ev

print("\n✅ 流程执行完毕！")


def plot_performance_comparison(y_train, y_train_pred, train_metric,
                                y_test, y_test_pred, test_metric,
                                model_name, save_path=None, show_after_save=False,
                                y_val=None, y_val_pred=None, val_metric=None,
                                sample_size=200000,
                                axis_range=(-12.0, 13.0)):
    """
    绘制训练集、验证集、测试集的 预测值 vs 真实值 散点图
    样式完全匹配教程 residuals.pdf：
      - 散点按残差着色（jet 色带，残差=预测-真实）
      - y=x 红色虚线参考线
      - 黑色线性回归拟合线
      - 95% 预测带 = 回归线 ± 1.96·σ(残差)（黑色虚线）
      - 灰色虚线网格
      - 右下角指标文本框（RMSE / R²）
      - 无框图例
    如果提供了 y_val/y_val_pred/val_metric，则显示 3 列；否则 2 列
    sample_size: 每个子图抽样的散点数（默认20万），设为 None 则全量绘制
    axis_range: 固定显示范围；范围外/非有限点不绘制且不参与色带、回归线和预测带
    提供 save_path 时，会在 tmp_metrics 保存 *_Plot_Data.npz 供后续直接重绘
    """
    import matplotlib.pyplot as plt
    import numpy as np

    has_val = y_val is not None and y_val_pred is not None and val_metric is not None
    ncols = 3 if has_val else 2
    # 多留宽度给 colorbar（每个子图右侧）
    figsize = (22, 7) if has_val else (15, 7)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True)

    datasets = [
        ("Train", y_train, y_train_pred, train_metric),
    ]
    if has_val:
        datasets.append(("Validation", y_val, y_val_pred, val_metric))
    datasets.append(("Test", y_test, y_test_pred, test_metric))

    # 保存绘图所需的逐样本数据；本轮重跑后可直接重画散点图，无需再次训练。
    plot_data_to_save = {}

    # 所有模型统一使用固定坐标范围，避免异常预测值改变图幅、妨碍横向比较。
    if axis_range is None or len(axis_range) != 2:
        raise ValueError("axis_range 必须是 (min, max) 二元组")
    target_min, target_max = map(float, axis_range)
    if not np.isfinite(target_min) or not np.isfinite(target_max) or target_min >= target_max:
        raise ValueError(f"无效 axis_range: {axis_range}")

    for i, (name, y_true, y_pred, metric) in enumerate(datasets):
        ax = axes[i]

        y_true_flat = np.array(y_true).flatten()
        y_pred_flat = np.array(y_pred).flatten()
        # 抽样绘制
        if sample_size is not None and len(y_true_flat) > sample_size:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y_true_flat), size=sample_size, replace=False)
            y_true_plot = y_true_flat[idx]
            y_pred_plot = y_pred_flat[idx]
        else:
            y_true_plot = y_true_flat
            y_pred_plot = y_pred_flat

        # 仅使用画布范围内的有限点进行绘制、色带归一化、回归与预测带拟合。
        # 旧逻辑把画布外不可见异常点也交给 scatter，导致这些点控制 jet 色带
        # 的 vmin/vmax，MLP 画布内正常点因此只能显示绿/青/蓝的一小段色带。
        finite_mask = np.isfinite(y_true_plot) & np.isfinite(y_pred_plot)
        in_range_mask = (
            finite_mask
            & (y_true_plot >= target_min) & (y_true_plot <= target_max)
            & (y_pred_plot >= target_min) & (y_pred_plot <= target_max)
        )
        y_true_visible = y_true_plot[in_range_mask]
        y_pred_visible = y_pred_plot[in_range_mask]
        if y_true_visible.size < 2:
            raise ValueError(
                f"{model_name} - {name}: 坐标范围 {axis_range} 内有效点不足 2 个"
            )

        excluded_count = int(y_true_plot.size - y_true_visible.size)
        print(
            f"  [{model_name} - {name}] 绘图点: {y_true_visible.size:,}; "
            f"范围外/非有限点不绘制: {excluded_count:,}"
        )

        split_key = name.lower()
        plot_data_to_save[f"{split_key}_y_true"] = np.asarray(
            y_true_plot, dtype=np.float32)
        plot_data_to_save[f"{split_key}_y_pred"] = np.asarray(
            y_pred_plot, dtype=np.float32)
        plot_data_to_save[f"{split_key}_in_range"] = np.asarray(
            in_range_mask, dtype=np.bool_)

        # 1. 残差着色散点（仅范围内点决定色带）
        residuals_visible = y_pred_visible - y_true_visible
        scatter = ax.scatter(y_true_visible, y_pred_visible, c=residuals_visible,
                             cmap='jet', s=30, edgecolor='none')

        # 2. y = x 红色虚线参考线
        ax.plot([target_min, target_max], [target_min, target_max],
                'r--', linewidth=1, label='y = x line')

        # 3. 黑色线性回归拟合线（np.polyfit 一阶，延伸整个画布）
        slope, intercept = np.polyfit(y_true_visible, y_pred_visible, 1)
        reg_line = slope * np.array([target_min, target_max]) + intercept
        ax.plot([target_min, target_max], reg_line, color='black',
                linewidth=1, label='Regression Line')

        # 4. 95% 预测带 = 回归线 ± 1.96·σ(残差)
        std_err = np.std(residuals_visible)
        ax.plot([target_min, target_max], reg_line + 1.96 * std_err,
                'k--', linewidth=0.8, label='95% Prediction Band')
        ax.plot([target_min, target_max], reg_line - 1.96 * std_err,
                'k--', linewidth=0.8)

        # 5. 灰色虚线网格
        ax.grid(True, linestyle='--', linewidth=0.8, color='#DCDCDC')

        # 6. 标题和标签
        ax.set_title(f"{model_name} - {name} Set", fontsize=16, fontweight='bold')
        ax.set_xlabel("True Value", fontsize=15, fontweight='bold')
        ax.set_ylabel("Predicted Value", fontsize=15, fontweight='bold')

        # 7. 强制应用设定的范围 + 等比例
        ax.set_xlim(target_min, target_max)
        ax.set_ylim(target_min, target_max)
        ax.set_aspect('equal')

        # 8. 无框图例（与教程 residuals.pdf 一致：fontsize=15）
        ax.legend(loc='upper left', fontsize=15, frameon=False)

        # 9. 右下角指标文本框（与教程 residuals.pdf 一致：RMSE/MAE/R² 三行）
        try:
            mae_score = metric[0][0]
            rmse_score = metric[0][3]
            r2_score = metric[0][4]
            text_str = (f'RMSE = {rmse_score:.3f}\n'
                        f'MAE = {mae_score:.3f}\n'
                        f'R² = {r2_score:.3f}')
        except Exception:
            text_str = "Metric Error"
        ax.text(0.95, 0.05, text_str, transform=ax.transAxes, fontsize=15,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if save_path:
        # 与聚合指标 CSV 同目录保存压缩逐样本绘图数据。
        # 文件包含固定抽样数据和 in_range 掩码，后续可独立重画。
        _plot_data_dir = globals().get('TMP_METRICS_DIR', os.path.dirname(save_path))
        os.makedirs(_plot_data_dir, exist_ok=True)
        _safe_model_name = ''.join(
            ch if (ch.isalnum() or ch in '-_') else '_'
            for ch in str(model_name)
        )
        _plot_data_path = os.path.join(
            _plot_data_dir, f'{_safe_model_name}_Plot_Data.npz')
        plot_data_to_save['axis_range'] = np.asarray(
            [target_min, target_max], dtype=np.float32)
        plot_data_to_save['sample_size_limit'] = np.asarray(
            [-1 if sample_size is None else int(sample_size)], dtype=np.int64)
        np.savez_compressed(_plot_data_path, **plot_data_to_save)
        print(f"  已保存可重绘散点数据: {_plot_data_path}")

        # 双格式保存：PDF（出版 dpi1200）+ PNG（预览 dpi300），并可预览
        _save_both(save_path, show=show_after_save)
    else:
        plt.show()


# ==========================================
# 调用部分
# ==========================================
print("\n📈 正在绘制优化后的散点图...")

# ==========================================
# 绘图：预测值 vs 真实值
# ==========================================
if TRAIN_MODE == 'all':
    # 全部数据模式：绘制散点图
    y_train_flat = np.array(y_train).flatten()
    y_test_flat = np.array(y_test).flatten()
    _yv = globals().get('y_val')
    y_val_flat = np.array(_yv).flatten() if _yv is not None else None

    y_train_pred_flat = np.array(y_train_pred).flatten()
    y_test_pred_flat = np.array(y_test_pred).flatten()
    _yvp = globals().get('y_val_pred')
    y_val_pred_flat = np.array(_yvp).flatten() if _yvp is not None else None

    plot_performance_comparison(
        y_train_flat, y_train_pred_flat, train_metrics,
        y_test_flat,  y_test_pred_flat,  test_metrics,
        method_label,
        y_val=y_val_flat, y_val_pred=y_val_pred_flat,
        val_metric=globals().get('val_metrics')
    )
elif TRAIN_MODE == 'by_climate':
    print(f"\n📊 绘制各气候带散点图...")
    for climate_code in climates_to_train:
        if climate_code not in climate_data:
            continue
        data = climate_data[climate_code]
        model_c = climate_models[climate_code]
        y_train_pred_c = model_c.predict(data['X_train'])
        y_test_pred_c = model_c.predict(data['X_test'])
        from sklearn.metrics import r2_score as _r2
        print(
            f"  气候带{climate_code} Test R2: {_r2(data['y_test'], y_test_pred_c):.4f}")
    print(f"✅ 各气候带模型训练完成")

elif TRAIN_MODE == 'by_season':
    print(f"\n📊 [by_season] 绘制夏冬单模型散点图（预览）...")
    season_name_map_plot = {'summer': '夏季', 'winter': '冬季'}

    for sk in season_data.keys():
        _d = season_data[sk]
        _cache = preview_cache[sk]  # 直接调用缓存
        _y_tr_p = _cache['y_train_pred']
        _y_te_p = _cache['y_test_pred']
        _y_va_p = _cache.get('y_val_pred')
        _tr_ev = _cache['train_metrics']
        _te_ev = _cache['test_metrics']
        _va_ev = _cache.get('val_metrics')

        print(f"\n  [{season_name_map_plot[sk]}] 绘图中...")
        plot_performance_comparison(
            np.array(_d['y_train']).flatten(),
            np.array(_y_tr_p).flatten(),
            _tr_ev,
            np.array(_d['y_test']).flatten(),
            np.array(_y_te_p).flatten(),
            _te_ev,
            f"{method_label}_{sk}_preview",
            y_val=np.array(_d['y_val']).flatten() if 'y_val' in _d else None,
            y_val_pred=np.array(_y_va_p).flatten() if _y_va_p is not None else None,
            val_metric=_va_ev if _va_ev else None
        )


# %%
# ==========================================
# Cell 5 后清理块：释放预览阶段残留大对象
# ==========================================
import gc
import matplotlib.pyplot as plt

# 1）关闭 cell5 遗留的所有图形对象
plt.close('all')

# 2）显式删除 cell5 单模型预览阶段遗留变量
for _v in ['y_train_pred', 'y_test_pred', 'best_model',
           'train_metrics', 'test_metrics',
           'y_train_flat', 'y_test_flat',
           'y_train_pred_flat', 'y_test_pred_flat',
           'y_val_pred', 'y_val_flat', 'y_val_pred_flat',
           'val_metrics']:
    if _v in globals():
        del globals()[_v]

# 3）兼容清理 preview_cache（by_season 模式产生）
if 'preview_cache' in globals():
    del globals()['preview_cache']

# 4）执行垃圾回收
gc.collect()

# 5）尝试将空闲堆内存归还给操作系统（Linux/Ubuntu 环境）
import ctypes
try:
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
    print("✅ 已尝试将空闲堆内存归还给操作系统")
except Exception as e:
    print(f"⚠️ malloc_trim 调用失败: {e}")

print("✅ Cell5 预览残留变量已清理，基础训练数据保留供 Cell6 使用")


# %%
# ==========================================
# Cell 6 重构：一个模型一个独立训练块
# ==========================================
import gc
import joblib

# ==========================================
# Cell 6 - 公共准备区（只执行一次）
# ==========================================
name_label_list_base = ['LightGBM', 'XGBoost', 'CatBoost', 'RF',
                        'MLP', 'DT', 'Bay', 'GBR', 'LinearRegression', 'xRFM']
evaluate_list = ['MAE', 'MCE', 'MSE', 'RMSE', 'R2', 'Bias', 'CVRMSE']
output_dir = METRICS_DIR

# 临时结果目录：单模型指标/模型/图落盘
TMP_METRICS_DIR = os.path.join(BASE_OUTPUT_DIR, "tmp_metrics")
TMP_MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "tmp_models")
TMP_PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "tmp_plots")
for _d in [TMP_METRICS_DIR, TMP_MODELS_DIR, TMP_PLOTS_DIR]:
    os.makedirs(_d, exist_ok=True)


def _model_feature_schema_payload(suite_key=None):
    return {
        'feature_cols': list(feature_cols),
        'run_version': _RUN_VERSION,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'number_of_features': len(feature_cols),
        'suite_key': suite_key,
    }


def _assert_lightgbm_schema_compatible(model):
    """Fail closed before SHAP if a stale FAR-model is loaded."""
    expected = list(feature_cols)
    n_features = getattr(model, 'n_features_in_', None)
    if n_features is None and hasattr(model, 'booster_'):
        n_features = model.booster_.num_feature()
    if n_features is None or int(n_features) != len(expected):
        raise RuntimeError(
            f"LightGBM feature-count mismatch: model={n_features}, "
            f"expected={len(expected)}"
        )
    stored_names = getattr(model, '_uhi_feature_cols', None)
    if stored_names is None:
        raise RuntimeError(
            "LightGBM 缺少 BC-noFAR feature schema 元数据；拒绝加载可能的旧 FAR 模型"
        )
    if list(stored_names) != expected:
        raise RuntimeError(
            f"LightGBM feature-name/order mismatch: model={list(stored_names)}, "
            f"expected={expected}"
        )
    if getattr(model, '_uhi_run_version', None) != _RUN_VERSION:
        raise RuntimeError(
            "LightGBM run_version 与当前 BC-noFAR 运行不一致；拒绝用于 SHAP"
        )

# SHAP 白名单：仅对 LightGBM 保留模型对象和 X_train
KEEP_MODEL_OBJECTS = {'LightGBM'}

# 确定训练套件
if TRAIN_MODE == 'by_season':
    training_suites = []
    if 'summer' in season_data:
        training_suites.append(('summer', 'Summer Model'))
    if 'winter' in season_data:
        training_suites.append(('winter', 'Winter Model'))
else:
    # suite_name 用纯英文，避免 Linux 服务器无中文字体导致图表标题乱码
    training_suites = [('all', 'All Data')]

# 当前激活的 suite 选择器（默认第一个）
ACTIVE_SUITE_KEY = training_suites[0][0]
ACTIVE_SUITE_NAME = training_suites[0][1]
print(f"当前激活套件: {ACTIVE_SUITE_KEY} ({ACTIVE_SUITE_NAME})")

# 轻量化缓存：仅保留 suite 元信息和 SHAP/Ensemble 需要的轻量引用
all_suite_artifacts = {}
for _sk, _sn in training_suites:
    all_suite_artifacts[_sk] = {
        'name': _sn,
        'model_registry': {},
        'X_train_ref': None,
        'model_results': [],
    }

def _get_suite_data(suite_key):
    """获取指定 suite 的训练/测试数据（含可选验证集）"""
    if TRAIN_MODE == 'by_season':
        _d = season_data[suite_key]
        _xv = _d.get('X_val')
        _yv = _d.get('y_val')
        return _d['X_train'], _d['y_train'], _d['X_test'], _d['y_test'], _xv, _yv
    else:
        _xv = globals().get('X_val')
        _yv = globals().get('y_val')
        return X_train, y_train, X_test, y_test, _xv, _yv


def run_single_model_block(method_label, suite_key, suite_name):
    """
    单模型完整流程：训练 → 评估 → 出图 → 落盘 → 释放
    返回轻量结果字典，不返回大数组或完整模型（除非白名单）
    """
    _X_train, _y_train, _X_test, _y_test, _X_val, _y_val = _get_suite_data(suite_key)
    prefix = f"{suite_key}_" if TRAIN_MODE == 'by_season' else ""

    print(f"\n{'='*65}")
    print(f"  [{suite_name}] 正在训练: {method_label}")
    print(f"{'='*65}")

    try:
        y_train_predict, y_test_predict, model, y_val_predict = train_muti_model(
            _X_train, _y_train, _X_test, _y_test, method_label,
            X_val=_X_val, y_val=_y_val
        )
        train_ev, test_ev, val_ev = test_method(
            y_train_predict, y_test_predict,
            _y_train, _y_test, method_label,
            val_y_predict=y_val_predict, val_y=_y_val
        )
    except Exception as e:
        print(f"❌ 模型 {method_label} 训练或评估失败: {e}")
        train_ev = [[np.nan] * len(evaluate_list)]
        test_ev = [[np.nan] * len(evaluate_list)]
        val_ev = []
        y_train_predict = None
        y_test_predict = None
        y_val_predict = None
        model = None

    # ── 构造单模型结果 DataFrame 并保存 CSV ────────────────────
    df_train_single = pd.DataFrame(
        data=train_ev if train_ev else [[np.nan]*len(evaluate_list)],
        columns=evaluate_list, index=[method_label])
    df_test_single = pd.DataFrame(
        data=test_ev if test_ev else [[np.nan]*len(evaluate_list)],
        columns=evaluate_list, index=[method_label])
    df_val_single = None
    if val_ev:
        df_val_single = pd.DataFrame(
            data=val_ev, columns=evaluate_list, index=[method_label])

    csv_train_path = os.path.join(
        TMP_METRICS_DIR, f'{prefix}{method_label}_Metrics_Train.csv')
    csv_test_path = os.path.join(
        TMP_METRICS_DIR, f'{prefix}{method_label}_Metrics_Test.csv')
    csv_val_path = os.path.join(
        TMP_METRICS_DIR, f'{prefix}{method_label}_Metrics_Val.csv')
    try:
        df_train_single.to_csv(csv_train_path, encoding='utf-8-sig')
        df_test_single.to_csv(csv_test_path, encoding='utf-8-sig')
        if df_val_single is not None:
            df_val_single.to_csv(csv_val_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"⚠️ 保存 {method_label} CSV 失败: {e}")
        csv_train_path = None
        csv_test_path = None
        csv_val_path = None

    # ── 绘制并保存散点图（即时显示 + 落盘） ─────────────────────
    plot_name = f"{suite_key}_{method_label}" if TRAIN_MODE == 'by_season' else method_label
    plot_path = os.path.join(TMP_PLOTS_DIR, f"{plot_name}_Performance.png")
    if y_train_predict is not None and y_test_predict is not None:
        try:
            plot_performance_comparison(
                _y_train, y_train_predict, train_ev,
                _y_test,  y_test_predict,  test_ev,
                plot_name, save_path=plot_path, show_after_save=True,
                y_val=_y_val, y_val_pred=y_val_predict,
                val_metric=val_ev if val_ev else None
            )
            print(f"  散点图已保存: {plot_path}")
        except Exception as e:
            print(f"⚠️ 保存 {method_label} 散点图失败: {e}")
            plot_path = None
    else:
        plot_path = None

    # ── 白名单模型：保留模型对象和 X_train ─────────────────────
    is_keep = method_label in KEEP_MODEL_OBJECTS
    saved_model_path = None
    if is_keep and model is not None:
        saved_model_path = os.path.join(
            TMP_MODELS_DIR, f'{prefix}{method_label}_model.pkl')
        try:
            model._uhi_feature_cols = list(feature_cols)
            model._uhi_run_version = _RUN_VERSION
            model._uhi_feature_schema_version = FEATURE_SCHEMA_VERSION
            joblib.dump(model, saved_model_path)
            _schema_payload = _model_feature_schema_payload(suite_key)
            _schema_paths = [
                os.path.join(TMP_MODELS_DIR, 'model_feature_schema.json'),
                os.path.join(
                    TMP_MODELS_DIR,
                    f'{prefix}{method_label}_model_feature_schema.json',
                ),
            ]
            for _schema_path in _schema_paths:
                with open(_schema_path, 'w', encoding='utf-8') as handle:
                    json.dump(_schema_payload, handle, ensure_ascii=False, indent=2)
            print(f"  白名单模型已保存: {saved_model_path}")
        except Exception as e:
            print(f"⚠️ 保存白名单模型失败: {e}")
            saved_model_path = None

    # ── 构建轻量结果字典 ──────────────────────────────────────
    result = {
        'method_label': method_label,
        'suite_key': suite_key,
        'suite_name': suite_name,
        'csv_train_path': csv_train_path,
        'csv_test_path': csv_test_path,
        'csv_val_path': csv_val_path if val_ev else None,
        'plot_path': plot_path,
        'saved_model_path': saved_model_path,
        'train_r2': train_ev[0][4] if train_ev and len(train_ev) > 0 else np.nan,
        'test_r2': test_ev[0][4] if test_ev and len(test_ev) > 0 else np.nan,
        'val_r2': val_ev[0][4] if val_ev and len(val_ev) > 0 else np.nan,
    }

    # ── 释放临时大对象 ────────────────────────────────────────
    del _X_train, _y_train, _X_test, _y_test
    if _X_val is not None:
        del _X_val, _y_val
    if y_train_predict is not None:
        del y_train_predict
    if y_test_predict is not None:
        del y_test_predict
    if y_val_predict is not None:
        del y_val_predict
    if train_ev is not None:
        del train_ev
    if test_ev is not None:
        del test_ev
    if val_ev is not None:
        del val_ev
    if model is not None and not is_keep:
        del model
    if df_train_single is not None:
        del df_train_single
    if df_test_single is not None:
        del df_test_single
    if df_val_single is not None:
        del df_val_single
    plt.close('all')
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    return result


def merge_final_metrics():
    """
    从临时 CSV 合并生成最终总指标表
    支持 by_season 分 suite 输出
    """
    for suite_key, suite_name in training_suites:
        prefix = f"{suite_key}_" if TRAIN_MODE == 'by_season' else ""
        print(f"\n{'='*60}")
        print(f"  合并 [{suite_name}] 最终指标表")
        print(f"{'='*60}")

        train_frames = []
        val_frames = []
        test_frames = []
        for mname in name_label_list_base:
            csv_t = os.path.join(TMP_METRICS_DIR, f'{prefix}{mname}_Metrics_Train.csv')
            csv_v = os.path.join(TMP_METRICS_DIR, f'{prefix}{mname}_Metrics_Val.csv')
            csv_e = os.path.join(TMP_METRICS_DIR, f'{prefix}{mname}_Metrics_Test.csv')
            try:
                if os.path.exists(csv_t):
                    train_frames.append(pd.read_csv(csv_t, index_col=0, encoding='utf-8-sig'))
                if os.path.exists(csv_v):
                    val_frames.append(pd.read_csv(csv_v, index_col=0, encoding='utf-8-sig'))
                if os.path.exists(csv_e):
                    test_frames.append(pd.read_csv(csv_e, index_col=0, encoding='utf-8-sig'))
            except Exception as e:
                print(f"⚠️ 读取 {mname} CSV 失败: {e}")

        if train_frames:
            df_train_final = pd.concat(train_frames)
            out_train = os.path.join(output_dir, f'{prefix}Metrics_Train_Final.csv')
            df_train_final.to_csv(out_train, encoding='utf-8-sig')
            print(f"  训练集总表已保存: {out_train}")
        if test_frames:
            df_test_final = pd.concat(test_frames)
            out_test = os.path.join(output_dir, f'{prefix}Metrics_Test_Final.csv')
            df_test_final.to_csv(out_test, encoding='utf-8-sig')
            print(f"  测试集总表已保存: {out_test}")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(f"\n  [{suite_name}] 测试集排名:")
            print(df_test_final.sort_values(by='R2', ascending=False))
        if val_frames:
            df_val_final = pd.concat(val_frames)
            out_val = os.path.join(output_dir, f'{prefix}Metrics_Val_Final.csv')
            df_val_final.to_csv(out_val, encoding='utf-8-sig')
            print(f"  验证集总表已保存: {out_val}")

    print("\n✅ 所有最终指标表合并完成！")


def load_suite_for_shap(suite_key):
    """
    为 SHAP 分析加载必要数据。
    返回测试集数据（SHAP 值应在测试集上计算——文献依据：
    Ponce-Bobadilla et al. 2024, PMC, 893 引用）。
    """
    if TRAIN_MODE == 'by_season':
        _X_shap = season_data[suite_key]['X_test']
        _y_shap = season_data[suite_key]['y_test']
        _scaler_shap = season_data[suite_key]['scaler']
        _group_ids_shap = season_data[suite_key].get('test_group_ids')
    else:
        _X_shap = X_test
        _y_shap = y_test
        _scaler_shap = globals().get('scaler')
        _group_ids_shap = globals().get('test_group_ids')

    model_path = None
    prefix = f"{suite_key}_" if TRAIN_MODE == 'by_season' else ""
    model_path = os.path.join(TMP_MODELS_DIR, f'{prefix}LightGBM_model.pkl')

    lgb_model = None
    if os.path.exists(model_path):
        try:
            lgb_model = joblib.load(model_path)
            _assert_lightgbm_schema_compatible(lgb_model)
            print(f"  已从磁盘加载 LightGBM 模型: {model_path}")
        except Exception as e:
            raise RuntimeError(f"加载 LightGBM 模型或 schema 校验失败: {e}") from e

    return _X_shap, _y_shap, lgb_model, _scaler_shap, _group_ids_shap


# Ensemble 辅助函数（load_model_for_ensemble / get_ensemble_data /
#   get_model_mae_from_csv / predict_with_model）已随 Ensemble 代码一并移除。


# %%
# ==========================================
# Cell 6A - LightGBM
# ==========================================
print(f"\n{'='*65}")
print(f"  Cell 6A: LightGBM [{ACTIVE_SUITE_NAME}]")
print(f"{'='*65}")
lightgbm_result = run_single_model_block('LightGBM', ACTIVE_SUITE_KEY, ACTIVE_SUITE_NAME)
all_suite_artifacts[ACTIVE_SUITE_KEY]['model_results'].append(lightgbm_result)

# %%
# ==========================================
# Cell 6B - XGBoost
# ==========================================
print(f"\n{'='*65}")
print(f"  Cell 6B: XGBoost [{ACTIVE_SUITE_NAME}]")
print(f"{'='*65}")
xgboost_result = run_single_model_block('XGBoost', ACTIVE_SUITE_KEY, ACTIVE_SUITE_NAME)
all_suite_artifacts[ACTIVE_SUITE_KEY]['model_results'].append(xgboost_result)

# %%
# ==========================================
# Cell 6C - CatBoost
# ==========================================
print(f"\n{'='*65}")
print(f"  Cell 6C: CatBoost [{ACTIVE_SUITE_NAME}]")
print(f"{'='*65}")
catboost_result = run_single_model_block('CatBoost', ACTIVE_SUITE_KEY, ACTIVE_SUITE_NAME)
all_suite_artifacts[ACTIVE_SUITE_KEY]['model_results'].append(catboost_result)

# %%
# ==========================================
# Cell 6D - RF
# ==========================================
print(f"\n{'='*65}")
print(f"  Cell 6D: RF [{ACTIVE_SUITE_NAME}]")
print(f"{'='*65}")
rf_result = run_single_model_block('RF', ACTIVE_SUITE_KEY, ACTIVE_SUITE_NAME)
all_suite_artifacts[ACTIVE_SUITE_KEY]['model_results'].append(rf_result)

# %%
# ==========================================
# Cell 6E - MLP
# ==========================================
print(f"\n{'='*65}")
print(f"  Cell 6E: MLP [{ACTIVE_SUITE_NAME}]")
print(f"{'='*65}")
mlp_result = run_single_model_block('MLP', ACTIVE_SUITE_KEY, ACTIVE_SUITE_NAME)
all_suite_artifacts[ACTIVE_SUITE_KEY]['model_results'].append(mlp_result)

# %%
# ==========================================
# Cell 6F - DT
# ==========================================
print(f"\n{'='*65}")
print(f"  Cell 6F: DT [{ACTIVE_SUITE_NAME}]")
print(f"{'='*65}")
dt_result = run_single_model_block('DT', ACTIVE_SUITE_KEY, ACTIVE_SUITE_NAME)
all_suite_artifacts[ACTIVE_SUITE_KEY]['model_results'].append(dt_result)

# %%
# ==========================================
# Cell 6G - Bay
# ==========================================
print(f"\n{'='*65}")
print(f"  Cell 6G: Bay [{ACTIVE_SUITE_NAME}]")
print(f"{'='*65}")
bay_result = run_single_model_block('Bay', ACTIVE_SUITE_KEY, ACTIVE_SUITE_NAME)
all_suite_artifacts[ACTIVE_SUITE_KEY]['model_results'].append(bay_result)

# %%
# ==========================================
# Cell 6H - GBR
# ==========================================
print(f"\n{'='*65}")
print(f"  Cell 6H: GBR [{ACTIVE_SUITE_NAME}]")
print(f"{'='*65}")
gbr_result = run_single_model_block('GBR', ACTIVE_SUITE_KEY, ACTIVE_SUITE_NAME)
all_suite_artifacts[ACTIVE_SUITE_KEY]['model_results'].append(gbr_result)

# %%
# ==========================================
# Cell 6I - LinearRegression
# ==========================================
print(f"\n{'='*65}")
print(f"  Cell 6I: LinearRegression [{ACTIVE_SUITE_NAME}]")
print(f"{'='*65}")
lr_result = run_single_model_block('LinearRegression', ACTIVE_SUITE_KEY, ACTIVE_SUITE_NAME)
all_suite_artifacts[ACTIVE_SUITE_KEY]['model_results'].append(lr_result)

# %%
# ==========================================
# Cell 6J - xRFM
# ==========================================
print(f"\n{'='*65}")
print(f"  Cell 6J: xRFM [{ACTIVE_SUITE_NAME}]")
print(f"{'='*65}")
xrfm_result = run_single_model_block('xRFM', ACTIVE_SUITE_KEY, ACTIVE_SUITE_NAME)
all_suite_artifacts[ACTIVE_SUITE_KEY]['model_results'].append(xrfm_result)


# %%
# ==========================================
# Cell 6K：最终汇总
# ==========================================
merge_final_metrics()

# %%
# ==========================================
# SHAP 可解释性分析（抽样版，适配轻量化结构）
# ==========================================
SHAP_SAMPLE_SIZE = 200000  # 抽样样本量，设为 None 则全量分析
SHAP_DEPENDENCE_SCATTER_SIZE = 8000  # 仅控制依赖图显示点数，不影响拟合与 CI

shap_suites = list(all_suite_artifacts.keys())

for suite_key in shap_suites:
    suite_name = all_suite_artifacts[suite_key]['name']

    print(f"\n{'='*60}")
    print(f"====== 🧠 正在进行 [{suite_name}] 的 SHAP 可解释性分析 ======")
    print(f"{'='*60}")

    # 从磁盘加载 LightGBM 模型和测试集数据
    # 文献依据：Ponce-Bobadilla et al. 2024 (893 引用)——
    #   SHAP 值应在测试集上计算，反映模型对未见数据的真实决策逻辑。
    (_X_test_shap, _y_test_shap, lgb_model,
     _shap_scaler, _shap_group_ids_full) = load_suite_for_shap(suite_key)

    if lgb_model is None:
        print(f"❌ 跳过 [{suite_name}] SHAP 分析: 无法加载 LightGBM 模型")
        continue

    # R² 始终在完整独立测试集上计算，不受 SHAP 抽样数量影响。
    try:
        _y_test_for_r2 = np.asarray(_y_test_shap).reshape(-1)
        _y_pred_lgb_full = lgb_model.predict(_X_test_shap)
        from sklearn.metrics import r2_score as _r2_fn
        _suite_r2 = round(float(_r2_fn(_y_test_for_r2, _y_pred_lgb_full)), 3)
        _suite_r2_n = len(_y_test_for_r2)
    except Exception as e:
        print(f"⚠️ [{suite_name}] 完整测试集 R² 计算失败: {e}")
        _suite_r2 = None
        _suite_r2_n = None

    # 把 Numpy 测试集重新打包为带列名的 DataFrame（保留原始索引以对齐 y）
    X_shap_df = pd.DataFrame(_X_test_shap, columns=feature_cols)
    _y_shap_full = pd.Series(_y_test_shap)
    _shap_full_rows = int(len(X_shap_df))
    _shap_full_grid_count = (
        int(np.unique(_shap_group_ids_full).size)
        if _shap_group_ids_full is not None else 0
    )

    # 完全复刻原版：从独立测试集逐行随机抽取 200,000 行。
    if SHAP_SAMPLE_SIZE is not None and len(X_shap_df) > SHAP_SAMPLE_SIZE:
        _shap_idx = _shap_random_subsample_indices(
            len(X_shap_df), SHAP_SAMPLE_SIZE, random_state=42,
        )
        _sampling_kind = '逐行随机抽样（原版）'
        print(f"SHAP 分析抽样(测试集, {_sampling_kind}): "
              f"{len(X_shap_df):,} → {len(_shap_idx):,}")
        X_shap_df = X_shap_df.iloc[_shap_idx].reset_index(drop=True)
        _y_shap_aligned = _y_shap_full.iloc[_shap_idx].values
        _shap_group_ids = (
            None if _shap_group_ids_full is None
            else np.asarray(_shap_group_ids_full)[_shap_idx]
        )
    else:
        print(f"SHAP 分析使用全量测试集: {len(X_shap_df):,}")
        _sampling_kind = '全量独立测试集'
        _y_shap_aligned = _y_shap_full.values
        _shap_group_ids = (
            None if _shap_group_ids_full is None
            else np.asarray(_shap_group_ids_full)
        )

    _shap_sampled_grid_count = (
        int(np.unique(_shap_group_ids).size)
        if _shap_group_ids is not None else 0
    )
    _grid_coverage = (
        _shap_sampled_grid_count / _shap_full_grid_count
        if _shap_full_grid_count else None
    )

    # 模型和 TreeExplainer 继续接收标准化输入；所有 SHAP 展示横轴统一
    # inverse_transform 回原始物理/数据单位。禁止静默退回 z-score，以免
    # 图、趋势 CSV 与单位元数据再次出现尺度混用。
    if _shap_scaler is None:
        raise RuntimeError(
            "未找到训练集 StandardScaler，无法生成原始尺度 SHAP 依赖图"
        )
    try:
        _raw_shap_values = _shap_scaler.inverse_transform(
            X_shap_df.to_numpy()
        )
        X_shap_display_df = pd.DataFrame(
            _raw_shap_values,
            columns=feature_cols,
        )
        del _raw_shap_values
    except Exception as e:
        raise RuntimeError(
            f"SHAP 横轴逆标准化失败，已停止以避免输出 z-score 图: {e}"
        ) from e
    print("SHAP 全局复合图与单变量依赖图横轴均已恢复为原始单位")

    print("正在构建 TreeExplainer 并计算 SHAP 值（测试集）...")
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_shap_df, check_additivity=False)

    # Disabling SHAP's internal assertion does not make unverified values valid.
    # Reconstruct every sampled prediction and stop before plotting on failure.
    _shap_predictions = lgb_model.predict(X_shap_df)
    _additivity = _validate_shap_additivity(
        shap_values, explainer.expected_value, _shap_predictions
    )
    _additivity_path = os.path.join(
        SHAP_DIR, f"{suite_key}_SHAP_additivity_check.csv"
    )
    pd.DataFrame([_additivity]).to_csv(_additivity_path, index=False)
    print(
        f"SHAP 加和验证: max |error|={_additivity['max_abs_error']:.3e}, "
        f"mean |error|={_additivity['mean_abs_error']:.3e}; "
        f"诊断={_additivity_path}"
    )
    if not _additivity['passed']:
        raise RuntimeError(
            "SHAP additivity validation failed; stale or incompatible model/data "
            "may be loaded. Plot generation has been stopped."
        )

    print(f"[{suite_name}] 正在绘制全局特征重要性图...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # ===== Nature Communications 风格 SHAP 左右组合图 =====
    nature_path = os.path.join(SHAP_DIR, f"{suite_key}_SHAP_nature_style")
    plot_shap_nature_style(
        X_df=X_shap_display_df,
        shap_values_matrix=shap_values,
        feature_cols=feature_cols,
        category_map=SHAP_CATEGORY_MAP,
        colors_map=SHAP_CATEGORY_COLORS,
        r2=_suite_r2,
        n_samples=_suite_r2_n,
        save_path=nature_path,
        dpi=_SAVE_DPI,
        lowess_frac=0.3,
    )

    base_img_save_path = os.path.join(SHAP_DIR, "Plots")
    base_data_save_path = os.path.join(SHAP_DIR, "Data")

    suite_img_save_path = os.path.join(base_img_save_path, suite_key)
    suite_data_save_path = os.path.join(base_data_save_path, suite_key)

    for path in [suite_img_save_path, suite_data_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    _continuous_features = [
        col for col in feature_cols
        if col not in {'LU_Type', 'LCZ_Function', 'Season_code', 'Class'}
    ]
    _missing_units = [
        col for col in _continuous_features
        if not SHAP_FEATURE_UNITS.get(col)
    ]
    if _missing_units:
        print(
            "AUTHOR_INPUT_NEEDED: 以下变量的源数据单位尚未确认，图中暂不猜测: "
            + ", ".join(_missing_units)
        )

    _unit_table = pd.DataFrame({
        'Feature': feature_cols,
        'Feature_Unit': [SHAP_FEATURE_UNITS.get(col) for col in feature_cols],
        'Dependence_X_Unit': [
            SHAP_FEATURE_UNITS.get(col) or 'original value (unit unconfirmed)'
            for col in feature_cols
        ],
        'Unit_Status': [
            'confirmed' if SHAP_FEATURE_UNITS.get(col) else 'AUTHOR_INPUT_NEEDED'
            for col in feature_cols
        ],
        'SHAP_Value_Unit': [
            '°C (model-output scale)' if target_col == 'UHII'
            else 'model-output unit'
            for _ in feature_cols
        ],
    })
    _unit_table.to_csv(
        os.path.join(suite_data_save_path, f'{suite_key}_SHAP_feature_units.csv'),
        index=False, encoding='utf-8-sig',
    )

    _run_metadata = {
        'schema_version': 'shap-dependence-raw-x-spline-cluster-ci-v20260822',
        'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'suite': suite_key,
        'data_split': 'independent test set',
        'full_test_rows': _shap_full_rows,
        'shap_rows': int(len(X_shap_df)),
        'full_test_grid_id_clusters': _shap_full_grid_count,
        'grid_id_clusters': _shap_sampled_grid_count,
        'grid_id_coverage': _grid_coverage,
        'sampling_method': _sampling_kind,
        'sampling_random_state': 42,
        'model_retrained_inside_ci': False,
        'dependence_x_scale': (
            'original feature value via training StandardScaler.inverse_transform'
        ),
        'ci_method': (
            'cubic B-spline mean-response 95% CI; Grid_ID cluster-robust '
            'covariance, with HC3 fallback when groups are unavailable'
        ),
        'curve_domain': (
            '1st to 99th percentile of inverse-transformed feature value'
        ),
        'curve_r2_role': 'descriptive spline-fit R2 within supported domain',
        'curve_p_value_displayed': False,
        'formula_mapping': 'polynomial equation not fitted or displayed',
        'shap_additivity_passed': _additivity['passed'],
        'shap_additivity_max_abs_error': _additivity['max_abs_error'],
        'model_run_version': _RUN_VERSION,
        'shap_result_version': _SHAP_RESULT_VERSION,
    }
    pd.DataFrame([_run_metadata]).to_csv(
        os.path.join(suite_data_save_path, f'{suite_key}_SHAP_run_metadata.csv'),
        index=False, encoding='utf-8-sig',
    )

    print(f"开始导出 {len(feature_cols)} 个变量的趋势分析至 {suite_key} 文件夹...")

    # 一次性构造 SHAP 值 DataFrame，避免循环内重复构造
    shap_values_df = pd.DataFrame(shap_values, columns=feature_cols)

    _LEGEND_POS_DEFAULT = 'lower right'
    _CATEGORICAL_SHAP_FEATURES = {
        'LU_Type', 'LCZ_Function', 'Season_code', 'Class'
    }

    for col_name in feature_cols:
        try:
            # 横轴保存并绘制 inverse_transform 后的原始特征值；SHAP 值
            # 仍保持模型输出单位（UHII 时为 °C）。
            col_index = feature_cols.index(col_name)
            trend_df = pd.DataFrame({
                'Feature_Value_raw': X_shap_display_df[col_name].values,
                'SHAP_Value': shap_values[:, col_index],
            }).sort_values(by='Feature_Value_raw')
            trend_df.insert(
                1,
                'Feature_Unit',
                SHAP_FEATURE_UNITS.get(col_name)
                or 'original value (unit unconfirmed)',
            )
            csv_filename = os.path.join(
                suite_data_save_path, f'{suite_key}_{col_name}_trend.csv')
            trend_df.to_csv(csv_filename, index=False)

            if col_name in _CATEGORICAL_SHAP_FEATURES:
                print(f"  跳过类别变量依赖图: {col_name}")
                continue

            dep_base = os.path.join(
                suite_img_save_path, f'{suite_key}_{col_name}_dependence')
            plot_shap_dependence(
                feature_name=col_name,
                X_data=X_shap_display_df,
                shap_values_df=shap_values_df,
                cluster_ids=_shap_group_ids,
                spline_df=6,
                save_pdf_name=dep_base,
                legend_location=_LEGEND_POS_DEFAULT,
                target_col=target_col,
                feature_unit=SHAP_FEATURE_UNITS.get(col_name),
                scatter_sample_size=SHAP_DEPENDENCE_SCATTER_SIZE,
            )

        except Exception as e:
            print(f"变量 {col_name} 导出失败: {e}")

    print(f"✅ [{suite_name}] 的 SHAP 导出完成！")

    # 释放临时 DataFrame
    del shap_values_df

    # 保存 mean |SHAP| 供因果推断复用
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = {feat: float(mean_abs_shap[i]) for i, feat in enumerate(feature_cols)}
    _shap_importance_table = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_Abs_SHAP': mean_abs_shap,
    }).sort_values('Mean_Abs_SHAP', ascending=False, kind='mergesort').reset_index(drop=True)
    _shap_importance_table['Rank'] = np.arange(1, len(_shap_importance_table) + 1)
    _shap_importance_path = os.path.join(
        suite_data_save_path, f'{suite_key}_SHAP_mean_abs_importance.csv'
    )
    _shap_importance_table.to_csv(
        _shap_importance_path, index=False, encoding='utf-8-sig'
    )
    for _focus_feature in ['BC', 'MBH']:
        _focus = _shap_importance_table.loc[
            _shap_importance_table['Feature'] == _focus_feature
        ]
        if not _focus.empty:
            print(
                f"  {_focus_feature}: mean|SHAP|="
                f"{float(_focus.iloc[0]['Mean_Abs_SHAP']):.8f}, "
                f"rank={int(_focus.iloc[0]['Rank'])}/{len(feature_cols)}"
            )
    all_suite_artifacts[suite_key]['shap_importance'] = shap_importance
    print(f"  ✅ SHAP 重要性已保存至 all_suite_artifacts")

    # SHAP 完成后释放
    del _X_test_shap, X_shap_df, X_shap_display_df
    del explainer, shap_values, mean_abs_shap
    if lgb_model is not None:
        del lgb_model
    plt.close('all')
    gc.collect()

print(f"\n所有处理完成！")
print(f"图片保存在: {base_img_save_path}")
print(f"数据保存在: {base_data_save_path}")



# %%
# ==========================================
# 因果推断 Part 1: 数据准备与因果发现 (Causal Discovery)
# ==========================================
import ctypes as _ctypes_ca
import networkx as nx
from causallearn.search.FCMBased import lingam as _lingam_ca


_EDGE_EFFECT_THRESHOLD = 0.05
_EDGE_SELECTION_THRESHOLD = 0.70
_EDGE_EFFECT_SENSITIVITY = (0.03, 0.05, 0.10)
_EDGE_SELECTION_SENSITIVITY = (0.60, 0.70, 0.80)


def _aggregate_stable_dag(edge_frequency, mean_coef,
                          threshold=_EDGE_SELECTION_THRESHOLD,
                          sink_idx=None):
    """Build an acyclic consensus DAG from threshold-selected edge frequencies.

    Adjacency matrices follow LiNGAM's convention: B[target, source] is the
    coefficient for source -> target. Frequency aggregation of individually
    acyclic graphs can itself contain a cycle, so candidate edges are added in
    descending vote-frequency order and cycle-closing edges are recorded and
    omitted from the final DAG.
    """
    edge_frequency = np.asarray(edge_frequency, dtype=float)
    mean_coef = np.asarray(mean_coef, dtype=float)
    if edge_frequency.shape != mean_coef.shape or edge_frequency.ndim != 2:
        raise ValueError("edge_frequency and mean_coef must be same-shape matrices")
    if edge_frequency.shape[0] != edge_frequency.shape[1]:
        raise ValueError("edge_frequency and mean_coef must be square matrices")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("selection threshold must be in [0, 1]")

    n_nodes = edge_frequency.shape[0]
    threshold_mask = edge_frequency >= threshold
    np.fill_diagonal(threshold_mask, False)
    if sink_idx is not None:
        if not 0 <= int(sink_idx) < n_nodes:
            raise IndexError("sink_idx is outside the adjacency matrix")
        threshold_mask[:, int(sink_idx)] = False

    candidates = [
        (float(edge_frequency[target, source]),
         abs(float(mean_coef[target, source])), target, source)
        for target, source in np.argwhere(threshold_mask)
    ]
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    final_mask = np.zeros_like(threshold_mask, dtype=bool)
    cycle_pruned = np.zeros_like(threshold_mask, dtype=bool)
    for _, _, target, source in candidates:
        graph.add_edge(source, target)
        if nx.is_directed_acyclic_graph(graph):
            final_mask[target, source] = True
        else:
            graph.remove_edge(source, target)
            cycle_pruned[target, source] = True

    aggregated = np.where(final_mask, mean_coef, 0.0)
    return aggregated, threshold_mask, final_mask, cycle_pruned


def _bootstrap_edge_statistics(adjacency_stack, effect_threshold):
    """Summarize directed-edge selection after within-replicate sparsification."""
    adjacency_stack = np.asarray(adjacency_stack, dtype=float)
    if adjacency_stack.ndim != 3:
        raise ValueError("adjacency_stack must have shape (bootstrap, node, node)")
    if adjacency_stack.shape[0] == 0:
        raise ValueError("adjacency_stack must contain at least one replicate")
    if adjacency_stack.shape[1] != adjacency_stack.shape[2]:
        raise ValueError("bootstrap adjacency matrices must be square")
    if effect_threshold < 0.0:
        raise ValueError("effect_threshold must be non-negative")

    selected = np.abs(adjacency_stack) >= effect_threshold
    diagonal = np.arange(adjacency_stack.shape[1])
    selected[:, diagonal, diagonal] = False

    directed_count = selected.sum(axis=0, dtype=int)
    directed_frequency = directed_count / adjacency_stack.shape[0]

    adjacency_selected = selected | np.swapaxes(selected, 1, 2)
    adjacency_frequency = adjacency_selected.mean(axis=0)
    direction_denominator = directed_count + directed_count.T
    direction_given_adjacency = np.divide(
        directed_count,
        direction_denominator,
        out=np.full_like(directed_frequency, np.nan, dtype=float),
        where=direction_denominator > 0,
    )

    positive_count = (selected & (adjacency_stack > 0)).sum(axis=0, dtype=int)
    negative_count = (selected & (adjacency_stack < 0)).sum(axis=0, dtype=int)
    sign_consistency = np.divide(
        np.maximum(positive_count, negative_count),
        directed_count,
        out=np.full_like(directed_frequency, np.nan, dtype=float),
        where=directed_count > 0,
    )

    selected_sum = np.where(selected, adjacency_stack, 0.0).sum(axis=0)
    mean_coefficient = np.divide(
        selected_sum,
        directed_count,
        out=np.zeros_like(directed_frequency),
        where=directed_count > 0,
    )
    coefficient_percentiles = {
        percentile: np.full_like(directed_frequency, np.nan, dtype=float)
        for percentile in (2.5, 50.0, 97.5)
    }
    for target in range(adjacency_stack.shape[1]):
        for source in range(adjacency_stack.shape[2]):
            coefficients = adjacency_stack[:, target, source]
            coefficients = coefficients[selected[:, target, source]]
            if coefficients.size:
                values = np.percentile(coefficients, [2.5, 50.0, 97.5])
                for percentile, value in zip((2.5, 50.0, 97.5), values):
                    coefficient_percentiles[percentile][target, source] = value

    return {
        'selected': selected,
        'directed_count': directed_count,
        'directed_frequency': directed_frequency,
        'adjacency_frequency': adjacency_frequency,
        'direction_given_adjacency': direction_given_adjacency,
        'sign_consistency': sign_consistency,
        'mean_coefficient': mean_coefficient,
        'coef_p025': coefficient_percentiles[2.5],
        'coef_p500': coefficient_percentiles[50.0],
        'coef_p975': coefficient_percentiles[97.5],
    }


def _edge_threshold_sensitivity_rows(adjacency_stack, labels, sink_idx):
    """Return the prespecified 3x3 edge-threshold sensitivity grid."""
    labels = list(labels)
    records = []
    masks = []
    for effect_threshold in _EDGE_EFFECT_SENSITIVITY:
        statistics = _bootstrap_edge_statistics(
            adjacency_stack, effect_threshold=effect_threshold)
        for selection_threshold in _EDGE_SELECTION_SENSITIVITY:
            _, threshold_mask, final_mask, cycle_pruned = _aggregate_stable_dag(
                statistics['directed_frequency'],
                statistics['mean_coefficient'],
                threshold=selection_threshold,
                sink_idx=sink_idx,
            )
            retained_edges = [
                f"{labels[source]}->{labels[target]}"
                for target, source in np.argwhere(final_mask)
            ]
            records.append({
                'effect_threshold': effect_threshold,
                'selection_threshold': selection_threshold,
                'candidate_edges': int(np.count_nonzero(threshold_mask)),
                'cycle_pruned_edges': int(np.count_nonzero(cycle_pruned)),
                'final_edges': int(np.count_nonzero(final_mask)),
                'retained_directed_edges': '; '.join(sorted(retained_edges)),
            })
            masks.append(final_mask)

    primary_index = next(
        index for index, record in enumerate(records)
        if np.isclose(record['effect_threshold'], _EDGE_EFFECT_THRESHOLD)
        and np.isclose(record['selection_threshold'], _EDGE_SELECTION_THRESHOLD)
    )
    primary_mask = masks[primary_index]
    for record, mask in zip(records, masks):
        union = np.count_nonzero(primary_mask | mask)
        intersection = np.count_nonzero(primary_mask & mask)
        record['jaccard_vs_primary'] = (
            1.0 if union == 0 else round(intersection / union, 4)
        )
    return records


def _dag_layout_spec(labels):
    """Return a publication layout for a supported stable UHI DAG."""
    labels = [str(label) for label in labels]
    label_set = set(labels)
    sink = next(
        (name for name in ("UHII", "Temperature") if name in label_set),
        None,
    )
    if sink is None:
        return None

    # Formal 7-node MBH+BC model.  Keep the native Graphviz/dot workflow used
    # by the three reference analyses, but provide topology-preserving ranks:
    # water -> Pop2025 -> Road_Dens -> BC -> {MBH, NDVI} -> UHII.
    # Only MBH and NDVI share a rank because no retained edge joins them.
    primary_bc_drivers = {
        'MBH', 'BC', 'NDVI', 'water', 'Pop2025', 'Road_Dens',
    }
    if label_set == primary_bc_drivers | {sink}:
        return {
            "rows": [
                ["water"], ["Pop2025"], ["Road_Dens"], ["BC"],
                ["MBH", "NDVI"], [sink],
            ],
            "groups": {},
            "edge_adjustment_profile": "right_routes_and_label_step20",
            "nodesep": "0.38",
            "ranksep": "0.72 equally",
            "manual_edge_adjustments": False,
        }

    # Primary no-MBH model: use the observed topological order rather than
    # grouping variables by theme.  The retained grid-level consensus graph
    # is close to a complete DAG, so placing causally connected variables on
    # the same rank creates horizontal/backtracking edges and a dense knot in
    # the middle of the panel.  One node per rank gives every retained edge a
    # consistent top-to-bottom direction without changing the graph itself.
    reduced_drivers = {
        "water", "Pop2025", "FAR", "COHESION", "Road_Dens", "NDVI",
    }
    if label_set == reduced_drivers | {sink}:
        return {
            "rows": [
                ["water"],
                ["Pop2025"],
                ["COHESION"],
                ["NDVI"],
                ["Road_Dens"],
                ["FAR"],
                [sink],
            ],
            "groups": {
                "water": "middle",
                "Pop2025": "middle",
                "COHESION": "middle",
                "NDVI": "middle",
                "Road_Dens": "middle",
                "FAR": "middle",
                sink: "middle",
            },
            "nodesep": "0.38",
            "ranksep": "0.72 equally",
            "manual_edge_adjustments": True,
        }

    # Legacy 10-node layout retained for archived/sensitivity runs.
    drivers = {
        "water", "Pop2025", "MBH", "FAR", "Road_Dens",
        "COHESION", "NPP_VIIRS", "PLAND", "NDVI",
    }
    if label_set != drivers | {sink}:
        return None

    return {
        "rows": [
            ["water", "Pop2025"],
            ["MBH", "FAR", "Road_Dens"],
            ["COHESION", "NPP_VIIRS"],
            ["PLAND", "NDVI"],
            [sink],
        ],
        "groups": {
            "MBH": "left",
            "COHESION": "left",
            "PLAND": "left",
            "FAR": "middle",
            sink: "middle",
            "Road_Dens": "right",
            "NPP_VIIRS": "right",
            "NDVI": "right",
        },
        "nodesep": "0.30",
        "ranksep": "0.95 equally",
        "manual_edge_adjustments": True,
    }


def _set_positioned_edge_quadratic_spline(
        positioned_dot, source_node_id, target_node_id,
        control_x_offset=-78.0, control_y_offset=0.0,
        start_dx=0.0, start_dy=0.0,
        terminal_tangent_to_target_center=False,
        terminal_arrow_gap=8.0):
    """Redraw one edge from exactly three logical spline points.

    The three points are the frozen start point, one adjustable control point,
    and the frozen end point.  The quadratic curve is converted to the cubic
    Bezier representation expected by Graphviz; nodes, labels, and all other
    edge paths remain untouched.
    """
    import math
    import re

    edge_pattern = re.compile(
        rf'({re.escape(source_node_id)}\s*->\s*'
        rf'{re.escape(target_node_id)}\s*\[.*?\bpos=")'
        r'([^"]+)(".*?\]\s*;?)',
        re.S,
    )
    match = edge_pattern.search(positioned_dot)
    if match is None:
        return positioned_dot, False

    tokens = match.group(2).replace('\\\n', ' ').split()
    if len(tokens) < 3 or not tokens[0].startswith('e,'):
        return positioned_dot, False
    try:
        start_x, start_y = map(float, tokens[1].split(','))
        end_x, end_y = map(float, tokens[-1].split(','))
    except (TypeError, ValueError):
        return positioned_dot, False

    start_x += float(start_dx)
    start_y += float(start_dy)

    control_x = (
        0.5 * (start_x + end_x) + float(control_x_offset)
    )
    control_y = (
        0.5 * (start_y + end_y) + float(control_y_offset)
    )
    arrow_token = tokens[0]
    if terminal_tangent_to_target_center:
        target_pattern = re.compile(
            rf'^\s*{re.escape(target_node_id)}\s+\[(.*?)\]\s*;?',
            re.M | re.S,
        )
        target_match = target_pattern.search(positioned_dot)
        if target_match is None:
            return positioned_dot, False
        target_attrs = target_match.group(1)
        position_match = re.search(r'\bpos="([^"]+)"', target_attrs)
        width_match = re.search(r'\bwidth=([0-9.]+)', target_attrs)
        height_match = re.search(r'\bheight=([0-9.]+)', target_attrs)
        if not (position_match and width_match and height_match):
            return positioned_dot, False
        try:
            target_x, target_y = map(
                float, position_match.group(1).split(','),
            )
            radius_x = float(width_match.group(1)) * 36.0
            radius_y = float(height_match.group(1)) * 36.0
        except (TypeError, ValueError):
            return positioned_dot, False
        radial_x = control_x - target_x
        radial_y = control_y - target_y
        radial_scale_denominator = math.hypot(
            radial_x / radius_x, radial_y / radius_y,
        )
        if radial_scale_denominator <= 1e-12:
            return positioned_dot, False
        radial_scale = 1.0 / radial_scale_denominator
        arrow_x = target_x + radial_scale * radial_x
        arrow_y = target_y + radial_scale * radial_y
        toward_center_x = target_x - arrow_x
        toward_center_y = target_y - arrow_y
        toward_center_norm = math.hypot(
            toward_center_x, toward_center_y,
        )
        if toward_center_norm <= 1e-12:
            return positioned_dot, False
        unit_x = toward_center_x / toward_center_norm
        unit_y = toward_center_y / toward_center_norm
        terminal_gap = float(terminal_arrow_gap)
        end_x = arrow_x - terminal_gap * unit_x
        end_y = arrow_y - terminal_gap * unit_y
        arrow_token = f'e,{arrow_x:.2f},{arrow_y:.2f}'

    cubic_1_x = start_x + (2.0 / 3.0) * (control_x - start_x)
    cubic_1_y = start_y + (2.0 / 3.0) * (control_y - start_y)
    cubic_2_x = end_x + (2.0 / 3.0) * (control_x - end_x)
    cubic_2_y = end_y + (2.0 / 3.0) * (control_y - end_y)

    adjusted_position = (
        f'{arrow_token} '
        f'{start_x:.2f},{start_y:.2f} '
        f'{cubic_1_x:.2f},{cubic_1_y:.2f} '
        f'{cubic_2_x:.2f},{cubic_2_y:.2f} '
        f'{end_x:.2f},{end_y:.2f}'
    )
    adjusted_dot = (
        positioned_dot[:match.start(2)]
        + adjusted_position
        + positioned_dot[match.end(2):]
    )
    return adjusted_dot, True


def _set_positioned_edge_cubic_spline(
        positioned_dot, source_node_id, target_node_id,
        control_1_dx=0.0, control_1_dy=0.0,
        control_2_dx=0.0, control_2_dy=0.0,
        start_dx=0.0, start_dy=0.0,
        terminal_tangent_to_target_center=False,
        terminal_arrow_gap=8.0, terminal_anchor_side=None):
    """Redraw one edge from four explicit cubic Bezier points."""
    import math
    import re

    edge_pattern = re.compile(
        rf'({re.escape(source_node_id)}\s*->\s*'
        rf'{re.escape(target_node_id)}\s*\[.*?\bpos=")'
        r'([^"]+)(".*?\]\s*;?)',
        re.S,
    )
    match = edge_pattern.search(positioned_dot)
    if match is None:
        return positioned_dot, False

    tokens = match.group(2).replace('\\\n', ' ').split()
    if len(tokens) < 3 or not tokens[0].startswith('e,'):
        return positioned_dot, False
    try:
        start_x, start_y = map(float, tokens[1].split(','))
        end_x, end_y = map(float, tokens[-1].split(','))
    except (TypeError, ValueError):
        return positioned_dot, False

    start_x += float(start_dx)
    start_y += float(start_dy)

    spline_points = [
        (start_x, start_y),
        (start_x + float(control_1_dx),
         start_y + float(control_1_dy)),
        (end_x + float(control_2_dx),
         end_y + float(control_2_dy)),
        (end_x, end_y),
    ]
    arrow_token = tokens[0]
    if terminal_anchor_side is not None:
        target_pattern = re.compile(
            rf'^\s*{re.escape(target_node_id)}\s+\[(.*?)\]\s*;?',
            re.M | re.S,
        )
        target_match = target_pattern.search(positioned_dot)
        if target_match is None:
            return positioned_dot, False
        target_attrs = target_match.group(1)
        position_match = re.search(r'\bpos="([^"]+)"', target_attrs)
        width_match = re.search(r'\bwidth=([0-9.]+)', target_attrs)
        if not (position_match and width_match):
            return positioned_dot, False
        try:
            target_x, target_y = map(
                float, position_match.group(1).split(','),
            )
            radius_x = float(width_match.group(1)) * 36.0
        except (TypeError, ValueError):
            return positioned_dot, False
        if terminal_anchor_side == "right":
            arrow_x = target_x + radius_x
            inward_unit_x = -1.0
        elif terminal_anchor_side == "left":
            arrow_x = target_x - radius_x
            inward_unit_x = 1.0
        else:
            raise ValueError(
                "terminal_anchor_side must be 'left', 'right', or None"
            )
        arrow_y = target_y
        terminal_gap = float(terminal_arrow_gap)
        end_x = arrow_x - terminal_gap * inward_unit_x
        end_y = arrow_y
        spline_points = [
            (start_x, start_y),
            (start_x + float(control_1_dx),
             start_y + float(control_1_dy)),
            (end_x + float(control_2_dx),
             end_y + float(control_2_dy)),
            (end_x, end_y),
        ]
        arrow_token = f'e,{arrow_x:.2f},{arrow_y:.2f}'
    elif terminal_tangent_to_target_center:
        target_pattern = re.compile(
            rf'^\s*{re.escape(target_node_id)}\s+\[(.*?)\]\s*;?',
            re.M | re.S,
        )
        target_match = target_pattern.search(positioned_dot)
        if target_match is None:
            return positioned_dot, False
        target_attrs = target_match.group(1)
        position_match = re.search(r'\bpos="([^"]+)"', target_attrs)
        width_match = re.search(r'\bwidth=([0-9.]+)', target_attrs)
        height_match = re.search(r'\bheight=([0-9.]+)', target_attrs)
        if not (position_match and width_match and height_match):
            return positioned_dot, False
        try:
            target_x, target_y = map(
                float, position_match.group(1).split(','),
            )
            radius_x = float(width_match.group(1)) * 36.0
            radius_y = float(height_match.group(1)) * 36.0
        except (TypeError, ValueError):
            return positioned_dot, False
        control_x, control_y = spline_points[-2]
        radial_x = control_x - target_x
        radial_y = control_y - target_y
        radial_scale_denominator = math.hypot(
            radial_x / radius_x, radial_y / radius_y,
        )
        if radial_scale_denominator <= 1e-12:
            return positioned_dot, False
        radial_scale = 1.0 / radial_scale_denominator
        arrow_x = target_x + radial_scale * radial_x
        arrow_y = target_y + radial_scale * radial_y
        arrow_token = f'e,{arrow_x:.2f},{arrow_y:.2f}'
        toward_center_x = target_x - arrow_x
        toward_center_y = target_y - arrow_y
        toward_center_norm = math.hypot(
            toward_center_x, toward_center_y,
        )
        if toward_center_norm <= 1e-12:
            return positioned_dot, False
        unit_x = toward_center_x / toward_center_norm
        unit_y = toward_center_y / toward_center_norm
        # Keep a non-zero terminal shaft behind the arrowhead.  The third
        # control point, fourth cubic point, arrow tip, and target centre are
        # collinear, so the arrow follows the terminal curve tangent inward.
        terminal_gap = float(terminal_arrow_gap)
        spline_points[-1] = (
            arrow_x - terminal_gap * unit_x,
            arrow_y - terminal_gap * unit_y,
        )

    adjusted_position = (
        f'{arrow_token} '
        + ' '.join(
            f'{x_coord:.2f},{y_coord:.2f}'
            for x_coord, y_coord in spline_points
        )
    )
    adjusted_dot = (
        positioned_dot[:match.start(2)]
        + adjusted_position
        + positioned_dot[match.end(2):]
    )
    return adjusted_dot, True


def _set_positioned_edge_rounded_spline(
        positioned_dot, source_node_id, target_node_id,
        inset_dx=-67.5, top_round=92.0, bottom_round=108.0,
        top_lift=0.0, exit_control_x_scale=0.91,
        exit_control_y_scale=0.15, exit_mid_x=None, exit_mid_y=None,
        exit_tangent_dx=45.5, exit_tangent_dy=-36.0,
        end_control_dx=-48.0, balanced_rounds=False,
        start_dx=0.0, start_dy=0.0,
        terminal_anchor_side=None, terminal_arrow_gap=8.0):
    """Redraw one edge with a vertical body and two generous round corners."""
    import re

    edge_pattern = re.compile(
        rf'({re.escape(source_node_id)}\s*->\s*'
        rf'{re.escape(target_node_id)}\s*\[.*?\bpos=")'
        r'([^"]+)(".*?\]\s*;?)',
        re.S,
    )
    match = edge_pattern.search(positioned_dot)
    if match is None:
        return positioned_dot, False

    tokens = match.group(2).replace('\\\n', ' ').split()
    if len(tokens) < 3 or not tokens[0].startswith('e,'):
        return positioned_dot, False
    try:
        start_x, start_y = map(float, tokens[1].split(','))
        end_x, end_y = map(float, tokens[-1].split(','))
        _, arrow_x, _ = tokens[0].split(',')
        arrow_x = float(arrow_x)
    except (TypeError, ValueError):
        return positioned_dot, False

    start_x += float(start_dx)
    start_y += float(start_dy)
    arrow_token = tokens[0]
    if terminal_anchor_side is not None:
        target_pattern = re.compile(
            rf'^\s*{re.escape(target_node_id)}\s+\[(.*?)\]\s*;?',
            re.M | re.S,
        )
        target_match = target_pattern.search(positioned_dot)
        if target_match is None:
            return positioned_dot, False
        target_attrs = target_match.group(1)
        position_match = re.search(r'\bpos="([^"]+)"', target_attrs)
        width_match = re.search(r'\bwidth=([0-9.]+)', target_attrs)
        if not (position_match and width_match):
            return positioned_dot, False
        try:
            target_x, target_y = map(
                float, position_match.group(1).split(','),
            )
            target_radius_x = float(width_match.group(1)) * 36.0
        except (TypeError, ValueError):
            return positioned_dot, False
        if terminal_anchor_side == "right":
            arrow_x = target_x + target_radius_x
            end_x = arrow_x + float(terminal_arrow_gap)
        elif terminal_anchor_side == "left":
            arrow_x = target_x - target_radius_x
            end_x = arrow_x - float(terminal_arrow_gap)
        else:
            raise ValueError(
                "terminal_anchor_side must be 'left', 'right', or None"
            )
        end_y = target_y
        arrow_token = f'e,{arrow_x:.2f},{end_y:.3f}'

    inset_x = start_x + float(inset_dx)
    top_round = float(top_round)
    bottom_round = float(bottom_round)
    top_lift = float(top_lift)
    exit_control_x_scale = float(exit_control_x_scale)
    exit_control_y_scale = float(exit_control_y_scale)
    top_turn_y = start_y - top_round
    bottom_turn_y = end_y + bottom_round
    spline_points = [
        (start_x, start_y),
        (start_x + 0.56 * float(inset_dx),
         start_y - 0.13 * top_round + top_lift),
        (inset_x, start_y - 0.44 * top_round + top_lift),
        (inset_x, top_turn_y),
        (inset_x, top_turn_y - 0.71 * top_round),
        (inset_x, bottom_turn_y + 0.69 * bottom_round),
        (inset_x, bottom_turn_y),
    ]
    if exit_mid_x is None and exit_mid_y is None:
        if balanced_rounds:
            # Mirror the normalized upper-corner handles at the lower turn.
            # This prevents a fixed absolute x handle from overshooting the
            # vertical channel when the source and target have different x.
            lower_span_x = end_x - inset_x
            spline_points.extend([
                (inset_x, end_y + 0.44 * bottom_round),
                (end_x - 0.56 * lower_span_x,
                 end_y + 0.13 * bottom_round),
                (end_x, end_y),
            ])
        else:
            spline_points.extend([
                (inset_x, end_y + 0.46 * bottom_round),
                (end_x - exit_control_x_scale * bottom_round,
                 end_y + exit_control_y_scale * bottom_round),
                (end_x, end_y),
            ])
    elif exit_mid_x is not None and exit_mid_y is not None:
        exit_mid_x = float(exit_mid_x)
        exit_mid_y = float(exit_mid_y)
        exit_tangent_dx = float(exit_tangent_dx)
        exit_tangent_dy = float(exit_tangent_dy)
        spline_points.extend([
            (inset_x, end_y + 0.71 * bottom_round),
            (exit_mid_x - exit_tangent_dx,
             exit_mid_y - exit_tangent_dy),
            (exit_mid_x, exit_mid_y),
            (exit_mid_x + exit_tangent_dx,
             exit_mid_y + exit_tangent_dy),
            (end_x + float(end_control_dx), end_y),
            (end_x, end_y),
        ])
    else:
        raise ValueError(
            "exit_mid_x and exit_mid_y must be provided together"
        )
    adjusted_position = (
        f'{arrow_token} '
        + ' '.join(
            f'{x_coord:.2f},{y_coord:.2f}'
            for x_coord, y_coord in spline_points
        )
    )
    adjusted_dot = (
        positioned_dot[:match.start(2)]
        + adjusted_position
        + positioned_dot[match.end(2):]
    )
    return adjusted_dot, True


def _offset_positioned_edge_label(
        positioned_dot, source_node_id, target_node_id,
        position_attr, dx=0.0, dy=0.0):
    """Move one frozen Graphviz edge label without changing its spline."""
    import re

    edge_pattern = re.compile(
        rf'({re.escape(source_node_id)}\s*->\s*'
        rf'{re.escape(target_node_id)}\s*\[.*?\b'
        rf'{re.escape(position_attr)}=")'
        r'([^,"]+),([^\"]+)(")',
        re.S,
    )
    match = edge_pattern.search(positioned_dot)
    if match is None:
        return positioned_dot, False
    try:
        label_x = float(match.group(2)) + float(dx)
        label_y = float(match.group(3)) + float(dy)
    except (TypeError, ValueError):
        return positioned_dot, False

    adjusted_dot = (
        positioned_dot[:match.start(2)]
        + f'{label_x:.2f},{label_y:.2f}'
        + positioned_dot[match.end(3):]
    )
    return adjusted_dot, True


def _set_positioned_edge_first_segment_three_point(
        positioned_dot, source_node_id, target_node_id,
        terminal_vertical=False):
    """Convert the first cubic segment to a smooth three-point spline."""
    import re

    edge_pattern = re.compile(
        rf'({re.escape(source_node_id)}\s*->\s*'
        rf'{re.escape(target_node_id)}\s*\[.*?\bpos=")'
        r'([^\"]+)(".*?\]\s*;?)',
        re.S,
    )
    match = edge_pattern.search(positioned_dot)
    if match is None:
        return positioned_dot, False
    tokens = match.group(2).replace('\\\n', ' ').split()
    if len(tokens) < 5 or not tokens[0].startswith('e,'):
        return positioned_dot, False
    try:
        points = [
            list(map(float, token.split(','))) for token in tokens[1:]
        ]
    except (TypeError, ValueError):
        return positioned_dot, False
    if len(points) < 4:
        return positioned_dot, False

    start, cubic_1, cubic_2, end = points[:4]
    midpoint = [
        (
            start[axis]
            + 3.0 * cubic_1[axis]
            + 3.0 * cubic_2[axis]
            + end[axis]
        ) / 8.0
        for axis in (0, 1)
    ]
    quadratic_control = [
        2.0 * midpoint[axis]
        - 0.5 * (start[axis] + end[axis])
        for axis in (0, 1)
    ]
    if terminal_vertical:
        # Match the next vertical segment's tangent at the shared endpoint.
        quadratic_control[0] = end[0]
    points[1] = [
        start[axis]
        + (2.0 / 3.0) * (quadratic_control[axis] - start[axis])
        for axis in (0, 1)
    ]
    points[2] = [
        end[axis]
        + (2.0 / 3.0) * (quadratic_control[axis] - end[axis])
        for axis in (0, 1)
    ]
    adjusted_position = (
        f'{tokens[0]} '
        + ' '.join(
            f'{point[0]:.3f},{point[1]:.3f}' for point in points
        )
    )
    adjusted_dot = (
        positioned_dot[:match.start(2)]
        + adjusted_position
        + positioned_dot[match.end(2):]
    )
    return adjusted_dot, True


def _snap_positioned_edge_labels_to_splines(
        positioned_dot, preferred_fractions=None, samples_per_curve=240):
    """Place every coefficient label directly on its own rendered spline."""
    import math
    import re

    preferred_fractions = preferred_fractions or {}
    edge_pattern = re.compile(
        r'(?P<source>[A-Za-z0-9_]+)\s*->\s*'
        r'(?P<target>[A-Za-z0-9_]+)\s*\[(?P<attrs>.*?)\]\s*;?',
        re.S,
    )
    replacements = []
    adjusted_count = 0

    for edge_match in edge_pattern.finditer(positioned_dot):
        attrs = edge_match.group("attrs")
        lp_match = re.search(
            r'\blp="([^,\"]+),([^\"]+)"', attrs,
        )
        pos_match = re.search(r'\bpos="([^\"]+)"', attrs, re.S)
        if lp_match is None or pos_match is None or "label=" not in attrs:
            continue
        try:
            current_lp = (float(lp_match.group(1)), float(lp_match.group(2)))
            tokens = pos_match.group(1).replace('\\\n', ' ').split()
            raw_points = [
                tuple(map(float, token.split(',')))
                for token in tokens[1:]
            ]
        except (TypeError, ValueError):
            continue
        if len(raw_points) < 2:
            continue

        sampled = []
        if len(raw_points) >= 4 and (len(raw_points) - 1) % 3 == 0:
            segment_count = (len(raw_points) - 1) // 3
            per_segment = max(
                24, int(samples_per_curve) // max(1, segment_count),
            )
            for segment_index in range(segment_count):
                p0, p1, p2, p3 = raw_points[
                    3 * segment_index:3 * segment_index + 4
                ]
                for sample_index in range(per_segment + 1):
                    if segment_index and sample_index == 0:
                        continue
                    t = sample_index / per_segment
                    one_minus_t = 1.0 - t
                    x_coord = (
                        one_minus_t ** 3 * p0[0]
                        + 3.0 * one_minus_t ** 2 * t * p1[0]
                        + 3.0 * one_minus_t * t ** 2 * p2[0]
                        + t ** 3 * p3[0]
                    )
                    y_coord = (
                        one_minus_t ** 3 * p0[1]
                        + 3.0 * one_minus_t ** 2 * t * p1[1]
                        + 3.0 * one_minus_t * t ** 2 * p2[1]
                        + t ** 3 * p3[1]
                    )
                    sampled.append((x_coord, y_coord))
        else:
            sampled = list(raw_points)
        if len(sampled) < 2:
            continue

        cumulative = [0.0]
        for point_1, point_2 in zip(sampled, sampled[1:]):
            cumulative.append(
                cumulative[-1] + math.hypot(
                    point_2[0] - point_1[0],
                    point_2[1] - point_1[1],
                )
            )
        edge_key = (
            edge_match.group("source"), edge_match.group("target"),
        )
        if edge_key in preferred_fractions and cumulative[-1] > 0.0:
            target_length = (
                max(0.0, min(1.0, float(preferred_fractions[edge_key])))
                * cumulative[-1]
            )
            candidate_index = min(
                range(len(sampled)),
                key=lambda index: abs(cumulative[index] - target_length),
            )
        else:
            candidate_index = min(
                range(len(sampled)),
                key=lambda index: (
                    (sampled[index][0] - current_lp[0]) ** 2
                    + (sampled[index][1] - current_lp[1]) ** 2
                ),
            )
        label_x, label_y = sampled[candidate_index]
        attrs_start = edge_match.start("attrs")
        replacements.append((
            attrs_start + lp_match.start(1),
            attrs_start + lp_match.end(2),
            f'{label_x:.2f},{label_y:.2f}',
        ))
        adjusted_count += 1

    adjusted_dot = positioned_dot
    for start_index, end_index, replacement in reversed(replacements):
        adjusted_dot = (
            adjusted_dot[:start_index]
            + replacement
            + adjusted_dot[end_index:]
        )
    return adjusted_dot, adjusted_count


def _set_positioned_edge_equal_corner_spline(
        positioned_dot, source_node_id, target_node_id,
        channel_dx, corner_radius, top_entry_y_offset=0.0,
        terminal_y=None, start_dx=0.0, start_dy=0.0,
        terminal_anchor_side=None, terminal_arrow_gap=8.0):
    """Redraw one edge with geometrically identical upper/lower corners."""
    import re

    edge_pattern = re.compile(
        rf'({re.escape(source_node_id)}\s*->\s*'
        rf'{re.escape(target_node_id)}\s*\[.*?\bpos=")'
        r'([^"]+)(".*?\]\s*;?)',
        re.S,
    )
    match = edge_pattern.search(positioned_dot)
    if match is None:
        return positioned_dot, False

    tokens = match.group(2).replace('\\\n', ' ').split()
    if len(tokens) < 3 or not tokens[0].startswith('e,'):
        return positioned_dot, False
    try:
        start_x, start_y = map(float, tokens[1].split(','))
        end_x, end_y = map(float, tokens[-1].split(','))
        _, arrow_x, _ = tokens[0].split(',')
        arrow_x = float(arrow_x)
    except (TypeError, ValueError):
        return positioned_dot, False

    start_x += float(start_dx)
    start_y += float(start_dy)

    if terminal_anchor_side is not None:
        target_pattern = re.compile(
            rf'^\s*{re.escape(target_node_id)}\s+\[(.*?)\]\s*;?',
            re.M | re.S,
        )
        target_match = target_pattern.search(positioned_dot)
        if target_match is None:
            return positioned_dot, False
        target_attrs = target_match.group(1)
        position_match = re.search(r'\bpos="([^"]+)"', target_attrs)
        width_match = re.search(r'\bwidth=([0-9.]+)', target_attrs)
        if not (position_match and width_match):
            return positioned_dot, False
        try:
            target_x, target_y = map(
                float, position_match.group(1).split(','),
            )
            target_radius_x = float(width_match.group(1)) * 36.0
        except (TypeError, ValueError):
            return positioned_dot, False
        if terminal_anchor_side == "right":
            arrow_x = target_x + target_radius_x
            end_x = arrow_x + float(terminal_arrow_gap)
        elif terminal_anchor_side == "left":
            arrow_x = target_x - target_radius_x
            end_x = arrow_x - float(terminal_arrow_gap)
        else:
            raise ValueError(
                "terminal_anchor_side must be 'left', 'right', or None"
            )
        end_y = target_y
        arrow_token = f'e,{arrow_x:.2f},{end_y:.3f}'
    elif terminal_y is not None:
        end_y = float(terminal_y)
        arrow_token = f'e,{arrow_x:.2f},{end_y:.3f}'
    else:
        arrow_token = tokens[0]

    channel_x = start_x + float(channel_dx)
    radius = abs(float(corner_radius))
    top_entry_y = start_y + float(top_entry_y_offset)
    top_turn_y = top_entry_y - radius
    bottom_turn_y = end_y + radius
    upper_side = 1.0 if channel_x >= start_x else -1.0
    lower_side = 1.0 if channel_x >= end_x else -1.0
    upper_entry_x = channel_x - upper_side * radius
    lower_exit_x = channel_x - lower_side * radius
    kappa = 0.5522847498

    lead_dx = upper_entry_x - start_x
    exit_dx = end_x - lower_exit_x
    vertical_span = bottom_turn_y - top_turn_y
    spline_points = [
        (start_x, start_y),
        (start_x + 0.32 * lead_dx,
         start_y + 0.20 * float(top_entry_y_offset)),
        (upper_entry_x - 0.24 * lead_dx, top_entry_y),
        (upper_entry_x, top_entry_y),
        (upper_entry_x + upper_side * kappa * radius, top_entry_y),
        (channel_x, top_turn_y + kappa * radius),
        (channel_x, top_turn_y),
        (channel_x, top_turn_y + vertical_span / 3.0),
        (channel_x, top_turn_y + 2.0 * vertical_span / 3.0),
        (channel_x, bottom_turn_y),
        (channel_x, bottom_turn_y - kappa * radius),
        (lower_exit_x + lower_side * kappa * radius, end_y),
        (lower_exit_x, end_y),
        (lower_exit_x + exit_dx / 3.0, end_y),
        (lower_exit_x + 2.0 * exit_dx / 3.0, end_y),
        (end_x, end_y),
    ]
    adjusted_position = (
        f'{arrow_token} '
        + ' '.join(
            f'{x_coord:.3f},{y_coord:.3f}'
            for x_coord, y_coord in spline_points
        )
    )
    adjusted_dot = (
        positioned_dot[:match.start(2)]
        + adjusted_position
        + positioned_dot[match.end(2):]
    )
    return adjusted_dot, True


def _get_positioned_edge_arrow_y(
        positioned_dot, source_node_id, target_node_id):
    """Return the frozen arrow-tip y coordinate for one positioned edge."""
    import re

    edge_pattern = re.compile(
        rf'{re.escape(source_node_id)}\s*->\s*'
        rf'{re.escape(target_node_id)}\s*\[.*?\bpos="'
        r'([^\"]+)"',
        re.S,
    )
    match = edge_pattern.search(positioned_dot)
    if match is None:
        return None
    tokens = match.group(1).replace('\\\n', ' ').split()
    if not tokens or not tokens[0].startswith('e,'):
        return None
    try:
        return float(tokens[0].split(',')[2])
    except (IndexError, TypeError, ValueError):
        return None


def _offset_positioned_node_with_incident_edges(
        positioned_dot, node_id, dx=0.0, dy=0.0,
        endpoint_control_count=3):
    """Move one frozen node and smoothly warp every incident edge."""
    import re

    dx = float(dx)
    dy = float(dy)
    node_pattern = re.compile(
        rf'(^\s*{re.escape(node_id)}\s+\[.*?\bpos=")'
        r'([^,\"]+),([^\"]+)(".*?\]\s*;?)',
        re.M | re.S,
    )
    node_match = node_pattern.search(positioned_dot)
    if node_match is None:
        return positioned_dot, False, 0
    try:
        node_x = float(node_match.group(2)) + dx
        node_y = float(node_match.group(3)) + dy
    except (TypeError, ValueError):
        return positioned_dot, False, 0

    replacements = [(
        node_match.start(2), node_match.end(3),
        f'{node_x:.3f},{node_y:.3f}',
    )]
    adjusted_edge_count = 0
    edge_pattern = re.compile(
        r'(?P<source>[A-Za-z0-9_]+)\s*->\s*'
        r'(?P<target>[A-Za-z0-9_]+)\s*\['
        r'.*?\bpos="(?P<position>[^\"]+)".*?\]\s*;?',
        re.S,
    )
    for edge_match in edge_pattern.finditer(positioned_dot):
        source_id = edge_match.group("source")
        target_id = edge_match.group("target")
        if source_id != node_id and target_id != node_id:
            continue
        tokens = edge_match.group("position").replace('\\\n', ' ').split()
        if len(tokens) < 3 or not tokens[0].startswith('e,'):
            continue
        try:
            arrow_parts = tokens[0].split(',')
            points = [
                list(map(float, token.split(','))) for token in tokens[1:]
            ]
        except (TypeError, ValueError):
            continue
        point_denominator = max(1, len(points) - 1)
        for point_index, point in enumerate(points):
            path_fraction = point_index / point_denominator
            shift_weight = 0.0
            if source_id == node_id:
                shift_weight += 1.0 - path_fraction
            if target_id == node_id:
                shift_weight += path_fraction
            point[0] += shift_weight * dx
            point[1] += shift_weight * dy
        if target_id == node_id:
            arrow_parts[1] = f'{float(arrow_parts[1]) + dx:.3f}'
            arrow_parts[2] = f'{float(arrow_parts[2]) + dy:.3f}'
        adjusted_position = (
            ','.join(arrow_parts)
            + ' '
            + ' '.join(
                f'{point[0]:.3f},{point[1]:.3f}' for point in points
            )
        )
        replacements.append((
            edge_match.start("position"),
            edge_match.end("position"),
            adjusted_position,
        ))
        adjusted_edge_count += 1

    adjusted_dot = positioned_dot
    for start_index, end_index, replacement in reversed(replacements):
        adjusted_dot = (
            adjusted_dot[:start_index]
            + replacement
            + adjusted_dot[end_index:]
        )
    return adjusted_dot, True, adjusted_edge_count


def _offset_positioned_edge_interior(
        positioned_dot, source_node_id, target_node_id,
        dx=0.0, dy=0.0):
    """Shift the middle of one spline while preserving both endpoint anchors."""
    import re

    edge_pattern = re.compile(
        rf'({re.escape(source_node_id)}\s*->\s*'
        rf'{re.escape(target_node_id)}\s*\[.*?\bpos=")'
        r'([^\"]+)(".*?\]\s*;?)',
        re.S,
    )
    match = edge_pattern.search(positioned_dot)
    if match is None:
        return positioned_dot, False
    tokens = match.group(2).replace('\\\n', ' ').split()
    if len(tokens) < 4 or not tokens[0].startswith('e,'):
        return positioned_dot, False
    try:
        points = [
            list(map(float, token.split(','))) for token in tokens[1:]
        ]
    except (TypeError, ValueError):
        return positioned_dot, False

    denominator = max(1, len(points) - 1)
    for point_index, point in enumerate(points):
        path_fraction = point_index / denominator
        interior_weight = 4.0 * path_fraction * (1.0 - path_fraction)
        point[0] += interior_weight * float(dx)
        point[1] += interior_weight * float(dy)
    adjusted_position = (
        f'{tokens[0]} '
        + ' '.join(
            f'{point[0]:.3f},{point[1]:.3f}' for point in points
        )
    )
    adjusted_dot = (
        positioned_dot[:match.start(2)]
        + adjusted_position
        + positioned_dot[match.end(2):]
    )
    return adjusted_dot, True


def _offset_positioned_edge_terminal(
        positioned_dot, source_node_id, target_node_id,
        arrow_dx=0.0, arrow_dy=0.0, end_dx=0.0, end_dy=0.0,
        horizontal_terminal=False):
    """Adjust one arrow terminal and its final tangent without relayout."""
    import re

    edge_pattern = re.compile(
        rf'({re.escape(source_node_id)}\s*->\s*'
        rf'{re.escape(target_node_id)}\s*\[.*?\bpos=")'
        r'([^"]+)(".*?\]\s*;?)',
        re.S,
    )
    match = edge_pattern.search(positioned_dot)
    if match is None:
        return positioned_dot, False

    tokens = match.group(2).replace('\\\n', ' ').split()
    if len(tokens) < 4 or not tokens[0].startswith('e,'):
        return positioned_dot, False
    try:
        _, arrow_x, arrow_y = tokens[0].split(',')
        arrow_x = float(arrow_x) + float(arrow_dx)
        arrow_y = float(arrow_y) + float(arrow_dy)
        spline_points = [
            list(map(float, token.split(','))) for token in tokens[1:]
        ]
    except (TypeError, ValueError):
        return positioned_dot, False

    spline_points[-1][0] += float(end_dx)
    spline_points[-1][1] += float(end_dy)
    if horizontal_terminal:
        spline_points[-2][1] = spline_points[-1][1]
    adjusted_position = (
        f'e,{arrow_x:.2f},{arrow_y:.2f} '
        + ' '.join(
            f'{x_coord:.2f},{y_coord:.2f}'
            for x_coord, y_coord in spline_points
        )
    )
    adjusted_dot = (
        positioned_dot[:match.start(2)]
        + adjusted_position
        + positioned_dot[match.end(2):]
    )
    return adjusted_dot, True


def _offset_positioned_edge_outer_channel(
        positioned_dot, source_node_id, target_node_id,
        dx=0.0, side="right", tolerance=0.05):
    """Move only the outer vertical channel of one positioned edge."""
    import re

    edge_pattern = re.compile(
        rf'({re.escape(source_node_id)}\s*->\s*'
        rf'{re.escape(target_node_id)}\s*\[.*?\bpos=")'
        r'([^"]+)(".*?\]\s*;?)',
        re.S,
    )
    match = edge_pattern.search(positioned_dot)
    if match is None:
        return positioned_dot, False

    tokens = match.group(2).replace('\\\n', ' ').split()
    if len(tokens) < 3 or not tokens[0].startswith('e,'):
        return positioned_dot, False
    try:
        spline_points = [
            list(map(float, token.split(','))) for token in tokens[1:]
        ]
    except (TypeError, ValueError):
        return positioned_dot, False

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    outer_x = (
        max(point[0] for point in spline_points)
        if side == "right"
        else min(point[0] for point in spline_points)
    )
    adjusted_count = 0
    for point in spline_points:
        if abs(point[0] - outer_x) <= float(tolerance):
            point[0] += float(dx)
            adjusted_count += 1
    if adjusted_count == 0:
        return positioned_dot, False

    adjusted_position = (
        f'{tokens[0]} '
        + ' '.join(
            f'{x_coord:.2f},{y_coord:.2f}'
            for x_coord, y_coord in spline_points
        )
    )
    adjusted_dot = (
        positioned_dot[:match.start(2)]
        + adjusted_position
        + positioned_dot[match.end(2):]
    )
    return adjusted_dot, True


def _set_positioned_graph_font_sizes(
        positioned_dot, node_fontsize=12.0, edge_fontsize=11.0,
        node_fontname="Times New Roman, Liberation Serif",
        edge_fontname="Arial"):
    """Set frozen-layout text size/font without moving graph geometry."""
    import re

    adjusted_dot = positioned_dot
    replacement_count = 0
    for element, fontsize in (
            ("node", node_fontsize), ("edge", edge_fontsize)):
        pattern = re.compile(
            rf'(^\s*{element}\s+\[.*?\bfontsize=)([0-9.]+)',
            re.M | re.S,
        )
        adjusted_dot, count = pattern.subn(
            rf'\g<1>{float(fontsize):g}', adjusted_dot, count=1,
        )
        replacement_count += count
    font_count = 0
    for element, fontname in (
            ("node", node_fontname), ("edge", edge_fontname)):
        font_pattern = re.compile(
            rf'(^\s*{element}\s+\[.*?\bfontname=)'
            r'(?:"[^"]*"|[^\s,\]]+)',
            re.M | re.S,
        )
        adjusted_dot, count = font_pattern.subn(
            lambda match, name=fontname: f'{match.group(1)}"{name}"',
            adjusted_dot,
            count=1,
        )
        font_count += count
    return adjusted_dot, replacement_count == 2 and font_count == 2


def _render_dag_graphviz(adj_matrix, labels, output_base, edge_threshold=0.0):
    """Render a LiNGAM DAG with a light, publication-oriented Graphviz layout."""
    try:
        import graphviz
        from graphviz.backend import ExecutableNotFound
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Python graphviz 包，无法绘制 DAG。请运行: "
            "pip install graphviz==0.21；Ubuntu 还需运行: "
            "sudo apt-get install -y graphviz"
        ) from exc

    adj = np.asarray(adj_matrix, dtype=float)
    labels = [str(label) for label in labels]
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"DAG 邻接矩阵必须是方阵，当前形状为 {adj.shape}")
    if adj.shape[0] != len(labels):
        raise ValueError(
            f"DAG 标签数 ({len(labels)}) 与矩阵维度 ({adj.shape[0]}) 不一致"
        )

    output_base = os.path.abspath(os.fspath(output_base))
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    layout_spec = _dag_layout_spec(labels)
    manual_edge_adjustments = bool(
        layout_spec
        and layout_spec.get("manual_edge_adjustments", False)
    )
    edge_adjustment_profile = (
        layout_spec.get("edge_adjustment_profile")
        if layout_spec else None
    )
    vertical_order = (
        list(layout_spec.get("vertical_order", []))
        if layout_spec else []
    )
    nodesep = layout_spec["nodesep"] if layout_spec else "0.42"
    ranksep = layout_spec["ranksep"] if layout_spec else "0.72 equally"
    search_limit = "20.0" if layout_spec else "8.0"
    graph = graphviz.Digraph(name="causal_dag", engine="dot")
    graph.attr(
        "graph", rankdir="TB", splines="spline", nodesep=nodesep,
        ranksep=ranksep, bgcolor="white", dpi="72",
        pad="0.16", margin="0", outputorder="edgesfirst",
        newrank="true", remincross="true", forcelabels="true",
        mclimit=search_limit, nslimit=search_limit, nslimit1=search_limit,
        fontname="Arial",
    )
    graph.attr(
        "node", shape="ellipse", style="filled", fillcolor="#FFFFFF",
        color="#1F1F1F", penwidth="0.9", fontname="Arial",
        fontsize="12", fontcolor="#1F1F1F", margin="0.14,0.08",
    )
    graph.attr(
        "edge", color="#1F1F1F", fontcolor="#1F1F1F",
        arrowsize="0.65", penwidth="0.9", fontname="Arial",
        fontsize="11", labelfloat="false",
    )

    node_ids = {label: f"node{index}" for index, label in enumerate(labels)}
    for label in labels:
        display_label = "Water" if label == "water" else label
        node_attrs = {"label": display_label}
        if layout_spec and label in layout_spec["groups"]:
            node_attrs["group"] = layout_spec["groups"][label]
        graph.node(node_ids[label], **node_attrs)

    # LiNGAM convention: adj[i, j] is the directed edge labels[j] -> labels[i].
    for target_index in range(adj.shape[0]):
        for source_index in range(adj.shape[1]):
            coefficient = adj[target_index, source_index]
            if (target_index == source_index
                    or not np.isfinite(coefficient)
                    or abs(coefficient) <= edge_threshold):
                continue
            source_label = labels[source_index]
            target_label = labels[target_index]
            coefficient_label = (
                '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" '
                'CELLPADDING="1" BGCOLOR="white"><TR><TD>'
                f'{coefficient:.3f}'
                '</TD></TR></TABLE>>'
            )
            edge_attrs = {"label": coefficient_label, "weight": "1"}
            # Local publication-layout adjustment requested for the
            # water -> Road_Dens path: enter Road_Dens from its upper-right
            # side and let the coefficient float outside the spline so the
            # curve is not kinked by a midpoint label box.
            if (manual_edge_adjustments
                    and (source_label, target_label)
                    == ("water", "Road_Dens")):
                edge_attrs = {
                    "headlabel": coefficient_label,
                    "labelangle": "-32.255",
                    "labeldistance": "9",
                    "weight": "1",
                }
            elif (manual_edge_adjustments
                    and source_label == "Pop2025"
                    and target_label in {"UHII", "Temperature"}):
                edge_attrs = {
                    "headlabel": coefficient_label,
                    "labelangle": "-10",
                    "labeldistance": "9",
                    "weight": "1",
                }
            graph.edge(
                node_ids[source_label], node_ids[target_label], **edge_attrs,
            )

    # Keep a causal-only DOT for downstream parsing.
    causal_dot_path = graph.save(
        filename=f"{output_base}_causal_only.dot",
    )
    if layout_spec:
        for row_index, row in enumerate(layout_spec["rows"]):
            with graph.subgraph(name=f"layout_rank_{row_index}") as rank:
                rank.attr(rank="same")
                for label in row:
                    rank.node(node_ids[label])
                for left_label, right_label in zip(row, row[1:]):
                    rank.edge(
                        node_ids[left_label], node_ids[right_label],
                        style="invis", weight="100", constraint="true",
                    )
        for source_label, target_label in zip(
                vertical_order, vertical_order[1:]):
            source_index = labels.index(source_label)
            target_index = labels.index(target_label)
            has_forward_edge = (
                abs(adj[target_index, source_index]) > edge_threshold
            )
            has_reverse_edge = (
                abs(adj[source_index, target_index]) > edge_threshold
            )
            if not has_forward_edge and not has_reverse_edge:
                graph.edge(
                    node_ids[source_label], node_ids[target_label],
                    style="invis", weight="100", constraint="true",
                )

    # The primary DOT includes layout-only invisible constraints so that
    # manual Graphviz rendering reproduces the publication layout.
    dot_path = graph.save(filename=f"{output_base}.dot")
    try:
        _use_local_water_ndvi_curve = (
            manual_edge_adjustments
            and "water" in node_ids
            and "NDVI" in node_ids
            and abs(adj[labels.index("NDVI"), labels.index("water")])
            > edge_threshold
        )
        _use_local_water_road_curve = (
            manual_edge_adjustments
            and "water" in node_ids
            and "Road_Dens" in node_ids
            and abs(adj[labels.index("Road_Dens"), labels.index("water")])
            > edge_threshold
        )
        _use_local_pop_far_curve = (
            manual_edge_adjustments
            and "Pop2025" in node_ids
            and "FAR" in node_ids
            and abs(adj[labels.index("FAR"), labels.index("Pop2025")])
            > edge_threshold
        )
        _use_local_pop_road_curve = (
            manual_edge_adjustments
            and "Pop2025" in node_ids
            and "Road_Dens" in node_ids
            and abs(adj[
                labels.index("Road_Dens"), labels.index("Pop2025"),
            ]) > edge_threshold
        )
        _use_local_pop_uhii_curve = (
            manual_edge_adjustments
            and "Pop2025" in node_ids
            and "UHII" in node_ids
            and abs(adj[labels.index("UHII"), labels.index("Pop2025")])
            > edge_threshold
        )
        _use_local_water_far_terminal = (
            manual_edge_adjustments
            and "water" in node_ids
            and "FAR" in node_ids
            and abs(adj[labels.index("FAR"), labels.index("water")])
            > edge_threshold
        )
        _use_local_water_uhii_curve = (
            manual_edge_adjustments
            and "water" in node_ids
            and "UHII" in node_ids
            and abs(adj[labels.index("UHII"), labels.index("water")])
            > edge_threshold
        )
        _use_primary_pop_mbh_curve = (
            edge_adjustment_profile in {
                "bc_mbh_reference2", "pop_mbh_step1",
                "pop_mbh_pop_uhii_step2", "pop_mbh_pop_uhii_step3",
                "pop_mbh_pop_uhii_step4",
                "pop_mbh_pop_uhii_step5",
                "pop_mbh_pop_uhii_step6",
                "water_mbh_and_label_clearance_step7",
                "water_mbh_left_rebalance_step8",
                "water_outer_curves_balanced_step9",
                "water_right_equal_corners_step10",
                "water_right_inset_corners_step11",
                "uhii_anchor_and_ndvi_cubic_step12",
                "water_start_and_uhii_inset_step13",
                "water_uhii_parallel_step14",
                "labels_on_edges_and_uhii_inset_step15",
                "water_routes_and_labels_step16",
                "nested_water_routes_step17",
                "road_node_left_step18",
                "water_ndvi_round_and_road_spline_step19",
                "right_routes_and_label_step20",
            }
            and "Pop2025" in node_ids
            and "MBH" in node_ids
            and abs(adj[labels.index("MBH"), labels.index("Pop2025")])
            > edge_threshold
        )
        _use_primary_pop_uhii_curve = (
            edge_adjustment_profile in {
                "bc_mbh_reference2", "pop_mbh_pop_uhii_step2",
                "pop_mbh_pop_uhii_step3",
                "pop_mbh_pop_uhii_step4",
                "pop_mbh_pop_uhii_step5",
                "pop_mbh_pop_uhii_step6",
                "water_mbh_and_label_clearance_step7",
                "water_mbh_left_rebalance_step8",
                "water_outer_curves_balanced_step9",
                "water_right_equal_corners_step10",
                "water_right_inset_corners_step11",
                "uhii_anchor_and_ndvi_cubic_step12",
                "water_start_and_uhii_inset_step13",
                "water_uhii_parallel_step14",
                "labels_on_edges_and_uhii_inset_step15",
                "water_routes_and_labels_step16",
                "nested_water_routes_step17",
                "road_node_left_step18",
                "water_ndvi_round_and_road_spline_step19",
                "right_routes_and_label_step20",
            }
            and "Pop2025" in node_ids
            and "UHII" in node_ids
            and abs(adj[labels.index("UHII"), labels.index("Pop2025")])
            > edge_threshold
        )
        _use_primary_label_cleanup = (
            edge_adjustment_profile in {
                "label_cleanup_only", "pop_mbh_step1",
                "pop_mbh_pop_uhii_step2", "pop_mbh_pop_uhii_step3",
                "pop_mbh_pop_uhii_step4",
                "pop_mbh_pop_uhii_step5",
                "pop_mbh_pop_uhii_step6",
                "water_mbh_and_label_clearance_step7",
                "water_mbh_left_rebalance_step8",
                "water_outer_curves_balanced_step9",
                "water_right_equal_corners_step10",
                "water_right_inset_corners_step11",
                "uhii_anchor_and_ndvi_cubic_step12",
                "water_start_and_uhii_inset_step13",
                "water_uhii_parallel_step14",
                "labels_on_edges_and_uhii_inset_step15",
                "water_routes_and_labels_step16",
                "nested_water_routes_step17",
                "road_node_left_step18",
                "water_ndvi_round_and_road_spline_step19",
                "right_routes_and_label_step20",
            }
        )
        _use_primary_water_mbh_curve = (
            edge_adjustment_profile
            in {
                "water_mbh_and_label_clearance_step7",
                "water_mbh_left_rebalance_step8",
                "water_outer_curves_balanced_step9",
                "water_right_equal_corners_step10",
                "water_right_inset_corners_step11",
                "uhii_anchor_and_ndvi_cubic_step12",
                "water_start_and_uhii_inset_step13",
                "water_uhii_parallel_step14",
                "labels_on_edges_and_uhii_inset_step15",
                "water_routes_and_labels_step16",
                "nested_water_routes_step17",
                "road_node_left_step18",
                "water_ndvi_round_and_road_spline_step19",
                "right_routes_and_label_step20",
            }
            and "water" in node_ids
            and "MBH" in node_ids
            and abs(adj[labels.index("MBH"), labels.index("water")])
            > edge_threshold
        )
        _use_primary_water_ndvi_curve = (
            edge_adjustment_profile
            in {
                "water_outer_curves_balanced_step9",
                "water_right_equal_corners_step10",
                "water_right_inset_corners_step11",
                "uhii_anchor_and_ndvi_cubic_step12",
                "water_start_and_uhii_inset_step13",
                "water_uhii_parallel_step14",
                "labels_on_edges_and_uhii_inset_step15",
                "water_routes_and_labels_step16",
                "nested_water_routes_step17",
                "road_node_left_step18",
                "water_ndvi_round_and_road_spline_step19",
                "right_routes_and_label_step20",
            }
            and "water" in node_ids
            and "NDVI" in node_ids
            and abs(adj[labels.index("NDVI"), labels.index("water")])
            > edge_threshold
        )
        _use_primary_water_uhii_equal_corner_curve = (
            edge_adjustment_profile
            in {
                "water_right_equal_corners_step10",
                "water_right_inset_corners_step11",
                "uhii_anchor_and_ndvi_cubic_step12",
                "water_start_and_uhii_inset_step13",
                "water_uhii_parallel_step14",
                "labels_on_edges_and_uhii_inset_step15",
                "water_routes_and_labels_step16",
                "nested_water_routes_step17",
                "road_node_left_step18",
                "water_ndvi_round_and_road_spline_step19",
                "right_routes_and_label_step20",
            }
            and "water" in node_ids
            and "UHII" in node_ids
            and abs(adj[labels.index("UHII"), labels.index("water")])
            > edge_threshold
        )
        _use_primary_road_node_shift = (
            edge_adjustment_profile
            in {
                "road_node_left_step18",
                "water_ndvi_round_and_road_spline_step19",
                "right_routes_and_label_step20",
            }
            and "Road_Dens" in node_ids
        )
        _use_primary_pop_ndvi_inset = (
            edge_adjustment_profile == "right_routes_and_label_step20"
            and "Pop2025" in node_ids
            and "NDVI" in node_ids
            and abs(adj[labels.index("NDVI"), labels.index("Pop2025")])
            > edge_threshold
        )
        if (_use_local_water_ndvi_curve
                or _use_local_water_road_curve
                or _use_local_pop_far_curve
                or _use_local_pop_road_curve
                or _use_local_pop_uhii_curve
                or _use_local_water_far_terminal
                or _use_local_water_uhii_curve
                or _use_primary_pop_mbh_curve
                or _use_primary_pop_uhii_curve
                or _use_primary_label_cleanup
                or _use_primary_water_mbh_curve
                or _use_primary_water_ndvi_curve
                or _use_primary_water_uhii_equal_corner_curve
                or _use_primary_road_node_shift
                or _use_primary_pop_ndvi_inset):
            import re

            # Freeze the complete dot layout first, then adjust only the
            # explicitly selected splines.  Supplying the positioned graph to
            # neato -n2 prevents every node, other edge, and label from moving.
            _positioned_dot = graph.pipe(format="dot", encoding="utf-8")
            if _use_local_water_ndvi_curve:
                _positioned_dot, _water_ndvi_adjusted = (
                    _set_positioned_edge_quadratic_spline(
                        _positioned_dot,
                        node_ids["water"], node_ids["NDVI"],
                        control_x_offset=40.0,
                        control_y_offset=0.0,
                        terminal_tangent_to_target_center=True,
                    )
                )
                if not _water_ndvi_adjusted:
                    warnings.warn(
                        "未能定位 water -> NDVI 的 Graphviz 样条"
                    )
                _positioned_dot, _water_ndvi_label_adjusted = (
                    _offset_positioned_edge_label(
                        _positioned_dot,
                        node_ids["water"], node_ids["NDVI"],
                        position_attr="lp", dx=34.0, dy=0.0,
                    )
                )
                if not _water_ndvi_label_adjusted:
                    warnings.warn(
                        "未能定位 water -> NDVI 的系数标注"
                    )
            if _use_local_water_road_curve:
                _positioned_dot, _water_road_adjusted = (
                    _set_positioned_edge_quadratic_spline(
                        _positioned_dot,
                        node_ids["water"], node_ids["Road_Dens"],
                        control_x_offset=160.0,
                        control_y_offset=35.0,
                    )
                )
                if not _water_road_adjusted:
                    warnings.warn(
                        "未能定位 water -> Road_Dens 的 Graphviz 样条"
                    )
                _positioned_dot, _water_road_label_adjusted = (
                    _offset_positioned_edge_label(
                        _positioned_dot,
                        node_ids["water"], node_ids["Road_Dens"],
                        position_attr="head_lp", dx=7.0, dy=171.0,
                    )
                )
                if not _water_road_label_adjusted:
                    warnings.warn(
                        "未能定位 water -> Road_Dens 的系数标注"
                    )
            if _use_local_pop_far_curve:
                _positioned_dot, _pop_far_adjusted = (
                    _set_positioned_edge_cubic_spline(
                        _positioned_dot,
                        node_ids["Pop2025"], node_ids["FAR"],
                        control_1_dx=-34.85667,
                        control_1_dy=-121.56333,
                        control_2_dx=-45.14333,
                        control_2_dy=121.56333,
                        terminal_tangent_to_target_center=True,
                    )
                )
                if not _pop_far_adjusted:
                    warnings.warn(
                        "未能定位 Pop2025 -> FAR 的 Graphviz 样条"
                    )
                _positioned_dot, _pop_far_label_adjusted = (
                    _offset_positioned_edge_label(
                        _positioned_dot,
                        node_ids["Pop2025"], node_ids["FAR"],
                        position_attr="lp", dx=45.0, dy=0.0,
                    )
                )
                if not _pop_far_label_adjusted:
                    warnings.warn(
                        "未能定位 Pop2025 -> FAR 的系数标注"
                    )
            if _use_local_pop_road_curve:
                _positioned_dot, _pop_road_adjusted = (
                    _set_positioned_edge_quadratic_spline(
                        _positioned_dot,
                        node_ids["Pop2025"], node_ids["Road_Dens"],
                        control_x_offset=-80.0,
                        control_y_offset=0.0,
                        terminal_tangent_to_target_center=True,
                    )
                )
                if not _pop_road_adjusted:
                    warnings.warn(
                        "未能定位 Pop2025 -> Road_Dens 的 Graphviz 样条"
                    )
            if _use_local_pop_uhii_curve:
                _positioned_dot, _pop_uhii_adjusted = (
                    _set_positioned_edge_rounded_spline(
                        _positioned_dot,
                        node_ids["Pop2025"], node_ids["UHII"],
                        inset_dx=-49.5,
                        top_round=92.0,
                        bottom_round=145.0,
                    )
                )
                if not _pop_uhii_adjusted:
                    warnings.warn(
                        "未能定位 Pop2025 -> UHII 的 Graphviz 样条"
                    )
            if _use_local_water_far_terminal:
                _positioned_dot, _water_far_curve_adjusted = (
                    _set_positioned_edge_rounded_spline(
                        _positioned_dot,
                        node_ids["water"], node_ids["FAR"],
                        inset_dx=-277.54,
                        top_round=160.0,
                        bottom_round=80.0,
                        top_lift=12.0,
                    )
                )
                if not _water_far_curve_adjusted:
                    warnings.warn(
                        "未能定位 water -> FAR 的 Graphviz 样条"
                    )
                _positioned_dot, _water_far_terminal_adjusted = (
                    _offset_positioned_edge_terminal(
                        _positioned_dot,
                        node_ids["water"], node_ids["FAR"],
                        arrow_dx=-1.45,
                        arrow_dy=-5.49,
                        end_dx=-1.46,
                        end_dy=-7.07,
                        horizontal_terminal=True,
                    )
                )
                if not _water_far_terminal_adjusted:
                    warnings.warn(
                        "未能定位 water -> FAR 的箭头末端"
                    )
                _positioned_dot, _water_far_label_adjusted = (
                    _offset_positioned_edge_label(
                        _positioned_dot,
                        node_ids["water"], node_ids["FAR"],
                        position_attr="lp", dx=60.0, dy=0.0,
                    )
                )
                if not _water_far_label_adjusted:
                    warnings.warn(
                        "未能定位 water -> FAR 的系数标注"
                    )
            if _use_local_water_uhii_curve:
                _positioned_dot, _water_uhii_adjusted = (
                    _offset_positioned_edge_outer_channel(
                        _positioned_dot,
                        node_ids["water"], node_ids["UHII"],
                        dx=-18.0,
                        side="right",
                    )
                )
                if not _water_uhii_adjusted:
                    warnings.warn(
                        "未能定位 water -> UHII 的外侧通道"
                    )
            if _use_primary_water_mbh_curve:
                _positioned_dot, _primary_water_mbh_adjusted = (
                    _set_positioned_edge_rounded_spline(
                        _positioned_dot,
                        node_ids["water"], node_ids["MBH"],
                        inset_dx=-145.0,
                        top_round=104.0,
                        bottom_round=104.0,
                        balanced_rounds=True,
                    )
                )
                if not _primary_water_mbh_adjusted:
                    warnings.warn(
                        "未能定位 water -> MBH 的内收等圆角样条"
                    )
                _positioned_dot, _primary_water_mbh_label_adjusted = (
                    _offset_positioned_edge_label(
                        _positioned_dot,
                        node_ids["water"], node_ids["MBH"],
                        position_attr="lp", dx=54.0, dy=0.0,
                    )
                )
                if not _primary_water_mbh_label_adjusted:
                    warnings.warn(
                        "未能定位 water -> MBH 的系数标注"
                    )
            if _use_primary_water_ndvi_curve:
                _positioned_dot, _primary_water_ndvi_adjusted = (
                    _set_positioned_edge_rounded_spline(
                        _positioned_dot,
                        node_ids["water"], node_ids["NDVI"],
                        inset_dx=212.0,
                        top_round=104.0,
                        bottom_round=34.0,
                        balanced_rounds=True,
                        start_dx=-9.5,
                        start_dy=-8.0,
                        terminal_anchor_side="right",
                    )
                )
                if not _primary_water_ndvi_adjusted:
                    warnings.warn(
                        "未能定位 water -> NDVI 的左移等圆角样条"
                    )
                _positioned_dot, _water_ndvi_top_three_point_adjusted = (
                    _set_positioned_edge_first_segment_three_point(
                        _positioned_dot,
                        node_ids["water"], node_ids["NDVI"],
                        terminal_vertical=True,
                    )
                )
                if not _water_ndvi_top_three_point_adjusted:
                    warnings.warn(
                        "未能将 water -> NDVI 上段改为三点中点样条"
                    )
                _positioned_dot, _primary_water_ndvi_label_adjusted = (
                    _offset_positioned_edge_label(
                        _positioned_dot,
                        node_ids["water"], node_ids["NDVI"],
                        position_attr="lp", dx=-90.0, dy=0.0,
                    )
                )
                if not _primary_water_ndvi_label_adjusted:
                    warnings.warn(
                        "未能定位 water -> NDVI 的系数标注"
                    )
            if _use_primary_water_uhii_equal_corner_curve:
                _uhii_reference_anchor_y = _get_positioned_edge_arrow_y(
                    _positioned_dot,
                    node_ids["Pop2025"], node_ids["UHII"],
                )
                if _uhii_reference_anchor_y is None:
                    warnings.warn(
                        "未能读取 Pop2025 -> UHII 的参考锚点高度"
                    )
                _positioned_dot, _primary_water_uhii_equal_adjusted = (
                    _set_positioned_edge_equal_corner_spline(
                        _positioned_dot,
                        node_ids["water"], node_ids["UHII"],
                        channel_dx=248.0,
                        corner_radius=84.0,
                        top_entry_y_offset=0.0,
                        terminal_y=_uhii_reference_anchor_y,
                    )
                )
                if not _primary_water_uhii_equal_adjusted:
                    warnings.warn(
                        "未能定位 water -> UHII 的上下等圆角样条"
                    )
                _positioned_dot, _primary_water_uhii_label_adjusted = (
                    _offset_positioned_edge_label(
                        _positioned_dot,
                        node_ids["water"], node_ids["UHII"],
                        position_attr="lp", dx=-102.0, dy=0.0,
                    )
                )
                if not _primary_water_uhii_label_adjusted:
                    warnings.warn(
                        "未能定位 water -> UHII 的系数标注"
                    )
            if _use_primary_pop_mbh_curve:
                _positioned_dot, _pop_mbh_adjusted = (
                    _set_positioned_edge_quadratic_spline(
                        _positioned_dot,
                        node_ids["Pop2025"], node_ids["MBH"],
                        control_x_offset=-115.0,
                        control_y_offset=0.0,
                        start_dx=-18.0,
                        start_dy=3.0,
                        terminal_tangent_to_target_center=True,
                    )
                )
                if not _pop_mbh_adjusted:
                    warnings.warn(
                        "未能定位 Pop2025 -> MBH 的三点样条"
                    )
                _positioned_dot, _pop_mbh_label_adjusted = (
                    _offset_positioned_edge_label(
                        _positioned_dot,
                        node_ids["Pop2025"], node_ids["MBH"],
                        position_attr="lp", dx=-30.0, dy=2.0,
                    )
                )
                if not _pop_mbh_label_adjusted:
                    warnings.warn(
                        "未能定位 Pop2025 -> MBH 的系数标注"
                    )
            if _use_primary_pop_uhii_curve:
                _positioned_dot, _primary_pop_uhii_adjusted = (
                    _set_positioned_edge_rounded_spline(
                        _positioned_dot,
                        node_ids["Pop2025"], node_ids["UHII"],
                        inset_dx=-96.0,
                        top_round=104.0,
                        bottom_round=104.0,
                    )
                )
                if not _primary_pop_uhii_adjusted:
                    warnings.warn(
                        "未能定位 Pop2025 -> UHII 的内收圆角样条"
                    )
                _positioned_dot, _primary_pop_uhii_label_adjusted = (
                    _offset_positioned_edge_label(
                        _positioned_dot,
                        node_ids["Pop2025"], node_ids["UHII"],
                        position_attr="lp", dx=24.0, dy=0.0,
                    )
                )
                if not _primary_pop_uhii_label_adjusted:
                    warnings.warn(
                        "未能定位 Pop2025 -> UHII 的系数标注"
                    )

            if _use_primary_road_node_shift:
                (_positioned_dot, _road_node_shifted,
                 _road_incident_edge_count) = (
                    _offset_positioned_node_with_incident_edges(
                        _positioned_dot,
                        node_ids["Road_Dens"],
                        dx=-18.0,
                        dy=0.0,
                        endpoint_control_count=3,
                    )
                )
                if not _road_node_shifted:
                    warnings.warn("未能将 Road_Dens 节点向左移动")
                elif _road_incident_edge_count < 5:
                    warnings.warn(
                        "Road_Dens 节点已移动，但部分相连曲线未同步调整"
                    )
                _positioned_dot, _road_ndvi_three_point_adjusted = (
                    _set_positioned_edge_quadratic_spline(
                        _positioned_dot,
                        node_ids["Road_Dens"], node_ids["NDVI"],
                        control_x_offset=16.0,
                        control_y_offset=0.0,
                        terminal_tangent_to_target_center=True,
                    )
                )
                if not _road_ndvi_three_point_adjusted:
                    warnings.warn(
                        "未能将 Road_Dens -> NDVI 改为三点样条"
                    )

            if _use_primary_pop_ndvi_inset:
                _positioned_dot, _pop_ndvi_inset_adjusted = (
                    _offset_positioned_edge_interior(
                        _positioned_dot,
                        node_ids["Pop2025"], node_ids["NDVI"],
                        dx=-18.0,
                        dy=0.0,
                    )
                )
                if not _pop_ndvi_inset_adjusted:
                    warnings.warn("未能将 Pop2025 -> NDVI 曲线向左内收")

            _large_label_offsets = (
                ("water", "COHESION", 12.0, 0.0),
                ("Pop2025", "COHESION", 10.0, 0.0),
                ("Pop2025", "Road_Dens", -42.0, 0.0),
                ("COHESION", "NDVI", -11.5, 22.0),
                ("COHESION", "Road_Dens", 5.0, 0.0),
                ("NDVI", "Road_Dens", -31.0, 5.0),
                ("NDVI", "UHII", 14.0, 0.0),
                ("Road_Dens", "UHII", 14.0, 0.0),
                ("BC", "MBH", -18.0, -7.0),
                ("BC", "NDVI", -18.0, -6.0),
                ("Road_Dens", "NDVI", 18.0, 8.0),
                ("COHESION", "FAR", 12.0, 0.0),
            )
            for (_source_label, _target_label,
                    _label_dx, _label_dy) in _large_label_offsets:
                if (_source_label not in node_ids
                        or _target_label not in node_ids
                        or abs(adj[
                            labels.index(_target_label),
                            labels.index(_source_label),
                        ]) <= edge_threshold):
                    continue
                _positioned_dot, _large_label_adjusted = (
                    _offset_positioned_edge_label(
                        _positioned_dot,
                        node_ids[_source_label], node_ids[_target_label],
                        position_attr="lp",
                        dx=_label_dx, dy=_label_dy,
                    )
                )
                if not _large_label_adjusted:
                    warnings.warn(
                        f"未能定位 {_source_label} -> {_target_label} 的系数标注"
                    )

            # Snap every coefficient back onto its final rendered spline.
            # Pop2025 -> NDVI is placed earlier along its own curve so its
            # label does not mask the routed Water -> NDVI spline.
            _preferred_label_fractions = {}
            if "Pop2025" in node_ids and "NDVI" in node_ids:
                _preferred_label_fractions[
                    (node_ids["Pop2025"], node_ids["NDVI"])
                ] = 0.30
            if "water" in node_ids and "BC" in node_ids:
                _preferred_label_fractions[
                    (node_ids["water"], node_ids["BC"])
                ] = 0.45
            if "BC" in node_ids and "MBH" in node_ids:
                _preferred_label_fractions[
                    (node_ids["BC"], node_ids["MBH"])
                ] = 0.35
            _positioned_dot, _snapped_label_count = (
                _snap_positioned_edge_labels_to_splines(
                    _positioned_dot,
                    preferred_fractions=_preferred_label_fractions,
                )
            )
            if _snapped_label_count == 0:
                warnings.warn("未能将 DAG 系数标注吸附到对应曲线上")

            _positioned_dot, _font_sizes_adjusted = (
                _set_positioned_graph_font_sizes(
                    _positioned_dot,
                    node_fontsize=12.0,
                    edge_fontsize=11.0,
                    node_fontname="Times New Roman, Liberation Serif",
                    edge_fontname="Arial",
                )
            )
            if not _font_sizes_adjusted:
                warnings.warn("未能放大冻结布局中的 DAG 字号；保留 Graphviz 默认字号")

            _positioned_dot_path = f"{output_base}_positioned.dot"
            with open(_positioned_dot_path, "w", encoding="utf-8") as _file:
                _file.write(_positioned_dot)

            _positioned_source = graphviz.Source(
                _positioned_dot, engine="neato",
            )
            svg_path = f"{output_base}.svg"
            pdf_path = f"{output_base}.pdf"
            png_path = f"{output_base}.png"
            with open(svg_path, "wb") as _file:
                _file.write(_positioned_source.pipe(
                    format="svg", neato_no_op=2,
                ))
            with open(pdf_path, "wb") as _file:
                _file.write(_positioned_source.pipe(
                    format="pdf", neato_no_op=2,
                ))
            _png_positioned_dot = re.sub(
                r'\bdpi\s*=\s*"?[0-9.]+"?', 'dpi=300',
                _positioned_dot, count=1,
            )
            with open(png_path, "wb") as _file:
                _file.write(graphviz.Source(
                    _png_positioned_dot, engine="neato",
                ).pipe(format="png", neato_no_op=2))
        else:
            svg_path = graph.render(
                filename=output_base, format="svg", cleanup=True,
            )
            pdf_path = graph.render(
                filename=output_base, format="pdf", cleanup=True,
            )
            graph.attr("graph", dpi="300")
            png_path = graph.render(
                filename=output_base, format="png", cleanup=True,
            )
    except ExecutableNotFound as exc:
        raise RuntimeError(
            "已安装 Python graphviz 包，但找不到 Graphviz 的 dot 可执行文件。"
            "Ubuntu 请运行: sudo apt-get install -y graphviz；"
            "Windows 请安装 Graphviz 并将其 bin 目录加入 PATH。"
        ) from exc

    return {
        "dot": dot_path, "causal_dot": causal_dot_path,
        "svg": svg_path, "pdf": pdf_path, "png": png_path,
    }


def _sample_complete_groups_to_row_cap(
        frame, row_cap, group_col='Grid_ID', random_state=42):
    """Subsample whole groups without replacement while respecting a row cap."""
    if len(frame) <= row_cap:
        return frame.copy()
    if group_col not in frame.columns:
        return frame.sample(n=row_cap, random_state=random_state).copy()

    group_sizes = frame.groupby(group_col, sort=False, observed=True).size()
    group_order = group_sizes.index.to_numpy(copy=True)
    rng = np.random.RandomState(random_state)
    rng.shuffle(group_order)
    ordered_sizes = group_sizes.reindex(group_order).to_numpy()
    cumulative_rows = np.cumsum(ordered_sizes)
    n_groups = int(np.searchsorted(cumulative_rows, row_cap, side='right'))
    n_groups = max(1, n_groups)
    selected_groups = group_order[:n_groups]
    sampled = frame.loc[frame[group_col].isin(selected_groups)].copy()

    # A whole-group sample may be slightly below row_cap; it must never split a grid.
    source_counts = group_sizes.reindex(selected_groups).sort_index()
    sampled_counts = sampled.groupby(
        group_col, sort=False, observed=True).size().reindex(selected_groups).sort_index()
    if not source_counts.equals(sampled_counts):
        raise RuntimeError(f"{group_col} 整组抽样破坏了块完整性")
    return sampled


_CAUSAL_UNIT_COL = 'Causal_Unit_ID'


def _build_grid_level_causal_frame(
        frame, value_cols, output_dir, dataset_tag,
        require_balanced_coverage=True):
    """Aggregate repeated hourly observations to one row per physical grid."""
    value_cols = list(dict.fromkeys(value_cols))
    required = ['Grid_ID'] + value_cols
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"网格级因果聚合缺少列: {missing}")
    if frame.empty:
        raise ValueError("网格级因果聚合收到空数据")
    if frame[required].isna().any().any():
        raise ValueError("网格级因果聚合前仍存在缺失值")

    # Grid_ID should identify a physical grid globally. If it is reused by
    # different cities, use the composite city-grid key while retaining both
    # source columns in the exported data.
    group_cols = ['Grid_ID']
    if 'City_Name' in frame.columns:
        city_counts = frame.groupby(
            'Grid_ID', observed=True, sort=False)['City_Name'].nunique(dropna=False)
        if bool((city_counts > 1).any()):
            group_cols = ['City_Name', 'Grid_ID']
            print("  [网格单位] Grid_ID 跨城市重复，自动使用 City_Name + Grid_ID")

    grouped = frame.groupby(group_cols, observed=True, sort=False)
    coverage = grouped.size().rename('n_rows').reset_index()
    coverage['coverage_fraction_of_max'] = (
        coverage['n_rows'] / float(coverage['n_rows'].max()))
    if 'Season' in frame.columns:
        season_counts = grouped['Season'].nunique(dropna=False).rename('n_seasons')
        coverage = coverage.merge(
            season_counts.reset_index(), on=group_cols, how='left')
    if 'Hour_cos' in frame.columns:
        hour_level_counts = grouped['Hour_cos'].nunique(
            dropna=False).rename('n_hour_cos_levels')
        coverage = coverage.merge(
            hour_level_counts.reset_index(), on=group_cols, how='left')

    # Compute mean, min and max in one grouped aggregation. The mean is the
    # analysis value; min/max diagnose whether nominally static treatments
    # actually vary within a grid over the study period.
    grouped_stats = grouped[value_cols].agg(['mean', 'min', 'max'])
    grid_values = grouped_stats.xs('mean', axis=1, level=1).reset_index()
    within_ranges = (
        grouped_stats.xs('max', axis=1, level=1)
        - grouped_stats.xs('min', axis=1, level=1)
    )

    diagnostics = []
    for col in value_cols:
        ranges = within_ranges[col].to_numpy(dtype=float)
        scale = max(float(np.nanmax(np.abs(grid_values[col]))), 1.0)
        tolerance = np.finfo(float).eps * scale * 100.0
        diagnostics.append({
            'variable': col,
            'aggregation': 'arithmetic mean over complete study-period rows',
            'n_grid_units': int(len(grid_values)),
            'grids_with_within_grid_variation': int(
                np.count_nonzero(ranges > tolerance)),
            'median_within_grid_range': float(np.nanmedian(ranges)),
            'max_within_grid_range': float(np.nanmax(ranges)),
            'between_grid_std_after_aggregation': float(
                grid_values[col].std(ddof=1)),
        })
    diagnostics = pd.DataFrame(diagnostics)

    grid_level = grid_values.copy()
    grid_level.insert(
        0, _CAUSAL_UNIT_COL,
        np.arange(len(grid_level), dtype=np.int64),
    )
    if group_cols == ['Grid_ID'] and 'City_Name' in frame.columns:
        city_first = grouped['City_Name'].first().rename('City_Name').reset_index()
        grid_level = grid_level.merge(city_first, on='Grid_ID', how='left')
    for coordinate in ('longitude', 'latitude'):
        if coordinate in frame.columns and coordinate not in grid_level.columns:
            coordinate_mean = grouped[coordinate].mean().rename(coordinate).reset_index()
            grid_level = grid_level.merge(
                coordinate_mean, on=group_cols, how='left')

    os.makedirs(output_dir, exist_ok=True)
    grid_path = os.path.join(
        output_dir, f'{dataset_tag}_grid_level_causal_data.csv')
    coverage_path = os.path.join(
        output_dir, f'{dataset_tag}_grid_coverage.csv')
    diagnostics_path = os.path.join(
        output_dir, f'{dataset_tag}_grid_aggregation_diagnostics.csv')
    grid_level.to_csv(grid_path, index=False, encoding='utf-8-sig')
    coverage.to_csv(coverage_path, index=False, encoding='utf-8-sig')
    diagnostics.to_csv(diagnostics_path, index=False, encoding='utf-8-sig')

    row_count_levels = np.sort(coverage['n_rows'].unique())
    coverage_errors = []
    if row_count_levels.size != 1:
        coverage_errors.append(
            f"观测行数层级={row_count_levels.tolist()}")
    if 'n_seasons' in coverage.columns:
        expected_seasons = int(frame['Season'].nunique(dropna=False))
        if bool((coverage['n_seasons'] != expected_seasons).any()):
            coverage_errors.append(
                f"部分网格未覆盖全部 {expected_seasons} 个季节层级")
    if 'n_hour_cos_levels' in coverage.columns:
        expected_hour_levels = int(frame['Hour_cos'].nunique(dropna=False))
        if bool((coverage['n_hour_cos_levels'] != expected_hour_levels).any()):
            coverage_errors.append(
                f"部分网格未覆盖全部 {expected_hour_levels} 个 Hour_cos 层级")
    if require_balanced_coverage and coverage_errors:
        raise RuntimeError(
            "各网格时间覆盖不一致，已停止主因果分析；请检查 "
            f"{coverage_path}。" + "；".join(coverage_errors)
        )
    if coverage_errors:
        print(f"  [网格覆盖] ⚠️ " + "；".join(coverage_errors))
    else:
        print(f"  [网格覆盖] 每个分析单位均为 {int(row_count_levels[0]):,} 行，"
              "季节/Hour_cos 层级完整")
    print(f"  [网格聚合] {len(frame):,} 行 → {len(grid_level):,} 个物理网格")
    print(f"  [网格聚合] 数据: {grid_path}")
    print(f"  [网格聚合] 覆盖诊断: {coverage_path}")
    print(f"  [网格聚合] 变量诊断: {diagnostics_path}")
    return grid_level, coverage, diagnostics


warnings.filterwarnings("ignore")

# === 唯一正式因果配置：MBH + BC；FAR/COHESION 不进入因果分析 ===
CAUSAL_ANALYSIS_VARIANT = "BC_MBH_noCOHESION_20260820"
CAUSAL_DIR = os.path.join(
    BASE_OUTPUT_DIR, f"5_Causal_Inference_{CAUSAL_ANALYSIS_VARIANT}"
)
CAUSAL_PLOTS_DIR = os.path.join(CAUSAL_DIR, "Plots")
CAUSAL_DATA_DIR = os.path.join(CAUSAL_DIR, "Data")
for _d in [CAUSAL_DIR, CAUSAL_PLOTS_DIR, CAUSAL_DATA_DIR]:
    os.makedirs(_d, exist_ok=True)

# The causal node set is pre-specified and is not selected by SHAP winners.
# FAR remains a raw-data lineage field only; COHESION remains an ML predictor
# only. Neither is allowed into the formal causal graph or DoWhy treatments.
CAUSAL_REPRESENTATIVE_EXCLUDE = {}
_CAUSAL_CORE_FEATURES = [
    'MBH', 'BC', 'NDVI', 'water', 'Pop2025', 'Road_Dens',
]
causal_features = list(_CAUSAL_CORE_FEATURES)
_missing_core_features = [f for f in causal_features if f not in df_clean.columns]
if _missing_core_features:
    raise ValueError(f"主因果变量缺失: {_missing_core_features}")
if 'Temp_2m' not in feature_cols:
    raise RuntimeError("正式配置要求 Temp_2m 保留在机器学习 feature_cols 中")
if 'Temp_2m' in causal_features:
    raise RuntimeError("Temp_2m 不得进入因果候选池或主因果变量集")
if 'FAR' in causal_features:
    raise RuntimeError("FAR 不得进入正式 causal_features")
if 'COHESION' in causal_features:
    raise RuntimeError("COHESION 不得进入正式 causal_features")
if 'Pop2025' not in causal_features:
    raise RuntimeError("v20260820 正式 causal_features 必须保留 Pop2025")
if 'MBH' not in causal_features or 'BC' not in causal_features:
    raise RuntimeError("正式 causal_features 必须同时包含 MBH 与 BC")
if causal_features != _CAUSAL_CORE_FEATURES or len(causal_features) != 6:
    raise RuntimeError(
        f"唯一正式因果模型必须严格为 {_CAUSAL_CORE_FEATURES}，实际={causal_features}"
    )

print(f"因果推断输出目录: {CAUSAL_DIR}")
print("\n===== 🎯 唯一正式因果模型（预设节点，不做 SHAP winner 筛选）=====")
print(f"进入 DirectLiNGAM/DoWhy 的 6 个处理变量: {causal_features}")
print(f"最终 7 个 DAG 节点: {causal_features + [_TARGET_LABEL]}")
print("FAR: raw-data lineage QA only；COHESION: ML predictor only")
print(f"目标变量: {target_col}")

# === 迭代所有套件 ===
causal_suites = list(all_suite_artifacts.keys())
all_suite_causal_artifacts = {}

for suite_key in causal_suites:
    suite_name = all_suite_artifacts[suite_key]['name']

    print(f"\n{'='*60}")
    print(f"=== 🧬 Causal Discovery [{suite_name}] ===")
    print(f"{'='*60}")

    # --- 1. 从当前 BC-noFAR df_clean 新建网格级数据 ---
    # 旧 FAR/no_MBH 因果缓存被明确禁用；每次正式新版本都从当前清洗数据
    # 重新聚合，以保证 MBH、BC 与 UHII 属于同一数据血缘。
    print(f"  [1/4] 从当前 BC-noFAR df_clean 构建 Grid_ID 级因果分析数据池...")
    dag_features = causal_features + [target_col]
    _grid_value_cols = list(dict.fromkeys(causal_features + [target_col]))
    CAUSAL_REUSE_GRID_LEVEL_CACHE = False
    if DATA_FORMAT != 'long' or 'Grid_ID' not in df_clean.columns:
        raise RuntimeError(
            "Grid_ID 级因果主分析要求 long 数据且 df_clean 保留 Grid_ID"
        )
    if TRAIN_MODE == 'by_season':
        _suite_season_code = {'summer': 1, 'winter': 0}.get(suite_key)
        if _suite_season_code is None:
            raise ValueError(f"未知季节套件: {suite_key}")
        _causal_source = df_clean.loc[
            df_clean['Season_code'] == _suite_season_code]
    else:
        _causal_source = df_clean

    CAUSAL_REQUIRE_BALANCED_GRID_COVERAGE = True
    df_causal_pool, _grid_coverage, _grid_aggregation_diagnostics = \
        _build_grid_level_causal_frame(
            _causal_source,
            value_cols=_grid_value_cols,
            output_dir=CAUSAL_DATA_DIR,
            dataset_tag=suite_key,
            require_balanced_coverage=CAUSAL_REQUIRE_BALANCED_GRID_COVERAGE,
        )
    del _causal_source

    _causal_metadata_cols = [
        col for col in (
            _CAUSAL_UNIT_COL, 'Grid_ID', 'City_Name', 'longitude', 'latitude'
        ) if col in df_causal_pool.columns
    ]
    df_causal = df_causal_pool[_causal_metadata_cols + dag_features].copy()
    _causal_unit_available = _CAUSAL_UNIT_COL in df_causal.columns
    if not _causal_unit_available:
        raise RuntimeError("网格级因果数据缺少 Causal_Unit_ID")

    dag_labels = [c for c in dag_features]   # 最后一列为 UHII
    print(f"  进入 LiNGAM 的特征 ({len(causal_features)} 个): {causal_features}")
    print(f"  LiNGAM 变量 ({len(dag_labels)} 个): {dag_labels}")
    print(f"  因果数据: {len(df_causal):,} 个网格单位 × {len(dag_labels)} 个建模变量")
    print("  分析单位: 一个物理 Grid_ID；UHII 与候选变量均取完整研究期均值")

    gc.collect()

    # --- 5. LiNGAM 因果发现（DirectLiNGAM + 物理先验约束）---
    print("[3/4] 运行 DirectLiNGAM 因果发现（最小先验：仅 UHII=sink）...")

    # ── 变量标准化（z-score）──
    # 消除量纲差异，使 LiNGAM 边系数可比，且稳定性判断不依赖变量量纲。
    # 注意：只对 dag_labels 中的列标准化（排除 Grid_ID 等元数据列）。
    df_causal_lingam = df_causal[dag_labels].copy()
    _causal_means = df_causal_lingam.mean(axis=0).to_numpy()
    # pandas ≥ 2.0 的 .std().values 返回只读视图，原地赋值会触发
    # ValueError: assignment destination is read-only → 必须 .copy() 再写
    _causal_stds = df_causal_lingam.std(axis=0).to_numpy().copy()
    _causal_stds[_causal_stds == 0] = 1.0   # 防止除零
    df_causal_std = (df_causal_lingam - _causal_means) / _causal_stds
    lingam_data = df_causal_std.to_numpy(dtype=np.float64)
    n_vars = lingam_data.shape[1]
    temp_idx = dag_labels.index(_TARGET_LABEL)

    # ── 构造先验知识矩阵 ──
    # causal-learn 约定（已从源码 _extract_partial_orders 的 [to, from] 注释确认）：
    #   prior[i, j] 中 j 是源(cause)、i 是靶(effect)，即 prior[i,j] 控制 j→i 这条路径
    #   -1 = 未知（算法自由探索），0 = 禁止 j→i，1 = 必须 j→i
    #
    # 先验约束设计原则（参考 Cao et al. 2022, Ind Eng Chem Res）：
    #   Cao 提出「四种过程知识」——forbidden edge / required edge /
    #   input variable(source) / output variable(sink)。
    #   其工业实验显示：同时约束 source 和 sink 时 DirectLiNGAM 的
    #   precision 从 0.21 提升到 0.56，假因果边(FP)从 11 降到 4。
    #
    #   但约束必须基于「物理不可能」，而非「物理可能但偏好某方向」
    #   （Mäkelä et al. 2022, Biogeosciences 警告：过度注入先验会让
    #    DAG 沦为「把假设画出来」，失去数据发现价值）。
    prior_knowledge_matrix = np.full((n_vars, n_vars), -1, dtype=int)
    np.fill_diagonal(prior_knowledge_matrix, 0)           # 禁止自环

    # 【约束1：sink】UHII(CUHI) 是因变量（物理 sink），它不能指向任何变量
    # → UHII 作"源"时全部禁止 → 清 UHII 所在【列】（不是行！）
    prior_knowledge_matrix[:, temp_idx] = 0
    prior_knowledge_matrix[temp_idx, temp_idx] = 0

    # MBH 与 BC 的方向和符号均不预先固定。正式结果由当前网格级数据、
    # DirectLiNGAM 稳定边与图调整效应共同决定，不按预期机制调阈值。

    print(f"  [先验] UHII 钉为 sink（清列 prior[:, temp_idx]=0）")
    print(f"  [先验] 其余方向全部留给数据发现（-1=未知，无硬分层）")

    # DirectLiNGAM：绕过 ICALiNGAM 的 ICA 置换搜索，逐对判向更稳健
    model_lingam = _lingam_ca.DirectLiNGAM(prior_knowledge=prior_knowledge_matrix)
    model_lingam.fit(lingam_data)
    adj_matrix = model_lingam.adjacency_matrix_

    # 验证约束生效：邻接矩阵 B[i,j]=j→i
    #   UHII 无出边 = B[:, temp_idx] 全 0（sink 约束）
    temp_out_sum = float(np.sum(np.abs(adj_matrix[:, temp_idx])))
    print(f"  [验证] UHII 出边绝对值之和（应≈0）: {temp_out_sum:.6f}")

    del lingam_data
    gc.collect()
    print("  LiNGAM 邻接矩阵:")
    adj_df = pd.DataFrame(
        np.round(adj_matrix, 4), index=dag_labels, columns=dag_labels)
    print(adj_df)

    temp_idx = dag_labels.index(_TARGET_LABEL)


    # --- 6. DAG 边的稳定性选择（网格单位 Bootstrap）---
    # 两条预设硬规则：
    #   ① 单次 Bootstrap 内，仅 |beta_std| >= 0.05 计为有向边被选择；
    #   ② 在有效 Bootstrap 中，有向选择频率 >= 0.70 才进入共识 DAG。
    # 邻接频率、条件方向概率、符号一致率和条件系数区间仅作诊断，
    # 不构成额外删边门槛。0.05 与 0.70 是本研究预设的稀疏化/稳定性规则，
    # 不解释为显著性水平、后验概率或 FDR 控制。
    # Formal submission run: 500 grid-unit resamples.  This supersedes the
    # exploratory B=100 run without overwriting its outputs.
    _STAB_N_BOOTSTRAP = 500
    _STAB_EFFECT_THRESHOLD = _EDGE_EFFECT_THRESHOLD
    _STAB_SELECTION_THRESHOLD = _EDGE_SELECTION_THRESHOLD

    # 准备标准化 numpy 数组（变量已在上文 z-score 标准化）
    _causal_np = df_causal_std.to_numpy(dtype=np.float64)
    _n_avail = _causal_np.shape[0]

    # df_causal 已是一行一个物理网格，因此标准 n-out-of-n 行 Bootstrap
    # 就是 Grid_ID 单位 Bootstrap。禁止退化为小时记录逐行抽样。
    if _CAUSAL_UNIT_COL not in df_causal.columns:
        raise RuntimeError("DAG Bootstrap 缺少网格分析单位标识")
    _unit_ids = df_causal[_CAUSAL_UNIT_COL].to_numpy()
    if pd.Series(_unit_ids).duplicated().any():
        raise RuntimeError("网格级因果数据中 Causal_Unit_ID 不唯一")
    _n_unique_units = len(_unit_ids)
    print(f"[稳定性] Grid-unit Bootstrap (N={_STAB_N_BOOTSTRAP}, "
          f"units={_n_unique_units:,}, "
          f"|beta_std|>={_STAB_EFFECT_THRESHOLD:.2f}, "
          f"directed_freq>={_STAB_SELECTION_THRESHOLD:.2f})...")

    _all_adj = []   # 存每次成功的完整邻接矩阵，用于筛选和诊断
    _all_causal_orders = []

    # ── 单次 bootstrap 函数（供 joblib 并行调用）──
    # 方法论定位：Bagged Causal Discovery with Grid-level Bootstrap
    #
    # 框架来源（两条独立但互补的文献线索）：
    #   ① Block Bootstrap for dependent data:
    #      Lahiri (2003) Resampling Methods for Dependent Data, Springer.
    #      → 小时记录已先聚合到 Grid_ID；每次有放回抽取 N 个物理网格。
    #      这处理分析单位的重抽样不确定性，但不等于控制相邻网格的空间
    #      自相关；空间块敏感性需另行基于连续地理分区实现。
    #
    #   ② Bootstrap aggregation (bagging) for causal discovery:
    #      Debeire et al. (2024) "Bootstrap aggregation and confidence
    #      measures to improve time series causal discovery", CLeaR/PMLR v236.
    #      → 对因果发现算法做 bootstrap 重采样，用频率投票判定边的稳定性。
    #
    #   ③ 聚合依据：Debeire et al. (2024) 与 Tarraga et al. (2024)
    #      支持对每轮独立学习的图进行 edge-level bootstrap aggregation。
    #      原文 PCMCI 的 p<0.05 不等同于本研究的 beta 阈值；这里的 0.05
    #      仅是预设的标准化系数稀疏化阈值，并通过 0.03/0.05/0.10 检验敏感性。
    #
    #   ④ 正式投稿版本使用 B=500；旧 B=100 结果仅作版本比较，不覆盖。
    def _run_single_bootstrap(_b):
        _rng_b = np.random.RandomState(42 + _b)
        _idx = _rng_b.randint(0, _n_unique_units, size=_n_unique_units)
        _boot_data = _causal_np[_idx]
        if _boot_data.shape[0] < n_vars + 1:
            return None, None, "insufficient bootstrap rows"
        try:
            _m = _lingam_ca.DirectLiNGAM(prior_knowledge=prior_knowledge_matrix)
            _m.fit(_boot_data)
            return _m.adjacency_matrix_.copy(), list(_m.causal_order_), None
        except Exception as _bootstrap_error:
            return None, None, f"{type(_bootstrap_error).__name__}: {_bootstrap_error}"[:500]

    # ── 多进程并行执行 bootstrap ──
    # 用 loky（多进程）后端：DirectLiNGAM 有大量纯 Python for 循环
    # （逐变量因果排序 + 互信息计算），不释放 GIL → threading 会退化成串行。
    # loky 每个子进程独立跑一个 DirectLiNGAM.fit()，真正并行。
    # 网格级矩阵通常仅数万行，进程间传输和单次拟合成本远低于小时长表。
    from joblib import Parallel, delayed
    import multiprocessing as _mp_boot
    _n_cpus = _mp_boot.cpu_count()
    _N_WORKERS = min(16, max(2, _n_cpus - 2))
    _boot_n_per = _n_unique_units
    print(f"  [并行] 启动 {_N_WORKERS} 进程 × {_STAB_N_BOOTSTRAP} bootstrap "
          f"(Grid_ID units = {_boot_n_per:,} 行/次)...")
    _adj_results = Parallel(n_jobs=_N_WORKERS, backend='loky', verbose=0)(
        delayed(_run_single_bootstrap)(_b) for _b in range(_STAB_N_BOOTSTRAP)
    )

    # ── 聚合结果 ──
    _success_count = 0
    _dag_bootstrap_audit = []
    for _bootstrap_id, _boot_result in enumerate(_adj_results, start=1):
        _adj_b, _order_b, _bootstrap_error = _boot_result
        if _adj_b is None:
            _dag_bootstrap_audit.append({
                'Suite': suite_name,
                'Bootstrap ID': _bootstrap_id,
                'Status': 'failed',
                'Error': _bootstrap_error,
            })
            continue
        _all_adj.append(_adj_b)
        _all_causal_orders.append(_order_b)
        _success_count += 1
        _dag_bootstrap_audit.append({
            'Suite': suite_name,
            'Bootstrap ID': _bootstrap_id,
            'Status': 'ok',
            'Error': None,
        })
    del _adj_results
    gc.collect()

    if _success_count == 0:
        raise RuntimeError("所有 bootstrap 均失败，无法计算边稳定性")
    print(f"  [稳定性] 成功 {_success_count}/{_STAB_N_BOOTSTRAP} 次")
    _dag_bootstrap_audit_path = os.path.join(
        CAUSAL_DATA_DIR, f'{suite_key}_DAG_Bootstrap_Audit.csv')
    pd.DataFrame(_dag_bootstrap_audit).to_csv(
        _dag_bootstrap_audit_path, encoding='utf-8-sig', index=False)
    print(f"  [审计] DAG Bootstrap 成功/失败明细: {_dag_bootstrap_audit_path}")

    _adj_stack = np.asarray(_all_adj, dtype=float)
    _edge_stats = _bootstrap_edge_statistics(
        _adj_stack, effect_threshold=_STAB_EFFECT_THRESHOLD)
    _edge_count = _edge_stats['directed_count']
    _stability = _edge_stats['directed_frequency']
    _adjacency_frequency = _edge_stats['adjacency_frequency']
    _direction_given_adjacency = _edge_stats['direction_given_adjacency']
    _sign_consistency = _edge_stats['sign_consistency']
    _stable_mean_coef = _edge_stats['mean_coefficient']
    _coef_p025 = _edge_stats['coef_p025']
    _coef_p500 = _edge_stats['coef_p500']
    _coef_p975 = _edge_stats['coef_p975']

    # 因果顺序稳定性：P(变量 i 在变量 j 之前)。
    _order_before_count = np.zeros((n_vars, n_vars), dtype=int)
    for _order_b in _all_causal_orders:
        if len(_order_b) != n_vars:
            continue
        _order_position = np.empty(n_vars, dtype=int)
        _order_position[np.asarray(_order_b, dtype=int)] = np.arange(n_vars)
        _order_before_count += (
            _order_position[:, None] < _order_position[None, :]).astype(int)
    _order_stability = _order_before_count / _success_count

    # 系数分位区间仅基于满足 |beta_std| >= 0.05 的重复，属于选择条件下的
    # 描述性稳定区间，不是未经选择偏差校正的总体参数置信区间。
    _ci_not_cross_zero = (_coef_p025 * _coef_p975) > 0

    # 有向边选择频率达到 0.70 即进入候选图。
    # 符号一致率和条件系数区间只报告不确定性，不参与结构选择。
    # 多个 bootstrap DAG 的阈值边集合仍可能形成 Condorcet 式有向环，
    # 因此按频率从高到低加入边，并记录会闭环而被剔除的候选边。
    (adj_masked, _threshold_mask, _stable_mask,
     _cycle_pruned_mask) = _aggregate_stable_dag(
        _stability, _stable_mean_coef,
        threshold=_STAB_SELECTION_THRESHOLD,
        sink_idx=temp_idx,
    )
    _n_edges = int(np.count_nonzero(adj_masked))
    _n_threshold = int(np.count_nonzero(_threshold_mask))
    _n_cycle_pruned = int(np.count_nonzero(_cycle_pruned_mask))
    _n_total_possible = n_vars * (n_vars - 1)
    print(f"  [稳定性] 有向边频率筛选（freq>={_STAB_SELECTION_THRESHOLD:.2f}）: "
          f"{_n_threshold} 条候选，剔除 {_n_cycle_pruned} 条成环边，"
          f"最终 {_n_edges} 条 / {_n_total_possible} 条可能边")

    # 稳定性报告
    print(f"  [稳定性报告] 各候选边（仅列频率>0.3的边）:")
    _stab_rows = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and _stability[i, j] > 0.3:
                _stab_rows.append((dag_labels[j], dag_labels[i],
                                   _stability[i, j], _sign_consistency[i, j],
                                   _stable_mean_coef[i, j],
                                   _coef_p025[i, j], _coef_p975[i, j]))
    for src, tgt, freq, scons, coef, ci_lo, ci_hi in \
            sorted(_stab_rows, key=lambda x: -x[2]):
        _src_idx = dag_labels.index(src)
        _tgt_idx = dag_labels.index(tgt)
        _passes = bool(_stable_mask[_tgt_idx, _src_idx])
        _pruned = bool(_cycle_pruned_mask[_tgt_idx, _src_idx])
        _tag = " ★频率保留" if _passes else ("  达阈值但成环" if _pruned else "  未达频率阈值")
        print(f"    {src:10s}→{tgt:12s}: freq={freq:.2f}, sign={scons:.2f}, "
              f"coef={coef:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]{_tag}")

    # 保存稳定性矩阵（含频率、符号一致率、系数 CI）
    _stab_data = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                _stab_data.append({
                    'from': dag_labels[j], 'to': dag_labels[i],
                    'frequency': round(_stability[i, j], 4),
                    'directed_selection_frequency': round(_stability[i, j], 4),
                    'reverse_frequency': round(_stability[j, i], 4),
                    'adjacency_frequency': round(_adjacency_frequency[i, j], 4),
                    'direction_given_adjacency': round(
                        _direction_given_adjacency[i, j], 4),
                    'selected_count': int(_edge_count[i, j]),
                    'valid_bootstrap_count': int(_success_count),
                    'from_before_to_frequency': round(_order_stability[j, i], 4),
                    'sign_consistency': round(_sign_consistency[i, j], 4),
                    'mean_coef': round(_stable_mean_coef[i, j], 6),
                    'ci_2.5%': round(_coef_p025[i, j], 6),
                    'ci_median': round(_coef_p500[i, j], 6),
                    'ci_97.5%': round(_coef_p975[i, j], 6),
                    'ci_basis': (
                        f'replicates with |beta_std| >= '
                        f'{_STAB_EFFECT_THRESHOLD:.2f}'),
                    'ci_cross_zero': (
                        None if np.isnan(_coef_p025[i, j])
                        else bool(not _ci_not_cross_zero[i, j])),
                    'aggregation_method': (
                        'within-bootstrap effect sparsification + '
                        'directed selection frequency'),
                    'effect_threshold': _STAB_EFFECT_THRESHOLD,
                    'selection_threshold': _STAB_SELECTION_THRESHOLD,
                    'selected_by_threshold': bool(_threshold_mask[i, j]),
                    'cycle_pruned': bool(_cycle_pruned_mask[i, j]),
                    'stable': bool(_stable_mask[i, j]),
                })
    _stab_df = pd.DataFrame(_stab_data)
    _stab_path = os.path.join(CAUSAL_DATA_DIR, f'{suite_key}_edge_stability.csv')
    _stab_df.to_csv(_stab_path, encoding='utf-8-sig', index=False)
    print(f"  [稳定性] 边稳定性明细已保存: {_stab_path}")

    _order_path = os.path.join(
        CAUSAL_DATA_DIR, f'{suite_key}_causal_order_stability.csv')
    pd.DataFrame(
        np.round(_order_stability, 4), index=dag_labels, columns=dag_labels,
    ).to_csv(_order_path, encoding='utf-8-sig')
    print(f"  [稳定性] 因果顺序稳定性矩阵已保存: {_order_path}")

    # 额外保存有向频率、邻接频率和条件方向概率宽表（论文 Supplementary 用）。
    _stab_wide = pd.DataFrame(np.round(_stability, 3),
                              index=dag_labels, columns=dag_labels)
    _stab_wide_path = os.path.join(CAUSAL_DATA_DIR, f'{suite_key}_edge_stability_matrix.csv')
    _stab_wide.to_csv(_stab_wide_path, encoding='utf-8-sig')
    print(f"  [稳定性] 频率矩阵已保存: {_stab_wide_path}")
    _adjacency_path = os.path.join(
        CAUSAL_DATA_DIR, f'{suite_key}_adjacency_frequency_matrix.csv')
    pd.DataFrame(
        np.round(_adjacency_frequency, 4),
        index=dag_labels, columns=dag_labels,
    ).to_csv(_adjacency_path, encoding='utf-8-sig')
    _direction_path = os.path.join(
        CAUSAL_DATA_DIR, f'{suite_key}_direction_given_adjacency_matrix.csv')
    pd.DataFrame(
        np.round(_direction_given_adjacency, 4),
        index=dag_labels, columns=dag_labels,
    ).to_csv(_direction_path, encoding='utf-8-sig')

    _threshold_sensitivity_path = os.path.join(
        CAUSAL_DATA_DIR, f'{suite_key}_edge_threshold_sensitivity.csv')
    pd.DataFrame(_edge_threshold_sensitivity_rows(
        _adj_stack, dag_labels, temp_idx,
    )).to_csv(_threshold_sensitivity_path, encoding='utf-8-sig', index=False)
    print(f"  [敏感性] 3×3 阈值组合已保存: {_threshold_sensitivity_path}")

    _bootstrap_matrix_path = os.path.join(
        CAUSAL_DATA_DIR, f'{suite_key}_bootstrap_adjacency_matrices.npz')
    np.savez_compressed(
        _bootstrap_matrix_path,
        adjacency_matrices=_adj_stack,
        labels=np.asarray(dag_labels, dtype=str),
        requested_bootstrap_count=_STAB_N_BOOTSTRAP,
        valid_bootstrap_count=_success_count,
        primary_effect_threshold=_STAB_EFFECT_THRESHOLD,
        primary_selection_threshold=_STAB_SELECTION_THRESHOLD,
    )
    print(f"  [复现] Bootstrap 邻接矩阵已保存: {_bootstrap_matrix_path}")
    # 注：此处不存 artifacts（该变量在本 Cell 未定义）；edge_stability/adj_masked
    #     在第8段存储字典里一并写入 all_suite_causal_artifacts。

    dag_img_name = f'{suite_key}_LiNGAM_causal_DAG'

    # DAG 可视化：Python graphviz，统一导出 DOT、矢量 PDF 和高分辨率 PNG。
    _n_dag_edges = int(np.count_nonzero(adj_masked))

    # 邻接矩阵 CSV 继续保留，供结果审计和补充材料使用。
    dag_csv_path = os.path.join(CAUSAL_DATA_DIR, f'{suite_key}_adj_masked.csv')
    _adj_export = pd.DataFrame(adj_masked, index=dag_labels, columns=dag_labels)
    _adj_export.to_csv(dag_csv_path, encoding='utf-8-sig')

    # 只有最终稳定 DAG 中直接指向 UHII 的边才能进入后续 DoWhy Top-3。
    cause_coefs = {}
    for i, feat in enumerate(dag_labels):
        coef = float(adj_masked[temp_idx, i])
        if i != temp_idx and coef != 0.0:
            cause_coefs[feat] = abs(coef)
    top3_treatments = sorted(
        cause_coefs, key=cause_coefs.get, reverse=True)[:3]
    print(f"  稳定 DAG 中直接指向 {_TARGET_LABEL} 的变量 "
          f"({len(cause_coefs)} 个):")
    for feat, coef_abs in sorted(cause_coefs.items(), key=lambda item: -item[1]):
        marker = " ★ Top-3" if feat in top3_treatments else ""
        print(f"    {feat}: |mean beta_std|={coef_abs:.4f}{marker}")

    # Hard guard against accidentally drawing the dense, single-fit LiNGAM
    # adjacency matrix.  Only edges surviving the Bootstrap aggregation are
    # allowed into the publication DAG.
    _plot_edge_mask = np.isfinite(adj_masked) & (np.abs(adj_masked) > 0.0)
    if not np.array_equal(_plot_edge_mask, _stable_mask):
        raise RuntimeError(
            "DAG 绘图矩阵与 Bootstrap 稳定边掩码不一致，已停止绘图，"
            "以避免误画未筛选的完整 LiNGAM 图"
        )
    if np.any(_plot_edge_mask & (_stability < _STAB_SELECTION_THRESHOLD)):
        raise RuntimeError(
            "DAG 绘图矩阵包含未达到 Bootstrap 频率阈值的边，已停止绘图"
        )
    _dag_matrix_for_plot = np.where(_plot_edge_mask, adj_masked, 0.0)

    _dag_output_base = os.path.join(CAUSAL_PLOTS_DIR, dag_img_name)
    try:
        _dag_paths = _render_dag_graphviz(
            _dag_matrix_for_plot, dag_labels, _dag_output_base,
        )
        print(f"  DAG (Python graphviz, {_n_dag_edges}条稳定边, "
              f"|beta_std|>={_STAB_EFFECT_THRESHOLD:.2f}, "
              f"freq>={_STAB_SELECTION_THRESHOLD:.2f}) 已保存:")
        print(f"    DOT: {_dag_paths['dot']}")
        print(f"    DOT (causal only): {_dag_paths['causal_dot']}")
        print(f"    SVG: {_dag_paths['svg']}")
        print(f"    PDF: {_dag_paths['pdf']}")
        print(f"    PNG: {_dag_paths['png']}")
    except Exception as _e_gv:
        print(f"  ⚠️ DAG 绘图失败: {_e_gv}")
        print(f"    邻接矩阵 CSV 已保存，可安装 Graphviz 后单独重新绘图: {dag_csv_path}")

    # --- 8. 存储 ---
    all_suite_causal_artifacts[suite_key] = {
        'suite_name': suite_name,
        'causal_features': causal_features,
        'top3_treatments': top3_treatments,
        'dag_features': dag_features,
        'dag_labels': dag_labels,
        'adj_matrix': adj_matrix,
        'adj_masked': adj_masked,              # bootstrap 稳定性过滤后的邻接矩阵（DoWhy 用）
        'edge_stability': _stability,
        'adjacency_frequency': _adjacency_frequency,
        'direction_given_adjacency': _direction_given_adjacency,
        'causal_order_stability': _order_stability,
        'dag_bootstrap_requested': _STAB_N_BOOTSTRAP,
        'dag_bootstrap_valid': _success_count,
        'df_causal': df_causal,
        'df_causal_pool': df_causal_pool,
        'grid_coverage': _grid_coverage,
        'grid_aggregation_diagnostics': _grid_aggregation_diagnostics,
        'cause_coefs': cause_coefs,
    }

    plt.close('all')
    gc.collect()
    try:
        _ctypes_ca.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

print(f"\n=== Cell CA 完成: 所有套件因果发现已结束 ===")

# %%
# ==========================================
# 因果推断 Part 2: 网格级 DoWhy ATE + Grid-unit Bootstrap CI + 反驳检验
# ==========================================
import ctypes as _ctypes_cb
import json
import dowhy
from dowhy import CausalModel as _CausalModel
from joblib import Parallel, delayed


def _adjacency_to_dot_str(adj_matrix, labels, threshold=0.0):
    # Edge existence has already been decided by bootstrap aggregation.
    # Here only exact zeros are excluded so plotting and DoWhy use the same DAG.
    idx = np.abs(adj_matrix) > threshold
    dirs = np.where(idx)
    lines = ["digraph {"]
    # Declare every node explicitly. Otherwise a treatment with no retained
    # edge disappears from DOT and DoWhy cannot distinguish an isolated node
    # from a misspelled/missing variable.
    for label in labels:
        lines.append(f'  "{label}";')
    for to, from_ in zip(dirs[0], dirs[1]):
        coef = adj_matrix[to, from_]
        lines.append(
            f'  "{labels[from_]}" -> "{labels[to]}" [label="{coef:.3f}"];')
    lines.append("}")
    return "\n".join(lines)


# fork CoW 共享字典：主进程加载一次 DataFrame，子进程通过 Copy-on-Write 零拷贝读取
_SHARED_DOWHY_DATA = {}
_DOWHY_ATE_N_BOOTSTRAP = 1000
_DOWHY_ATE_MIN_VALID_FRACTION = 0.80


def _plot_ate_forest_python(main_results, pdf_path, png_path, suite_name):
    """Plot journal-ready one-treatment-SD effects with grid-bootstrap CIs."""
    plot_df = main_results.copy()
    plot_df = plot_df.dropna(subset=[
        'Effect per Treatment SD',
        'SD Effect CI Lower',
        'SD Effect CI Upper',
    ])
    if plot_df.empty:
        raise ValueError("没有可用于 ATE forest plot 的主估计")

    # A transparent magnitude ordering makes the strongest effects immediately
    # visible while retaining the sign and exact interval for every treatment.
    plot_df['_abs_effect'] = plot_df['Effect per Treatment SD'].abs()
    plot_df = plot_df.sort_values(
        '_abs_effect', ascending=False, kind='mergesort').reset_index(drop=True)

    y = np.arange(len(plot_df))
    effect = plot_df['Effect per Treatment SD'].to_numpy(dtype=float)
    ci_low = plot_df['SD Effect CI Lower'].to_numpy(dtype=float)
    ci_high = plot_df['SD Effect CI Upper'].to_numpy(dtype=float)

    display_names = {
        'MBH': 'Mean building height',
        'BC': 'Building coverage',
        'NDVI': 'NDVI',
        'water': 'Water',
        'Pop2025': 'Population (2025)',
        'Road_Dens': 'Road density',
    }
    treatment_names = [
        display_names.get(str(name), str(name).replace('_', ' '))
        for name in plot_df['Treatment']
    ]

    # Submission figure uses a restrained grayscale palette.  Sign is encoded
    # primarily by position relative to zero; tone is only a secondary cue.
    cooling_color = '#2B2B2B'
    warming_color = '#777777'
    neutral_dark = '#252525'
    neutral_mid = '#6B6B6B'
    neutral_light = '#E7E7E7'
    row_fill = '#F6F6F4'
    point_colors = np.where(effect < 0.0, cooling_color, warming_color)

    max_abs = float(np.nanmax(np.abs(np.r_[ci_low, ci_high, 0.0])))
    if not np.isfinite(max_abs) or max_abs == 0.0:
        max_abs = 1.0
    x_limit = max_abs * 1.14

    # Compact three-column table/forest layout:
    # treatment label | estimate and CI | exact numerical estimate.
    # The forest panel is the visual anchor; metadata and footnotes stay quiet.
    fig_height = max(3.65, 0.46 * len(plot_df) + 0.95)
    fig = plt.figure(figsize=(7.2, fig_height), dpi=300, facecolor='white')
    grid = fig.add_gridspec(
        # Equal-width text columns keep the central forest axis centred on
        # the full figure canvas rather than on an asymmetric table region.
        nrows=1, ncols=3, width_ratios=(1.82, 3.86, 1.82), wspace=0.025,
        left=0.045, right=0.985, bottom=0.205, top=0.82,
    )
    ax_names = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1], sharey=ax_names)
    ax_values = fig.add_subplot(grid[0, 2], sharey=ax_names)

    for axis in (ax_names, ax, ax_values):
        axis.set_ylim(len(plot_df) - 0.5, -0.5)
        for row_index in range(len(plot_df)):
            if row_index % 2 == 0:
                axis.axhspan(
                    row_index - 0.5, row_index + 0.5,
                    color=row_fill, linewidth=0, zorder=-10,
                )

    ax_names.set_xlim(0.0, 1.0)
    ax_names.set_xticks([])
    ax_names.set_yticks([])
    for spine in ax_names.spines.values():
        spine.set_visible(False)
    for yi, label in zip(y, treatment_names):
        ax_names.text(
            0.025, yi, label, ha='left', va='center',
            fontsize=8.8, color=neutral_dark,
        )

    ax.axvline(
        0.0, color=neutral_mid, linestyle='-', linewidth=0.9, zorder=1,
    )
    for yi, estimate_i, lower_i, upper_i, color_i in zip(
            y, effect, ci_low, ci_high, point_colors):
        ax.errorbar(
            estimate_i, yi,
            xerr=[[max(estimate_i - lower_i, 0.0)],
                  [max(upper_i - estimate_i, 0.0)]],
            fmt='o', markersize=3.6, markerfacecolor=color_i,
            markeredgecolor='white', markeredgewidth=0.45,
            color=color_i, ecolor=color_i, elinewidth=1.35,
            capsize=3.2, capthick=1.05, zorder=3,
        )
    ax.set_xlim(-x_limit, x_limit)
    ax.set_yticks([])
    ax.set_xlabel(
        'Effect on mean UHII per 1-SD increase (°C)',
        fontsize=8.7, color=neutral_dark, labelpad=7,
    )
    ax.tick_params(
        axis='x', labelsize=7.8, direction='out', length=2.8,
        width=0.75, colors=neutral_dark,
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(axis='x', color=neutral_light, linewidth=0.55, zorder=-9)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color(neutral_dark)
    ax.spines['bottom'].set_linewidth(0.8)

    ax_values.set_xlim(0.0, 1.0)
    ax_values.set_xticks([])
    ax_values.set_yticks([])
    for spine in ax_values.spines.values():
        spine.set_visible(False)
    for yi, estimate_i, lower_i, upper_i in zip(y, effect, ci_low, ci_high):
        ax_values.text(
            0.98, yi,
            f'{estimate_i:+.3f} [{lower_i:+.3f}, {upper_i:+.3f}]',
            ha='right', va='center', fontsize=8.15, color=neutral_dark,
        )

    ax_names.set_title(
        'Treatment', loc='left', fontsize=8.4, fontweight='bold',
        color=neutral_dark, pad=9,
    )
    ax.set_title(
        'Effect per 1 SD and 95% CI', fontsize=8.4, fontweight='bold',
        color=neutral_dark, pad=9,
    )
    ax_values.set_title(
        'Estimate [95% CI], °C', loc='right', fontsize=8.4,
        fontweight='bold', color=neutral_dark, pad=9,
    )
    for axis in (ax_names, ax, ax_values):
        axis.plot(
            [0.0, 1.0], [1.035, 1.035], transform=axis.transAxes,
            color=neutral_dark, linewidth=0.7, clip_on=False,
        )

    suite_display = str(suite_name).strip()
    fig.text(
        0.045, 0.965, 'Effects on mean urban heat island intensity',
        ha='left', va='top', fontsize=11.2, fontweight='bold',
        color=neutral_dark,
    )
    if suite_display:
        fig.text(
            0.985, 0.957, suite_display,
            ha='right', va='top', fontsize=7.4, color=neutral_mid,
        )

    n_units = None
    if 'N Units' in plot_df.columns:
        n_values = plot_df['N Units'].dropna().unique()
        if len(n_values) == 1:
            n_units = int(n_values[0])
    n_bootstrap = None
    if 'Bootstrap Requested' in plot_df.columns:
        b_values = plot_df['Bootstrap Requested'].dropna().unique()
        if len(b_values) == 1:
            n_bootstrap = int(b_values[0])
    footer_parts = [
        'Points show linear effect estimates; horizontal lines show 95% '
        'grid-unit percentile-bootstrap CIs.'
    ]
    if n_units is not None:
        footer_parts.append(f'n = {n_units:,} physical grids.')
    if n_bootstrap is not None:
        footer_parts.append(f'B = {n_bootstrap:,} resamples.')
    fig.text(
        0.045, 0.052, ' '.join(footer_parts),
        ha='left', va='bottom', fontsize=6.65, color=neutral_mid,
    )

    # Preserve editable text in vector exports and provide a high-resolution
    # raster preview. The SVG path mirrors the existing PDF output name.
    previous_svg_fonttype = plt.rcParams.get('svg.fonttype', 'path')
    previous_pdf_fonttype = plt.rcParams.get('pdf.fonttype', 3)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    svg_path = os.path.splitext(pdf_path)[0] + '.svg'
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.rcParams['svg.fonttype'] = previous_svg_fonttype
    plt.rcParams['pdf.fonttype'] = previous_pdf_fonttype
    plt.close(fig)
    return {'svg': svg_path, 'pdf': pdf_path, 'png': png_path}

for suite_key in all_suite_causal_artifacts:
    artifacts = all_suite_causal_artifacts[suite_key]
    suite_name = artifacts['suite_name']
    causal_features = artifacts['causal_features']
    # 用【稳定性过滤后】的邻接矩阵（adj_masked）生成 DoWhy 的 DAG，保证：
    #   展示给审稿人的 make_dot DAG 图 与 DoWhy 后门调整用的图【完全一致】。
    #   adj_masked 仅含达到预设有向选择频率且未造成有向环的边，
    #   其余边归零 → 不参与后门集识别，保证 ATE 识别图与展示图一致。
    adj_matrix = artifacts.get('adj_masked', artifacts['adj_matrix'])
    dag_labels = artifacts['dag_labels']
    df_causal = artifacts['df_causal']

    graph_dot_str = _adjacency_to_dot_str(adj_matrix, dag_labels)
    all_treatment_results = []
    all_bootstrap_results = []
    n_treatments = len(causal_features)

    # DoWhy 使用全部网格级样本。Grid_ID/Causal_Unit_ID 仅用于构造分析
    # 单位，不进入 DAG 或回归设计矩阵，也不再做 50 万小时行随机抽样。
    df_dowhy = df_causal[dag_labels].copy()

    # 预转换 float64 并存入 fork 共享字典（CoW 零拷贝，无需写磁盘）
    for c in df_dowhy.columns:
        if df_dowhy[c].dtype.name == 'category':
            df_dowhy[c] = df_dowhy[c].astype(float)
        else:
            df_dowhy[c] = df_dowhy[c].astype(np.float64)

    print(f"\n{'='*60}")
    _dowhy_workers = min(8, max(1, n_treatments))
    print(f"DoWhy 图调整线性效应估计: {n_treatments} 个处理变量 "
          f"(并行 × {_dowhy_workers} worker, {len(df_dowhy):,} 个网格) "
          f"[{suite_name}]")
    print(f"{'='*60}")
    print(f"  [准备] 使用全部网格级数据；效应 95% CI = "
          f"{_DOWHY_ATE_N_BOOTSTRAP} 次 Grid-unit percentile Bootstrap")
    print("  [限定] 效应 Bootstrap 固定当前共识 DAG；每轮重抽网格并重新识别/估计，"
          "但不重新学习 LiNGAM 结构")

    _SHARED_DOWHY_DATA['df'] = df_dowhy
    del df_dowhy
    gc.collect()

    # --- Worker 函数：每个进程独立处理一个处理变量 ---
    def _dowhy_single(treatment_var):
        import pandas as pd
        import numpy as np
        from dowhy import CausalModel
        import gc
        rows = []
        bootstrap_rows = []
        backdoor_record = {
            'Treatment': treatment_var,
            'Outcome': _TARGET_LABEL,
            'Backdoor adjustment variables': 'None / empty set',
            'Number of adjustment variables': 0,
            'Identification status': 'failed',
            'Estimator': 'backdoor.linear_regression',
            'DAG version / causal-analysis variant': CAUSAL_ANALYSIS_VARIANT,
            'DAG bootstrap replicates': _STAB_N_BOOTSTRAP,
            'Within-bootstrap effect threshold': _EDGE_EFFECT_THRESHOLD,
            'Directed selection-frequency threshold': _EDGE_SELECTION_THRESHOLD,
            'Effect bootstrap replicates': _DOWHY_ATE_N_BOOTSTRAP,
            'Error': None,
        }
        df = _SHARED_DOWHY_DATA['df']  # fork CoW 共享读取，零 IO
        try:
            def _estimate_on_frame(estimate_df):
                model = CausalModel(
                    data=estimate_df,
                    treatment=treatment_var,
                    outcome=_TARGET_LABEL,
                    graph=graph_dot_str,
                )
                estimand = model.identify_effect(
                    proceed_when_unidentifiable=False)
                estimate = model.estimate_effect(
                    estimand,
                    method_name="backdoor.linear_regression",
                    control_value=0.0,
                    treatment_value=1.0,
                    confidence_intervals=False,
                    test_significance=False,
                )
                value = float(np.asarray(estimate.value).ravel()[0])
                return model, estimand, estimate, value

            cm, ie, est, ate_val = _estimate_on_frame(df)
            # 打印后门调整集（审计用）
            try:
                _bd_vars = list(ie.get_backdoor_variables() or [])
            except Exception as _backdoor_error:
                raise RuntimeError(
                    f"无法提取 {treatment_var} 的后门调整集: {_backdoor_error}"
                ) from _backdoor_error
            backdoor_record.update({
                'Backdoor adjustment variables': (
                    '; '.join(_bd_vars) if _bd_vars else 'None / empty set'),
                'Number of adjustment variables': len(_bd_vars),
                'Identification status': 'identified',
            })
            print(f"    [{treatment_var}] 后门调整集: {_bd_vars}")

            treatment_sd = float(df[treatment_var].std(ddof=1))
            if not np.isfinite(treatment_sd) or treatment_sd <= 0.0:
                raise ValueError(f"{treatment_var} 在网格级数据中没有有效变异")

            treatment_index = causal_features.index(treatment_var)
            rng = np.random.RandomState(42000 + treatment_index * 1000)
            bootstrap_effects = []
            n_units = len(df)
            for bootstrap_id in range(_DOWHY_ATE_N_BOOTSTRAP):
                sampled_idx = rng.randint(0, n_units, size=n_units)
                bootstrap_df = df.iloc[sampled_idx].reset_index(drop=True)
                try:
                    _, _, _, bootstrap_ate = _estimate_on_frame(bootstrap_df)
                    bootstrap_effects.append(bootstrap_ate)
                    bootstrap_rows.append({
                        'Suite': suite_name,
                        'Treatment': treatment_var,
                        'Bootstrap ID': bootstrap_id + 1,
                        'Estimated Effect': bootstrap_ate,
                        'Effect per Treatment SD': bootstrap_ate * treatment_sd,
                        'Status': 'ok',
                        'Error': None,
                    })
                except Exception as bootstrap_error:
                    bootstrap_rows.append({
                        'Suite': suite_name,
                        'Treatment': treatment_var,
                        'Bootstrap ID': bootstrap_id + 1,
                        'Estimated Effect': None,
                        'Effect per Treatment SD': None,
                        'Status': 'failed',
                        'Error': str(bootstrap_error)[:240],
                    })

            bootstrap_effects = np.asarray(bootstrap_effects, dtype=float)
            min_valid = int(np.ceil(
                _DOWHY_ATE_N_BOOTSTRAP * _DOWHY_ATE_MIN_VALID_FRACTION))
            if bootstrap_effects.size < min_valid:
                raise RuntimeError(
                    f"{treatment_var} ATE Bootstrap 仅成功 "
                    f"{bootstrap_effects.size}/{_DOWHY_ATE_N_BOOTSTRAP} 次"
                )

            ci_lower, ci_upper = np.percentile(
                bootstrap_effects, [2.5, 97.5])
            bootstrap_se = float(bootstrap_effects.std(ddof=1))
            effect_per_sd = ate_val * treatment_sd
            sd_ci_lower = float(ci_lower * treatment_sd)
            sd_ci_upper = float(ci_upper * treatment_sd)

            rows.append({
                "Suite": suite_name, "Treatment": treatment_var,
                "Test": "Main Estimate",
                "Estimated Effect": ate_val,
                "CI Lower": float(ci_lower), "CI Upper": float(ci_upper),
                "Effect per Treatment SD": effect_per_sd,
                "SD Effect CI Lower": sd_ci_lower,
                "SD Effect CI Upper": sd_ci_upper,
                "Treatment SD": treatment_sd,
                "Treatment Contrast": "1 raw-unit increase (linear slope)",
                "CI Method": "Grid-unit percentile bootstrap",
                "Bootstrap SE": bootstrap_se,
                "Bootstrap Requested": _DOWHY_ATE_N_BOOTSTRAP,
                "Bootstrap Valid": int(bootstrap_effects.size),
                "Analysis Unit": "physical Grid_ID",
                "N Units": int(n_units),
                "p-value": None,
                "Status": "ok",
                "Error": None,
            })

            if ate_val is not None:
                for ref_name, ref_method in [
                    ("Random Common Cause", "random_common_cause"),
                    ("Placebo Treatment", "placebo_treatment_refuter"),
                    ("Data Subset", "data_subset_refuter"),
                ]:
                    try:
                        # num_simulations=50：默认 100 次，减半够用（p 值仍稳定）
                        refute = cm.refute_estimate(ie, est, method_name=ref_method,
                                                    num_simulations=50)
                        ref_p = None
                        if hasattr(refute, 'refutation_result'):
                            ref_p = refute.refutation_result.get("p_value", None)
                        ref_effect = float(np.asarray(refute.new_effect).ravel()[0])
                        rows.append({
                            "Suite": suite_name, "Treatment": treatment_var,
                            "Test": ref_name,
                            "Estimated Effect": ref_effect,
                            "CI Lower": None, "CI Upper": None,
                            "Effect per Treatment SD": ref_effect * treatment_sd,
                            "SD Effect CI Lower": None,
                            "SD Effect CI Upper": None,
                            "Treatment SD": treatment_sd,
                            "Treatment Contrast": "1 raw-unit increase (linear slope)",
                            "CI Method": None,
                            "Bootstrap SE": None,
                            "Bootstrap Requested": None,
                            "Bootstrap Valid": None,
                            "Analysis Unit": "physical Grid_ID",
                            "N Units": int(n_units),
                            "p-value": ref_p,
                            "Status": "ok",
                            "Error": None,
                        })
                    except Exception as refuter_error:
                        rows.append({
                            "Suite": suite_name,
                            "Treatment": treatment_var,
                            "Test": ref_name,
                            "Estimated Effect": None,
                            "CI Lower": None,
                            "CI Upper": None,
                            "Effect per Treatment SD": None,
                            "SD Effect CI Lower": None,
                            "SD Effect CI Upper": None,
                            "Treatment SD": treatment_sd,
                            "Treatment Contrast": "1 raw-unit increase (linear slope)",
                            "CI Method": None,
                            "Bootstrap SE": None,
                            "Bootstrap Requested": None,
                            "Bootstrap Valid": None,
                            "Analysis Unit": "physical Grid_ID",
                            "N Units": int(n_units),
                            "p-value": None,
                            "Status": "failed",
                            "Error": (
                                f"{type(refuter_error).__name__}: {refuter_error}"
                            )[:500],
                        })
        except Exception as e:
            backdoor_record.update({
                'Identification status': 'failed',
                'Error': f"{type(e).__name__}: {e}"[:500],
            })
            rows.append({
                "Suite": suite_name, "Treatment": treatment_var,
                "Test": "Error", "Estimated Effect": str(e),
                "CI Lower": None, "CI Upper": None,
                "Effect per Treatment SD": None,
                "SD Effect CI Lower": None, "SD Effect CI Upper": None,
                "Treatment SD": None, "Treatment Contrast": None,
                "CI Method": None, "Bootstrap SE": None,
                "Bootstrap Requested": _DOWHY_ATE_N_BOOTSTRAP,
                "Bootstrap Valid": int(sum(
                    row.get('Status') == 'ok' for row in bootstrap_rows)),
                "Analysis Unit": "physical Grid_ID",
                "N Units": int(len(df)), "p-value": None,
                "Status": "failed",
                "Error": f"{type(e).__name__}: {e}"[:500],
            })
            _bd_vars = []
        del df
        gc.collect()
        return treatment_var, rows, _bd_vars, bootstrap_rows, backdoor_record

    # --- 并行执行 ---
    print(f"  [执行] 启动 {n_treatments} 个处理变量并行估计 "
          f"(n_jobs={_dowhy_workers})...")
    worker_results = Parallel(
        n_jobs=_dowhy_workers, backend='multiprocessing', verbose=10)(
        delayed(_dowhy_single)(t) for t in causal_features
    )

    # 收集后门调整集、主结果和每轮网格 Bootstrap 估计。
    backdoor_sets = {}
    backdoor_records = []
    for t_var, res, bd_vars, bootstrap_rows, backdoor_record in worker_results:
        backdoor_sets[t_var] = bd_vars if bd_vars else []
        backdoor_records.append(backdoor_record)
        if bootstrap_rows:
            all_bootstrap_results.append(pd.DataFrame(bootstrap_rows))
        if res:
            main_row = [r for r in res if r["Test"] == "Main Estimate"]
            if main_row:
                ate_v = main_row[0]["Estimated Effect"]
                ci_l = main_row[0].get("CI Lower")
                ci_u = main_row[0].get("CI Upper")
                sd_effect = main_row[0].get("Effect per Treatment SD")
                valid_boot = main_row[0].get("Bootstrap Valid")
                if isinstance(ate_v, (int, float)):
                    ci_info = f", 95% CI=[{ci_l:.6f}, {ci_u:.6f}]" if ci_l is not None else ""
                    print(f"  ✓ {t_var}: raw-unit ATE={ate_v:.6f}{ci_info}, "
                          f"effect/SD={sd_effect:.6f}, "
                          f"valid bootstrap={valid_boot}/{_DOWHY_ATE_N_BOOTSTRAP}")
                else:
                    print(f"  ✗ {t_var}: {ate_v}")
            all_treatment_results.append(pd.DataFrame(res))

    # 每个 treatment 均保留一行，包括空后门集合和识别失败情形。
    if {row['Treatment'] for row in backdoor_records} != set(causal_features):
        raise RuntimeError(
            f"后门调整集审计表未完整覆盖 {len(causal_features)} 个预设 treatment")
    backdoor_path = os.path.join(
        CAUSAL_DATA_DIR,
        f'{suite_key}_DoWhy_Backdoor_Adjustment_Sets.csv')
    pd.DataFrame(backdoor_records).to_csv(
        backdoor_path, index=False, encoding='utf-8-sig')
    print(f"  后门调整集审计表已保存: {backdoor_path}")

    # --- 清理共享数据引用 ---
    _SHARED_DOWHY_DATA.pop('df', None)
    gc.collect()
    print("  [清理] 共享字典 df 引用已释放")

    if all_treatment_results:
        combined = pd.concat(all_treatment_results, ignore_index=True)
        csv_path = os.path.join(
            CAUSAL_DATA_DIR, f'{suite_key}_DoWhy_Refutation_Results.csv')
        combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n  反驳结果已保存: {csv_path}")
        print(f"  总计 {len(combined)} 条记录, {n_treatments} 个处理变量")

        # Main text compares per-1-SD effects; raw-scale slopes are exported
        # separately for Supplementary reporting.  Unverified source units are
        # explicitly flagged instead of inferred from the stored magnitudes.
        _treatment_unit_audit = {
            'MBH': ('m', 'confirmed'),
            'BC': ('fraction (0-1)', 'confirmed from authoritative Fishnet'),
            'NDVI': ('dimensionless index', 'confirmed'),
            'water': ('stored grid-level water value', 'AUTHOR_INPUT_NEEDED'),
            'Pop2025': ('stored WorldPop grid value', 'AUTHOR_INPUT_NEEDED'),
            'Road_Dens': (
                'stored road-density unit; verify conversion to km km^-2',
                'AUTHOR_INPUT_NEEDED'),
        }
        _raw_effect_rows = []
        _backdoor_lookup = {
            row['Treatment']: row['Backdoor adjustment variables']
            for row in backdoor_records
        }
        for _, _effect_row in combined.loc[
                combined['Test'] == 'Main Estimate'].iterrows():
            _treatment = str(_effect_row['Treatment'])
            _unit, _unit_status = _treatment_unit_audit.get(
                _treatment, ('not reported', 'AUTHOR_INPUT_NEEDED'))
            _raw_effect_rows.append({
                'Treatment': _treatment,
                'Outcome': _TARGET_LABEL,
                'Raw-unit linear slope': _effect_row['Estimated Effect'],
                'Raw-unit CI lower': _effect_row['CI Lower'],
                'Raw-unit CI upper': _effect_row['CI Upper'],
                'Original unit': _unit,
                'Unit verification status': _unit_status,
                'Treatment sample SD': _effect_row['Treatment SD'],
                'Effect per 1-SD increase': _effect_row['Effect per Treatment SD'],
                'Per-SD CI lower': _effect_row['SD Effect CI Lower'],
                'Per-SD CI upper': _effect_row['SD Effect CI Upper'],
                'Backdoor adjustment variables': _backdoor_lookup.get(
                    _treatment, 'NOT_EXPORTED'),
                'Estimator': 'backdoor.linear_regression',
                'Analysis unit': 'physical Grid_ID',
                'N physical grids': _effect_row['N Units'],
                'Effect bootstrap replicates requested': _DOWHY_ATE_N_BOOTSTRAP,
                'Effect uncertainty conditional on fixed consensus DAG': True,
            })
        _raw_effect_path = os.path.join(
            CAUSAL_DATA_DIR,
            f'{suite_key}_DoWhy_Raw_Unit_Effects_Supplementary.csv')
        pd.DataFrame(_raw_effect_rows).to_csv(
            _raw_effect_path, index=False, encoding='utf-8-sig')
        print(f"  原始单位效应补充表已保存: {_raw_effect_path}")

        _causal_metadata = {
            'analysis_variant': CAUSAL_ANALYSIS_VARIANT,
            'suite': suite_name,
            'analysis_unit': 'one globally unique physical Grid_ID',
            'n_physical_grids': int(len(artifacts['df_causal'])),
            'outcome': _TARGET_LABEL,
            'outcome_aggregation': 'study-period arithmetic mean within Grid_ID',
            'main_treatments': '; '.join(causal_features),
            'Temp_2m_in_predictive_feature_cols': bool('Temp_2m' in feature_cols),
            'Pop2025_in_predictive_feature_cols': bool('Pop2025' in feature_cols),
            'Temp_2m_in_causal_features': bool('Temp_2m' in causal_features),
            'Pop2025_in_causal_features': bool('Pop2025' in causal_features),
            'causal_discovery': 'DirectLiNGAM',
            'prior_knowledge': 'self-loops forbidden; UHII fixed as sink only',
            'dag_bootstrap_design': 'n-out-of-n physical-grid bootstrap',
            'dag_bootstrap_requested': _STAB_N_BOOTSTRAP,
            'dag_bootstrap_valid': int(artifacts['dag_bootstrap_valid']),
            'within_bootstrap_abs_beta_std_threshold': _EDGE_EFFECT_THRESHOLD,
            'directed_selection_frequency_threshold': _EDGE_SELECTION_THRESHOLD,
            'cycle_pruning': 'frequency-first acyclic aggregation',
            'effect_estimator': 'DoWhy backdoor.linear_regression',
            'effect_bootstrap_design': 'n-out-of-n physical-grid bootstrap',
            'effect_bootstrap_requested_per_treatment': _DOWHY_ATE_N_BOOTSTRAP,
            'effect_ci': '2.5th and 97.5th percentile of valid estimates',
            'effect_uncertainty_scope': 'conditional on fixed consensus DAG',
            'dag_relearned_inside_effect_bootstrap': False,
            'software_note': 'See revision script and environment lockfile/log',
        }
        _metadata_csv_path = os.path.join(
            CAUSAL_DATA_DIR, f'{suite_key}_Causal_Analysis_Metadata.csv')
        _metadata_json_path = os.path.join(
            CAUSAL_DATA_DIR, f'{suite_key}_Causal_Analysis_Metadata.json')
        pd.DataFrame([_causal_metadata]).to_csv(
            _metadata_csv_path, index=False, encoding='utf-8-sig')
        with open(_metadata_json_path, 'w', encoding='utf-8') as _metadata_file:
            json.dump(_causal_metadata, _metadata_file,
                      ensure_ascii=False, indent=2)
        print(f"  因果分析元数据已保存: {_metadata_csv_path}")

        bootstrap_csv_path = os.path.join(
            CAUSAL_DATA_DIR,
            f'{suite_key}_DoWhy_ATE_Grid_Bootstrap_Replicates.csv')
        if all_bootstrap_results:
            bootstrap_combined = pd.concat(
                all_bootstrap_results, ignore_index=True)
            bootstrap_combined.to_csv(
                bootstrap_csv_path, index=False, encoding='utf-8-sig')
            print(f"  ATE Bootstrap 明细已保存: {bootstrap_csv_path}")

        # Forest plot 使用每个处理变量增加 1 SD 的效应，避免把 BC、人口、
        # 道路密度等不同原始单位的系数直接放在同一数轴上比较。
        ate_pdf = os.path.join(CAUSAL_PLOTS_DIR, f'{suite_key}_ATE_forest.pdf')
        ate_png = os.path.join(CAUSAL_PLOTS_DIR, f'{suite_key}_ATE_forest.png')
        try:
            _main_for_plot = combined.loc[
                combined['Test'] == 'Main Estimate'].copy()
            _ate_forest_paths = _plot_ate_forest_python(
                _main_for_plot, ate_pdf, ate_png, suite_name,
            )
            print("  ATE Forest Plot (Python, effect per treatment SD) 已保存:")
            print(f"    SVG: {_ate_forest_paths['svg']}")
            print(f"    PDF: {_ate_forest_paths['pdf']}")
            print(f"    PNG: {_ate_forest_paths['png']}")
        except Exception as _e_ate:
            print(f"  ATE 绘图异常: {_e_ate}")

        artifacts['combined_results'] = combined
        artifacts['graph_dot_str'] = graph_dot_str
    else:
        print(f"⚠ [{suite_name}] 无有效 DoWhy 结果")
    artifacts['backdoor_sets'] = backdoor_sets

    gc.collect()
    try:
        _ctypes_cb.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

print(f"\n=== 因果推断完成: DoWhy 效应估计与反驳检验结束 ===")
gc.collect()
print(f"因果推断输出目录: {CAUSAL_DIR}")

# %%
# ================= Stacking/Ensemble 模型融合（已移除） =================
# 原 Ensemble 2/3/4 三段加权融合代码已移除。
# 本研究保留 LightGBM/CatBoost/XGBoost/RF/GBR/DT/MLP/LinearRegression/BayesianRidge
# 九个基模型的独立预测结果。SHAP 和因果推断基于单个最优模型（LightGBM）进行。
# =================


print(f"\n✅ 全部流程完成。输出目录: {BASE_OUTPUT_DIR}")
