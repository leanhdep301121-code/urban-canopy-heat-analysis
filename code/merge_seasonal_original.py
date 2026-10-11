"""
数据集扩充脚本：将原始数据集扩展为夏季+冬季版本
====================================================
原始数据：每个 Grid_ID 24行（Hour 0-23）
输出数据：每个 Grid_ID 48行（夏季24行 + 冬季24行）

变化说明：
  因变量 Temperature  → 替换为对应季节的 UHI 强度（℃）
  自变量 Rural_Temp   → 替换为对应季节的郊区背景温度（℃）
  自变量 Temp_2m      → 替换为对应季节的 2m 均值气温（℃）
  新增列 Season       → 'summer' / 'winter'

静态自变量（MBH/FAR/NDVI 等）在两个季节行中保持不变。
"""

import pandas as pd
import numpy as np
import os

# ==============================================================================
# 配置区
# ==============================================================================

ORIGINAL_CSV = r"H:\数据集\Forth_export_long_format_final.csv"
SUMMER_UHI   = r"D:\1km_00_hourly_avg\UHI_result\summer_UHI_All_Hours.csv"
WINTER_UHI   = r"D:\1km_00_hourly_avg\UHI_result\winter_UHI_All_Hours.csv"
OUTPUT_CSV   = r"H:\数据集\Fifth_export_seasonal_long_format.csv"

# ==============================================================================
# 辅助函数
# ==============================================================================

def uhi_wide_to_long(df_wide, season_label):
    """
    将宽表 UHI 结果（每行一个 Grid_ID，24列时次）
    转为长表（每行一个 Grid_ID × Hour），包含：
      Grid_ID, Hour, Temperature, Rural_Temp, Temp_2m, Season
    """
    print(f"  [{season_label}] 宽表 → 长表转换（{len(df_wide)} 个格子 × 24小时）...")
    frames = []
    for h in range(24):
        col_uhi   = f"UHI_BJT_{h:02d}"
        col_rural = f"Rural_Temp_BJT_{h:02d}"
        col_temp  = f"Temp_BJT_{h:02d}"

        if col_uhi not in df_wide.columns:
            print(f"    ⚠ 缺少列 {col_uhi}，跳过第 {h} 小时")
            continue

        frame = pd.DataFrame({
            "Grid_ID"    : df_wide["Grid_ID"].values,
            "Hour"       : h,
            "Temperature": df_wide[col_uhi].values,    # 因变量：UHI强度
            "Rural_Temp" : df_wide[col_rural].values,  # 背景温度
            "Temp_2m"    : df_wide[col_temp].values,   # 2m气温
            "Season"     : season_label,
        })
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    print(f"  [{season_label}] 长表行数：{len(result):,}")
    return result


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    print("=" * 65)
    print("  数据集季节化扩充：原始 24行/格 → 夏+冬 48行/格")
    print(f"  输入：{ORIGINAL_CSV}")
    print(f"  输出：{OUTPUT_CSV}")
    print("=" * 65)

    # ── 1. 读取原始数据集 ────────────────────────────────────────────
    print("\n[1/5] 读取原始数据集...")
    orig = pd.read_csv(ORIGINAL_CSV)
    print(f"  ✓ 原始形状：{orig.shape}，Grid_ID 数量：{orig['Grid_ID'].nunique():,}")

    # 确保 Grid_ID 为整数
    orig["Grid_ID"] = orig["Grid_ID"].astype(int)

    # 静态列：除时间和因变量外，剩余列都是每个 Grid_ID 不变的
    DYNAMIC_COLS = {"Hour", "Hour_sin", "Hour_cos", "Temperature", "Rural_Temp", "Temp_2m"}
    static_cols  = [c for c in orig.columns if c not in DYNAMIC_COLS]

    # 提取静态特征（每个 Grid_ID 取第一条即可，静态特征各小时相同）
    print(f"  静态列（{len(static_cols)}个）：{static_cols}")
    static_df = orig.drop_duplicates(subset=["Grid_ID"])[static_cols].copy()
    print(f"  ✓ 静态特征表：{static_df.shape}")

    # ── 2. 读取 UHI 结果并转为长表 ──────────────────────────────────
    print("\n[2/5] 读取 UHI 季节均值数据...")
    summer_wide = pd.read_csv(SUMMER_UHI)
    winter_wide = pd.read_csv(WINTER_UHI)
    summer_wide["Grid_ID"] = summer_wide["Grid_ID"].astype(int)
    winter_wide["Grid_ID"] = winter_wide["Grid_ID"].astype(int)
    print(f"  夏季 UHI 格子数：{len(summer_wide):,}，冬季：{len(winter_wide):,}")

    print("\n[3/5] 宽表 → 逐小时长表...")
    summer_long = uhi_wide_to_long(summer_wide, "summer")
    winter_long = uhi_wide_to_long(winter_wide, "winter")

    # ── 3. 合并夏冬 UHI 长表 ────────────────────────────────────────
    print("\n[4/5] 合并夏冬数据并加入静态特征...")
    uhi_long = pd.concat([summer_long, winter_long], ignore_index=True)
    print(f"  合并后行数：{len(uhi_long):,}")

    # 重新计算 Hour_sin / Hour_cos（与原始数据保持一致）
    uhi_long["Hour_sin"] = np.sin(2 * np.pi * uhi_long["Hour"] / 24)
    uhi_long["Hour_cos"] = np.cos(2 * np.pi * uhi_long["Hour"] / 24)

    # 左连接静态特征（保留 UHI 中存在的格子；
    # 若某 Grid_ID 在原始数据中无静态特征则 NaN）
    result = uhi_long.merge(static_df, on="Grid_ID", how="left")

    # ── 4. 整理列顺序（与原始数据保持一致，新增 Season 在最后）────
    orig_col_order = list(orig.columns)
    # 把 Season 插入到所有列末尾
    final_cols = orig_col_order + ["Season"]
    # 只保留实际存在的列
    final_cols = [c for c in final_cols if c in result.columns]
    result = result[final_cols]

    print(f"  ✓ 最终数据集形状：{result.shape}")
    print(f"  ✓ Grid_ID 数量：{result['Grid_ID'].nunique():,}")
    print(f"  ✓ 每个 Grid_ID 行数分布：")
    rows_per_grid = result['Grid_ID'].value_counts()
    print(f"     mean={rows_per_grid.mean():.0f}，min={rows_per_grid.min()}，max={rows_per_grid.max()}")

    # 抽样检查：打印一个 Grid_ID 的前几行
    sample_id = result['Grid_ID'].iloc[0]
    print(f"\n  样本 Grid_ID={sample_id} 的前4行：")
    sample = result[result['Grid_ID'] == sample_id][
        ['Grid_ID', 'Hour', 'Season', 'Temperature', 'Rural_Temp', 'Temp_2m']
    ].head(4)
    print(sample.to_string(index=False))

    # ── 5. 导出 ─────────────────────────────────────────────────────
    print(f"\n[5/5] 导出结果到：{OUTPUT_CSV}")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    size_mb = os.path.getsize(OUTPUT_CSV) / 1024 / 1024
    print(f"  ✅ 完成！文件大小：{size_mb:.1f} MB")

    print("\n" + "=" * 65)
    print("  列说明：")
    print("    Temperature  → 季节均值 UHI 强度（℃）")
    print("    Rural_Temp   → 季节均值郊区背景温度（℃）")
    print("    Temp_2m      → 季节均值 2m 气温（℃）")
    print("    Season       → 'summer'(7-8月) / 'winter'(12-1月)")
    print("=" * 65)


if __name__ == "__main__":
    main()
