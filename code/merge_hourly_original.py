"""将四个月逐日逐时热岛结果整合到原始长表数据集中。

当前版本采用“先完成整合”的策略：
1. 以北京时间自然月为时间骨架，保证每个 Grid_ID 的每个小时都有一行。
2. 若热岛月表缺少某个网格某个时次，则自动补出该行，热岛相关数值填为 NaN。
3. 若某个北京时间小时整槽缺失，也自动补出整槽，便于后续统一建模。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import math
import os

import pandas as pd


# ==============================================================================
# 配置区
# ==============================================================================
ORIGINAL_CSV = r"H:\数据集\Fifth_export_seasonal_long_format.csv"
UHI_DIR = r"F:\result_hourly_long"
UHI_MONTH_FILES = [
    "UHI_Long_BJT_2025_07.csv",
    "UHI_Long_BJT_2025_08.csv",
    "UHI_Long_BJT_2025_12.csv",
    "UHI_Long_BJT_2026_01.csv",
]
OUTPUT_CSV = r"H:\数据集\Sixth_export_daily_hourly_long_format.csv"
GRID_ID_COL = "Grid_ID"
PATCH_FILE_FLAG = "_patch_"

# 原始数据中这些列会被逐日逐时热岛结果替换。
DYNAMIC_COLS = {
    "Hour",
    "Hour_sin",
    "Hour_cos",
    "Temperature",
    "Rural_Temp",
    "Temp_2m",
    "Season",
}

ORIGINAL_CHUNK_SIZE = 200_000
UHI_CHUNK_SIZE = 300_000

UHI_REQUIRED_COLS = [
    "Grid_ID",
    "Date_BJT",
    "Hour_BJT",
    "DateTime_BJT",
    "Date_UTC",
    "Hour_UTC",
    "T_Grid_Raw",
    "Temp_2m_Celsius",
    "Bg_Temp_Rural_Celsius",
    "UHI_Intensity",
    "Month_BJT",
    "Season",
    "Source_TIF",
]

EXTRA_OUTPUT_COLS = [
    "Date_BJT",
    "DateTime_BJT",
    "Date_UTC",
    "Hour_UTC",
    "Month_BJT",
    "T_Grid_Raw",
    "Source_TIF",
]

SLOT_DYNAMIC_COLS = [
    "Hour",
    "Hour_sin",
    "Hour_cos",
    "Temperature",
    "Rural_Temp",
    "Temp_2m",
    "Season",
    "Date_BJT",
    "DateTime_BJT",
    "Date_UTC",
    "Hour_UTC",
    "Month_BJT",
    "T_Grid_Raw",
    "Source_TIF",
]


def ensure_exists(path_str: str) -> None:
    """检查输入路径是否存在。"""
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"路径不存在：{path_str}")


def build_uhi_file_paths() -> list[str]:
    """构造热岛月表完整路径。"""
    file_paths = [os.path.join(UHI_DIR, file_name) for file_name in UHI_MONTH_FILES]
    for file_path in file_paths:
        ensure_exists(file_path)
    return file_paths


def build_patch_file_map() -> dict[str, list[str]]:
    """扫描热岛目录中的补丁文件，并按主月表文件名建立索引。"""
    patch_file_map: dict[str, list[str]] = {file_name: [] for file_name in UHI_MONTH_FILES}

    for file_name in sorted(os.listdir(UHI_DIR)):
        if not file_name.endswith(".csv") or PATCH_FILE_FLAG not in file_name:
            continue

        for month_file in UHI_MONTH_FILES:
            month_prefix = month_file.replace(".csv", "")
            if file_name.startswith(f"{month_prefix}{PATCH_FILE_FLAG}"):
                patch_path = os.path.join(UHI_DIR, file_name)
                ensure_exists(patch_path)
                patch_file_map[month_file].append(patch_path)
                break

    return patch_file_map


def read_original_columns() -> list[str]:
    """读取原始表头，确定字段顺序。"""
    ensure_exists(ORIGINAL_CSV)
    columns = pd.read_csv(ORIGINAL_CSV, nrows=0).columns.tolist()
    if GRID_ID_COL not in columns:
        raise ValueError(f"原始数据集缺少字段：{GRID_ID_COL}")
    return columns


def extract_static_features(original_columns: list[str]) -> pd.DataFrame:
    """提取每个 Grid_ID 唯一的一份静态特征。"""
    print("\n[1/6] 提取原始数据集中的静态特征...")
    static_cols = [column for column in original_columns if column not in DYNAMIC_COLS]
    usecols = list(dict.fromkeys([GRID_ID_COL, *static_cols]))

    static_parts: list[pd.DataFrame] = []
    seen_grid_ids: set[int] = set()
    total_rows = 0

    for chunk_index, chunk in enumerate(
        pd.read_csv(ORIGINAL_CSV, usecols=usecols, chunksize=ORIGINAL_CHUNK_SIZE),
        start=1,
    ):
        total_rows += len(chunk)
        chunk[GRID_ID_COL] = chunk[GRID_ID_COL].astype(int)
        chunk = chunk.drop_duplicates(subset=[GRID_ID_COL], keep="first")
        chunk = chunk[~chunk[GRID_ID_COL].isin(seen_grid_ids)].copy()

        if not chunk.empty:
            seen_grid_ids.update(chunk[GRID_ID_COL].tolist())
            static_parts.append(chunk)

        print(f"  [{chunk_index}] 已扫描原始数据 {total_rows:,} 行，已提取静态 Grid_ID {len(seen_grid_ids):,}")

    if not static_parts:
        raise ValueError("未能从原始数据集中提取任何静态特征")

    static_df = pd.concat(static_parts, ignore_index=True)
    static_df = static_df.drop_duplicates(subset=[GRID_ID_COL], keep="first")
    print(f"  静态特征提取完成：{len(static_df):,} 个 Grid_ID")
    return static_df


def read_coverage_report() -> dict[str, int]:
    """读取热岛覆盖报告，获取每个月理论应有的北京时间小时数。"""
    coverage_path = os.path.join(UHI_DIR, "UHI_Month_Coverage_Report.csv")
    ensure_exists(coverage_path)

    coverage_df = pd.read_csv(coverage_path)
    required_cols = {"Month_BJT", "Date_Count"}
    if not required_cols.issubset(set(coverage_df.columns)):
        raise ValueError(f"覆盖报告缺少必要字段：{sorted(required_cols)}")

    expected_rows_by_month: dict[str, int] = {}
    for _, row in coverage_df.iterrows():
        month_bjt = str(row["Month_BJT"])
        expected_rows_by_month[month_bjt] = int(row["Date_Count"]) * 24

    return expected_rows_by_month


def validate_uhi_columns(file_path: str) -> None:
    """检查热岛月表字段完整性。"""
    header_cols = set(pd.read_csv(file_path, nrows=0).columns.tolist())
    missing_cols = [column for column in UHI_REQUIRED_COLS if column not in header_cols]
    if missing_cols:
        raise ValueError(f"文件 {file_path} 缺少字段：{missing_cols}")


def collect_all_uhi_grid_ids(file_paths: list[str]) -> set[int]:
    """扫描四个月热岛月表，收集实际出现过的 Grid_ID。"""
    print("\n[2/6] 扫描热岛月表中的 Grid_ID 范围...")
    observed_grid_ids: set[int] = set()

    for file_path in file_paths:
        rows = 0
        for chunk_index, chunk in enumerate(
            pd.read_csv(file_path, usecols=[GRID_ID_COL], chunksize=UHI_CHUNK_SIZE),
            start=1,
        ):
            chunk[GRID_ID_COL] = chunk[GRID_ID_COL].astype(int)
            observed_grid_ids.update(chunk[GRID_ID_COL].unique().tolist())
            rows += len(chunk)
            print(f"  {os.path.basename(file_path)} [{chunk_index}] 已扫描 {rows:,} 行，累计 Grid_ID {len(observed_grid_ids):,}")

    return observed_grid_ids


def load_patch_slot_maps(
    patch_paths: list[str],
) -> tuple[dict[datetime, pd.DataFrame], dict[datetime, dict[str, object]]]:
    """读取单月补丁文件，构建按北京时间时次索引的补丁映射。"""
    patch_slot_df_map: dict[datetime, pd.DataFrame] = {}
    patch_slot_meta_map: dict[datetime, dict[str, object]] = {}

    for patch_path in patch_paths:
        patch_df = pd.read_csv(patch_path, usecols=UHI_REQUIRED_COLS)
        if patch_df.empty:
            continue

        patch_df = transform_uhi_chunk(patch_df)
        for (date_bjt, hour_bjt), group in patch_df.groupby(["Date_BJT", "Hour_BJT"], sort=False):
            slot_dt = slot_dt_from_values(date_bjt, int(hour_bjt))
            slot_meta = {
                "Date_BJT": date_bjt,
                "Hour": int(hour_bjt),
                "Hour_sin": math.sin(2 * math.pi * int(hour_bjt) / 24),
                "Hour_cos": math.cos(2 * math.pi * int(hour_bjt) / 24),
                "Season": str(group["Season"].iloc[0]),
                "DateTime_BJT": str(group["DateTime_BJT"].iloc[0]),
                "Date_UTC": str(group["Date_UTC"].iloc[0]),
                "Hour_UTC": int(group["Hour_UTC"].iloc[0]),
                "Month_BJT": str(group["Month_BJT"].iloc[0]),
                "Source_TIF": str(group["Source_TIF"].iloc[0]),
            }

            if slot_dt in patch_slot_df_map:
                patch_slot_df_map[slot_dt] = pd.concat(
                    [patch_slot_df_map[slot_dt], group],
                    ignore_index=True,
                )
            else:
                patch_slot_df_map[slot_dt] = group.copy()
            patch_slot_meta_map[slot_dt] = slot_meta

    return patch_slot_df_map, patch_slot_meta_map


def overlay_patch_rows_for_slot(
    slot_dt: datetime,
    slot_df: pd.DataFrame | None,
    slot_meta: dict[str, object] | None,
    patch_slot_df_map: dict[datetime, pd.DataFrame],
    patch_slot_meta_map: dict[datetime, dict[str, object]],
) -> tuple[pd.DataFrame | None, dict[str, object] | None]:
    """将同一时次的补丁记录覆盖到主月表结果上。"""
    patch_df = patch_slot_df_map.get(slot_dt)
    patch_meta = patch_slot_meta_map.get(slot_dt)

    if patch_df is not None and not patch_df.empty:
        if slot_df is None or slot_df.empty:
            slot_df = patch_df.copy()
        else:
            # 主月表在前，补丁在后，后续 keep="last" 时即可让补丁值生效。
            slot_df = pd.concat([slot_df, patch_df], ignore_index=True)

    if slot_meta is None and patch_meta is not None:
        slot_meta = patch_meta.copy()

    return slot_df, slot_meta


def build_canonical_static_df(static_df: pd.DataFrame, observed_grid_ids: set[int]) -> pd.DataFrame:
    """构造整合时使用的标准 Grid_ID 骨架。"""
    static_df = static_df.copy()
    static_df[GRID_ID_COL] = static_df[GRID_ID_COL].astype(int)
    static_df = static_df.drop_duplicates(subset=[GRID_ID_COL], keep="first")
    static_df = static_df.set_index(GRID_ID_COL, drop=True)

    canonical_grid_ids = sorted(set(static_df.index.astype(int).tolist()) | observed_grid_ids)
    canonical_static_df = static_df.reindex(canonical_grid_ids)
    canonical_static_df.index.name = GRID_ID_COL

    print(f"  原始静态 Grid_ID 数量：{len(static_df):,}")
    print(f"  热岛观测 Grid_ID 数量：{len(observed_grid_ids):,}")
    print(f"  最终整合骨架 Grid_ID 数量：{len(canonical_static_df):,}")
    return canonical_static_df


def build_final_columns(original_columns: list[str]) -> list[str]:
    """构建最终输出列顺序。"""
    final_cols = original_columns.copy()
    for column in EXTRA_OUTPUT_COLS:
        if column not in final_cols:
            final_cols.append(column)
    return final_cols


def transform_uhi_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """将热岛逐时表转换为兼容原表字段命名的结构。"""
    chunk = chunk.copy()
    chunk[GRID_ID_COL] = chunk[GRID_ID_COL].astype(int)
    chunk["Hour_BJT"] = chunk["Hour_BJT"].astype(int)
    chunk["Hour"] = chunk["Hour_BJT"]

    chunk["Hour_sin"] = chunk["Hour"].map(lambda hour: math.sin(2 * math.pi * hour / 24))
    chunk["Hour_cos"] = chunk["Hour"].map(lambda hour: math.cos(2 * math.pi * hour / 24))
    chunk["Temperature"] = chunk["UHI_Intensity"]
    chunk["Rural_Temp"] = chunk["Bg_Temp_Rural_Celsius"]
    chunk["Temp_2m"] = chunk["Temp_2m_Celsius"]
    chunk["Season"] = chunk["Season"].astype(str)
    return chunk


def build_expected_slots(month_bjt: str) -> list[datetime]:
    """按北京时间自然月构造该月每一个小时的完整时次骨架。"""
    year = int(month_bjt[:4])
    month = int(month_bjt[5:7])
    start_dt = datetime(year, month, 1, 0, 0)

    if month == 12:
        next_month_dt = datetime(year + 1, 1, 1, 0, 0)
    else:
        next_month_dt = datetime(year, month + 1, 1, 0, 0)

    expected_slots: list[datetime] = []
    current_dt = start_dt
    while current_dt < next_month_dt:
        expected_slots.append(current_dt)
        current_dt += timedelta(hours=1)
    return expected_slots


def slot_dt_from_values(date_bjt: str, hour_bjt: int) -> datetime:
    """将日期和小时拼成北京时间时次。"""
    return datetime.strptime(f"{date_bjt} {hour_bjt:02d}", "%Y-%m-%d %H")


def make_empty_slot_meta(slot_dt: datetime) -> dict[str, object]:
    """当整个时次缺失时，为该时次构造最小元数据。"""
    utc_dt = slot_dt - timedelta(hours=8)
    return {
        "Date_BJT": slot_dt.strftime("%Y-%m-%d"),
        "Hour": slot_dt.hour,
        "Hour_sin": math.sin(2 * math.pi * slot_dt.hour / 24),
        "Hour_cos": math.cos(2 * math.pi * slot_dt.hour / 24),
        "Season": "summer" if slot_dt.month in {7, 8} else "winter",
        "DateTime_BJT": slot_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Date_UTC": utc_dt.strftime("%Y-%m-%d"),
        "Hour_UTC": utc_dt.hour,
        "Month_BJT": slot_dt.strftime("%Y-%m"),
        "Source_TIF": pd.NA,
    }


def finalize_slot_output(
    slot_dt: datetime,
    slot_df: pd.DataFrame | None,
    slot_meta: dict[str, object] | None,
    canonical_static_reset: pd.DataFrame,
    canonical_grid_ids: list[int],
    final_columns: list[str],
) -> pd.DataFrame:
    """将单个北京时间时次补齐成完整 Grid_ID 骨架，并与静态特征拼接。"""
    output_df = canonical_static_reset.copy()

    # 先填充整槽固定的时间元数据。
    if slot_meta is None:
        slot_meta = make_empty_slot_meta(slot_dt)

    output_df["Hour"] = int(slot_meta["Hour"])
    output_df["Hour_sin"] = float(slot_meta["Hour_sin"])
    output_df["Hour_cos"] = float(slot_meta["Hour_cos"])
    output_df["Season"] = slot_meta["Season"]
    output_df["Date_BJT"] = slot_meta["Date_BJT"]
    output_df["DateTime_BJT"] = slot_meta["DateTime_BJT"]
    output_df["Date_UTC"] = slot_meta["Date_UTC"]
    output_df["Hour_UTC"] = slot_meta["Hour_UTC"]
    output_df["Month_BJT"] = slot_meta["Month_BJT"]
    output_df["Source_TIF"] = slot_meta["Source_TIF"]

    # 默认把热岛动态值全部置为空，后续仅用真实观测覆盖。
    output_df["Temperature"] = pd.NA
    output_df["Rural_Temp"] = pd.NA
    output_df["Temp_2m"] = pd.NA
    output_df["T_Grid_Raw"] = pd.NA

    if slot_df is not None and not slot_df.empty:
        # 若主月表与补丁同时包含同一 Grid_ID，则优先保留补丁值。
        slot_df = slot_df.drop_duplicates(subset=[GRID_ID_COL], keep="last")
        aligned_df = slot_df.set_index(GRID_ID_COL).reindex(canonical_grid_ids)

        # 用真实值覆盖缺省空值，这样缺测网格会自然保留为 NaN。
        output_df["Temperature"] = aligned_df["Temperature"].to_numpy()
        output_df["Rural_Temp"] = aligned_df["Rural_Temp"].to_numpy()
        output_df["Temp_2m"] = aligned_df["Temp_2m"].to_numpy()
        output_df["T_Grid_Raw"] = aligned_df["T_Grid_Raw"].to_numpy()

    return output_df[final_columns]


def flush_one_slot(
    slot_dt: datetime,
    slot_df: pd.DataFrame | None,
    slot_meta: dict[str, object] | None,
    canonical_static_reset: pd.DataFrame,
    canonical_grid_ids: list[int],
    final_columns: list[str],
    row_counter: dict[int, int],
    output_path: str,
    is_first_write: bool,
) -> bool:
    """将一个北京时间时次补齐并写入总表。"""
    output_df = finalize_slot_output(
        slot_dt=slot_dt,
        slot_df=slot_df,
        slot_meta=slot_meta,
        canonical_static_reset=canonical_static_reset,
        canonical_grid_ids=canonical_grid_ids,
        final_columns=final_columns,
    )

    output_df.to_csv(
        output_path,
        mode="w" if is_first_write else "a",
        index=False,
        header=is_first_write,
        encoding="utf-8-sig",
    )

    for grid_id in canonical_grid_ids:
        row_counter[grid_id] += 1

    return False


def merge_and_write_monthly_file(
    file_path: str,
    month_bjt: str,
    patch_paths: list[str],
    canonical_static_df: pd.DataFrame,
    final_columns: list[str],
    row_counter: dict[int, int],
    is_first_write: bool,
) -> bool:
    """按月流式处理热岛结果，并自动补齐缺失网格/缺失时次。"""
    print(f"\n[4/6] 开始整合 {month_bjt} ...")
    expected_slots = build_expected_slots(month_bjt)
    canonical_grid_ids = canonical_static_df.index.astype(int).tolist()
    canonical_static_reset = canonical_static_df.reset_index()
    patch_slot_df_map, patch_slot_meta_map = load_patch_slot_maps(patch_paths)
    print(f"  {month_bjt} 已识别补丁文件 {len(patch_paths)} 个，补丁时次 {len(patch_slot_df_map)} 个")

    expected_slot_index = 0
    observed_slot_count = 0
    flushed_slot_count = 0

    current_slot_dt: datetime | None = None
    current_slot_parts: list[pd.DataFrame] = []
    current_slot_meta: dict[str, object] | None = None

    for chunk_index, chunk in enumerate(
        pd.read_csv(file_path, usecols=UHI_REQUIRED_COLS, chunksize=UHI_CHUNK_SIZE),
        start=1,
    ):
        chunk = transform_uhi_chunk(chunk)

        for (date_bjt, hour_bjt), group in chunk.groupby(["Date_BJT", "Hour_BJT"], sort=False):
            slot_dt = slot_dt_from_values(date_bjt, int(hour_bjt))
            slot_meta = {
                "Date_BJT": date_bjt,
                "Hour": int(hour_bjt),
                "Hour_sin": math.sin(2 * math.pi * int(hour_bjt) / 24),
                "Hour_cos": math.cos(2 * math.pi * int(hour_bjt) / 24),
                "Season": str(group["Season"].iloc[0]),
                "DateTime_BJT": str(group["DateTime_BJT"].iloc[0]),
                "Date_UTC": str(group["Date_UTC"].iloc[0]),
                "Hour_UTC": int(group["Hour_UTC"].iloc[0]),
                "Month_BJT": str(group["Month_BJT"].iloc[0]),
                "Source_TIF": str(group["Source_TIF"].iloc[0]),
            }

            if current_slot_dt is None:
                current_slot_dt = slot_dt
                current_slot_parts = [group]
                current_slot_meta = slot_meta
                continue

            if slot_dt == current_slot_dt:
                current_slot_parts.append(group)
                continue

            # 在当前观测槽位到来之前，先把所有应有但缺失的时次槽补齐。
            while expected_slot_index < len(expected_slots) and expected_slots[expected_slot_index] < current_slot_dt:
                missing_slot_dt = expected_slots[expected_slot_index]
                missing_slot_df, missing_slot_meta = overlay_patch_rows_for_slot(
                    slot_dt=missing_slot_dt,
                    slot_df=None,
                    slot_meta=None,
                    patch_slot_df_map=patch_slot_df_map,
                    patch_slot_meta_map=patch_slot_meta_map,
                )
                is_first_write = flush_one_slot(
                    slot_dt=missing_slot_dt,
                    slot_df=missing_slot_df,
                    slot_meta=missing_slot_meta,
                    canonical_static_reset=canonical_static_reset,
                    canonical_grid_ids=canonical_grid_ids,
                    final_columns=final_columns,
                    row_counter=row_counter,
                    output_path=OUTPUT_CSV,
                    is_first_write=is_first_write,
                )
                expected_slot_index += 1
                flushed_slot_count += 1

            if expected_slot_index >= len(expected_slots) or expected_slots[expected_slot_index] != current_slot_dt:
                raise ValueError(f"{month_bjt} 出现了不在预期北京时间骨架中的时次：{current_slot_dt}")
            merged_slot_df = pd.concat(current_slot_parts, ignore_index=True)
            merged_slot_df, merged_slot_meta = overlay_patch_rows_for_slot(
                slot_dt=current_slot_dt,
                slot_df=merged_slot_df,
                slot_meta=current_slot_meta,
                patch_slot_df_map=patch_slot_df_map,
                patch_slot_meta_map=patch_slot_meta_map,
            )
            is_first_write = flush_one_slot(
                slot_dt=current_slot_dt,
                slot_df=merged_slot_df,
                slot_meta=merged_slot_meta,
                canonical_static_reset=canonical_static_reset,
                canonical_grid_ids=canonical_grid_ids,
                final_columns=final_columns,
                row_counter=row_counter,
                output_path=OUTPUT_CSV,
                is_first_write=is_first_write,
            )
            expected_slot_index += 1
            observed_slot_count += 1
            flushed_slot_count += 1

            current_slot_dt = slot_dt
            current_slot_parts = [group]
            current_slot_meta = slot_meta

        print(f"  [{chunk_index}] {month_bjt} 已处理观测时次 {observed_slot_count} 个，已写出槽位 {flushed_slot_count} 个")

    # 处理文件结束时仍在缓冲中的最后一个时次槽。
    if current_slot_dt is not None:
        while expected_slot_index < len(expected_slots) and expected_slots[expected_slot_index] < current_slot_dt:
            missing_slot_dt = expected_slots[expected_slot_index]
            missing_slot_df, missing_slot_meta = overlay_patch_rows_for_slot(
                slot_dt=missing_slot_dt,
                slot_df=None,
                slot_meta=None,
                patch_slot_df_map=patch_slot_df_map,
                patch_slot_meta_map=patch_slot_meta_map,
            )
            is_first_write = flush_one_slot(
                slot_dt=missing_slot_dt,
                slot_df=missing_slot_df,
                slot_meta=missing_slot_meta,
                canonical_static_reset=canonical_static_reset,
                canonical_grid_ids=canonical_grid_ids,
                final_columns=final_columns,
                row_counter=row_counter,
                output_path=OUTPUT_CSV,
                is_first_write=is_first_write,
            )
            expected_slot_index += 1
            flushed_slot_count += 1

        if expected_slot_index >= len(expected_slots) or expected_slots[expected_slot_index] != current_slot_dt:
            raise ValueError(f"{month_bjt} 最后一个观测时次不在预期北京时间骨架中：{current_slot_dt}")
        merged_slot_df = pd.concat(current_slot_parts, ignore_index=True)
        merged_slot_df, merged_slot_meta = overlay_patch_rows_for_slot(
            slot_dt=current_slot_dt,
            slot_df=merged_slot_df,
            slot_meta=current_slot_meta,
            patch_slot_df_map=patch_slot_df_map,
            patch_slot_meta_map=patch_slot_meta_map,
        )
        is_first_write = flush_one_slot(
            slot_dt=current_slot_dt,
            slot_df=merged_slot_df,
            slot_meta=merged_slot_meta,
            canonical_static_reset=canonical_static_reset,
            canonical_grid_ids=canonical_grid_ids,
            final_columns=final_columns,
            row_counter=row_counter,
            output_path=OUTPUT_CSV,
            is_first_write=is_first_write,
        )
        expected_slot_index += 1
        observed_slot_count += 1
        flushed_slot_count += 1

    # 将该月剩余未出现的时次槽全部补齐。
    while expected_slot_index < len(expected_slots):
        missing_slot_dt = expected_slots[expected_slot_index]
        missing_slot_df, missing_slot_meta = overlay_patch_rows_for_slot(
            slot_dt=missing_slot_dt,
            slot_df=None,
            slot_meta=None,
            patch_slot_df_map=patch_slot_df_map,
            patch_slot_meta_map=patch_slot_meta_map,
        )
        is_first_write = flush_one_slot(
            slot_dt=missing_slot_dt,
            slot_df=missing_slot_df,
            slot_meta=missing_slot_meta,
            canonical_static_reset=canonical_static_reset,
            canonical_grid_ids=canonical_grid_ids,
            final_columns=final_columns,
            row_counter=row_counter,
            output_path=OUTPUT_CSV,
            is_first_write=is_first_write,
        )
        expected_slot_index += 1
        flushed_slot_count += 1

    print(
        f"  {month_bjt} 整合完成：观测时次 {observed_slot_count} 个，"
        f"最终输出槽位 {flushed_slot_count} 个，理论槽位 {len(expected_slots)} 个"
    )
    return is_first_write


def validate_total_row_counts(row_counter: dict[int, int], expected_total_rows_per_grid: int) -> None:
    """校验最终每个 Grid_ID 的总行数是否完整。"""
    invalid_totals = {
        grid_id: count
        for grid_id, count in row_counter.items()
        if count != expected_total_rows_per_grid
    }
    if invalid_totals:
        sample_items = list(invalid_totals.items())[:10]
        raise ValueError(
            f"整合后存在 Grid_ID 总行数不完整的情况，理论应为 {expected_total_rows_per_grid} 行，"
            f"样例：{sample_items}"
        )


def main() -> None:
    """主流程：提取静态特征、补齐热岛时间骨架并输出总表。"""
    print("=" * 80)
    print("四个月逐日逐时热岛数据与原始长表数据集整合脚本")
    print(f"ORIGINAL_CSV : {ORIGINAL_CSV}")
    print(f"UHI_DIR      : {UHI_DIR}")
    print(f"OUTPUT_CSV   : {OUTPUT_CSV}")
    print("=" * 80)

    ensure_exists(ORIGINAL_CSV)
    ensure_exists(UHI_DIR)
    uhi_file_paths = build_uhi_file_paths()
    patch_file_map = build_patch_file_map()
    original_columns = read_original_columns()
    final_columns = build_final_columns(original_columns)

    static_df = extract_static_features(original_columns)
    expected_rows_by_month = read_coverage_report()

    print("\n[3/6] 检查热岛月表字段并构建统一 Grid_ID 骨架...")
    for file_path in uhi_file_paths:
        validate_uhi_columns(file_path)
        print(f"  已检查字段：{os.path.basename(file_path)}")

    all_uhi_sources = uhi_file_paths + [
        patch_path
        for patch_paths in patch_file_map.values()
        for patch_path in patch_paths
    ]
    observed_grid_ids = collect_all_uhi_grid_ids(all_uhi_sources)
    canonical_static_df = build_canonical_static_df(static_df, observed_grid_ids)

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    row_counter: dict[int, int] = defaultdict(int)
    is_first_write = True

    for file_path in uhi_file_paths:
        month_bjt = os.path.basename(file_path).replace("UHI_Long_BJT_", "").replace(".csv", "").replace("_", "-")
        if month_bjt not in expected_rows_by_month:
            raise ValueError(f"覆盖报告中未找到月份：{month_bjt}")

        is_first_write = merge_and_write_monthly_file(
            file_path=file_path,
            month_bjt=month_bjt,
            patch_paths=patch_file_map.get(os.path.basename(file_path), []),
            canonical_static_df=canonical_static_df,
            final_columns=final_columns,
            row_counter=row_counter,
            is_first_write=is_first_write,
        )

    print("\n[5/6] 校验四个月整合后的总小时覆盖...")
    expected_total_rows_per_grid = sum(expected_rows_by_month.values())
    validate_total_row_counts(row_counter, expected_total_rows_per_grid)
    print(f"  每个 Grid_ID 总行数均为 {expected_total_rows_per_grid}")

    print("\n[6/6] 输出结果汇总...")
    print(f"  已输出文件：{OUTPUT_CSV}")
    print(f"  输出 Grid_ID 数量：{len(row_counter):,}")
    print("  缺失网格或缺失时次已自动补齐为 NaN")
    print("整合完成。")


if __name__ == "__main__":
    main()
