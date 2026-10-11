"""
城市形态指标计算脚本
根据建筑高度TIF和渔网SHP计算2D和3D UFPs指标
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 配置参数
# ==========================================
# 输入文件路径
BUILDING_HEIGHT_TIF = r"H:\数据集\106城市建筑矢量\nDSM\city_building_collection2.tif"  # 修改为你的建筑高度TIF路径
FISHNET_SHP = r"H:\数据集\106城市建筑矢量\Fishnet\Fishnet.shp"  # 修改为你的渔网SHP路径

# 输出文件路径
OUTPUT_CSV = r'H:\数据集\城市形态指标结果.csv'

# 空间单元大小（米）
GRID_SIZE = 1000  # 30m × 30m
PIXEL_SIZE = 10  # TIF数据的像元大小（米），根据实际情况修改

# 高层建筑阈值（米）
HIGH_RISE_THRESHOLD = 24  # 高于24米视为高层建筑

print("=" * 60)
print("城市形态指标计算工具")
print("=" * 60)

# ==========================================
# 1. 读取数据
# ==========================================
print("\n[1/5] 读取数据...")

# 读取渔网
fishnet = gpd.read_file(FISHNET_SHP)
print(f"   渔网格子数量: {len(fishnet)}")
print(f"   渔网CRS: {fishnet.crs}")

# 打开TIF但不读取到内存（避免内存溢出）
tif_src = rasterio.open(BUILDING_HEIGHT_TIF)
print(f"   TIF形状: {tif_src.shape}")
print(f"   TIF CRS: {tif_src.crs}")
print(f"   ✅ 使用按需读取模式，节省内存")

# 确保CRS一致
if fishnet.crs != tif_src.crs:
    print(f"   ⚠️ CRS不一致，正在转换渔网CRS...")
    fishnet = fishnet.to_crs(tif_src.crs)

# ==========================================
# 2. 定义指标计算函数
# ==========================================
print("\n[2/5] 定义计算函数...")

def calculate_2d_3d_metrics(grid_geom, tif_source, pixel_size, grid_size):
    """
    从TIF中裁剪单个格子并计算指标

    参数:
        grid_geom: 渔网格子的几何形状
        tif_source: 打开的TIF数据源
        pixel_size: 像元大小（米）
        grid_size: 格子大小（米）

    返回:
        dict: 包含所有指标的字典
    """
    try:
        # 从TIF中裁剪当前格子（只读取这个格子的数据）
        from rasterio.mask import mask as rio_mask

        out_image, out_transform = rio_mask(tif_source, [grid_geom], crop=True, filled=False)
        grid_heights = out_image[0]

        # 处理masked array
        if hasattr(grid_heights, 'filled'):
            grid_heights = grid_heights.filled(0)

        # 处理NoData和负值
        grid_heights[grid_heights < 0] = 0
        grid_heights[np.isnan(grid_heights)] = 0

        # 有效高度
        valid_heights = grid_heights[grid_heights > 0]

        # 像元和格子面积
        pixel_area = pixel_size ** 2
        grid_area = grid_size ** 2

        # 建筑像元数
        building_pixels = np.sum(grid_heights > 0)

        # 初始化结果
        metrics = {}

        # BC - 建筑覆盖率
        building_area = building_pixels * pixel_area
        metrics['BC'] = (building_area / grid_area) * 100

        # HBC - 高层建筑覆盖率
        high_rise_pixels = np.sum(grid_heights > HIGH_RISE_THRESHOLD)
        high_rise_area = high_rise_pixels * pixel_area
        metrics['HBC'] = (high_rise_area / grid_area) * 100

        if len(valid_heights) > 0:
            # MBH - 平均建筑高度
            metrics['MBH'] = float(np.mean(valid_heights))

            # MBV - 平均建筑体积
            total_volume = np.sum(valid_heights) * pixel_area
            metrics['MBV'] = total_volume / grid_area

            # FAR - 容积率
            floor_height = 3
            total_floor_area = np.sum(valid_heights / floor_height) * pixel_area
            metrics['FAR'] = total_floor_area / grid_area
        else:
            metrics['MBH'] = 0
            metrics['MBV'] = 0
            metrics['FAR'] = 0

        # 暂未实现的指标
        metrics['CAR'] = 0
        metrics['SVF'] = 0
        metrics['MTH'] = 0
        metrics['FAI'] = 0
        metrics['CBC'] = 0

        return metrics

    except Exception as e:
        return {
            'BC': 0, 'HBC': 0, 'MBH': 0, 'MBV': 0,
            'FAR': 0, 'CAR': 0, 'SVF': 0, 'MTH': 0,
            'FAI': 0, 'CBC': 0
        }

# ==========================================
# 3. 批量计算所有格子的指标
# ==========================================
print("\n[3/5] 开始计算指标...")

results = []

for idx, row in tqdm(fishnet.iterrows(), total=len(fishnet), desc="计算进度"):
    # 计算指标（直接从TIF读取，不占用大量内存）
    metrics = calculate_2d_3d_metrics(
        row.geometry,
        tif_src,
        PIXEL_SIZE,
        GRID_SIZE
    )

    # 使用渔网中的Grid_ID
    if 'Grid_ID' in fishnet.columns:
        metrics['Grid_ID'] = row['Grid_ID']
    else:
        metrics['Grid_ID'] = idx  # 如果没有Grid_ID列，使用索引

    # 如果渔网有其他属性，也可以添加
    if 'City_Name' in fishnet.columns:
        metrics['City_Name'] = row['City_Name']

    results.append(metrics)

# 关闭TIF文件
tif_src.close()

# ==========================================
# 4. 整理结果
# ==========================================
print("\n[4/5] 整理结果...")

df_results = pd.DataFrame(results)

# 调整列顺序
cols_order = ['Grid_ID']
if 'City_Name' in df_results.columns:
    cols_order.append('City_Name')
cols_order.extend(['BC', 'HBC', 'MBH', 'MBV', 'FAR', 'CAR', 'SVF', 'MTH', 'FAI', 'CBC'])

df_results = df_results[cols_order]

# 显示统计信息
print(f"\n计算完成！共处理 {len(df_results)} 个格子")
print("\n指标统计:")
print(df_results[['BC', 'HBC', 'MBH', 'MBV', 'FAR']].describe())

# ==========================================
# 5. 保存结果
# ==========================================
print(f"\n[5/5] 保存结果到: {OUTPUT_CSV}")
df_results.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print("✅ 计算完成！")
print("=" * 60)

