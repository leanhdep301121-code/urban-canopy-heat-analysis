import geopandas as gpd
import pandas as pd
import warnings

# 忽略 geopandas 的一些投影警告，保持控制台整洁
warnings.filterwarnings('ignore')

def process_urban_road_metrics(fishnet_path, osm_roads_path, output_path, target_crs=3857):
    """
    计算城市路网的长度密度与不透水面面积占比
    
    参数:
    fishnet_path (str): 输入的渔网 Shapefile 路径
    osm_roads_path (str): 输入的 OSM 道路 Shapefile 路径
    output_path (str): 结果保存的 Shapefile 路径
    target_crs (int): 投影坐标系 EPSG 代码 (默认为 3857 Web Mercator。若需更高精度，请替换为当地 UTM 投影)
    """
    print("开始执行道路空间特征提取工作流...")
    
    # ==========================================
    # 1. 数据读取与投影一致性审查 (First Principles)
    # ==========================================
    print("正在读取矢量数据并执行投影转换...")
    grids = gpd.read_file(fishnet_path)
    roads = gpd.read_file(osm_roads_path)

    # 确保渔网有唯一标识符
    if 'Grid_ID' not in grids.columns:
        grids['Grid_ID'] = range(1, len(grids) + 1)

    # 强制投影转换：将经纬度转换为以“米”为单位的平面投影坐标系，这是物理计算的基础
    grids = grids.to_crs(epsg=target_crs)
    roads = roads.to_crs(epsg=target_crs)

    # 计算网格自身的物理面积并在字段中显示 (单位: 平方米)
    grids['grid_area'] = grids.geometry.area

    # ==========================================
    # 2. 数据清洗与物理宽度映射 (基于 GB 55011-2021)
    # ==========================================
    print("正在依据最新国标进行不透水面宽度映射与数据清洗...")
    # 剥离强制绿化带后的纯硬化铺装宽度字典 (单位: 米)
    width_dict_cn_2024 = {
        'motorway': 28.0,       # 高速路/城市快速路主线
        'trunk': 26.0,          # 快速路辅路/干线
        'primary': 22.0,        # 城市主干路 (扣除隔离带)
        'secondary': 16.0,      # 城市次干路
        'tertiary': 9.0,        # 城市支路
        'residential': 6.0,     # 居住区内部道路
        'unclassified': 6.5,    # 城乡未分类公路
        'living_street': 4.0,   # 街坊路
        'pedestrian': 3.0       # 独立步行街
    }

    # 优先保留已有宽度，缺失值则依据国标字典填充，兜底值为4米
    if 'width' in roads.columns:
        roads['width_numeric'] = pd.to_numeric(roads['width'], errors='coerce')
        roads['width_est'] = roads['width_numeric'].fillna(roads['fclass'].map(width_dict_cn_2024))
    else:
        roads['width_est'] = roads['fclass'].map(width_dict_cn_2024)
    
    roads['width_est'] = roads['width_est'].fillna(4.0)

    # ==========================================
    # 3. 跨网格空间几何切割 (消除边缘效应)
    # ==========================================
    print("正在执行严格的空间几何切割 (Intersection)，请稍候...")
    # 物理切断跨越网格边界的道路线段，防止数值膨胀
    intersected_roads = gpd.overlay(roads, grids[['Grid_ID', 'geometry']], how='intersection')

    # ==========================================
    # 4. 物理量计算与网格聚合
    # ==========================================
    print("正在计算道路长度、密度与加权平均宽度...")
    # 计算切割后路段的实际长度与此路段“长度×宽度”的乘积权重
    intersected_roads['seg_length'] = intersected_roads.geometry.length
    intersected_roads['length_x_width'] = intersected_roads['seg_length'] * intersected_roads['width_est']

    # 按网格聚合求和
    agg_stats = intersected_roads.groupby('Grid_ID').agg(
        tot_length=('seg_length', 'sum'),
        sum_lw=('length_x_width', 'sum') # 用于后续计算加权平均宽度
    ).reset_index()

    # 将统计结果连接回原始渔网
    grids = grids.merge(agg_stats, on='Grid_ID', how='left')

    # 异常值处理：填充无道路网格的空值为0
    grids['tot_length'] = grids['tot_length'].fillna(0)
    grids['sum_lw'] = grids['sum_lw'].fillna(0)

    # 计算核心变量：密度 (km/km² 或者 m/m²) 和 加权平均宽度 (m)
    # 此处 road_dens 单位为: 米(路长) / 平方米(网格面积)
    grids['road_dens'] = grids['tot_length'] / grids['grid_area'] 
    
    # 计算平均宽度：总的(长度*宽度) / 总长度 
    # 若该网格没有路(总长度为0)，则宽度填充为 0
    import numpy as np
    grids['mean_width'] = np.where(grids['tot_length'] > 0, 
                                   grids['sum_lw'] / grids['tot_length'], 
                                   0)

    # ==========================================
    # 5. 结果规范化输出
    # ==========================================
    print(f"正在保存结果至: {output_path}")
    # 整理输出字段，保留必需特征 (总长度, 长度密度, 平均宽度)
    export_columns = ['Grid_ID', 'grid_area', 'tot_length', 'road_dens', 'mean_width', 'geometry']
    export_grids = grids[export_columns]
    
    # 导出文件
    export_grids.to_file(output_path, encoding='utf-8')
    print("处理完成！你可以直接将此数据作为特征送入你的机器学习流水线。")

# ==========================================
# 运行触发区
# ==========================================
if __name__ == "__main__":
    # 请在这里输入你自己的本地文件路径
    # 注意 Windows 路径中的斜杠最好用正斜杠 '/' 或者在前面加 'r'
    USER_FISHNET = r"H:\数据集\106城市建筑矢量\Fishnet\Fishnet.shp"
    USER_ROADS = r"H:\数据集\道路数据\【立方数据学社】全国范围的道路数据\全国道路数据.shp"
    USER_OUTPUT = r"H:\数据集\道路数据\processed_road_metrics.shp"
    
    process_urban_road_metrics(
        fishnet_path=USER_FISHNET,
        osm_roads_path=USER_ROADS,
        output_path=USER_OUTPUT
    )