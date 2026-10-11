import os
import subprocess
import rasterio
from rasterio.windows import Window
from osgeo import gdal
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

# ================= 🔧 核心配置战情室 🔧 =================
# 1. SAGA 命令行路径
SAGA_CMD = r"C:\Users\Yinaoqi\Desktop\saga-9.11.1_msw\saga_cmd.exe"

# 2. 输入：你的原始大文件
INPUT_RAW_FILE = r"D:\DSM\Overlay_Result_3072.tif"

# 3. 中间文件：重投影后的临时大文件 (脚本会自动生成)
# 建议放在 SSD 上
PROJECTED_FILE = r"F:\Temp_SAGA_Work\China_DSM_Albers_Native.tif"

# 4. 输出：SVF 最终结果文件夹
OUTPUT_DIR = r"E:\DSM\Full_China_SVF_Result_Native"
TEMP_TILE_DIR = os.path.join(OUTPUT_DIR, "Temp_Tiles") 

# 5. 重投影参数
TARGET_CRS = "EPSG:102028"  # 中国 Albers 投影
# ⚠️ 注意：这里不再设置 TARGET_RES，让 GDAL 自动保持原始分辨率

# 6. SVF 计算参数
TILE_SIZE = 3072    # 切片大小
BUFFER = 200        # 缓冲区
SEARCH_RADIUS = 10000 # SVF 搜索半径
MAX_WORKERS = 12     # 并行核心数
# ========================================================

def step1_reproject_native():
    """
    阶段一：重投影 (保持原始像元大小)
    """
    print("🌊 [阶段 1/2] 正在进行重投影 (保持原始分辨率)...")
    print(f"   目标坐标系: {TARGET_CRS}")
    print(f"   输出文件: {PROJECTED_FILE}")

    if os.path.exists(PROJECTED_FILE):
        print("   ⚠️ 投影文件已存在，跳过。")
        return

    os.makedirs(os.path.dirname(PROJECTED_FILE), exist_ok=True)

    # 使用 GDAL Warp
    # 【修改点】：去掉了 xRes 和 yRes 参数
    # GDAL 会自动计算 Output Size 以匹配输入文件的细节程度
    ds = gdal.Warp(
        PROJECTED_FILE,
        INPUT_RAW_FILE,
        format="GTiff",
        dstSRS=TARGET_CRS,
        # xRes, yRes 已移除，启用自动计算
        resampleAlg=gdal.GRA_NearestNeighbour, # 依然必须用最近邻，防止建筑变形
        dstNodata=-9999,
        creationOptions=[
            "COMPRESS=LZW", 
            "BIGTIFF=YES", 
            "TILED=YES", 
            "NUM_THREADS=ALL_CPUS"
        ],
        multithread=True,
        callback=gdal.TermProgress_nocb
    )
    ds = None
    print("   ✅ 重投影完成！GDAL 已自动匹配最佳分辨率。")

def calculate_single_tile(args):
    """
    单个切片的计算逻辑 (与之前一致)
    """
    col, row, width, height, src_profile, input_tif_path = args
    
    tile_id = f"{col}_{row}"
    temp_in = os.path.join(TEMP_TILE_DIR, f"in_{tile_id}.tif")
    temp_out = os.path.join(TEMP_TILE_DIR, f"out_{tile_id}.tif")
    final_out = os.path.join(OUTPUT_DIR, f"SVF_{tile_id}.tif")

    if os.path.exists(final_out):
        return final_out 

    try:
        with rasterio.open(input_tif_path) as src:
            # 1. 计算窗口
            read_col = col - BUFFER
            read_row = row - BUFFER
            read_w = TILE_SIZE + 2 * BUFFER
            read_h = TILE_SIZE + 2 * BUFFER
            
            window = Window(read_col, read_row, read_w, read_h)
            data = src.read(1, window=window, boundless=True, fill_value=src.nodata)
            
            # 跳过无效块
            if src.nodata is not None:
                if np.all(data == src.nodata): return None
            elif np.all(data == 0): return None

            # 2. 保存临时输入
            win_transform = src.window_transform(window)
            profile = src_profile.copy()
            profile.update({
                'height': read_h, 'width': read_w,
                'transform': win_transform,
                'compress': 'lzw', 'driver': 'GTiff', 'bigtiff': 'NO'
            })
            
            with rasterio.open(temp_in, 'w', **profile) as dst:
                dst.write(data, 1)

        # 3. 调用 SAGA CMD
        cmd = [
            SAGA_CMD,
            "ta_lighting", "2", 
            "-DEM", temp_in,
            "-SVF", temp_out,
            "-RADIUS", str(SEARCH_RADIUS),
            "-NDIRS", "16",
            "-METHOD", "1"
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        # 4. 去边并保存
        if os.path.exists(temp_out):
            with rasterio.open(temp_out) as src_res:
                valid_w = min(TILE_SIZE, width - col)
                valid_h = min(TILE_SIZE, height - row)
                
                result_data = src_res.read(1, window=Window(BUFFER, BUFFER, valid_w, valid_h))
                
                with rasterio.open(input_tif_path) as big_src:
                     final_transform = big_src.window_transform(Window(col, row, valid_w, valid_h))

                final_profile = profile.copy()
                final_profile.update({
                    'height': valid_h, 'width': valid_w,
                    'transform': final_transform,
                    'dtype': 'float32', 'nodata': -99999
                })
                
                with rasterio.open(final_out, 'w', **final_profile) as dst_final:
                    dst_final.write(result_data, 1)
        
        return final_out

    except Exception as e:
        return None
    finally:
        if os.path.exists(temp_in): os.remove(temp_in)
        if os.path.exists(temp_out): os.remove(temp_out)

def step2_calculate_svf():
    """
    阶段二：流式计算
    """
    print("\n🚀 [阶段 2/2] 启动全图 SVF 计算引擎...")
    
    if not os.path.exists(PROJECTED_FILE):
        print("❌ 找不到投影文件！")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_TILE_DIR, exist_ok=True)

    tasks = []
    with rasterio.open(PROJECTED_FILE) as src:
        width = src.width
        height = src.height
        profile = src.profile.copy()
        # 打印一下 GDAL 自动算出来的分辨率
        res_x, res_y = src.res
        print(f"   数据全貌: {width}x{height} | {src.crs}")
        print(f"   ✅ 当前像元大小: {res_x:.4f} x {abs(res_y):.4f} (原始精度)")

        for col in range(0, width, TILE_SIZE):
            for row in range(0, height, TILE_SIZE):
                tasks.append((col, row, width, height, profile, PROJECTED_FILE))
    
    print(f"   📦 总切片数: {len(tasks)}")
    print(f"   🔥 并行核心: {MAX_WORKERS}")

    created_files = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(calculate_single_tile, tasks), total=len(tasks), unit="tile"))
        for res in results:
            if res: created_files.append(res)

    try:
        shutil.rmtree(TEMP_TILE_DIR)
    except: pass

    print("   🧩 构建 VRT...")
    vrt_path = os.path.join(OUTPUT_DIR, "China_Full_SVF.vrt")
    gdal.BuildVRT(vrt_path, created_files)
    
    print("-" * 30)
    print(f"✅ 完成！结果在: {vrt_path}")

def main():
    if not os.path.exists(SAGA_CMD):
        print("❌ 找不到 saga_cmd.exe！")
        return
    step1_reproject_native()
    step2_calculate_svf()

if __name__ == "__main__":
    main()