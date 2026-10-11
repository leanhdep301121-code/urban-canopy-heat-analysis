[English](README.md) | 简体中文

# 城市冠层热岛分析代码

论文 **"Predictive importance diverges from graph-adjusted associations with canopy heat islands across China"**（预测重要性与图调整关联在中国城市冠层热岛中的分歧）所用代码。

本仓库**原样存档**作者提供的分析脚本（保留原始文件名与本地输入/输出路径；`source_manifest.json` 记录各文件哈希）。脚本目前尚非一键复现包：运行前需将输入/输出路径指向你自己的数据位置。

## 文件说明

| 文件 | 作用 |
|---|---|
| `code/analysis_original.py` | 主流程：样本构建、八模型比较（保留 LightGBM）、SHAP 归因、DirectLiNGAM 图发现、DoWhy 图调整估计 |
| `code/svf_saga_original.py` | 论文 SVF 诊断所用的 SAGA 天空开敞度计算（16 个搜索方向、10,000 m 搜索半径） |
| `code/building_metrics_original.py` | 由建筑高度栅格计算建筑覆盖率与建筑高度指标 |
| `code/road_metrics_original.py` | 由 OpenStreetMap 计算道路密度与分级宽度代理 |
| `code/merge_hourly_original.py` | 逐时网格数据集组装 |
| `code/merge_seasonal_original.py` | 季节数据集组装 |
| `source_manifest.json` | 各存档脚本的 SHA-256 与角色 |
| `import_inventory.json` | 各脚本的导入模块清单 |

## 说明

- 分析源码中还包含探索性模型分支（贝叶斯岭回归、xRFM、K 近邻），它们**不属于**论文报告的八模型比较。
- 分析环境为 Python 3.13 与现行包版本（包括 scikit-learn 1.7.1、numpy 2.3.2、scipy 1.16.1、xgboost 3.1.0）；完整环境规格见 `requirements.txt`。
- 仓库不含任何商业气温数据（天机 TJ-CN）及其派生温度场，这些数据受供应商许可条件约束。图件源数据见期刊 Source Data 文件及版本化 Zenodo 存档。

## 许可证

MIT — 见 [LICENSE](LICENSE)。
