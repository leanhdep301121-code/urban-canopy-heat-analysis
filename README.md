# Urban canopy heat analysis

Code for the study **"Predictive importance diverges from graph-adjusted associations with canopy heat islands across China"**.

The repository archives the author-supplied analysis scripts **unchanged** (original file names and local input/output paths are preserved; `source_manifest.json` records file hashes). The scripts are not yet a one-command reproduction package: before running, point the input/output paths at your own data locations.

## Contents

| File | Role |
|---|---|
| `code/analysis_original.py` | Main omnibus pipeline: sample construction, eight-model comparison (LightGBM retained), SHAP attribution, DirectLiNGAM graph discovery, DoWhy graph-adjusted estimation |
| `code/svf_saga_original.py` | SAGA sky-view-factor computation used for the paper's SVF diagnostic (16 search directions, 10,000 m search radius) |
| `code/building_metrics_original.py` | Building coverage and height metrics from the building-height raster |
| `code/road_metrics_original.py` | Road density and class-assigned width proxies from OpenStreetMap |
| `code/merge_hourly_original.py` | Hourly grid-level dataset assembly |
| `code/merge_seasonal_original.py` | Seasonal dataset assembly |
| `source_manifest.json` | SHA-256 and role of each archived script |
| `import_inventory.json` | Imported modules per script |

## Notes

- The analysis source also contains exploratory model branches (Bayesian ridge, xRFM, k-nearest-neighbour models) that were **not** part of the eight-model comparison reported in the paper.
- The analysis used Python 3.13 with current package versions (including scikit-learn 1.7.1, numpy 2.3.2, scipy 1.16.1 and xgboost 3.1.0); the full environment specification is provided here: `requirements.txt`.
- No commercial temperature data (TianJi TJ-CN) or temperature-derived fields are included; they are subject to the provider's licensing conditions. Figure source data are available in the journal's Source Data files and a versioned Zenodo archive.

## License

MIT — see [LICENSE](LICENSE).
