# Planetary Aerodynamic Model-Validity Maps

Research code and data accompanying **Planetary Aerodynamic Model-Validity Maps: A Dimensionless Comparison of Earth, Mars and Titan** (James Taylor, University of Bristol, 2026).

Applies a five-category flat-plate freestream screening hierarchy using NASA GRAM-derived atmospheres, Reynolds, Mach and Knudsen numbers for Earth, Mars and Titan. The repository contains the processed atmospheric inputs, model-selection and plotting scripts, sensitivity results, and all nine figures presented in the research manuscript.

## Quick start

Requires Python 3.10+ with NumPy, SciPy, pandas and Matplotlib. From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_all.py
python scripts/verify_results.py
```

`reproduce_all.py` stages source scripts and inputs in a temporary folder and regenerates Figs. 1–9 as PNG, PDF and SVG in `figures/`, plus numeric CSVs under `results/reproduced/`. Regeneration from the included **processed** reference CSVs does not require NASA GRAM.

## Figure to source map

| Final paper | Script | Details |
|---|---|---|
| Fig. 1 | `scripts/figures/fig01_combined_maps.py` | Three-planet validity maps |
| Fig. 2 | `scripts/figures/fig02_length_effects.py` | Characteristic length effect |
| Fig. 3 | `scripts/figures/fig03_mars_length_maps.py` | Mars L=1 m versus L=10 m |
| Fig. 4 | `scripts/figures/fig04_master_map.py` | Dimensionless master map and trajectories |
| Fig. 5 | `scripts/figures/fig05_analytical_kn.py` | Isothermal versus GRAM Kn |
| Fig. 6 | `scripts/figures/fig06_local_lambda_gradient.py` | Local mean-free-path gradient |
| Fig. 7 | `scripts/figures/fig07_sensitivity_zeros.py` | Nominal sensitivity, zero Mars Re-transition bars |
| Fig. 8 | `scripts/figures/fig08_retransition_sensitivity.py` | Wider laminar-transition cases |
| Fig. 9 | `scripts/figures/fig09_atmospheric_variability.py` | 96 saved crossing cases for Earth/Mars |

The `scripts/gram` directory contains the Earth/Mars reference-atmosphere sampling scripts, viscosity processing, dimensionless CSV generation, and scientific audit scripts. Regenerating GRAM atmospheric profiles requires the separately distributed GRAM executables, their source input templates and auxiliary climatology files. The processed data and source-case results included here reproduce the reported figures without those external GRAM dependencies.

For optional atmospheric resampling (which requires a separately installed GRAM Suite and associated input data), the Earth and Mars pipeline scripts accept `--gram-root` and `--out-dir`. The original Titan sampling executable/configuration is not included; the provided nominal Titan reference CSV supports all nine published figures.

## Data

- `data/earth3.csv`, `data/mars3.csv`, `data/titan3.csv`: final processed 1 km atmospheric profiles from 0–2200 km.
- `data/sampling/individual_48profile_Kn_crossings.csv`: 48 Earth and 48 Mars boundary locations and their weights.
- `data/sampling/individual_profile_summary.json`: summary statistics used in Fig. 9.
- `data/nominal_one_at_a_time_sensitivity.csv`: source nominal sensitivity values for Fig. 7.
- Additional input/reference datasets and results are in `data/sampling/` and `results/`.

## Validation boundary

The scripts reproduce the published screening calculations from **processed atmospheric inputs**. They do not independently validate local non-equilibrium, shocks, transitions, turbulent closures, or mission-specific flow physics. Sampling uncertainty is not an established planetary confidence interval. See `docs/PROVENANCE.md` and `results/reproduction_report.json`.

## NASA GRAM

Original GRAM 2024b Earth/Mars and 2020d Titan profiles used GRAM Suite 2.1.1, available separately via https://software.nasa.gov/software/MFS-33888-1 . GRAM proprietary or externally distributed binaries and large auxiliary datasets are not bundled here. Verify the permissions associated with redistributing the processed GRAM outputs in this archive.

## Citation and archive

Citation metadata are provided in `CITATION.cff`. A version-specific archived DOI, when available, can be cited alongside the GitHub repository URL.

## Licensing

No public reuse license is specified for the code or datasets. Public access does not by itself grant redistribution or modification rights. NASA GRAM software and auxiliary files are separately distributed by NASA.
