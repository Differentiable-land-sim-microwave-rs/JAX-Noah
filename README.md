# JAX-Noah

A JAX-based Noah land-surface model with soil-moisture data assimilation and parameter-optimization scripts.

## Project Attribution

JAX-Noah is developed through collaboration between the Institute of Tibetan Plateau Research, Chinese Academy of Sciences (中国科学院青藏高原研究所), and the College of Hydrology and Water Resources, Hohai University (河海大学水文水资源学院).

- Project lead: Researcher Donghai Zheng (郑东海), Institute of Tibetan Plateau Research, Chinese Academy of Sciences.
- JAX-Noah coding and testing: PhD student Yu Zhang (张羽), College of Hydrology and Water Resources, Hohai University.
- Research collaboration: Postdoctoral researcher Pei Zhang (张佩), Institute of Tibetan Plateau Research, Chinese Academy of Sciences; Professor Haishen Lü (吕海深), College of Hydrology and Water Resources, Hohai University.

This attribution records project collaboration and contributions. Copyright ownership and licensing of original project code and research outputs are subject to applicable institutional policies and collaboration agreements. Third-party code, parameter tables, and datasets remain subject to their respective source licenses and terms; this attribution does not grant rights to redistribute them.

## Requirements

- Python 3.10 or newer
- Packages listed in `requirements.txt`
- The Microsoft Visual C++ Redistributable may be needed for JAX on Windows

Create an environment and install the Python dependencies from the project directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

JAX's native Windows CPU support is experimental. Native Windows does not support JAX NVIDIA GPU execution; use Linux or WSL2 for NVIDIA GPU acceleration and follow the JAX installation guide for the matching CUDA setup: <https://docs.jax.dev/en/latest/installation.html>.

## Included data and parameter tables

The default experiment configurations use these input and parameter files:

- `wudaoliang-forcing_cmfd.txt`
- `wudaoliang-smap_apr01_oct31_valid.csv`
- `wudaoliang-smap_apr01_oct31_valid-CDF.csv` (available for CDF-based experiments)
- `parameter_new/SOILPARM.TBL`
- `parameter_new/VEGPARM-USGS.TBL`
- `parameter_new/VEGPARM-IGBP.TBL`

`wudaoliang-smap.csv` is also included, but is not referenced by the current default script configurations.

Check the source, license, redistribution terms, and repository size of the forcing and observation data before making the repository public. If you do not have permission to redistribute the data, provide instructions for obtaining it and keep the files out of the repository.

## Running

Run commands from the project directory so the data and parameter paths used by the assimilation scripts resolve correctly.

```powershell
python cycling_enkf_jax_relative_error.py
python cycling_ekf_jax_relative_error.py
python optimization_minibatch_margin.py
python optimization_lhs_multistart.py
python enkf_relative_error_optimize_optuna_soil_params_smap_mse.py
```

The scripts contain experiment-specific input paths and settings near their configuration sections. Review and adjust those settings before running. Optimization and assimilation runs can be computationally intensive; JAX may initially compile functions before execution.

## Dependency notes

`requirements.txt` lists the third-party packages imported by the project, including `cmaes` for Optuna's CMA-ES sampler. It intentionally does not pin package versions; exact version pins should be added after validating a known working environment.
