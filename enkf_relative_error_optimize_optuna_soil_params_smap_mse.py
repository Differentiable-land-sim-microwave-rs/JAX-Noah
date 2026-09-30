# file name: enkf_relative_error_optimize_optuna_soil_params_smap_mse.py
#
# EnKF Soil Parameter Optimization (Optuna)
# Data-assimilation workflow detail.
#
# optimization objective: 
#   - 4soilparameteroptimization: BB, SATDK, SATPSI, MAXSMC
# Model-parameter handling.
# Data-assimilation workflow detail.
#
# createdate: 2025-12-03
#
from pathlib import Path

# ============================================================================
# ============================== Configuration parameters ===============================
# ============================================================================

# --- Implementation details ---
ROOT_DIR = Path(__file__).resolve().parent
FORCING_FILE = str(ROOT_DIR /"wudaoliang-forcing_cmfd.txt")
OBS_FILE = str(ROOT_DIR /"wudaoliang-smap_apr01_oct31_valid.csv")

# Data-assimilation workflow detail.
COMPARISON_FILE_SMAP = OBS_FILE

OUTPUT_DIR = str(ROOT_DIR / "optuna_optimization_enkf_mse-wdl_soil_params_smap_20200601")  # output directory

# --- Optimization parameter ranges ---
# Data-assimilation workflow detail.
# Implementation detail.
PARAM_BOUNDS = {
    'BB': (2.79, 11.55),
    'SATDK': (9.74e-7, 1.41e-5),
    'SATPSI': (0.036, 0.759),
    'MAXSMC': (0.339, 0.476),
}

# --- optimizationdefaultsettings ---
# Model-parameter handling.
DEFAULT_N_INIT = 40  # Optuna startup trials
DEFAULT_N_ITER = 200  # Optuna optimization iterations
DEFAULT_EARLY_STOP_THRESHOLD = 0.015  # Implementation detail.
DEFAULT_PATIENCE = None  # Implementation detail.
DEFAULT_MIN_IMPROVEMENT = 0.001  # minimum threshold for a meaningful improvement
DEFAULT_EXEC_MODE = "multi"  # options: "single", "multi"
DEFAULT_WORKERS = 4  # used when mode="multi"
DEFAULT_STARTUP_PARALLEL = False  # True: run startup trials in worker processes
DEFAULT_THREADS_PER_WORKER = 1  # worker-internal CPU threads (used in mode="multi")
DISABLE_TRIAL_FILE_OUTPUT = True  # avoid heavy per-trial pkl/csv I/O during optimization

# --- Optuna sampler configuration ---
OPTUNA_SAMPLER = "TPE"  # Implementation detail.
OPTUNA_SAMPLER_KWARGS = {
    "TPE": {
        "n_startup_trials": DEFAULT_N_INIT,  # startup trial count (overridden by CLI/init)
        "n_ei_candidates": 25,  # EInumber of candidates(larger is more accurate but slower)
        "multivariate": True,  # multivariate TPE(account for parameter correlations)⚠️experimental feature
        "constant_liar": "max",  # Helps when running in multi-process mode
    },
    "CmaEs": {
        "n_startup_trials": 0,
        "restart_strategy": "ipop",  # restart strategy
    },
    "QMC": {
        "qmc_type": "sobol",  # Sobolsequence(quasi-random)
        "scramble": True,
    },
    "Random": {}
}

# --- loss functionmetric weights ---
LOSS_METRIC = "mse"  # optimization target metric

# Defaults used by --mode multi (can be overridden by CLI args).
STUDY_STORAGE = "sqlite:///enkf_mse_smap_optuna.db"
STUDY_NAME = "enkf_mse_smap_optimization"

# Keep this experiment separate from older CDF-based Optuna studies/results.
STUDY_STORAGE = "sqlite:///enkf_mse_smap_20200601_optuna.db"
STUDY_NAME = "enkf_mse_smap_20200601_optimization"

# --- data comparison window(used for evaluation) ---
LOSS_WEIGHT_SMAP = 1.0  # Implementation detail.

# --- outlier handling ---
ERROR_RETURN_VALUE = 1e6  # large loss returned on failure

# --- visualization configuration ---
PLOT_FIGURE_SIZE = (16, 12)  # figure size
PLOT_DPI = 150  # figure resolution
ENABLE_PLOT_DISPLAY = False  # whether to display figures(False=onlysave)
FONT_CONFIG = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # font configuration



# ============================================================================
# ============================== Import dependencies ====================================
# ============================================================================

import os
import sys
import json
import pickle
import subprocess
import time
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, QMCSampler, RandomSampler
from optuna.trial import TrialState
from typing import Dict, List, Tuple
from datetime import datetime

# Avoid Windows GBK console crashes on emoji/unicode logs.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

# visualization libraries
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = FONT_CONFIG
matplotlib.rcParams['axes.unicode_minus'] = False

# checkcycling_enkf_jax_relative_error.pywhether it exists
enkf_script_name = "cycling_enkf_jax_relative_error.py"
if not os.path.exists(enkf_script_name):
    print(f"ERROR: {enkf_script_name} not found in current directory!")
    print("Please make sure the EnKF deterministic script is in the same directory.")
    sys.exit(1)

# import the deterministic EnKF module
import importlib.util
spec = importlib.util.spec_from_file_location("enkf_module", enkf_script_name)
enkf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enkf_module)

# Force the deterministic EnKF module to use the current Wudaoliang forcing
# and cropped SMAP file for all optimization trials.
enkf_module.FORCING_FILE = FORCING_FILE
enkf_module.OBS_CSV_FILE = OBS_FILE


# ============================================================================
# ============================== helper functions ====================================
# ============================================================================

def compute_metrics(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    """Compute diagnostics between predictions and observations."""
    # Remove NaN values before metric calculations.
    mask = ~(np.isnan(pred) | np.isnan(obs))
    pred_clean = pred[mask]
    obs_clean = obs[mask]

    if len(pred_clean) < 2:
        return {
            'mse': 1.0,
            'rmse': 1.0,
            'r': 0.0,
            'mae': 1.0,
            'bias': 0.0,
        }

    r = np.corrcoef(pred_clean, obs_clean)[0, 1]
    if np.isnan(r):
        r = 0.0

    mse = np.mean((pred_clean - obs_clean) ** 2)
    rmse = np.sqrt(mse)
    bias = np.mean(pred_clean - obs_clean)
    mae = np.mean(np.abs(pred_clean - obs_clean))

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'r': float(r),
        'mae': float(mae),
        'bias': float(bias),
    }


def compute_loss_from_metrics(metrics: Dict[str, float]) -> float:
    """Use MSE as the optimization loss."""
    total_loss = float(metrics.get('mse', ERROR_RETURN_VALUE))
    metrics['total_loss'] = float(total_loss)
    return total_loss



# ============================================================================
# ============================== optimizer class ====================================
# ============================================================================

class EnKFRelativeErrorOptimizer:
    """
    EnKF soil parameter optimizer.

    Optimized parameters:
    - BB
    - SATDK
    - SATPSI
    - MAXSMC
    """

    def __init__(self,
                 n_init: int = DEFAULT_N_INIT,
                 n_iter: int = DEFAULT_N_ITER,
                 early_stop_threshold: float = DEFAULT_EARLY_STOP_THRESHOLD,
                 patience: int = DEFAULT_PATIENCE,
                 min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
                 output_dir: str = OUTPUT_DIR):
        """Technical documentation."""
        self.n_init = n_init
        self.n_iter = n_iter
        self.n_optuna = n_iter  # Implementation detail.
        self.startup_target = int(n_init)  # expected startup trial count for phase labeling
        self.phase2_target = int(n_iter)  # expected phase-2 trial count for phase labeling
        self.early_stop_threshold = early_stop_threshold
        self.patience = patience
        self.min_improvement = min_improvement
        self.output_dir = output_dir

        # createoutput directory
        os.makedirs(self.output_dir, exist_ok=True)

        # optimization state
        self.best_score = float('inf')
        self.best_params = None
        self.evaluation_count = 0
        self.loss_history = []
        self.optimization_history = []
        self.best_assimilation_pkl = os.path.join(self.output_dir, 'best_loss_assimilation.pkl')
        self.best_assimilation_csv = os.path.join(self.output_dir, 'best_loss_assimilation.csv')
        self.best_assimilation_meta = os.path.join(self.output_dir, 'best_loss_assimilation_meta.json')
        self.best_assimilation_lock = os.path.join(self.output_dir, 'best_loss_assimilation.lock')

        # Phase 1/2 state
        self.phase1_active = True
        self.no_improvement_count = 0

        # load comparison data
        self._load_comparison_data()
        self._init_soil_param_context()

        print("=" * 80)
        print("EnKF Soil Parameter Optimization (Optuna)")
        print("=" * 80)
        print(f"Parameters to optimize: {len(PARAM_BOUNDS)}")
        for key, bounds in PARAM_BOUNDS.items():
            print(f"  - {key}: {bounds}")
        print(f"\nSoil parameter target:")
        print(f"  - Source file: {self.base_soil_param_file}")
        print(f"  - Soil type row: {self.target_soil_type}")
        print(f"\nData files:")
        print(f"  - Forcing file: {enkf_module.FORCING_FILE}")
        print(f"  - Assimilation SMAP file: {enkf_module.OBS_CSV_FILE}")
        print(f"  - Loss comparison file: {COMPARISON_FILE_SMAP}")
        print(f"\nOptimization settings:")
        print(f"  - Startup trials: {n_init}")
        print(f"  - Optuna iterations: {n_iter}")
        print(f"  - Total evaluations: {n_init + n_iter}")
        print(f"  - Sampler: {OPTUNA_SAMPLER}")
        print(f"  - Early stop threshold: {early_stop_threshold}")
        if patience is not None:
            print(f"  - Patience: {patience} iterations")
        print(f"\nOutput directory: {output_dir}")
        print("=" * 80)

    def _load_comparison_data(self):
        """load comparison data(used for evaluation)"""
        self.comparison_data = {}

        if os.path.exists(COMPARISON_FILE_SMAP):
            df = pd.read_csv(COMPARISON_FILE_SMAP, parse_dates=['Date'], index_col='Date')
            self.comparison_data['SMAP'] = df
            print(f"Loaded comparison data: {COMPARISON_FILE_SMAP} ({len(df)} records)")
        else:
            print(f"WARNING: {COMPARISON_FILE_SMAP} not found")
            self.comparison_data['SMAP'] = None

    def _init_soil_param_context(self):
        """Prepare baseline soil metadata for trial-time parameter overrides."""
        self.base_soil_param_file = str(getattr(enkf_module, 'SOIL_PARAM_FILE', ''))

        if not self.base_soil_param_file or not os.path.exists(self.base_soil_param_file):
            raise FileNotFoundError(f"SOIL_PARAM_FILE not found: {self.base_soil_param_file}")

        self.base_soil_df = pd.read_csv(
            self.base_soil_param_file,
            sep=r',\s*',
            engine='python',
            header=0,
            index_col=0,
            usecols=range(11),
            dtype=np.float32,
        )
        self.target_soil_type = self._detect_target_soil_type()
        if self.target_soil_type not in self.base_soil_df.index:
            fallback_pos = int(np.clip(int(self.target_soil_type) - 1, 0, len(self.base_soil_df.index) - 1))
            self.target_soil_type = self.base_soil_df.index[fallback_pos]
        self.default_params = self._get_default_params()

    def _get_default_params(self) -> Dict[str, float]:
        """Read baseline soil parameters for the current soil type row."""
        row = self.base_soil_df.loc[self.target_soil_type]
        return {
            'BB': float(row['BB']),
            'SATDK': float(row['SATDK']),
            'SATPSI': float(row['SATPSI']),
            'MAXSMC': float(row['MAXSMC']),
        }

    def enqueue_default_trial(self, study: optuna.Study) -> None:
        """Use model default soil parameters as the first queued Optuna trial."""
        default_params = dict(self.default_params)
        study.enqueue_trial(default_params, user_attrs={'seed_source': 'default_soil_params'})
        print("Enqueued default soil parameters as the first trial:")
        for key, value in default_params.items():
            if key == 'SATDK':
                print(f"  - {key}: {value:.8e}")
            else:
                print(f"  - {key}: {value:.6f}")

    def _detect_target_soil_type(self):
        """Infer the soil type row used by the deterministic EnKF script."""
        try:
            forcing_tuple = enkf_module.open_forcing_file(enkf_module.FORCING_FILE)
            stype = np.asarray(forcing_tuple[18]).ravel()
            if stype.size == 0:
                raise ValueError("Empty STYPE returned from forcing file.")
            return int(stype[0])
        except Exception as e:
            fallback = int(self.base_soil_df.index[0])
            print(f"WARNING: Failed to infer STYPE from forcing file ({e}), fallback to row {fallback}")
            return fallback

    def objective(self, trial: optuna.Trial) -> float:
        """Technical documentation."""
        # from trial ingetparameter
        bb = trial.suggest_float('BB', *PARAM_BOUNDS['BB'])
        satdk = trial.suggest_float('SATDK', *PARAM_BOUNDS['SATDK'], log=True)
        satpsi = trial.suggest_float('SATPSI', *PARAM_BOUNDS['SATPSI'])
        maxsmc = trial.suggest_float('MAXSMC', *PARAM_BOUNDS['MAXSMC'])

        self.evaluation_count += 1

        print(f"\n{'─' * 80}")
        # Implementation detail.
        startup_target = max(0, int(getattr(self, 'startup_target', self.n_init)))
        phase2_target = max(1, int(getattr(self, 'phase2_target', self.n_optuna)))

        in_startup = startup_target > 0 and trial.number < startup_target
        if startup_target == 0 and self.phase1_active and self.evaluation_count <= max(0, self.n_init):
            in_startup = True

        if in_startup:
            startup_idx = trial.number + 1 if startup_target > 0 else self.evaluation_count
            startup_total = startup_target if startup_target > 0 else max(1, self.n_init)
            phase_info = f"[Startup {startup_idx}/{startup_total}] "
        else:
            if startup_target > 0:
                optuna_idx = max(1, trial.number - startup_target + 1)
            else:
                optuna_idx = max(1, self.evaluation_count - max(0, self.n_init))
            phase_info = f"[Optuna {optuna_idx}/{phase2_target}] "
        print(f"{phase_info}[Evaluation {self.evaluation_count} | Trial {trial.number}]")
        print(f"Parameters:")
        print(f"  BB: {bb:.6f}")
        print(f"  SATDK: {satdk:.8e}")
        print(f"  SATPSI: {satpsi:.6f}")
        print(f"  MAXSMC: {maxsmc:.6f}")

        # save the original parameters
        original_params = self._save_original_params()

        try:
            # apply the new parameters
            self._apply_params(bb, satdk, satpsi, maxsmc)

            # run EnKF
            print("  Running EnKF...")
            results = enkf_module.run_cycling_deterministic()

            # calculate the loss function
            loss, metrics = self._compute_loss(results)

            print(f"  Total Loss: {loss:.6f}")

            # record history
            self._record_history(bb, satdk, satpsi, maxsmc, loss, metrics)

            # update the best parameters
            is_new_best = False
            if not self.phase1_active:
                improvement = self.best_score - loss
                if loss < self.best_score:
                    if improvement > self.min_improvement:
                        self.no_improvement_count = 0
                        print(f"  New best loss: {loss:.6f} (improvement: {improvement:.6f})")
                    else:
                        self.no_improvement_count += 1
                        print(f"  New best loss: {loss:.6f} (improvement too small: {improvement:.6f})")

                    self.best_score = loss
                    self.best_params = {
                        'BB': bb,
                        'SATDK': satdk,
                        'SATPSI': satpsi,
                        'MAXSMC': maxsmc,
                    }
                    is_new_best = True
                else:
                    self.no_improvement_count += 1

                if self.patience is not None and self.no_improvement_count > 0:
                    print(f"  No improvement count: {self.no_improvement_count}/{self.patience}")
            else:
                # Phase 1
                if loss < self.best_score:
                    self.best_score = loss
                    self.best_params = {
                        'BB': bb,
                        'SATDK': satdk,
                        'SATPSI': satpsi,
                        'MAXSMC': maxsmc,
                    }
                    is_new_best = True
                    print(f"  Phase 1 best: {loss:.6f}")

            if is_new_best:
                saved = self._save_best_assimilation_results(
                    results=results,
                    loss=loss,
                    params=self.best_params,
                    trial_number=trial.number,
                    metrics=metrics,
                )
                if saved:
                    print(f"  Saved best assimilation results to: {self.best_assimilation_pkl}")

            self.loss_history.append(loss)

            return loss

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return ERROR_RETURN_VALUE

        finally:
            # restore the original parameters
            self._restore_original_params(original_params)

    def _save_original_params(self) -> Dict:
        """save the original parameters"""
        return {
            'SOIL_PARAM_OVERRIDES': dict(getattr(enkf_module, 'SOIL_PARAM_OVERRIDES', {})),
            'VERBOSE': enkf_module.VERBOSE,
            'OUTPUT_FILE': enkf_module.OUTPUT_FILE,
        }

    def _apply_params(self, bb, satdk, satpsi, maxsmc):
        """
        Apply trial soil parameters to the deterministic EnKF module.
        """
        enkf_module.SOIL_PARAM_OVERRIDES = {
            'BB': float(bb),
            'SATDK': float(satdk),
            'SATPSI': float(satpsi),
            'MAXSMC': float(maxsmc),
        }

        # disable verbose output
        enkf_module.VERBOSE = False
        if DISABLE_TRIAL_FILE_OUTPUT:
            enkf_module.OUTPUT_FILE = os.devnull

    def _restore_original_params(self, original_params: Dict):
        """restore the original parameters"""
        for key, val in original_params.items():
            setattr(enkf_module, key, val)

    def _compute_loss(self, results: Dict) -> Tuple[float, Dict]:
        """
        calculate the loss function

        parameter:
            results: EnKFresultsdictionary

        return:
            (total_loss, metrics_dict)
        """
        # extract analysis states
        analysis_states = results['analysis_states']  # (n_steps, n_states)
        dates = results['metadata']['dates']

        # state indices(cycling_enkf_jax_relative_error.pyuse 12 states)
        # [STC(4), SMC(4), SH2O(4)]
        idx_sh2o1 = 8

        # extract predicted SH2O values
        sh2o1_pred = analysis_states[:, idx_sh2o1]

        metrics_smap = {}

        if self.comparison_data['SMAP'] is not None:
            df_comp = self.comparison_data['SMAP']
            dates_pd = pd.to_datetime(dates)
            df_pred = pd.DataFrame({'SH2O(1)': sh2o1_pred}, index=dates_pd)
            df_merged = df_pred.join(df_comp[['SH2O(1)']], how='inner', rsuffix='_obs')

            if len(df_merged) > 0:
                pred = df_merged['SH2O(1)'].values
                obs = df_merged['SH2O(1)_obs'].values
                metrics_smap = compute_metrics(pred, obs)
            else:
                print("  WARNING: No overlapping dates for SMAP")
                metrics_smap = {'mse': 1.0, 'rmse': 1.0, 'r': 0.0, 'mae': 1.0, 'bias': 0.0}
        else:
            metrics_smap = {'mse': 1.0, 'rmse': 1.0, 'r': 0.0, 'mae': 1.0, 'bias': 0.0}

        loss_smap = compute_loss_from_metrics(metrics_smap)
        total_loss = LOSS_WEIGHT_SMAP * loss_smap

        print(f"  SMAP: MSE={metrics_smap.get('mse', 0):.6f}, RMSE={metrics_smap.get('rmse', 0):.4f}, Loss={loss_smap:.6f}")

        return total_loss, {'SMAP': metrics_smap}

    def _record_history(self, bb, satdk, satpsi, maxsmc, loss, metrics):
        """Technical documentation."""
        record = {
            'evaluation': self.evaluation_count,
            'timestamp': datetime.now().isoformat(),
            'params': {
                'BB': float(bb),
                'SATDK': float(satdk),
                'SATPSI': float(satpsi),
                'MAXSMC': float(maxsmc),
            },
            'loss': float(loss),
            'metrics': metrics
        }
        self.optimization_history.append(record)

    def _acquire_lock(self, lock_path: str, timeout_seconds: float = 120.0):
        """Acquire a simple file lock for best-result writes."""
        start_time = time.time()
        while True:
            try:
                return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.time() - start_time >= timeout_seconds:
                    raise TimeoutError(f"Timeout waiting for lock: {lock_path}")
                time.sleep(0.2)

    def _release_lock(self, lock_fd, lock_path: str) -> None:
        """Release a file lock created with _acquire_lock."""
        try:
            if lock_fd is not None:
                os.close(lock_fd)
        finally:
            if lock_fd is not None and os.path.exists(lock_path):
                os.remove(lock_path)

    def _atomic_write_json(self, path: str, payload: Dict) -> None:
        temp_path = f"{path}.tmp.{os.getpid()}"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, path)

    def _atomic_write_pickle(self, path: str, payload) -> None:
        temp_path = f"{path}.tmp.{os.getpid()}"
        with open(temp_path, 'wb') as f:
            pickle.dump(payload, f)
        os.replace(temp_path, path)

    def _atomic_write_csv(self, path: str, df: pd.DataFrame) -> None:
        temp_path = f"{path}.tmp.{os.getpid()}"
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, path)

    def _build_assimilation_dataframe(self, results: Dict) -> pd.DataFrame:
        """Build the same CSV-style table as the deterministic EnKF script."""
        metadata = results.get('metadata', {})
        dates = metadata.get('dates', [])
        detailed_states = results.get('analysis_states_detailed')
        analysis_states = np.asarray(results.get('analysis_states'))

        nsoil = int(metadata.get('nsoil', 0))
        if nsoil <= 0 and analysis_states.ndim == 2:
            nsoil = analysis_states.shape[1] // 3
        if nsoil <= 0:
            raise ValueError("Could not infer nsoil from optimization results.")

        rows = []
        if detailed_states is not None and len(detailed_states) == len(dates):
            for idx, state_dict in enumerate(detailed_states):
                row = {
                    'Date': dates[idx],
                    'step': int(state_dict.get('step_index', idx)),
                    'has_obs': bool(state_dict.get('has_observation', False)),
                }
                stc = np.asarray(state_dict.get('STC', []), dtype=float)
                smc = np.asarray(state_dict.get('SMC', []), dtype=float)
                sh2o = np.asarray(state_dict.get('SH2O', []), dtype=float)
                for i in range(nsoil):
                    row[f'STC({i + 1})'] = round(float(stc[i]), 3)
                    row[f'SMC({i + 1})'] = round(float(smc[i]), 6)
                    row[f'SH2O({i + 1})'] = round(float(sh2o[i]), 6)
                rows.append(row)
            return pd.DataFrame(rows)

        if analysis_states.ndim != 2 or analysis_states.shape[0] != len(dates):
            raise ValueError("Incompatible result structure for saving best assimilation output.")

        for idx in range(analysis_states.shape[0]):
            row = {
                'Date': dates[idx],
                'step': idx,
                'has_obs': False,
            }
            stc = analysis_states[idx, 0:nsoil]
            smc = analysis_states[idx, nsoil:2 * nsoil]
            sh2o = analysis_states[idx, 2 * nsoil:3 * nsoil]
            for i in range(nsoil):
                row[f'STC({i + 1})'] = round(float(stc[i]), 3)
                row[f'SMC({i + 1})'] = round(float(smc[i]), 6)
                row[f'SH2O({i + 1})'] = round(float(sh2o[i]), 6)
            rows.append(row)

        return pd.DataFrame(rows)

    def _save_best_assimilation_results(
        self,
        results: Dict,
        loss: float,
        params: Dict[str, float],
        trial_number: int,
        metrics: Dict,
    ) -> bool:
        """Persist the assimilation output for the current best-loss trial."""
        lock_fd = None
        try:
            lock_fd = self._acquire_lock(self.best_assimilation_lock)

            existing_best = float('inf')
            if os.path.exists(self.best_assimilation_meta):
                try:
                    with open(self.best_assimilation_meta, 'r', encoding='utf-8') as f:
                        existing_meta = json.load(f)
                    existing_best = float(existing_meta.get('best_loss', float('inf')))
                except Exception:
                    existing_best = float('inf')

            if loss >= existing_best:
                return False

            df = self._build_assimilation_dataframe(results)
            self._atomic_write_pickle(self.best_assimilation_pkl, results)
            self._atomic_write_csv(self.best_assimilation_csv, df)

            meta = {
                'best_loss': float(loss),
                'trial_number': int(trial_number),
                'saved_at': datetime.now().isoformat(),
                'params': {key: float(val) for key, val in params.items()},
                'metrics': metrics,
                'soil_type_row': int(self.target_soil_type),
                'num_assimilations': int(results.get('metadata', {}).get('num_assimilations', 0)),
                'n_output_steps': int(len(results.get('metadata', {}).get('dates', []))),
                'files': {
                    'pkl': os.path.basename(self.best_assimilation_pkl),
                    'csv': os.path.basename(self.best_assimilation_csv),
                },
            }
            self._atomic_write_json(self.best_assimilation_meta, meta)
            return True
        finally:
            self._release_lock(lock_fd, self.best_assimilation_lock)

    def optimize(self):
        """
        execute the complete optimization workflow

        Phase 1: Optuna startup trials
        Phase 2: Optunaoptimization(with early stopping)
        """

        # ===== Phase 1: startup trials =====
        print("\n" + "=" * 80)
        print("PHASE 1: Optuna Startup Trials")
        print("=" * 80)

        # create the Optuna study
        sampler = self._create_sampler()

        # Single-process path always uses in-memory study.
        storage = None
        study_name = f"enkf_mse_smap_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        load_if_exists = False

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists
        )
        self.enqueue_default_trial(study)

        # Implementation detail.
        study.optimize(self.objective, n_trials=self.n_init, show_progress_bar=False)

        print(f"\nPhase 1 complete: Best loss = {self.best_score:.6f}")
        print(f"Best parameters: {self.best_params}")

        # ===== Phase 2: Optuna TPE optimization =====
        print("\n" + "=" * 80)
        print("PHASE 2: Optuna TPE Optimization")
        print("=" * 80)

        self.phase1_active = False

        # Implementation detail.
        def early_stopping_callback(study, trial):
            # Implementation detail.
            if self.best_score < self.early_stop_threshold:
                print(f"\n Early stopping: Loss {self.best_score:.6f} < threshold {self.early_stop_threshold}")
                study.stop()

            # checkpatience
            if self.patience is not None and self.no_improvement_count >= self.patience:
                print(f"\n  Early stopping: No improvement for {self.patience} iterations")
                study.stop()

        # Implementation detail.
        study.optimize(
            self.objective,
            n_trials=self.n_iter,
            callbacks=[early_stopping_callback],
            show_progress_bar=False
        )

        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Total evaluations: {self.evaluation_count}")
        print(f"Best loss: {self.best_score:.6f}")
        print(f"Best parameters:")
        for key, val in self.best_params.items():
            if key == 'SATDK':
                print(f"  {key}: {val:.8e}")
            else:
                print(f"  {key}: {val:.6f}")

        # generate visualizations
        self._plot_optimization_results(study)

        # save results
        self._save_results(study)

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """create the Optuna sampler"""
        sampler_name = OPTUNA_SAMPLER
        kwargs = dict(OPTUNA_SAMPLER_KWARGS.get(sampler_name, {}))

        if sampler_name == "TPE":
            kwargs["n_startup_trials"] = int(self.n_init)
            return TPESampler(**kwargs)
        elif sampler_name == "CmaEs":
            return CmaEsSampler(**kwargs)
        elif sampler_name == "QMC":
            return QMCSampler(**kwargs)
        elif sampler_name == "Random":
            return RandomSampler(**kwargs)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def _plot_optimization_results(self, study: optuna.Study):
        """generate optimization-process visualizations(Optuna native style + custom analysis)"""
        if len(self.loss_history) == 0:
            return

        try:
            # Implementation detail.
            print("\n Generating Optuna-style visualization...")

            try:
                import optuna.visualization as vis

                # 1. Optimization History(optimizationhistory)
                fig_history = vis.plot_optimization_history(study)
                fig_history.update_layout(
                    title="Optimization History Plot",
                    width=1200,
                    height=500,
                    font=dict(size=12),
                    plot_bgcolor='rgba(240, 242, 246, 0.5)',
                )
                history_file = os.path.join(self.output_dir, 'optuna_history.html')
                fig_history.write_html(history_file)
                print(f"   Optimization History: {history_file}")

                # Model-parameter handling.
                if len(study.trials) > 10:
                    fig_importance = vis.plot_param_importances(study)
                    fig_importance.update_layout(title="Parameter Importances", width=1000, height=600)
                    importance_file = os.path.join(self.output_dir, 'optuna_param_importance.html')
                    fig_importance.write_html(importance_file)
                    print(f"   Parameter Importances: {importance_file}")

                # Implementation detail.
                fig_parallel = vis.plot_parallel_coordinate(study, params=list(PARAM_BOUNDS.keys()))
                fig_parallel.update_layout(title="Parallel Coordinate Plot", width=1400, height=600)
                parallel_file = os.path.join(self.output_dir, 'optuna_parallel_coordinate.html')
                fig_parallel.write_html(parallel_file)
                print(f"   Parallel Coordinate: {parallel_file}")

                # Implementation detail.
                if len(study.trials) > 20 and len(PARAM_BOUNDS) >= 2:
                    try:
                        importance = optuna.importance.get_param_importances(study)
                        top_params = list(importance.keys())[:2]
                        if len(top_params) == 2:
                            fig_contour = vis.plot_contour(study, params=top_params)
                            fig_contour.update_layout(
                                title=f"Contour Plot: {top_params[0]} vs {top_params[1]}",
                                width=800, height=700
                            )
                            contour_file = os.path.join(self.output_dir, 'optuna_contour.html')
                            fig_contour.write_html(contour_file)
                            print(f"   Contour Plot: {contour_file}")
                    except:
                        pass

                print(f"\n   Open the HTML file to view interactive charts (recommended!)")

            except ImportError:
                print("    Need to install plotly to generate Optuna native charts:")
                print("     pip install plotly kaleido")
            except Exception as e:
                print(f"    Optuna visualization part failed: {e}")

            # Implementation detail.
            print("\n Generating custom matplotlib analysis...")

            fig, axes = plt.subplots(2, 3, figsize=PLOT_FIGURE_SIZE)
            iterations = range(1, len(self.loss_history) + 1)
            n_init = min(self.n_init, len(self.loss_history))

            # Implementation detail.
            ax1 = axes[0, 0]
            phase1_iters = list(iterations[:n_init])
            phase2_iters = list(iterations[n_init:])
            phase1_losses = self.loss_history[:n_init]
            phase2_losses = self.loss_history[n_init:]

            ax1.scatter(phase1_iters, phase1_losses, c='#3498db', s=30, alpha=0.6,
                       label='Phase 1 (Startup)', edgecolors='none')
            if len(phase2_losses) > 0:
                ax1.scatter(phase2_iters, phase2_losses, c='#e74c3c', s=30, alpha=0.6,
                           label='Phase 2 (Optuna)', edgecolors='none')

            ax1.axhline(y=self.best_score, color='#c0392b', linestyle='-', linewidth=2.5,
                       label=f'Best Value: {self.best_score:.4f}', alpha=0.8)

            ax1.set_xlabel('Trial Number', fontsize=11)
            ax1.set_ylabel('Objective Value', fontsize=11)
            ax1.set_title('Optimization History Plot', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=9)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_facecolor('#f0f2f6')

            # 2. cumulative best value(convergence curve)
            ax2 = axes[0, 1]
            cumulative_best = []
            current_best = float('inf')
            for loss in self.loss_history:
                if loss < current_best:
                    current_best = loss
                cumulative_best.append(current_best)

            ax2.plot(iterations, cumulative_best, color='#27ae60', linewidth=2.5, alpha=0.8)
            ax2.fill_between(iterations, cumulative_best, alpha=0.2, color='#27ae60')
            ax2.axvline(x=n_init, color='#f39c12', linestyle='--', linewidth=2, alpha=0.7, label='Phase 1→2')
            ax2.set_xlabel('Trial Number', fontsize=11)
            ax2.set_ylabel('Best Value', fontsize=11)
            ax2.set_title('Convergence Curve', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_facecolor('#f0f2f6')

            # 3. loss-distribution histogram
            ax3 = axes[0, 2]
            if len(phase2_losses) > 0:
                ax3.hist(phase1_losses, bins=20, alpha=0.6, label='Phase 1',
                        color='#3498db', edgecolor='white')
                ax3.hist(phase2_losses, bins=20, alpha=0.6, label='Phase 2',
                        color='#e74c3c', edgecolor='white')

                std1 = np.std(phase1_losses)
                std2 = np.std(phase2_losses)
                convergence_ratio = std2 / std1 if std1 > 0 else 1.0
                convergence_status = " Converged" if convergence_ratio < 0.6 else " Exploring"

                ax3.set_title(f'Loss Distribution\nConv. Ratio: {convergence_ratio:.2f} {convergence_status}',
                             fontsize=12, fontweight='bold')
                ax3.text(0.05, 0.95, f'Phase 1 Std: {std1:.4f}\nPhase 2 Std: {std2:.4f}',
                        transform=ax3.transAxes, verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))
            else:
                ax3.hist(phase1_losses, bins=20, alpha=0.7, color='#3498db', edgecolor='white')
                ax3.set_title('Loss Distribution (Phase 1 only)', fontsize=12, fontweight='bold')

            ax3.set_xlabel('Loss Value', fontsize=11)
            ax3.set_ylabel('Frequency', fontsize=11)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax3.set_facecolor('#f0f2f6')

            # 4-6. key-parameter evolution
            key_params = list(PARAM_BOUNDS.keys())[:3]  # take the first 3 parameters
            param_axes = [axes[1, 0], axes[1, 1], axes[1, 2]]
            param_colors = ['#e74c3c', '#3498db', '#2ecc71']

            for ax, param, color in zip(param_axes, key_params, param_colors):
                values = [entry['params'].get(param, 0) for entry in self.optimization_history]
                if len(values) > 0:
                    # Phase 1
                    ax.scatter(phase1_iters, values[:n_init], c=color, s=25, alpha=0.4, edgecolors='none')
                    # Phase 2
                    if len(values) > n_init:
                        ax.scatter(phase2_iters, values[n_init:], c=color, s=25, alpha=0.7, edgecolors='none')

                    # trend line(moving average)
                    if len(values) > 10:
                        window = min(20, len(values) // 5)
                        moving_avg = pd.Series(values).rolling(window=window, center=True).mean()
                        ax.plot(iterations[:len(values)], moving_avg, color=color, linewidth=2,
                               alpha=0.8, linestyle='-')

                    ax.axvline(x=n_init, color='#f39c12', linestyle='--', linewidth=1.5, alpha=0.5)
                    ax.set_xlabel('Trial Number', fontsize=11)
                    ax.set_ylabel(param, fontsize=11)
                    ax.set_title(f'{param} Evolution', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.set_facecolor('#f0f2f6')

            plt.tight_layout()

            plot_file = os.path.join(self.output_dir, 'optuna_custom_analysis.png')
            plt.savefig(plot_file, dpi=PLOT_DPI, bbox_inches='tight')
            print(f"   Custom Analysis: {plot_file}")

            if ENABLE_PLOT_DISPLAY:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"  Error creating plots: {e}")
            import traceback
            traceback.print_exc()

    def _sync_history_from_study(self, study: optuna.Study):
        """Rebuild local history from persisted study trials (for multiprocess mode)."""
        complete_trials = [
            t for t in study.trials
            if t.state == TrialState.COMPLETE and t.value is not None
        ]
        complete_trials.sort(key=lambda t: t.number)

        self.loss_history = []
        self.optimization_history = []

        for idx, trial in enumerate(complete_trials, start=1):
            params = dict(trial.params)

            metrics = trial.user_attrs.get('metrics', {})
            if not isinstance(metrics, dict):
                metrics = {}

            record = {
                'evaluation': idx,
                'timestamp': trial.user_attrs.get('timestamp', datetime.now().isoformat()),
                'params': {
                    key: float(params.get(key, np.nan))
                    for key in PARAM_BOUNDS.keys()
                },
                'loss': float(trial.value),
                'metrics': {
                    'SMAP': metrics.get('SMAP', {}),
                }
            }
            self.optimization_history.append(record)
            self.loss_history.append(float(trial.value))

        self.evaluation_count = len(self.loss_history)
        if len(complete_trials) > 0:
            self.best_score = float(study.best_value)
            self.best_params = {
                key: float(val)
                for key, val in study.best_trial.params.items()
            }

    def _save_results(self, study: optuna.Study):
        complete_trial_count = sum(
            1 for t in study.trials
            if t.state == TrialState.COMPLETE and t.value is not None
        )
        if complete_trial_count != len(self.optimization_history):
            self._sync_history_from_study(study)
        """Save optimization results"""
        best_assimilation_output = None
        if os.path.exists(self.best_assimilation_meta):
            try:
                with open(self.best_assimilation_meta, 'r', encoding='utf-8') as f:
                    best_assimilation_output = json.load(f)
            except Exception:
                best_assimilation_output = None

        # save JSON
        output_file = os.path.join(self.output_dir, 'optimization_results.json')
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_evaluations': self.evaluation_count,
            'param_bounds': PARAM_BOUNDS,
            'soil_target': {
                'soil_param_file': self.base_soil_param_file,
                'soil_type_row': int(self.target_soil_type),
            },
            'loss_definition': {
                'metric': LOSS_METRIC,
                'smap_weight': LOSS_WEIGHT_SMAP,
            },
            'best_assimilation_output': best_assimilation_output,
            'optimization_history': self.optimization_history
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # save the Python configuration file
        config_file = os.path.join(self.output_dir, 'best_config.py')

        with open(config_file, 'w') as f:
            f.write("# Optimized Soil Parameters for cycling_enkf_jax_relative_error.py\n")
            f.write(f"# Optimization date: {datetime.now()}\n")
            f.write(f"# Best loss: {self.best_score:.6f}\n\n")
            f.write(f"SOIL_TYPE_ROW = {int(self.target_soil_type)}\n")
            f.write("SOIL_PARAM_OVERRIDES = {\n")
            for key in PARAM_BOUNDS.keys():
                value = float(self.best_params[key])
                if key == 'SATDK':
                    f.write(f"    '{key}': {value:.8e},\n")
                else:
                    f.write(f"    '{key}': {value:.6f},\n")
            f.write("}\n")

        print(f"Config file saved to: {config_file}")

        # save the Study object
        study_file = os.path.join(self.output_dir, 'optuna_study.pkl')
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)

        print(f"Study object saved to: {study_file}")

        # save CSV
        csv_file = self._save_optimization_history_csv()
        print(f"CSV history saved to: {csv_file}")

    def _save_optimization_history_csv(self) -> str:
        """save optimization history as a CSV file"""
        csv_data = []
        for entry in self.optimization_history:
            row = {
                'Trial': entry['evaluation'],
                'Timestamp': entry['timestamp'],
                'Total_Loss': entry['loss'],
            }

            for key in PARAM_BOUNDS.keys():
                row[key] = entry['params'].get(key, np.nan)

            # add metrics for each variable
            for var in ['SMAP']:
                if var in entry['metrics']:
                    metrics = entry['metrics'][var]
                    row[f'{var}_MSE'] = metrics.get('mse', np.nan)
                    row[f'{var}_R'] = metrics.get('r', np.nan)
                    row[f'{var}_RMSE'] = metrics.get('rmse', np.nan)
                    row[f'{var}_MAE'] = metrics.get('mae', np.nan)
                    row[f'{var}_Bias'] = metrics.get('bias', np.nan)
                    row[f'{var}_Loss'] = metrics.get('total_loss', np.nan)

            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.output_dir, 'optimization_history.csv')
        df.to_csv(csv_path, index=False, float_format='%.8e')

        return csv_path


def _split_trials(total_trials: int, n_workers: int) -> List[int]:
    """Split total trials into near-equal chunks for worker processes."""
    n_workers = max(1, int(n_workers))
    total_trials = max(0, int(total_trials))
    base, rem = divmod(total_trials, n_workers)
    return [base + (1 if i < rem else 0) for i in range(n_workers)]


def _build_worker_env(threads_per_worker: int) -> Dict[str, str]:
    """Create worker env vars for controlled per-process CPU threading."""
    threads = max(1, int(threads_per_worker))
    env = os.environ.copy()
    thread_vars = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for key in thread_vars:
        env[key] = str(threads)

    # Do not inject XLA_FLAGS here: accepted flag names vary across JAX/XLA builds.
    return env


def _count_complete_trials(study: optuna.Study) -> int:
    """Count completed trials with valid objective values."""
    return sum(
        1 for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None
    )


def _run_worker_mode(args) -> int:
    """Run phase-2 trials as a worker process attached to an existing study."""
    global STUDY_STORAGE, STUDY_NAME
    startup_target = max(0, int(getattr(args, "startup_target", 0)))
    phase2_target = max(1, int(getattr(args, "phase2_target", args.n_iter)))

    if args.study_storage is not None:
        STUDY_STORAGE = args.study_storage
    if args.study_name is not None:
        STUDY_NAME = args.study_name

    optimizer = EnKFRelativeErrorOptimizer(
        n_init=0,
        n_iter=args.n_iter,
        early_stop_threshold=args.early_stop,
        patience=args.patience,
        min_improvement=args.min_improvement,
        output_dir=args.output_dir,
    )
    optimizer.phase1_active = False
    optimizer.startup_target = startup_target
    optimizer.phase2_target = phase2_target

    if OPTUNA_SAMPLER == "TPE":
        kwargs = dict(OPTUNA_SAMPLER_KWARGS.get("TPE", {}))
        kwargs["n_startup_trials"] = startup_target
        sampler = TPESampler(**kwargs)
    else:
        sampler = optimizer._create_sampler()
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=STUDY_NAME,
        storage=STUDY_STORAGE,
        load_if_exists=True,
    )
    if args.n_iter > 0:
        study.optimize(optimizer.objective, n_trials=args.n_iter, show_progress_bar=False)
    return 0


def _run_parallel_mode(args, startup_trials: int) -> int:
    """Run startup sequentially, then phase-2 in multiple subprocess workers."""
    global STUDY_STORAGE, STUDY_NAME
    if args.study_storage is not None:
        STUDY_STORAGE = args.study_storage

    if args.study_name is not None:
        STUDY_NAME = args.study_name
    else:
        STUDY_NAME = f"{STUDY_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(args.output_dir, exist_ok=True)

    optimizer = EnKFRelativeErrorOptimizer(
        n_init=startup_trials,
        n_iter=args.n_iter,
        early_stop_threshold=args.early_stop,
        patience=args.patience,
        min_improvement=args.min_improvement,
        output_dir=args.output_dir,
    )

    sampler = optimizer._create_sampler()
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=STUDY_NAME,
        storage=STUDY_STORAGE,
        load_if_exists=False,
    )
    optimizer.enqueue_default_trial(study)
    default_trial_consumed = False

    # The queued default trial is a single WAITING trial. If multiple worker
    # processes start at the same time, SQLite-backed storage can let more than
    # one worker grab that same queued trial, which later crashes with
    # "Cannot tell a COMPLETE trial." Consume it once in the manager process
    # before spawning workers to avoid duplicate Trial 0 execution.
    if args.mode == "multi":
        if startup_trials <= 0:
            optimizer.phase1_active = False
        print("\n[Manager] Running queued default trial in the main process to avoid worker race on Trial 0...")
        study.optimize(optimizer.objective, n_trials=1, show_progress_bar=False)
        default_trial_consumed = True

    startup_parallel = bool(getattr(args, "startup_parallel", False))
    if startup_parallel and startup_trials > 0:
        print("\n[Manager] Running startup trials in worker processes...")
        worker_startup_target = int(startup_trials)
        parallel_total_trials = max(0, int(startup_trials + args.n_iter - (1 if default_trial_consumed else 0)))
    else:
        if startup_trials > 0:
            print("\n[Manager] Running startup trials in main process...")
            remaining_startup_trials = max(0, int(startup_trials - (1 if default_trial_consumed else 0)))
            if remaining_startup_trials > 0:
                study.optimize(optimizer.objective, n_trials=remaining_startup_trials, show_progress_bar=False)
        worker_startup_target = int(startup_trials)
        parallel_total_trials = max(0, int(args.n_iter - (1 if default_trial_consumed and startup_trials <= 0 else 0)))

    optimizer.phase1_active = False
    trial_chunks = _split_trials(parallel_total_trials, args.workers)
    worker_env = _build_worker_env(args.threads_per_worker)
    print(
        f"[Manager] Parallel config: workers={args.workers}, "
        f"threads_per_worker={max(1, int(args.threads_per_worker))}"
    )

    worker_procs = []
    script_path = os.path.abspath(__file__)
    for idx, n_trials in enumerate(trial_chunks, start=1):
        if n_trials <= 0:
            continue

        log_path = os.path.join(args.output_dir, f"worker_{idx:02d}.log")
        cmd = [
            sys.executable,
            script_path,
            "--worker",
            "--n_init", "0",
            "--startup_target", str(worker_startup_target),
            "--phase2_target", str(max(1, int(args.n_iter))),
            "--n_iter", str(n_trials),
            "--early_stop", str(args.early_stop),
            "--min_improvement", str(args.min_improvement),
            "--output_dir", args.output_dir,
            "--study_name", STUDY_NAME,
            "--study_storage", STUDY_STORAGE,
        ]
        if args.patience is not None:
            cmd.extend(["--patience", str(args.patience)])

        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            env=worker_env,
        )
        worker_procs.append((idx, n_trials, proc, log_file, log_path))
        trial_label = "total_trials" if worker_startup_target > 0 else "phase2_trials"
        print(f"[Manager] Worker-{idx} started: {trial_label}={n_trials}, log={log_path}")

    failed_workers = []
    for idx, n_trials, proc, log_file, log_path in worker_procs:
        return_code = proc.wait()
        log_file.close()
        if return_code != 0:
            failed_workers.append((idx, return_code, log_path))
        else:
            print(f"[Manager] Worker-{idx} finished (phase2_trials={n_trials}).")

    total_target_trials = int(startup_trials + args.n_iter)
    if failed_workers:
        for idx, rc, log_path in failed_workers:
            print(f"[Manager] Worker-{idx} failed (code={rc}), log={log_path}")

        study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_STORAGE)
        completed_trials = _count_complete_trials(study)
        remaining_trials = max(0, total_target_trials - completed_trials)

        print(
            f"[Manager] Partial completion detected: "
            f"completed={completed_trials}, target={total_target_trials}, remaining={remaining_trials}"
        )

        if remaining_trials > 0:
            print(
                "[Manager] Resuming remaining trials sequentially in the main process "
                "to avoid losing the whole optimization."
            )
            optimizer.phase1_active = False
            resume_sampler = optimizer._create_sampler()
            study = optuna.create_study(
                direction="minimize",
                sampler=resume_sampler,
                study_name=STUDY_NAME,
                storage=STUDY_STORAGE,
                load_if_exists=True,
            )
            study.optimize(optimizer.objective, n_trials=remaining_trials, show_progress_bar=False)
    else:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_STORAGE)

    optimizer._sync_history_from_study(study)

    print("\n" + "=" * 80)
    print("PARALLEL OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Study name: {STUDY_NAME}")
    print(f"Storage: {STUDY_STORAGE}")
    print(f"Total evaluations: {optimizer.evaluation_count}")
    print(f"Best loss: {optimizer.best_score:.6f}")

    optimizer._plot_optimization_results(study)
    optimizer._save_results(study)
    return 0


# ============================================================================
# ============================== main function ======================================
# ============================================================================

def main():
    """main function"""
    import argparse

    parser = argparse.ArgumentParser(description='EnKF SMAP MSE Soil Parameter Optimization')
    parser.add_argument('--n_init', type=int, default=DEFAULT_N_INIT,
                        help='Number of startup trials (kept for compatibility)')
    parser.add_argument('--n_startup', type=int, default=None,
                        help='Number of Optuna startup trials (overrides --n_init)')
    parser.add_argument('--n_iter', type=int, default=DEFAULT_N_ITER,
                        help='Number of Optuna iterations')
    parser.add_argument('--early_stop', type=float, default=DEFAULT_EARLY_STOP_THRESHOLD,
                        help='Early stopping loss threshold')
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE,
                        help='Patience for early stopping (None to disable)')
    parser.add_argument('--min_improvement', type=float, default=DEFAULT_MIN_IMPROVEMENT,
                        help='Minimum improvement threshold')
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], default=DEFAULT_EXEC_MODE,
                        help='Execution mode: single-process or multi-process')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help='Number of internal worker processes for phase-2 optimization')
    parser.add_argument('--threads_per_worker', type=int, default=DEFAULT_THREADS_PER_WORKER,
                        help='CPU threads for each worker process (effective in mode=multi)')
    parser.add_argument('--startup_parallel', action='store_true', default=DEFAULT_STARTUP_PARALLEL,
                        help='Run startup trials in worker processes when mode=multi')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--startup_target', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--phase2_target', type=int, default=DEFAULT_N_ITER, help=argparse.SUPPRESS)
    parser.add_argument('--study_name', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--study_storage', type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.threads_per_worker < 1:
        print("WARNING: threads_per_worker<1, forcing threads_per_worker=1")
        args.threads_per_worker = 1

    startup_trials = args.n_startup if args.n_startup is not None else args.n_init
    if args.worker:
        _run_worker_mode(args)
        return

    if args.mode == 'single':
        args.workers = 1
        args.startup_parallel = False
    else:
        if args.workers < 2:
            print("WARNING: mode='multi' but workers<2, forcing workers=2")
            args.workers = 2

    if args.mode == 'multi':
        _run_parallel_mode(args, startup_trials)
        return

    # Implementation detail.
    optimizer = EnKFRelativeErrorOptimizer(
        n_init=startup_trials,
        n_iter=args.n_iter,
        early_stop_threshold=args.early_stop,
        patience=args.patience,
        min_improvement=args.min_improvement,
        output_dir=args.output_dir
    )

    # run optimization
    optimizer.optimize()


if __name__ == "__main__":
    main()
