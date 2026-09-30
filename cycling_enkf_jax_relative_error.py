# file name: cycling_enkf_jax_relative_error.py
#
# Data-assimilation workflow detail.
# Difference from standard EnKF: during assimilation, use  K @ innovation update the deterministic trajectory directly, 
# Implementation detail.
#
# Change history:
#    2025-01-15:
# Data-assimilation workflow detail.
#      * use deterministic updates during assimilation: deterministic_state += K @ innovation
# Implementation detail.
#
import os
import time
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import pickle
import jax.tree_util as tree_util
from functools import partial
from typing import Dict, Tuple
from datetime import datetime

# Import the Noah model
from Noah_jax_simple import (
    open_forcing_file,
    run_noah_simulation_pure,
    soil_veg_gen_parm,
    month_d
)

# ============================================================================
# ============================== Configuration parameters ===============================
# ============================================================================

# --- Debug output control ---
VERBOSE = False  # Output handling.

# --- File path settings ---n
FORCING_FILE = "wudaoliang-forcing_cmfd.txt"
OBS_CSV_FILE = "wudaoliang-smap_apr01_oct31_valid.csv"
PARAM_DIR = "parameter_new"
VEG_PARAM_FILE = os.path.join(PARAM_DIR, "VEGPARM.TBL")
SOIL_PARAM_FILE = os.path.join(PARAM_DIR, "SOILPARM.TBL")
OUTPUT_FILE = "wdl100_deenkf_analysis-ob_bmlim-0.05.pkl"
EXCEPTION_LOG_FILE = "wdl100_deenkf_exceptions.log"
SOIL_PARAM_OVERRIDES = {}

# --- Time-step settings ---
# Implementation detail.
# - Noahinternal time step: 1hour(NOAH_TIMESTEP=3600)
# - assimilation interval: 6hour(KHOUR=6)
# - observation interval: 6hour(obs.dat)
ASSIMILATION_TIMESTEP = 1  # Noahinternal time step(hour)
STEPS_PER_FORCING_INTERVAL = 3  # Data-assimilation workflow detail.
ASSIMILATION_INTERVAL = 1

# --- Implementation details ---
SPINUP_BEFORE_OBS = 6
MAX_HOURS_WITHOUT_RESET = 24

# ---  EnKF Algorithm parameters ---
ENSEMBLE_SIZE = 100  # ensemble size - used only to calculate K

# --- Numerical stability options ---
USE_COVARIANCE_INFLATION = False
INFLATION_FACTOR = 1.0
MIN_VARIANCE_FLOOR = 1e-6

# --- Error covariance parameters (relative errors, percentage form) ---
OBS_ERROR_SH2O_RELATIVE = 0.05
BACKGROUND_ERROR_STC_RELATIVE = [0.01, 0.005, 0.002, 0.001]  # soil temperature (err_soilt_1-4)
BACKGROUND_ERROR_SMC_RELATIVE = [0.15, 0.06, 0.05, 0.03]
BACKGROUND_ERROR_SH2O_RELATIVE = [0.15, 0.06, 0.05, 0.03]   # liquid water (err_soilm_1-4)
MODEL_ERROR_RATIO_RELATIVE = 0.0

# --- increment limits ---
MAX_INCREMENT_STC = 1e10
MAX_INCREMENT_SMC = 1e10
MAX_INCREMENT_SH2O = 1e10

# --- Physical ranges of state variables ---
STATE_CLIP_STC_MIN = 200.0
STATE_CLIP_STC_MAX = 350.0
STATE_CLIP_SMC_MIN = 0.02
STATE_CLIP_SH2O_MIN = 0.02

# --- Vertical localization configuration ---
VERTICAL_LOCALIZATION_SH2O = [1.0, 0.0, 0.0, 0.0]
VERTICAL_LOCALIZATION_SMC = [1.0, 0.0, 0.0, 0.0]

# --- JAX random numbers ---
RNG_SEED = 42
RNG_KEY = jax.random.PRNGKey(RNG_SEED)


# ============================================================================
# ========================== Exception logging system ====================================
# ============================================================================

class EnKFExceptionLogger:
    """EnKFException logger"""

    def __init__(self, log_file):
        self.log_file = log_file
        self.exception_count = {
            'invalid_innovation_cov': 0,
            'invalid_kalman_gain': 0,
            'nan_in_forecast_cov': 0,
            'negative_forecast_cov': 0,
            'large_forecast_cov': 0,
            'nan_in_state_after_forecast': 0,
            'nan_in_state_after_analysis': 0,
            'ensemble_propagation_failed': 0,
            'cholesky_decomposition_failed': 0
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Deterministic DA Exception Log - Started at {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")

    def log_exception(self, step_idx, exception_type, details):
        self.exception_count[exception_type] += 1
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[Step {step_idx}] {datetime.now()}\n")
            f.write(f"Exception Type: {exception_type}\n")
            f.write(f"Details:\n")
            for key, value in details.items():
                f.write(f"  {key}: {value}\n")
            f.write("-" * 80 + "\n\n")

    def print_summary(self):
        print("\n" + "=" * 80)
        print("=== Exception Summary ===")
        print("=" * 80)
        total = sum(self.exception_count.values())
        if total == 0:
            print(" No exceptions occurred!")
        else:
            print(f"Total exceptions: {total}")
            for exc_type, count in self.exception_count.items():
                if count > 0:
                    print(f"  - {exc_type}: {count} times")
        print(f"\nDetailed log saved to: {self.log_file}")
        print("=" * 80)


exception_logger = EnKFExceptionLogger(EXCEPTION_LOG_FILE)


# ============================================================================
# ========================== helper functions ========================================
# ============================================================================

def load_observations(csv_file, nsoil, forcing_startdate=None):
    """Technical documentation."""
    print(f"Loading observations from {csv_file}...")
    if forcing_startdate:
        print(f"  Forcing startdate: {forcing_startdate}")
    
    try:
        df = pd.read_csv(csv_file, parse_dates=['Date'], index_col='Date')
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None, None

    sh2o_surf_col = 'SH2O(1)'
    if sh2o_surf_col not in df.columns:
        print(f"Warning: Missing observation column: {sh2o_surf_col}")
        df = df.reindex(columns=df.columns.union([sh2o_surf_col]))

    # Data-assimilation workflow detail.
    if forcing_startdate:
        try:
            forcing_start = pd.to_datetime(forcing_startdate, format='%Y%m%d%H%M')
            if forcing_start in df.index:
                df = df[df.index >= forcing_start]
                print(f"   Observations aligned to forcing startdate")
                print(f"    Obs start: {df.index[0]}, Forcing start: {forcing_start}")
            else:
                print(f"   Warning: Forcing startdate {forcing_start} not found in observations")
                print(f"    Obs range: {df.index[0]} to {df.index[-1]}")
        except Exception as e:
            print(f"   Warning: Could not align observations: {e}")
    
    obs_sh2o_surf_all = jnp.array(df[sh2o_surf_col].values, dtype=jnp.float32)
    dates = df.index.tolist()
    
    valid_count = int(jnp.sum(~jnp.isnan(obs_sh2o_surf_all)))
    print(f"  Total observation records: {len(obs_sh2o_surf_all)}")
    print(f"  Valid observations: {valid_count}")

    obs_data = {
        'SH2O_surf': obs_sh2o_surf_all,
        'dates': dates
    }

    return obs_data, dates


def find_next_observation(obs_array, current_idx, max_search_steps=500):
    """Find the next valid observation"""
    for i in range(current_idx + 1, min(current_idx + max_search_steps, len(obs_array))):
        if not jnp.isnan(obs_array[i]):
            return i
    return None


def apply_covariance_inflation_to_ensemble(ensemble_states, inflation_factor=1.02, min_variance=1e-6):
    mean = jnp.mean(ensemble_states, axis=0)
    anomalies = ensemble_states - mean
    anomalies_inflated = anomalies * jnp.sqrt(inflation_factor)
    ensemble_inflated = mean + anomalies_inflated
    return ensemble_inflated


def apply_soil_param_overrides(soil_params_jax, stype_idx: int):
    """Apply external soil-parameter overrides for the active soil type."""
    if not SOIL_PARAM_OVERRIDES:
        return soil_params_jax

    for key, value in SOIL_PARAM_OVERRIDES.items():
        if key not in soil_params_jax:
            raise KeyError(f"Unknown soil parameter override: {key}")
        soil_params_jax[key] = soil_params_jax[key].at[stype_idx].set(
            jnp.asarray(value, dtype=soil_params_jax[key].dtype)
        )
    return soil_params_jax


# ============================================================================
# ========================== Model forecast function ====================================
# ============================================================================

@partial(jax.jit, static_argnames=(
        'nsoil', 'sfcdif_option_jax', 'vegtyp_jax', 'slopetyp_jax',
        'sbeta_option_jax', 'df_option_jax', 'root_option_jax',
        'inf_option_jax', 'soc_option_ks_jax', 'soc_option2_thermal_jax',
        'ric_option_jax', 'blim_option_jax', 'ck_option_jax', 'iz0tlnd_jax',
        'forcing_data_len', 'steps_per_forcing', 'nroot_config', 'stype_idx'
))
def run_ensemble_member_step(
        state_vector,
        other_states,
        forcing_data_win,
        static_inputs,
        veg_params,
        gen_params,
        soil_params,
        date_months_win,
        date_days_win,
        date_years_win,
        nsoil: int,
        sfcdif_option_jax: int,
        vegtyp_jax: int,
        slopetyp_jax: int,
        sbeta_option_jax: int,
        df_option_jax: int,
        root_option_jax: int,
        inf_option_jax: int,
        soc_option_ks_jax: int,
        soc_option2_thermal_jax: int,
        ric_option_jax: int,
        blim_option_jax: int,
        ck_option_jax: int,
        iz0tlnd_jax: int,
        forcing_data_len: int,
        steps_per_forcing: int,
        nroot_config: int,
        stype_idx: int
):
    """Model forecast function"""
    stc = state_vector[:nsoil]
    sh2o = state_vector[nsoil:2 * nsoil]
    smc = state_vector[2 * nsoil:3 * nsoil]

    smcmax_value = soil_params['MAXSMC'][stype_idx]

    stc = jnp.clip(stc, STATE_CLIP_STC_MIN, STATE_CLIP_STC_MAX)
    smc = jnp.clip(smc, STATE_CLIP_SMC_MIN, smcmax_value)
    sh2o = jnp.clip(sh2o, STATE_CLIP_SH2O_MIN, smcmax_value)
    sh2o = jnp.minimum(sh2o, smc)

    full_veg_params = {**veg_params, 'NROOT': nroot_config}
    Q1_current = other_states.get('Q1', jnp.array(0.0, dtype=jnp.float32))

    final_outputs = run_noah_simulation_pure(
        forcing_data_jax=forcing_data_win,
        NSOIL=nsoil,
        SLDPTH_jax=static_inputs['SLDPTH'],
        ZSOIL_jax=static_inputs['ZSOIL'],
        DT_jax=static_inputs['DT'],
        EMISSI_jax=other_states.get('EMISSI', static_inputs['EMISSI']),
        ALBEDO_init=other_states.get('ALBEDO', static_inputs['ALBEDO']),
        Z0_init=other_states.get('Z0', static_inputs['Z0']),
        Z0BRD_init=other_states['Z0BRD'],
        CZIL_jax=static_inputs['CZIL'],
        CH_init=other_states.get('CH', static_inputs['CH']),
        CM_init=other_states.get('CM', static_inputs['CM']),
        sfcdif_option_jax=sfcdif_option_jax,
        CMC_init=other_states['CMC'],
        T1_init=other_states['T1'],
        STC_init=stc,
        SMC_init=smc,
        SH2O_init=sh2o,
        SNOWH_init=other_states['SNOWH'],
        SNEQV_init=other_states['SNEQV'],
        TBOT_jax=static_inputs['TBOT'],
        ZLVL_jax=static_inputs['ZLVL'],
        ZLVL_WIND_jax=static_inputs['ZLVL_WIND'],
        SHDMIN_jax=static_inputs['SHDMIN'],
        SHDMAX_jax=static_inputs['SHDMAX'],
        SNOALB_jax=static_inputs['SNOALB'],
        STYPE_jax=static_inputs['STYPE'],
        VEGTYP_jax=vegtyp_jax,
        SLOPETYP_jax=slopetyp_jax,
        RDLAI2D_jax=static_inputs['RDLAI2D'],
        USEMONALB_jax=static_inputs['USEMONALB'],
        SOC_jax=static_inputs['SOC'],
        SBETA_OPTION_jax=sbeta_option_jax,
        DF_OPTION_jax=df_option_jax,
        ROOT_OPTION_jax=root_option_jax,
        INF_OPTION_jax=inf_option_jax,
        SOC_OPTION_KS_jax=soc_option_ks_jax,
        SOC_OPTION2_THERMAL_jax=soc_option2_thermal_jax,
        RIC_OPTION_jax=ric_option_jax,
        BLIM_OPTION_jax=blim_option_jax,
        CK_OPTION_jax=ck_option_jax,
        IZ0TLND_jax=iz0tlnd_jax,
        albedo_monthly_idx=static_inputs['albedo_monthly'],
        shdfac_monthly_idx=static_inputs['shdfac_monthly'],
        lai_monthly_idx=static_inputs['lai_monthly'],
        XLAI_init_jax=other_states.get('XLAI', static_inputs['XLAI_init']),
        date_months=date_months_win,
        date_days=date_days_win,
        date_years=date_years_win,
        total_steps=1,
        forcing_data_len=forcing_data_len,
        soil_params_jax=soil_params,
        veg_params_dict=full_veg_params,
        gen_params_dict=gen_params,
        steps_per_forcing=steps_per_forcing,
        NROOT_jax=nroot_config,
        Q1_init=Q1_current
    )

    stc_new = final_outputs['STC_Kelvin'][-1]
    smc_new = final_outputs['SMC'][-1]
    sh2o_new = final_outputs['SH2O'][-1]

    new_state_vector = jnp.concatenate([stc_new, sh2o_new, smc_new])

    new_other_states = {
        'T1': final_outputs['T1_final'][-1],
        'CMC': final_outputs['CMC'][-1],
        'SNOWH': final_outputs['SNOWH'][-1],
        'SNEQV': final_outputs['SNEQV'][-1],
        'Z0BRD': final_outputs['Z0BRD_out'][-1],
        'Q1': final_outputs['Q1'][-1],
        'CH': final_outputs['CH'][-1],
        'CM': final_outputs['CM'][-1],
        'ALBEDO': final_outputs['ALBEDO'][-1],
        'PC': final_outputs['PC'][-1],
        'XLAI': final_outputs['XLAI_out'][-1],
        'Z0': final_outputs['Z0_out'][-1],
        'EMISSI': final_outputs['EMISSI'][-1],
        'SNOTIME1': final_outputs['SNOTIME1'][-1],
        'SFCTMP': final_outputs['SFCTMP'][-1],
        'storage_m_prev': final_outputs['storage_m_prev'][-1]
    }

    return new_state_vector, new_other_states



@partial(jax.jit, static_argnames=(
        'nsoil', 'n_states', 'n_ens', 'stype_idx',
        'sfcdif_option_jax', 'vegtyp_jax', 'slopetyp_jax',
        'sbeta_option_jax', 'df_option_jax', 'root_option_jax',
        'inf_option_jax', 'soc_option_ks_jax', 'soc_option2_thermal_jax',
        'ric_option_jax', 'blim_option_jax', 'ck_option_jax', 'iz0tlnd_jax',
        'forcing_data_len', 'steps_per_forcing', 'nroot_config', 'use_relative_error',
        'model_error_ratio'
))
def enkf_forecast_step_jit(
        key,
        ensemble_states,
        ensemble_other_states,
        Q_base,
        relative_error_array,
        forcing_data_win, static_model_inputs, veg_params_dict_py,
        gen_params_dict_py, soil_params_jax,
        date_months_win, date_days_win, date_years_win,
        nsoil: int, n_states: int, n_ens: int, stype_idx: int,
        sfcdif_option_jax: int, vegtyp_jax: int, slopetyp_jax: int,
        sbeta_option_jax: int, df_option_jax: int, root_option_jax: int,
        inf_option_jax: int, soc_option_ks_jax: int, soc_option2_thermal_jax: int,
        ric_option_jax: int, blim_option_jax: int, ck_option_jax: int, iz0tlnd_jax: int,
        forcing_data_len: int, steps_per_forcing: int, nroot_config: int,
        use_relative_error: bool,
        model_error_ratio: float
):
    """EnKF forecast step"""
    vmapped_model = jax.vmap(
        run_ensemble_member_step,
        in_axes=(0, 0, None, None, None, None, None, None, None, None) + (None,) * 18,
        out_axes=0
    )

    forecast_ensemble_states, forecast_ensemble_other_states = vmapped_model(
        ensemble_states,
        ensemble_other_states,
        forcing_data_win,
        static_model_inputs,
        veg_params_dict_py,
        gen_params_dict_py,
        soil_params_jax,
        date_months_win,
        date_days_win,
        date_years_win,
        nsoil,
        sfcdif_option_jax,
        vegtyp_jax,
        slopetyp_jax,
        sbeta_option_jax,
        df_option_jax,
        root_option_jax,
        inf_option_jax,
        soc_option_ks_jax,
        soc_option2_thermal_jax,
        ric_option_jax,
        blim_option_jax,
        ck_option_jax,
        iz0tlnd_jax,
        forcing_data_len,
        steps_per_forcing,
        nroot_config,
        stype_idx
    )

    if use_relative_error:
        if model_error_ratio > 1e-9:
            current_mean = jnp.mean(forecast_ensemble_states, axis=0)
            Q_variances = (jnp.abs(current_mean) * relative_error_array * model_error_ratio) ** 2
            Q_dynamic = jnp.diag(Q_variances)
            model_noise = jax.random.multivariate_normal(
                key, jnp.zeros(n_states), Q_dynamic, shape=(n_ens,)
            )
        else:
            model_noise = jnp.zeros((n_ens, n_states))
    else:
        Q_dynamic = Q_base
        model_noise = jax.random.multivariate_normal(
            key, jnp.zeros(n_states), Q_dynamic, shape=(n_ens,)
        )

    forecast_ensemble_states = forecast_ensemble_states + model_noise

    return forecast_ensemble_states, forecast_ensemble_other_states


@partial(jax.jit, static_argnames=(
        'n_states', 'n_obs', 'n_ens', 'nsoil', 'use_relative_error'
))
def enkf_analysis_step_jit(
        key,
        forecast_ensemble_states,
        observation,
        H,
        R_base,
        obs_error_relative,
        localization_weights,
        n_states, n_obs, n_ens, nsoil,
        use_relative_error: bool
):
    """Technical documentation."""
    x_f_mean = jnp.mean(forecast_ensemble_states, axis=0)
    X_f_anom = forecast_ensemble_states - x_f_mean

    y_f_ensemble = jax.vmap(lambda x: H @ x)(forecast_ensemble_states)
    y_f_mean = jnp.mean(y_f_ensemble, axis=0)
    Y_f_anom = y_f_ensemble - y_f_mean

    sh2o_smc_forecast = forecast_ensemble_states[:, nsoil:3 * nsoil]
    sh2o_smc_mean = jnp.mean(sh2o_smc_forecast, axis=0)
    X_sh2o_smc_anom = sh2o_smc_forecast - sh2o_smc_mean

    Pf_xy_sh2o_smc = X_sh2o_smc_anom.T @ Y_f_anom / (n_ens - 1)
    Pf_xy_sh2o_smc = Pf_xy_sh2o_smc * localization_weights[:, None]

    Pf_xy = jnp.zeros((n_states, n_obs))
    Pf_xy = Pf_xy.at[nsoil:3 * nsoil, :].set(Pf_xy_sh2o_smc)

    Pf_yy = Y_f_anom.T @ Y_f_anom / (n_ens - 1)

    if use_relative_error:
        obs_val = observation[0]
        min_obs_err = 0.0005
        obs_std = jnp.maximum(jnp.abs(obs_val) * obs_error_relative, min_obs_err)
        R = jnp.array([[obs_std ** 2]])
    else:
        R = R_base

    S = Pf_yy + R
    S_inv = jnp.linalg.pinv(S)
    kalman_gain = Pf_xy @ S_inv

    # Data-assimilation workflow detail.
    return kalman_gain, y_f_mean, x_f_mean



def run_cycling_deterministic():
    """Technical documentation."""
    global RNG_KEY
    RNG_KEY = jax.random.PRNGKey(RNG_SEED)

    print("=" * 80)
    print("=== Deterministic DA ===")
    print("=== Computing K using EnKF, deterministic trajectory update ===")
    print("=" * 80)
    print(f"Ensemble Size (for K calculation): {ENSEMBLE_SIZE}")
    print(f"Assimilation timestep: {ASSIMILATION_TIMESTEP} hour(s)")
    print(f"Steps per forcing interval: {STEPS_PER_FORCING_INTERVAL}")

    print(f"Observation error (relative): {OBS_ERROR_SH2O_RELATIVE:.1%}")
    print(f"Background error STC (relative): {BACKGROUND_ERROR_STC_RELATIVE}")
    print(f"Background error SMC (relative): {BACKGROUND_ERROR_SMC_RELATIVE}")
    print(f"Background error SH2O (relative): {BACKGROUND_ERROR_SH2O_RELATIVE}")
    print(f"Model error ratio: {MODEL_ERROR_RATIO_RELATIVE:.4f}")

    print("\n--- output file ---")
    print(f"Results: {OUTPUT_FILE}")
    print(f"Exception log: {EXCEPTION_LOG_FILE}")

    # --- 1. load model settings and data ---
    print("\n[1/5] Loading model setup and data...")
    try:
        (Date_pd, forcing_data_full, _, _, _, NSOIL, startdate, _, loop_for_a_while,
         latitude, longitude, forcing_timestep, noahlsm_timestep, _, T1_init, STC_init, SMC_init, SH2O_init, STYPE,
         SLDPTH, CMC_init, SNOWH_init, SNEQV_init, TBOT, VEGTYP, SOILTYP, SLOPETYP, SNOALB, ZLVL, ZLVL_WIND,
         albedo_monthly, shdfac_monthly, z0brd_monthly, lai_monthly, _, _,
         SHDMIN, SH2O_MAX, USEMONALB, RDLAI2D, LLANDUSE, SBETA_OPTION, DF_OPTION, ROOT_OPTION, INF_OPTION, SOC_OPTION_KS,
         SOC_OPTION2_THERMAL, RIC_OPTION, BLIM_OPTION, CK_OPTION, IZ0TLND, sfcdif_option, SOC) = open_forcing_file(
            FORCING_FILE)

        if LLANDUSE.upper() == "USGS":
            veg_param_path = os.path.join(PARAM_DIR, "VEGPARM-USGS.TBL")
        elif LLANDUSE.upper() == "IGBP":
            veg_param_path = os.path.join(PARAM_DIR, "VEGPARM-IGBP.TBL")

        veg_parameter_df = pd.read_csv(veg_param_path, sep=r',\s*', engine='python', header=0, index_col=0,
                                       usecols=range(18), dtype=np.float32)
        soil_parameter_df = pd.read_csv(SOIL_PARAM_FILE, sep=r',\s*', engine='python', header=0, index_col=0,
                                        usecols=range(11), dtype=np.float32)
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
        return
    except Exception as e:
        print(f"Error loading input: {e}")
        return

    soil_params_jax = {k: jnp.array(soil_parameter_df[k].to_numpy(), dtype=jnp.float32)
                       for k in ['BB', 'MAXSMC', 'SATDK', 'SATPSI', 'QTZ']}
    soil_params_jax['BLIMBX'] = jnp.full_like(soil_params_jax['BB'], 4.0)
    veg_params_dict_py = {k: veg_parameter_df.at[VEGTYP, k.upper()] for k in
                          ['SNUP', 'RS', 'RGL', 'HS', 'EMISSMIN', 'EMISSMAX', 'LAIMIN', 'LAIMAX',
                           'Z0MIN', 'Z0MAX', 'ALBEDOMIN', 'ALBEDOMAX', 'ROOTA', 'ROOTB']}
    gen_params_dict_py = soil_veg_gen_parm(LLANDUSE)
    gen_params_dict_py['SLOPE_DATA'] = jnp.array(gen_params_dict_py['SLOPE_DATA'])
    STYPE_IDX = int(STYPE[0] - 1)

    SMCMAX_base = float(soil_params_jax['MAXSMC'][STYPE_IDX])
    SOC_value = float(SOC[0])
    if SOC_value > 0:
        FTSOC = SOC_value * 2700 * (1 - SMCMAX_base) / (
                SOC_value * 2700 * (1 - SMCMAX_base) + (1 - SOC_value) * 130 + 1e-9
        )
        SMCMAX_CORRECTED = (1 - FTSOC) * SMCMAX_base + FTSOC * 0.83
        print(f"\n Soil Organic Carbon (SOC) correction applied:")
        print(f"  SOC-corrected MAXSMC: {SMCMAX_CORRECTED:.4f}")
    else:
        SMCMAX_CORRECTED = SMCMAX_base
        print(f"\n No SOC in soil, MAXSMC: {SMCMAX_CORRECTED:.4f}")

    soil_params_jax['MAXSMC'] = soil_params_jax['MAXSMC'].at[STYPE_IDX].set(SMCMAX_CORRECTED)
    soil_params_jax = apply_soil_param_overrides(soil_params_jax, STYPE_IDX)

    ZSOIL = jnp.zeros_like(SH2O_init)
    ZSOIL = ZSOIL.at[0].set(-SLDPTH[0])
    for i in range(1, NSOIL):
        ZSOIL = ZSOIL.at[i].set(-SLDPTH[i] + ZSOIL[i - 1])
    DT = jnp.array(noahlsm_timestep, dtype=jnp.float32)
    EMISSI = jnp.array(0.96, dtype=jnp.float32)
    ALBEDO = month_d(albedo_monthly, startdate)
    Z0 = month_d(z0brd_monthly, startdate)
    Z0BRD = Z0 if sfcdif_option == 1 else jnp.array(-1.E36)
    CZIL = gen_params_dict_py['CZIL_DATA']
    CH, CM = jnp.array(1.E-4, dtype=jnp.float32), jnp.array(1.E-4, dtype=jnp.float32)

    Date_filtered = Date_pd[Date_pd >= startdate]
    date_months_full = jnp.array([d.month for d in Date_filtered], dtype=jnp.int32)
    date_days_full = jnp.array([d.day for d in Date_filtered], dtype=jnp.int32)
    date_years_full = jnp.array([d.year for d in Date_filtered], dtype=jnp.int32)
    date_info_full = {'months': date_months_full, 'days': date_days_full, 'years': date_years_full}

    forcing_data_len = len(forcing_data_full)
    N_STEPS_TOTAL = forcing_data_len * STEPS_PER_FORCING_INTERVAL

    print(f"Forcing data length: {forcing_data_len}")
    print(f"Total model steps: {N_STEPS_TOTAL}")
    print(f"Soil layers: {NSOIL}")

    # --- 2. loadobservationdata ---
    print("\n[2/5] Loading observation data...")
    # Data-assimilation workflow detail.
    obs_data, obs_dates = load_observations(OBS_CSV_FILE, NSOIL, forcing_startdate=startdate)
    if obs_data is None:
        print("Failed to load observations. Exiting.")
        return
    
    obs_sh2o_surf = obs_data['SH2O_surf']
    
    valid_obs_count = int(jnp.sum(~jnp.isnan(obs_sh2o_surf)))
    print(f"Valid observations: {valid_obs_count} out of {len(obs_sh2o_surf)}")
    
    obs_error_sh2o_placeholder = 0.01
    R_matrix = jnp.array([[obs_error_sh2o_placeholder ** 2]])
    N_OBS = R_matrix.shape[0]

    # --- 3. initializestate ---
    print("\n[3/5] Initializing states...")

    current_state_mean = jnp.concatenate([STC_init, SH2O_init, SMC_init])
    N_STATES = len(current_state_mean)
    
    stc_mean = jnp.clip(current_state_mean[:NSOIL], STATE_CLIP_STC_MIN, STATE_CLIP_STC_MAX)
    sh2o_mean = jnp.clip(current_state_mean[NSOIL:2 * NSOIL], STATE_CLIP_SH2O_MIN, SMCMAX_CORRECTED)
    smc_mean = jnp.clip(current_state_mean[2 * NSOIL:3 * NSOIL], STATE_CLIP_SMC_MIN, SMCMAX_CORRECTED)
    sh2o_mean = jnp.minimum(sh2o_mean, smc_mean)
    current_state_mean = jnp.concatenate([stc_mean, sh2o_mean, smc_mean])

    relative_errors = jnp.array(
        BACKGROUND_ERROR_STC_RELATIVE +
        BACKGROUND_ERROR_SH2O_RELATIVE +
        BACKGROUND_ERROR_SMC_RELATIVE,
        dtype=jnp.float32
    )
    background_variances = (jnp.abs(current_state_mean) * relative_errors) ** 2
    P_b = jnp.diag(background_variances)
    Q_base = P_b * (MODEL_ERROR_RATIO_RELATIVE ** 2)

    # generate the initial ensemble(for calculating K)
    RNG_KEY, init_key = jax.random.split(RNG_KEY)
    ensemble_states = jax.random.multivariate_normal(
        init_key,
        current_state_mean,
        P_b,
        shape=(ENSEMBLE_SIZE,)
    )
    
    def clip_state(state_vec):
        stc = jnp.clip(state_vec[:NSOIL], STATE_CLIP_STC_MIN, STATE_CLIP_STC_MAX)
        sh2o = jnp.clip(state_vec[NSOIL:2 * NSOIL], STATE_CLIP_SH2O_MIN, SMCMAX_CORRECTED)
        smc = jnp.clip(state_vec[2 * NSOIL:3 * NSOIL], STATE_CLIP_SMC_MIN, SMCMAX_CORRECTED)
        sh2o = jnp.minimum(sh2o, smc)
        return jnp.concatenate([stc, sh2o, smc])
    
    ensemble_states = jax.vmap(clip_state)(ensemble_states)

    print(f"State vector size (N_STATES): {N_STATES}")
    print(f"Ensemble size (for K): {ENSEMBLE_SIZE}")

    albedo_monthly_idx = jnp.array(albedo_monthly[:12], dtype=jnp.float32)
    shdfac_monthly_idx = jnp.array(shdfac_monthly[:12], dtype=jnp.float32)
    lai_monthly_idx = jnp.array(lai_monthly[:12], dtype=jnp.float32)
    XLAI_init = jnp.where(RDLAI2D.astype(bool), month_d(lai_monthly, startdate), jnp.array(-1.E36))

    static_model_inputs = {
        'SLDPTH': SLDPTH, 'ZSOIL': ZSOIL, 'DT': DT, 'EMISSI': EMISSI, 'ALBEDO': ALBEDO, 'Z0': Z0, 'Z0BRD': Z0BRD,
        'CZIL': CZIL, 'CH': CH, 'CM': CM, 'T1': T1_init, 'CMC': CMC_init, 'SNOWH': SNOWH_init, 'SNEQV': SNEQV_init,
        'TBOT': jnp.array(TBOT), 'ZLVL': jnp.array(ZLVL), 'ZLVL_WIND': jnp.array(ZLVL_WIND),
        'SHDMIN': jnp.array(SHDMIN), 'SHDMAX': jnp.array(SH2O_MAX), 'SNOALB': jnp.array(SNOALB),
        'STYPE': STYPE, 'RDLAI2D': jnp.array(RDLAI2D), 'USEMONALB': jnp.array(USEMONALB), 'SOC': SOC,
        'albedo_monthly': albedo_monthly_idx, 'shdfac_monthly': shdfac_monthly_idx,
        'lai_monthly': lai_monthly_idx, 'XLAI_init': XLAI_init
    }

    static_config_scalars = {
        'sfcdif_option_jax': int(sfcdif_option), 'vegtyp_jax': int(VEGTYP), 'slopetyp_jax': int(SLOPETYP),
        'nroot_config': int(veg_parameter_df.at[VEGTYP, 'NROOT']),
        'sbeta_option_jax': int(SBETA_OPTION), 'df_option_jax': int(DF_OPTION), 'root_option_jax': int(ROOT_OPTION),
        'inf_option_jax': int(INF_OPTION), 'soc_option_ks_jax': int(SOC_OPTION_KS),
        'soc_option2_thermal_jax': int(SOC_OPTION2_THERMAL), 'ric_option_jax': int(RIC_OPTION),
        'blim_option_jax': int(BLIM_OPTION), 'ck_option_jax': int(CK_OPTION), 'iz0tlnd_jax': int(IZ0TLND),
        'forcing_data_len': 1, 'steps_per_forcing': STEPS_PER_FORCING_INTERVAL
    }

    smc_clipped = current_state_mean[2 * NSOIL:3 * NSOIL]
    initial_storage_m = jnp.sum(smc_clipped * jnp.array([-ZSOIL[i] for i in range(NSOIL)]))
    current_other_states_mean = {
        'T1': T1_init, 'CMC': CMC_init, 'SNOWH': SNOWH_init, 'SNEQV': SNEQV_init, 'Z0BRD': Z0BRD,
        'Q1': jnp.array(0.0, dtype=jnp.float32), 'CH': CH, 'CM': CM, 'ALBEDO': ALBEDO,
        'PC': jnp.array(-1.E36, dtype=jnp.float32), 'XLAI': XLAI_init, 'Z0': Z0, 'EMISSI': EMISSI,
        'SNOTIME1': jnp.array(0.0, dtype=jnp.float32), 'SFCTMP': jnp.array(0.0, dtype=jnp.float32),
        'storage_m_prev': initial_storage_m
    }

    ensemble_other_states = jax.tree_util.tree_map(
        lambda x: jnp.array([x] * ENSEMBLE_SIZE),
        current_other_states_mean
    )

    idx_sh2o1 = NSOIL
    H_matrix = jnp.zeros((N_OBS, N_STATES))
    H_matrix = H_matrix.at[0, idx_sh2o1].set(1.0)

    localization_weights = jnp.array(
        VERTICAL_LOCALIZATION_SH2O + VERTICAL_LOCALIZATION_SMC,
        dtype=jnp.float32
    )
    print(f"\n--- Vertical Localization Configuration ---")
    print(f"  SH2O layer weights: {VERTICAL_LOCALIZATION_SH2O}")
    print(f"  SMC layer weights: {VERTICAL_LOCALIZATION_SMC}")

    # initializedeterministic trajectory
    deterministic_state = current_state_mean.copy()
    deterministic_other_states = current_other_states_mean.copy()
    print(f"\n[Deterministic DA] Initialized deterministic trajectory")
    print(f"  Initial SH2O: {deterministic_state[NSOIL:2 * NSOIL]}")


    # --- 4. start the assimilation loop ---
    print("\n[4/5] Starting deterministic assimilation cycle...")
    print(f"Total steps: {N_STEPS_TOTAL}")

    total_cycle_time = 0
    num_assimilations = 0
    all_analysis_states = []
    last_regeneration_step = 0

    # process the initial state
    initial_month = int(date_info_full['months'][0])
    in_assim_season_init = (1 <= initial_month <= 12)
    
    if 0 < len(obs_sh2o_surf) and not jnp.isnan(obs_sh2o_surf[0]) and in_assim_season_init:
        print(f"\n   Initial state correction with obs[0]={obs_sh2o_surf[0]:.6f}")
        observation_vector = jnp.array([obs_sh2o_surf[0]])
        
        RNG_KEY, init_analysis_key = jax.random.split(RNG_KEY)
        K, y_f_mean, x_f_mean = enkf_analysis_step_jit(
            init_analysis_key,
            ensemble_states,
            observation_vector,
            H_matrix,
            R_matrix,
            OBS_ERROR_SH2O_RELATIVE,
            localization_weights,
            n_states=N_STATES,
            n_obs=N_OBS,
            n_ens=ENSEMBLE_SIZE,
            nsoil=NSOIL,
            use_relative_error=True
        )
        
        # Output handling.
        deterministic_obs_pred = H_matrix @ deterministic_state
        innovation = observation_vector - deterministic_obs_pred
        increment = K[:, 0] * innovation[0]
        
        increment_STC = jnp.clip(increment[:NSOIL], -MAX_INCREMENT_STC, MAX_INCREMENT_STC)
        increment_SH2O = jnp.clip(increment[NSOIL:2 * NSOIL], -MAX_INCREMENT_SH2O, MAX_INCREMENT_SH2O)
        increment_SMC = jnp.clip(increment[2 * NSOIL:3 * NSOIL], -MAX_INCREMENT_SMC, MAX_INCREMENT_SMC)
        limited_increment = jnp.concatenate([increment_STC, increment_SH2O, increment_SMC])
        
        deterministic_state = deterministic_state + limited_increment
        deterministic_state = clip_state(deterministic_state)
        
        num_assimilations += 1
        print(f"   Initial state corrected: SH2O(1)={deterministic_state[idx_sh2o1]:.6f}")
    
    # save the initial state
    all_analysis_states.append({
        'step_index': 0,
        'has_observation': (0 < len(obs_sh2o_surf) and not jnp.isnan(obs_sh2o_surf[0]) and in_assim_season_init),
        'STC': deterministic_state[:NSOIL],
        'SH2O': deterministic_state[NSOIL:2 * NSOIL],
        'SMC': deterministic_state[2 * NSOIL:3 * NSOIL],
        'std_STC': jnp.zeros(NSOIL),
        'std_SH2O': jnp.zeros(NSOIL),
        'std_SMC': jnp.zeros(NSOIL)
    })

    # main loop
    for step_idx in range(1, N_STEPS_TOTAL):
        step_start_time = time.time()
        RNG_KEY, forecast_key, analysis_key = jax.random.split(RNG_KEY, 3)

        if VERBOSE and step_idx % 240 == 0:
            print(f"\n--- Step {step_idx}/{N_STEPS_TOTAL} ---")
            print(f"   SH2O(2)={float(deterministic_state[NSOIL+1]):.6f}")

        # ensemble reset
        should_reset_ensemble = False
        next_obs_step = find_next_observation(obs_sh2o_surf, step_idx - 1)
        if next_obs_step is not None:
            hours_until_obs = next_obs_step - step_idx
            if hours_until_obs == SPINUP_BEFORE_OBS:
                should_reset_ensemble = True
        
        time_since_last_regen = step_idx - last_regeneration_step
        if time_since_last_regen >= MAX_HOURS_WITHOUT_RESET:
            should_reset_ensemble = True
        
        if should_reset_ensemble:
            reset_variances = (jnp.abs(deterministic_state) * relative_errors) ** 2
            reset_variances = jnp.maximum(reset_variances, 1e-8)
            P_reset = jnp.diag(reset_variances)
            
            reset_key = jax.random.fold_in(RNG_KEY, step_idx * 10000)
            ensemble_states = jax.random.multivariate_normal(
                reset_key,
                deterministic_state,
                P_reset,
                shape=(ENSEMBLE_SIZE,)
            )
            ensemble_states = jax.vmap(clip_state)(ensemble_states)
            
            ensemble_other_states = jax.tree_util.tree_map(
                lambda x: jnp.array([x] * ENSEMBLE_SIZE),
                deterministic_other_states
            )
            
            last_regeneration_step = step_idx

        # Data-assimilation workflow detail.
        forcing_idx_before = step_idx // STEPS_PER_FORCING_INTERVAL
        forcing_idx_after = min(forcing_idx_before + 1, forcing_data_len - 1)
        forcing_idx_before = forcing_idx_before % forcing_data_len
        forcing_idx_after = forcing_idx_after % forcing_data_len
        time_substep = step_idx % STEPS_PER_FORCING_INTERVAL
        weight_after = float(time_substep) / float(STEPS_PER_FORCING_INTERVAL)
        weight_before = 1.0 - weight_after
        forcing_before = forcing_data_full[forcing_idx_before]
        forcing_after = forcing_data_full[forcing_idx_after]
        forcing_interpolated = forcing_before * weight_before + forcing_after * weight_after
        forcing_interpolated = forcing_interpolated.at[7].set(forcing_before[7])
        forcing_data_win = forcing_interpolated.reshape(1, -1)

        current_forcing_idx = forcing_idx_before
        date_months_win = jax.lax.dynamic_slice_in_dim(date_info_full['months'], current_forcing_idx, 1, axis=0)
        date_days_win = jax.lax.dynamic_slice_in_dim(date_info_full['days'], current_forcing_idx, 1, axis=0)
        date_years_win = jax.lax.dynamic_slice_in_dim(date_info_full['years'], current_forcing_idx, 1, axis=0)

        # Implementation detail.
        deterministic_forecast_state, deterministic_forecast_other = run_ensemble_member_step(
            deterministic_state,
            deterministic_other_states,
            forcing_data_win,
            static_model_inputs,
            veg_params_dict_py,
            gen_params_dict_py,
            soil_params_jax,
            date_months_win,
            date_days_win,
            date_years_win,
            NSOIL,
            static_config_scalars['sfcdif_option_jax'],
            static_config_scalars['vegtyp_jax'],
            static_config_scalars['slopetyp_jax'],
            static_config_scalars['sbeta_option_jax'],
            static_config_scalars['df_option_jax'],
            static_config_scalars['root_option_jax'],
            static_config_scalars['inf_option_jax'],
            static_config_scalars['soc_option_ks_jax'],
            static_config_scalars['soc_option2_thermal_jax'],
            static_config_scalars['ric_option_jax'],
            static_config_scalars['blim_option_jax'],
            static_config_scalars['ck_option_jax'],
            static_config_scalars['iz0tlnd_jax'],
            static_config_scalars['forcing_data_len'],
            static_config_scalars['steps_per_forcing'],
            static_config_scalars['nroot_config'],
            STYPE_IDX
        )
        deterministic_state = deterministic_forecast_state
        deterministic_other_states = deterministic_forecast_other

        # ensemble forecast(for calculating K)
        forecast_ensemble_states, forecast_ensemble_other_states = enkf_forecast_step_jit(
            forecast_key,
            ensemble_states,
            ensemble_other_states,
            Q_base,
            relative_errors,
            forcing_data_win,
            static_model_inputs,
            veg_params_dict_py,
            gen_params_dict_py,
            soil_params_jax,
            date_months_win,
            date_days_win,
            date_years_win,
            nsoil=NSOIL,
            n_states=N_STATES,
            n_ens=ENSEMBLE_SIZE,
            stype_idx=STYPE_IDX,
            sfcdif_option_jax=static_config_scalars['sfcdif_option_jax'],
            vegtyp_jax=static_config_scalars['vegtyp_jax'],
            slopetyp_jax=static_config_scalars['slopetyp_jax'],
            sbeta_option_jax=static_config_scalars['sbeta_option_jax'],
            df_option_jax=static_config_scalars['df_option_jax'],
            root_option_jax=static_config_scalars['root_option_jax'],
            inf_option_jax=static_config_scalars['inf_option_jax'],
            soc_option_ks_jax=static_config_scalars['soc_option_ks_jax'],
            soc_option2_thermal_jax=static_config_scalars['soc_option2_thermal_jax'],
            ric_option_jax=static_config_scalars['ric_option_jax'],
            blim_option_jax=static_config_scalars['blim_option_jax'],
            ck_option_jax=static_config_scalars['ck_option_jax'],
            iz0tlnd_jax=static_config_scalars['iz0tlnd_jax'],
            forcing_data_len=static_config_scalars['forcing_data_len'],
            steps_per_forcing=static_config_scalars['steps_per_forcing'],
            nroot_config=static_config_scalars['nroot_config'],
            use_relative_error=True,
            model_error_ratio=MODEL_ERROR_RATIO_RELATIVE
        )

        ensemble_states = forecast_ensemble_states
        ensemble_other_states = forecast_ensemble_other_states

        current_month = int(date_months_win[0])
        in_assim_season = (1 <= current_month <= 12)

        # Data-assimilation workflow detail.
        has_observation = False
        # Data-assimilation workflow detail.
        if step_idx % ASSIMILATION_INTERVAL == 0:
            obs_idx = step_idx  # observations are also hourly
            if obs_idx < len(obs_sh2o_surf) and not jnp.isnan(obs_sh2o_surf[obs_idx]) and in_assim_season:
                has_observation = True
                num_assimilations += 1

                observation_vector = jnp.array([obs_sh2o_surf[obs_idx]])

                # calculate K from the ensemble
                K, y_f_mean, x_f_mean = enkf_analysis_step_jit(
                    analysis_key,
                    forecast_ensemble_states,
                    observation_vector,
                    H_matrix,
                    R_matrix,
                    OBS_ERROR_SH2O_RELATIVE,
                    localization_weights,
                    n_states=N_STATES,
                    n_obs=N_OBS,
                    n_ens=ENSEMBLE_SIZE,
                    nsoil=NSOIL,
                    use_relative_error=True
                )

                # Output handling.
                deterministic_obs_pred = H_matrix @ deterministic_state
                innovation = observation_vector - deterministic_obs_pred
                increment = K[:, 0] * innovation[0]
                
                increment_STC = jnp.clip(increment[:NSOIL], -MAX_INCREMENT_STC, MAX_INCREMENT_STC)
                increment_SH2O = jnp.clip(increment[NSOIL:2 * NSOIL], -MAX_INCREMENT_SH2O, MAX_INCREMENT_SH2O)
                increment_SMC = jnp.clip(increment[2 * NSOIL:3 * NSOIL], -MAX_INCREMENT_SMC, MAX_INCREMENT_SMC)
                limited_increment = jnp.concatenate([increment_STC, increment_SH2O, increment_SMC])
                
                deterministic_state = deterministic_state + limited_increment
                deterministic_state = clip_state(deterministic_state)

                if VERBOSE and step_idx % 48 == 6:
                    print(f"   Step {step_idx}: K[SH2O(1)]={float(K[NSOIL, 0]):.6f}, K[SH2O(2)]={float(K[NSOIL+1, 0]):.6f}")
                    print(f"   Obs={float(observation_vector[0]):.4f}, Innovation={float(innovation[0]):.4f}")
                    print(f"   Analysis SH2O(1)={float(deterministic_state[NSOIL]):.4f}")
                    print(f"   Increment SH2O(1)={float(limited_increment[NSOIL]):.6f}, SH2O(2)={float(limited_increment[NSOIL+1]):.6f}")

                # regenerate the ensemble after assimilation
                regenerate_key = jax.random.fold_in(analysis_key, step_idx)
                analysis_variances = (jnp.abs(deterministic_state) * relative_errors) ** 2
                analysis_variances = jnp.maximum(analysis_variances, 1e-8)
                P_analysis_regen = jnp.diag(analysis_variances)
                
                ensemble_states = jax.random.multivariate_normal(
                    regenerate_key,
                    deterministic_state,
                    P_analysis_regen,
                    shape=(ENSEMBLE_SIZE,)
                )
                ensemble_states = jax.vmap(clip_state)(ensemble_states)
                
                ensemble_other_states = jax.tree_util.tree_map(
                    lambda x: jnp.array([x] * ENSEMBLE_SIZE),
                    deterministic_other_states
                )
                
                last_regeneration_step = step_idx
        else:
            deterministic_state = clip_state(deterministic_state)

        # savestate
        state_dict = {
            'step_index': step_idx,
            'has_observation': has_observation,
            'STC': deterministic_state[:NSOIL],
            'SH2O': deterministic_state[NSOIL:2 * NSOIL],
            'SMC': deterministic_state[2 * NSOIL:3 * NSOIL],
            'std_STC': jnp.zeros(NSOIL),
            'std_SH2O': jnp.zeros(NSOIL),
            'std_SMC': jnp.zeros(NSOIL)
        }
        all_analysis_states.append(state_dict)

        #if step_idx % 24 == 0:
            #print(f"   Step {step_idx}: SH2O(1)={float(deterministic_state[NSOIL]):.4f}, SH2O(2)={float(deterministic_state[NSOIL+1]):.4f}")

    print(f"\nAssimilation complete! Number of assimilations: {num_assimilations}")


    # --- 5. save results ---
    print("\n[5/5] Saving results...")

    try:
        all_analysis_states_np = tree_util.tree_map(
            lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
            all_analysis_states
        )
        
        num_states_out = len(all_analysis_states_np)
        output_freq = pd.to_timedelta(noahlsm_timestep, unit='s')
        
        dates_for_output = pd.date_range(
            start=startdate,
            periods=num_states_out,
            freq=output_freq
        )
        
        analysis_states_array = np.zeros((num_states_out, NSOIL * 3))
        for i, state_dict in enumerate(all_analysis_states_np):
            analysis_states_array[i, 0:NSOIL] = state_dict['STC']
            analysis_states_array[i, NSOIL:NSOIL*2] = state_dict['SMC']
            analysis_states_array[i, NSOIL*2:NSOIL*3] = state_dict['SH2O']

        results_dict = {
            'analysis_states': analysis_states_array,
            'analysis_states_detailed': all_analysis_states_np,
            'analysis_covariances': None,
            'metadata': {
                'nsoil': NSOIL,
                'obs_error_relative': OBS_ERROR_SH2O_RELATIVE,
                'background_error_sh2o_relative': BACKGROUND_ERROR_SH2O_RELATIVE,
                'model_error_ratio_relative': MODEL_ERROR_RATIO_RELATIVE,
                'num_assimilations': num_assimilations,
                'total_time': total_cycle_time,
                'ensemble_size': ENSEMBLE_SIZE,
                'algorithm': 'Deterministic_DA',
                'error_type': 'relative',
                'dates': dates_for_output.strftime('%Y-%m-%d %H:%M:%S').tolist()
            }
        }

        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"Results saved to {OUTPUT_FILE}")

        csv_file = OUTPUT_FILE.replace('.pkl', '.csv')
        
        df_list = []
        for idx, state_dict in enumerate(all_analysis_states_np):
            row = {
                'Date': dates_for_output[idx].strftime('%Y-%m-%d %H:%M:%S'),
                'step': state_dict['step_index'],
                'has_obs': state_dict['has_observation'],
            }
            for i in range(NSOIL):
                row[f'STC({i + 1})'] = round(float(state_dict['STC'][i]), 3)
                row[f'SMC({i + 1})'] = round(float(state_dict['SMC'][i]), 6)
                row[f'SH2O({i + 1})'] = round(float(state_dict['SH2O'][i]), 6)
            df_list.append(row)

        df = pd.DataFrame(df_list)
        df.to_csv(csv_file, index=False)
        print(f"CSV results saved to {csv_file}")

    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()

    exception_logger.print_summary()

    print("\n" + "=" * 80)
    print("=== Deterministic DA Complete ===")
    print("=" * 80)
    
    return results_dict


if __name__ == "__main__":
    try:
        with open(FORCING_FILE, 'r') as f:
            content = f.read()
            ftime = float(
                [line.split('=')[1].strip() for line in content.splitlines() if 'Forcing_Timestep' in line][0])
            mtime_lines = [line for line in content.splitlines() if 'Noahlsm_Timestep' in line and '=' in line]
            if mtime_lines:
                mtime = float(mtime_lines[0].split('=')[1].strip())
            else:
                mtime = 3600.0

            # check the time stepsettings
            expected_ratio = ftime / mtime
            if expected_ratio != STEPS_PER_FORCING_INTERVAL:
                print("=" * 60)
                print(f"Warning: forcing.txt settings do not match the code!")
                print(f"  Forcing_Timestep = {ftime}s")
                print(f"  Noahlsm_Timestep = {mtime}s")
                print(f"  Ratio (Forcing/Noah) = {expected_ratio}")
                print(f"  STEPS_PER_FORCING_INTERVAL = {STEPS_PER_FORCING_INTERVAL}")
                print("=" * 60)
            else:
                print(f"✓ Time step settings are correct: Forcing={ftime}s, Noah={mtime}s, Ratio={expected_ratio}")
                print(f"  Assimilation interval: {ASSIMILATION_INTERVAL} hours")
    except Exception as e:
        print(f"Unable to check forcing.txt: {e}")

    run_cycling_deterministic()
