# file name: cycling_ekf_jax_relative_error.py
#
# Extended Kalman Filter (EKF) data assimilation implementation - relative-error version
# Fully replicates the calculation logic of DA_modify.py (without LST assimilation).
#
# Change history:
#    2026-01-14:
# Data-assimilation workflow detail.
#        - reset matrix P immediately after assimilation(recompute from the analysis statebackground-error covariance)
# Data-assimilation workflow detail.
#        - use the same observation file and configuration
#    2025-01-07:
#      * Fix Pxy calculation: SH2OandSMCcompute gains independently from each variable's covariance with the observation
# Covariance and uncertainty handling.
#    2025-11-24:
#      * reset the error covariance matrix every 6 hours (Re-initialization)
#      * no model process noise (Q=0)
# Data-assimilation workflow detail.
#      * update soil moisture only (STC gain=0)
# Model-parameter handling.
# State-variable handling.
# Covariance and uncertainty handling.
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

# --- File path settings ---
FORCING_FILE = "wudaoliang-forcing_cmfd.txt"
OBS_CSV_FILE = "wudaoliang-smap_apr01_oct31_valid.csv"
PARAM_DIR = "parameter_new"
VEG_PARAM_FILE = os.path.join(PARAM_DIR, "VEGPARM.TBL")
SOIL_PARAM_FILE = os.path.join(PARAM_DIR, "SOILPARM.TBL")
OUTPUT_FILE = "wdl_ekf_exceptions-ob_blim-0.05.pkl"
EXCEPTION_LOG_FILE = "wdl_ekf_exceptions.log"

# --- Time-step settings ---
ASSIMILATION_TIMESTEP = 1
STEPS_PER_FORCING_INTERVAL = 3

# --- Covariance reset strategy ---
# Data-assimilation workflow detail.
SPINUP_BEFORE_OBS = 6  # reset covariance matrix P 6 hours before observations(to give the system sufficient spin-up time)
MAX_HOURS_WITHOUT_RESET = 24  # Implementation detail.

# ---  EKF Algorithm parameters ---
# Covariance and uncertainty handling.

# --- Numerical stability options ---
USE_COVARIANCE_INFLATION = False  # Implementation detail.
INFLATION_FACTOR = 1.0  # not used
MIN_VARIANCE_FLOOR = 1e-6  # Implementation detail.

# --- Error covariance parameters (relative errors, percentage form) ---
# with DA_modify(3).py fully consistent parameter settings:
# DA_modify(3).py:              cycling_ekf_jax_relative_error.py:
# err_soilm_o = 0.05      -->   OBS_ERROR_SH2O_RELATIVE = 0.05
# err_soilm_1-4 = [0.15, 0.06, 0.05, 0.03]  -->  BACKGROUND_ERROR_SH2O_RELATIVE
# err_soilt_1-4 = [0.01, 0.005, 0.002, 0.001]  -->  BACKGROUND_ERROR_STC_RELATIVE
# en_numb = 10            -->   ENSEMBLE_SIZE = 10
# no process noise              -->   MODEL_ERROR_RATIO_RELATIVE = 0.0

OBS_ERROR_SH2O_RELATIVE = 0.05  # surface soil-moisture observation error (err_soilm_o)
BACKGROUND_ERROR_STC_RELATIVE = [0.01, 0.005, 0.002, 0.001]  # soil temperature (err_soilt_1-4)
BACKGROUND_ERROR_SMC_RELATIVE = [0.15, 0.06, 0.05, 0.03]
BACKGROUND_ERROR_SH2O_RELATIVE = [0.15, 0.06, 0.05, 0.03]   # liquid water (err_soilm_1-4)
MODEL_ERROR_RATIO_RELATIVE = 0.0  # no process noise (Q=0)

# --- increment limits (applied to the analysis mean) ---
# Implementation detail.
MAX_INCREMENT_STC = 1e10  # effectively unlimited
MAX_INCREMENT_SMC = 1e10  # effectively unlimited
MAX_INCREMENT_SH2O = 1e10  # Implementation detail.

# --- Physical ranges of state variables ---
STATE_CLIP_STC_MIN = 200.0
STATE_CLIP_STC_MAX = 350.0
STATE_CLIP_SMC_MIN = 0.02
STATE_CLIP_SH2O_MIN = 0.02

# --- Vertical localization configuration ---
# control the influence weights of surface observations on each layer
# Weight range [0, 1]: 0 = not updated at all, 1 = fully updated
# Output handling.
VERTICAL_LOCALIZATION_SH2O = [1.0, 0.0, 0.0, 0.0]  # State-variable handling.
VERTICAL_LOCALIZATION_SMC = [1.0, 0.0, 0.0, 0.0]   # State-variable handling.
# exampleconfiguration: 
# [1.0, 0.0, 0.0, 0.0] - update layer 1 only
# [1.0, 0.5, 0.0, 0.0] - fully update layer 1 and half-update layer 2
# [1.0, 0.7, 0.3, 0.1] - decrease with depth
# [1.0, 1.0, 1.0, 1.0] - update all layers(without localization)

# --- JAX random numbers ---
RNG_SEED = 42
RNG_KEY = jax.random.PRNGKey(RNG_SEED)


# ============================================================================
# ========================== Exception logging system ====================================
# Implementation detail.
# ============================================================================

class EKFExceptionLogger:
    """EKFException logger"""

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
        # Implementation detail.
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"EKF Exception Log - Started at {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")

    def log_exception(self, step_idx, exception_type, details):
        """Technical documentation."""
        self.exception_count[exception_type] += 1

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[Step {step_idx}] {datetime.now()}\n")
            f.write(f"Exception Type: {exception_type}\n")
            f.write(f"Details:\n")
            for key, value in details.items():
                f.write(f"  {key}: {value}\n")
            f.write("-" * 80 + "\n\n")

    def print_summary(self):
        """Print an exception summary"""
        print("\n" + "=" * 80)
        print("=== EKF Exception Summary ===")
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


# Implementation detail.
exception_logger = EKFExceptionLogger(EXCEPTION_LOG_FILE)


# ============================================================================
# ========================== helper functions ========================================
# Implementation detail.
# ============================================================================

def load_observations(csv_file, start_index, n_steps, nsoil):
    """Load the observation CSV file"""
    print(f"Loading observations from {csv_file} (start_index={start_index}, n_steps={n_steps})...")
    try:
        df = pd.read_csv(csv_file, parse_dates=['Date'], index_col='Date')
        full_len = len(df)
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None, 0

    sh2o_surf_col = 'SH2O(1)'
    if sh2o_surf_col not in df.columns:
        print(f"Warning: Missing observation column: {sh2o_surf_col}")
        df = df.reindex(columns=df.columns.union([sh2o_surf_col]))

    obs_sh2o_surf_all = jnp.array(df[sh2o_surf_col].values, dtype=jnp.float32)

    # Implementation detail.
    end_index = start_index + n_steps
    if start_index >= len(obs_sh2o_surf_all):
        obs_sh2o_surf_win = jnp.full((n_steps,), jnp.nan, dtype=jnp.float32)
    elif end_index > len(obs_sh2o_surf_all):
        pad_width = end_index - len(obs_sh2o_surf_all)
        obs_sh2o_surf_win = jnp.pad(obs_sh2o_surf_all[start_index:], (0, pad_width), constant_values=jnp.nan)
    else:
        obs_sh2o_surf_win = obs_sh2o_surf_all[start_index:end_index]

    print(f"Loaded {len(obs_sh2o_surf_win)} observation steps for SH2O(1).")

    return {
        'SH2O_surf': obs_sh2o_surf_win,
        'STC': jnp.full((n_steps, nsoil), jnp.nan, dtype=jnp.float32),
        'SMC': jnp.full((n_steps, nsoil), jnp.nan, dtype=jnp.float32)
    }, full_len


def find_next_observation(obs_array, current_idx, max_search_steps=500):
    """Find the next valid observation"""
    for i in range(current_idx + 1, min(current_idx + max_search_steps, len(obs_array))):
        if not jnp.isnan(obs_array[i]):
            return i
    return None


def apply_covariance_inflation_to_ensemble(ensemble_states, inflation_factor=1.02, min_variance=1e-6):
    """
    Apply covariance inflation to the ensemble
    P_inflated = factor * P
    This is equivalent to: anom_inflated = sqrt(factor) * anom
    """
    mean = jnp.mean(ensemble_states, axis=0)
    anomalies = ensemble_states - mean

    # Implementation detail.
    anomalies_inflated = anomalies * jnp.sqrt(inflation_factor)

    ensemble_inflated = mean + anomalies_inflated

    return ensemble_inflated


# ============================================================================
# Implementation detail.
# ============================================================================

#
# single-step model forecast function, for EKF Jacobian calculation and state forecasting
#
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
    """Technical documentation."""
    # State-variable handling.
    #  state-vector order: [STC(0:4), SH2O(4:8), SMC(8:12)]
    stc = state_vector[:nsoil]
    sh2o = state_vector[nsoil:2 * nsoil]       #  the middle segment is SH2O
    smc = state_vector[2 * nsoil:3 * nsoil]    #  the final segment is SMC

    # 2. fromsoilparameteringetactualof SMCMAX
    smcmax_value = soil_params['MAXSMC'][stype_idx]

    # State-variable handling.
    stc = jnp.clip(stc, STATE_CLIP_STC_MIN, STATE_CLIP_STC_MAX)
    smc = jnp.clip(smc, STATE_CLIP_SMC_MIN, smcmax_value)
    sh2o = jnp.clip(sh2o, STATE_CLIP_SH2O_MIN, smcmax_value)
    # State-variable handling.
    sh2o = jnp.minimum(sh2o, smc)

    # Implementation detail.
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

    # State-variable handling.
    stc_new = final_outputs['STC_Kelvin'][-1]
    smc_new = final_outputs['SMC'][-1]
    sh2o_new = final_outputs['SH2O'][-1]

    # State-variable handling.
    # State-variable handling.
    new_state_vector = jnp.concatenate([stc_new, sh2o_new, smc_new])

    # 7. updateother_states
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
        'storage_m_prev': final_outputs['storage_m_prev'][-1],
        # newflux outputs
        'SHEAT': final_outputs['SHEAT'][-1],
        'ETA': final_outputs['ETA'][-1],
        'SSOIL': final_outputs['SSOIL'][-1],
    }

    return new_state_vector, new_other_states


# ============================================================================
# Data-assimilation workflow detail.
# ============================================================================

def observation_operator(state_vector, nsoil):
    """
    observation operator h(x): extract the observation from the state vector
    hereobservationsurface soil moisture SH2O(1)
    """
    #  state-vector order: [STC(0:4), SH2O(4:8), SMC(8:12)]
    sh2o_idx = nsoil  # State-variable handling.
    return state_vector[sh2o_idx]


def compute_jacobian_H(state_vector, nsoil):
    """
    calculate the Jacobian matrix of the observation operator H = ∂h/∂x
    for a linear observation operator, this is a constant matrix
    """

    def obs_func(sv):
        return observation_operator(sv, nsoil)

    # useJAXautodiff
    jacobian_H = jax.jacfwd(obs_func)(state_vector)
    return jacobian_H


def compute_jacobian_M(
        state_vector, other_states, forcing_data_win,
        static_inputs, veg_params, gen_params, soil_params,
        date_months_win, date_days_win, date_years_win,
        nsoil, static_config_scalars, stype_idx
):
    """
    calculate the model Jacobian matrix M = ∂f/∂x
    use JAX autodiff
    """

    # define a function that accepts only state_vector
    def model_func(sv):
        new_sv, _ = run_ensemble_member_step(
            sv, other_states, forcing_data_win,
            static_inputs, veg_params, gen_params, soil_params,
            date_months_win, date_days_win, date_years_win,
            nsoil=nsoil,
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
            stype_idx=stype_idx
        )
        return new_sv

    # calculateJacobian
    jacobian_M = jax.jacfwd(model_func)(state_vector)

    # clip the Jacobian matrix to prevent covariance explosion
    MAX_JACOBIAN_ELEMENT = 200.0
    max_element = jnp.max(jnp.abs(jacobian_M))
    if max_element > MAX_JACOBIAN_ELEMENT:
        jacobian_M = jnp.clip(jacobian_M, -MAX_JACOBIAN_ELEMENT, MAX_JACOBIAN_ELEMENT)
        if VERBOSE:
            max_idx = jnp.unravel_index(jnp.argmax(jnp.abs(jacobian_M)), jacobian_M.shape)
            print(f"  Jacobian clipped: {max_element:.2f} -> {MAX_JACOBIAN_ELEMENT}")

    return jacobian_M


# ============================================================================
# Data-assimilation workflow detail.
# ============================================================================

def ekf_forecast_step(
        state_analysis,  # State-variable handling.
        P_analysis,  # Covariance and uncertainty handling.
        Q_base,  # Covariance and uncertainty handling.
        relative_error_array,  # Implementation detail.
        other_states,
        forcing_data_win,
        static_inputs,
        veg_params,
        gen_params,
        soil_params,
        date_months_win,
        date_days_win,
        date_years_win,
        nsoil,
        static_config_scalars,
        stype_idx,
        use_relative_error=True,
        model_error_ratio=0.0,
        use_inflation=False,
        inflation_factor=1.0
):
    """
    EKF forecast step
    x_f = M(x_a)
    P_f = M * P_a * M^T + Q
    """
    # State-variable handling.
    state_forecast, new_other_states = run_ensemble_member_step(
        state_analysis, other_states, forcing_data_win,
        static_inputs, veg_params, gen_params, soil_params,
        date_months_win, date_days_win, date_years_win,
        nsoil=nsoil,
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
        stype_idx=stype_idx
    )

    # Jacobian and gradient handling.
    M = compute_jacobian_M(
        state_analysis, other_states, forcing_data_win,
        static_inputs, veg_params, gen_params, soil_params,
        date_months_win, date_days_win, date_years_win,
        nsoil, static_config_scalars, stype_idx
    )

    # Implementation detail.
    if use_relative_error and model_error_ratio > 1e-9:
        Q_variances = (jnp.abs(state_forecast) * relative_error_array * model_error_ratio) ** 2
        Q = jnp.diag(Q_variances)
    else:
        Q = Q_base

    # 4. forecast error covariance
    # P_f = M * P_a * M^T + Q
    P_forecast = M @ P_analysis @ M.T + Q

    # 5. numericalsafeguard
    P_diag = jnp.diag(P_forecast)
    if jnp.any(jnp.isnan(P_diag)) or jnp.any(P_diag < 0) or jnp.max(P_diag) > 1e6:
        P_forecast = Q.copy()

    # 6. apply covariance inflation
    if use_inflation:
        P_forecast = P_forecast * inflation_factor
        P_forecast = jnp.maximum(P_forecast, MIN_VARIANCE_FLOOR)

    # 7. symmetrize and enforce positive definiteness
    P_forecast = 0.5 * (P_forecast + P_forecast.T)
    P_forecast = P_forecast + jnp.eye(P_forecast.shape[0]) * 1e-8

    return state_forecast, P_forecast, new_other_states


def ekf_analysis_step(
        state_forecast,  # State-variable handling.
        P_forecast,  # Covariance and uncertainty handling.
        observation,  # Data-assimilation workflow detail.
        R_base,  # Data-assimilation workflow detail.
        obs_error_relative,  # Data-assimilation workflow detail.
        localization_weights_sh2o,  # State-variable handling.
        localization_weights_smc,   # State-variable handling.
        nsoil,
        smcmax,
        use_relative_error=True
):
    """
    EKF analysis step - update soil moisture (SMC, SH2O) only, not temperature (STC)
    support vertical localization: through  localization_weights control the update weight of each layer
    
    consistent with EnKF: SMC usewith SH2O the same gain calculation method
    (based on SH2O's own covariance, rather than the SMC-SH2O cross-covariance)
    
    K = P_f * H^T * (H * P_f * H^T + R)^{-1}
    x_a = x_f + K * (y - h(x_f))
    P_a = (I - K*H) * P_f
    """
    # 1. calculateobservation operatorofJacobian H
    H = compute_jacobian_H(state_forecast, nsoil)
    H = H.reshape(1, -1)  # State-variable handling.

    # 2. calculate the predicted observation and innovation
    h_x = observation_operator(state_forecast, nsoil)
    innovation = observation - h_x

    # Implementation detail.
    if use_relative_error:
        min_obs_err = 0.0005
        obs_std = jnp.maximum(jnp.abs(observation) * obs_error_relative, min_obs_err)
        R = obs_std ** 2
    else:
        R = R_base

    # Covariance and uncertainty handling.
    # EnKF: Pxy = X_anom.T @ Y_anom / (n_ens - 1)
    # Covariance and uncertainty handling.
    #
    #  state-vector order: [STC(0:4), SH2O(4:8), SMC(8:12)]
    sh2o_idx_start = nsoil       # SH2Ostart index
    sh2o_idx_end = 2 * nsoil     # SH2Oend index
    smc_idx_start = 2 * nsoil    # SMCstart index
    smc_idx_end = 3 * nsoil      # SMCend index

    # extract the SH2O column from matrix H(observation sensitivity to SH2O, observationrelated only to the first SH2O layer)
    H_sh2o = H[:, sh2o_idx_start:sh2o_idx_end]  # (1, 4)

    # Covariance and uncertainty handling.
    P_sh2o_sh2o = P_forecast[sh2o_idx_start:sh2o_idx_end, sh2o_idx_start:sh2o_idx_end]  # (4, 4)
    Pf_yy = H_sh2o @ P_sh2o_sh2o @ H_sh2o.T  # (1, 1)
    S = Pf_yy + R  # Implementation detail.

    # numericalsafeguard
    MIN_INNOVATION_COV = 1e-8
    S_scalar = float(S.squeeze())
    if jnp.isnan(S_scalar) or S_scalar < MIN_INNOVATION_COV:
        S = MIN_INNOVATION_COV

    # Covariance and uncertainty handling.
    # State-variable handling.
    # Covariance and uncertainty handling.
    # Covariance and uncertainty handling.
    # 
    # Data-assimilation workflow detail.
    #   Pf_xy = X_anom.T @ Y_anom / (n_ens - 1)
    # State-variable handling.
    # 
    # Covariance and uncertainty handling.
    # Covariance and uncertainty handling.
    # Implementation detail.
    
    # Covariance and uncertainty handling.
    Pf_xy_sh2o = P_sh2o_sh2o @ H_sh2o.T  # (4, 1)
    
    #  apply vertical localization weightsto SH2O
    Pf_xy_sh2o = Pf_xy_sh2o * localization_weights_sh2o[:, None]  # (4, 1)
    
    # Covariance and uncertainty handling.
    # State-variable handling.
    # inEnKFin, SMCwithSH2O(1)ofcovariancethrough ensemblecalculate naturally
    # Covariance and uncertainty handling.
    Pf_xy_sh2o_raw = P_sh2o_sh2o @ H_sh2o.T  # Covariance and uncertainty handling.
    Pf_xy_smc = Pf_xy_sh2o_raw * localization_weights_smc[:, None]  # State-variable handling.

    Pf_xy = jnp.zeros((P_forecast.shape[0], 1))
    Pf_xy = Pf_xy.at[sh2o_idx_start:sh2o_idx_end, :].set(Pf_xy_sh2o)  # Covariance and uncertainty handling.
    Pf_xy = Pf_xy.at[smc_idx_start:smc_idx_end, :].set(Pf_xy_smc)     # SMC: use the same gain as SH2Ostructure
    # State-variable handling.

    # 6. calculate the Kalman gain
    K = Pf_xy / S_scalar  # State-variable handling.

    # 7. updatestate
    state_analysis = state_forecast + K[:, 0] * innovation

    # 8. update the covariance(simplified form)
    I_KH = jnp.eye(P_forecast.shape[0]) - K @ H
    P_analysis = I_KH @ P_forecast

    # Implementation detail.
    P_analysis = 0.5 * (P_analysis + P_analysis.T)

    return state_analysis, P_analysis, K, innovation


def run_cycling_ekf():
    """Technical documentation."""
    global RNG_KEY
    RNG_KEY = jax.random.PRNGKey(RNG_SEED)

    print("=" * 80)
    print("=" * 80)
    print(f"Algorithm: Extended Kalman Filter (EKF)")
    print(f"Assimilation timestep: {ASSIMILATION_TIMESTEP} hour(s)")
    print(f"Steps per forcing interval: {STEPS_PER_FORCING_INTERVAL}")
    print(f"Using covariance inflation: {USE_COVARIANCE_INFLATION}")
    if USE_COVARIANCE_INFLATION:
        print(f"  Inflation factor: {INFLATION_FACTOR}")

    print(f"Observation error (relative): {OBS_ERROR_SH2O_RELATIVE:.1%}")
    print(f"Background error STC (relative): {BACKGROUND_ERROR_STC_RELATIVE}")
    print(f"Background error SMC (relative): {BACKGROUND_ERROR_SMC_RELATIVE}")
    print(f"Background error SH2O (relative): {BACKGROUND_ERROR_SH2O_RELATIVE}")
    print(f"Model error ratio: {MODEL_ERROR_RATIO_RELATIVE:.4f}")

    print("\n--- output file ---")
    print(f"Results: {OUTPUT_FILE}")
    print(f"Exception log: {EXCEPTION_LOG_FILE}")

    # --- Implementation details ---
    print("\n[1/5] Loading model setup and data...")
    try:
        (Date_pd, forcing_data_full, _, _, _, NSOIL, startdate, _, loop_for_a_while,
         latitude, longitude, forcing_timestep, noahlsm_timestep, _, T1_init, STC_init, SMC_init, SH2O_init, STYPE,
         SLDPTH, CMC_init, SNOWH_init, SNEQV_init, TBOT, VEGTYP, SOILTYP, SLOPETYP, SNOALB, ZLVL, ZLVL_WIND,
         albedo_monthly, shdfac_monthly, z0brd_monthly, lai_monthly, _, _,
         SHDMIN, SH2O_MAX, USEMONALB, RDLAI2D, LLANDUSE, SBETA_OPTION, DF_OPTION, ROOT_OPTION, INF_OPTION, SOC_OPTION_KS,
         SOC_OPTION2_THERMAL, RIC_OPTION, BLIM_OPTION, CK_OPTION, IZ0TLND, sfcdif_option, SOC) = open_forcing_file(
            FORCING_FILE)

        # select the VEGPARM table for Landuse_dataset
        if LLANDUSE.upper() == "USGS":
            veg_param_path = os.path.join(PARAM_DIR, "VEGPARM-USGS.TBL")
        elif LLANDUSE.upper() == "IGBP":
            veg_param_path = os.path.join(PARAM_DIR, "VEGPARM-IGBP.TBL")

        veg_parameter_df = pd.read_csv(veg_param_path, sep=r',\s*', engine='python', header=0, index_col=0,
                                       usecols=range(18), dtype=np.float32)
        soil_parameter_df = pd.read_csv(SOIL_PARAM_FILE, sep=r',\s*', engine='python', header=0, index_col=0,
                                        usecols=range(11), dtype=np.float32)
    except FileNotFoundError as e:
        print(f"error： no input file：{e}")
        return
    except Exception as e:
        print(f"wrong in inputdata: {e}")
        return

    # Model-parameter handling.
    soil_params_jax = {k: jnp.array(soil_parameter_df[k].to_numpy(), dtype=jnp.float32)
                       for k in ['BB', 'MAXSMC', 'SATDK', 'SATPSI', 'QTZ']}
    soil_params_jax['BLIMBX'] = jnp.full_like(soil_params_jax['BB'], 4.0)
    veg_params_dict_py = {k: veg_parameter_df.at[VEGTYP, k.upper()] for k in
                          ['SNUP', 'RS', 'RGL', 'HS', 'EMISSMIN', 'EMISSMAX', 'LAIMIN', 'LAIMAX',
                           'Z0MIN', 'Z0MAX', 'ALBEDOMIN', 'ALBEDOMAX', 'ROOTA', 'ROOTB']}
    gen_params_dict_py = soil_veg_gen_parm(LLANDUSE)
    gen_params_dict_py['SLOPE_DATA'] = jnp.array(gen_params_dict_py['SLOPE_DATA'])
    STYPE_IDX = int(STYPE[0] - 1)

    # calculate MAXSMC with the organic-matter correction (consistent with the Noah model internals)
    # Implementation detail.
    SMCMAX_base = float(soil_params_jax['MAXSMC'][STYPE_IDX])
    SOC_value = float(SOC[0])
    if SOC_value > 0:
        # State-variable handling.
        FTSOC = SOC_value * 2700 * (1 - SMCMAX_base) / (
                SOC_value * 2700 * (1 - SMCMAX_base) + (1 - SOC_value) * 130 + 1e-9
        )
        SMCMAX_CORRECTED = (1 - FTSOC) * SMCMAX_base + FTSOC * 0.83
        print(f"\n Soil Organic Carbon (SOC) correction applied:")
        print(f"  SOC content: {SOC_value:.4f} (SOC_OPTION_KS={SOC_OPTION_KS})")
        print(f"  Original MAXSMC: {SMCMAX_base:.4f}")
        print(f"  FTSOC (organic fraction): {FTSOC:.4f}")
        print(f"  SOC-corrected MAXSMC: {SMCMAX_CORRECTED:.4f}")
        print(f"  Increase: +{(SMCMAX_CORRECTED - SMCMAX_base):.4f} ({(SMCMAX_CORRECTED/SMCMAX_base - 1)*100:.1f}%)")
    else:
        SMCMAX_CORRECTED = SMCMAX_base
        print(f"\n No SOC in soil (SOC={SOC_value:.4f})")
        print(f"  MAXSMC: {SMCMAX_CORRECTED:.4f}")

    # State-variable handling.
    # Implementation detail.
    soil_params_jax['MAXSMC'] = soil_params_jax['MAXSMC'].at[STYPE_IDX].set(SMCMAX_CORRECTED)
    print(f"   Updated soil_params_jax['MAXSMC'][{STYPE_IDX}] = {SMCMAX_CORRECTED:.4f}")

    # Implementation detail.
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
    observations_dict, obs_full_len = load_observations(OBS_CSV_FILE, 0, N_STEPS_TOTAL, NSOIL)
    if observations_dict is None:
        print("Failed to load observations. Exiting.")
        return

    obs_sh2o_surf = observations_dict['SH2O_surf']
    # Numerical-stability safeguard.
    obs_error_sh2o_placeholder = 0.01  # Implementation detail.
    R_matrix = jnp.array([[obs_error_sh2o_placeholder ** 2]])  # Implementation detail.
    N_OBS = R_matrix.shape[0]

    # --- 3.  initialize the EKF state ---
    print("\n[3/5] Initializing EKF state and covariance...")

    # 3a. initial state vector
    # State-variable handling.
    current_state = jnp.concatenate([STC_init, SH2O_init, SMC_init])
    N_STATES = len(current_state)

    # 3b. initial background-error covariance (P_b) - based on relative errors
    # State-variable handling.
    relative_errors = jnp.array(
        BACKGROUND_ERROR_STC_RELATIVE +
        BACKGROUND_ERROR_SH2O_RELATIVE +
        BACKGROUND_ERROR_SMC_RELATIVE,
        dtype=jnp.float32
    )
    # State-variable handling.
    background_variances = (jnp.abs(current_state) * relative_errors) ** 2
    P_current = jnp.diag(background_variances)

    # 3c. model-error covariance (Q) - as the base, actualusewhencalculate dynamically
    Q_base = P_current * (MODEL_ERROR_RATIO_RELATIVE ** 2)

    print(f"State vector size (N_STATES): {N_STATES}")
    print(f"Observation size (N_OBS): {N_OBS}")
    print(f"Background error covariance trace: {jnp.trace(P_current):.6f}")
    print(f"Model error covariance trace (initial): {jnp.trace(Q_base):.6f}")
    print(f"Using relative error mode: True")

    # 3d. initialize vertical localization weights
    localization_weights_sh2o = jnp.array(VERTICAL_LOCALIZATION_SH2O, dtype=jnp.float32)
    localization_weights_smc = jnp.array(VERTICAL_LOCALIZATION_SMC, dtype=jnp.float32)
    print(f"\n--- Vertical localization configuration ---")
    print(f"  SH2O Layer weights: {VERTICAL_LOCALIZATION_SH2O}")
    print(f"  SMC Layer weights: {VERTICAL_LOCALIZATION_SMC}")

    # Implementation detail.
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

    # [JIT errorfix] this dictionary is now used only for CPU-side argument passing
    static_config_scalars = {
        'sfcdif_option_jax': int(sfcdif_option), 'vegtyp_jax': int(VEGTYP), 'slopetyp_jax': int(SLOPETYP),
        'nroot_config': int(veg_parameter_df.at[VEGTYP, 'NROOT']),
        'sbeta_option_jax': int(SBETA_OPTION), 'df_option_jax': int(DF_OPTION), 'root_option_jax': int(ROOT_OPTION),
        'inf_option_jax': int(INF_OPTION), 'soc_option_ks_jax': int(SOC_OPTION_KS),
        'soc_option2_thermal_jax': int(SOC_OPTION2_THERMAL), 'ric_option_jax': int(RIC_OPTION),
        'blim_option_jax': int(BLIM_OPTION), 'ck_option_jax': int(CK_OPTION), 'iz0tlnd_jax': int(IZ0TLND),
        'forcing_data_len': 1, 'steps_per_forcing': STEPS_PER_FORCING_INTERVAL
    }

    # 3f.  initialize "other_states"
    initial_storage_m = jnp.sum(SMC_init * jnp.array([-ZSOIL[i] for i in range(NSOIL)]))
    current_other_states = {
        'T1': T1_init, 'CMC': CMC_init, 'SNOWH': SNOWH_init, 'SNEQV': SNEQV_init, 'Z0BRD': Z0BRD,
        'Q1': jnp.array(0.0, dtype=jnp.float32), 'CH': CH, 'CM': CM, 'ALBEDO': ALBEDO,
        'PC': jnp.array(-1.E36, dtype=jnp.float32), 'XLAI': XLAI_init, 'Z0': Z0, 'EMISSI': EMISSI,
        'SNOTIME1': jnp.array(0.0, dtype=jnp.float32), 'SFCTMP': jnp.array(0.0, dtype=jnp.float32),
        'storage_m_prev': initial_storage_m
    }

    #  state-vector order: [STC(0:4), SH2O(4:8), SMC(8:12)]
    idx_sh2o1 = NSOIL  # State-variable handling.
    print(f"Initial state STC: {current_state[:NSOIL]}")
    print(f"Initial state SH2O(1): {current_state[idx_sh2o1]:.6f}")

    # Data-assimilation workflow detail.
    # Implementation detail.
    print(f"\n[EKF Logic] Using single deterministic trajectory with periodic P reset")

    # --- 4. start the assimilation loop ---
    print("\n[4/5] Starting EKF assimilation cycle...")
    print(f"Total steps: {N_STEPS_TOTAL}")

    total_cycle_time = 0
    num_assimilations = 0
    all_analysis_states = []  # store states and diagonal variances

    # State-variable handling.
    # if there is obs[0], use it first to correct the initial state
    initial_month = int(date_info_full['months'][0])
    in_assim_season_init = (1 <= initial_month <= 12)

    if 0 < len(obs_sh2o_surf) and not jnp.isnan(obs_sh2o_surf[0]) and in_assim_season_init:
        print(f"\n   Initial state correction with obs[0]={obs_sh2o_surf[0]:.6f}")
        observation_value = float(obs_sh2o_surf[0])

        # State-variable handling.
        state_analysis, P_analysis, K, innovation = ekf_analysis_step(
            current_state,
            P_current,
            observation_value,
            obs_error_sh2o_placeholder ** 2,
            OBS_ERROR_SH2O_RELATIVE,
            localization_weights_sh2o,  # State-variable handling.
            localization_weights_smc,   # State-variable handling.
            NSOIL,
            SMCMAX_CORRECTED,
            use_relative_error=True
        )

        # increment limits
        increment = state_analysis - current_state
        increment_STC = jnp.clip(increment[:NSOIL], -MAX_INCREMENT_STC, MAX_INCREMENT_STC)
        increment_SH2O = jnp.clip(increment[NSOIL:2 * NSOIL], -MAX_INCREMENT_SH2O, MAX_INCREMENT_SH2O)
        increment_SMC = jnp.clip(increment[2 * NSOIL:3 * NSOIL], -MAX_INCREMENT_SMC, MAX_INCREMENT_SMC)
        limited_increment = jnp.concatenate([increment_STC, increment_SH2O, increment_SMC])

        current_state = current_state + limited_increment
        P_current = P_analysis

        # physical constraints
        def clip_state_init(state_vec):
            stc = jnp.clip(state_vec[:NSOIL], STATE_CLIP_STC_MIN, STATE_CLIP_STC_MAX)
            sh2o = jnp.clip(state_vec[NSOIL:2 * NSOIL], STATE_CLIP_SH2O_MIN, SMCMAX_CORRECTED)
            smc = jnp.clip(state_vec[2 * NSOIL:3 * NSOIL], STATE_CLIP_SMC_MIN, SMCMAX_CORRECTED)
            sh2o = jnp.minimum(sh2o, smc)
            return jnp.concatenate([stc, sh2o, smc])

        current_state = clip_state_init(current_state)
        num_assimilations += 1
        print(f"   Initial state corrected: SH2O(1)={current_state[idx_sh2o1]:.6f}")

    # State-variable handling.
    current_std = jnp.sqrt(jnp.diag(P_current))
    all_analysis_states.append({
        'step_index': 0,
        'has_observation': (0 < len(obs_sh2o_surf) and not jnp.isnan(obs_sh2o_surf[0]) and in_assim_season_init),
        'STC': current_state[:NSOIL],
        'SH2O': current_state[NSOIL:2 * NSOIL],      #  the middle segment is SH2O
        'SMC': current_state[2 * NSOIL:3 * NSOIL],   #  the final segment is SMC
        'std_STC': current_std[:NSOIL],
        'std_SH2O': current_std[NSOIL:2 * NSOIL],    #  the middle segment is SH2O
        'std_SMC': current_std[2 * NSOIL:3 * NSOIL],  #  the final segment is SMC
        # Output handling.
        'T1': current_other_states.get('T1', jnp.nan),
        'SHEAT': jnp.nan,  # Implementation detail.
        'ETA': jnp.nan,
        'SSOIL': jnp.nan,
        'std_T1': jnp.abs(current_other_states.get('T1', 0.0)) * 0.01,
        'std_SHEAT': jnp.nan,
        'std_ETA': jnp.nan,
        'std_SSOIL': jnp.nan,
    })
    print(f"   Saved initial state (step_idx=0)")

    # Data-assimilation workflow detail.
    last_regeneration_step = -999  # Covariance and uncertainty handling.

    for step_idx in range(1, N_STEPS_TOTAL):
        step_start_time = time.time()

        if VERBOSE and step_idx % 240 == 0:  # Implementation detail.
            print(f"\n--- Step {step_idx}/{N_STEPS_TOTAL} (预报到时刻t={step_idx}) ---")

        # Data-assimilation workflow detail.
        # Data-assimilation workflow detail.
        # Data-assimilation workflow detail.

        should_reset_covariance = False
        reset_reason = ""

        # Data-assimilation workflow detail.
        next_obs_step = find_next_observation(obs_sh2o_surf, step_idx - 1, max_search_steps=MAX_HOURS_WITHOUT_RESET)

        if next_obs_step is not None:
            hours_until_obs = next_obs_step - step_idx
            # Data-assimilation workflow detail.
            if hours_until_obs == SPINUP_BEFORE_OBS:
                should_reset_covariance = True
                reset_reason = f"Pre-obs spin-up (obs in {hours_until_obs}h at t={next_obs_step})"

        # Implementation detail.
        time_since_last_regen = step_idx - last_regeneration_step
        if time_since_last_regen >= MAX_HOURS_WITHOUT_RESET:
            should_reset_covariance = True
            reset_reason = f"Fallback reset ({time_since_last_regen}h since last reset)"

        if should_reset_covariance:
            # Covariance and uncertainty handling.
            reset_variances = (jnp.abs(current_state) * relative_errors) ** 2
            reset_variances = jnp.maximum(reset_variances, 1e-8)
            P_current = jnp.diag(reset_variances)

            last_regeneration_step = step_idx  # 🔧 update the reset record

            if VERBOSE:
                print(f"   [{reset_reason}] P matrix reset at t={step_idx}")
                print(f"     State SH2O(1): {float(current_state[idx_sh2o1]):.4f}")
                print(f"     P diagonal (SH2O1): {float(P_current[idx_sh2o1, idx_sh2o1]):.6f}")

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
        forcing_interpolated = forcing_interpolated.at[7].set(forcing_before[7])  # Implementation detail.
        forcing_data_win = forcing_interpolated.reshape(1, -1)

        current_forcing_idx = forcing_idx_before
        date_months_win = jax.lax.dynamic_slice_in_dim(date_info_full['months'], current_forcing_idx, 1, axis=0)
        date_days_win = jax.lax.dynamic_slice_in_dim(date_info_full['days'], current_forcing_idx, 1, axis=0)
        date_years_win = jax.lax.dynamic_slice_in_dim(date_info_full['years'], current_forcing_idx, 1, axis=0)

        # 4b. EKF forecast step
        state_forecast, P_forecast, new_other_states = ekf_forecast_step(
            current_state,
            P_current,
            Q_base,
            relative_errors,
            current_other_states,
            forcing_data_win,
            static_model_inputs,
            veg_params_dict_py,
            gen_params_dict_py,
            soil_params_jax,
            date_months_win,
            date_days_win,
            date_years_win,
            NSOIL,
            static_config_scalars,
            STYPE_IDX,
            use_relative_error=True,
            model_error_ratio=MODEL_ERROR_RATIO_RELATIVE,
            use_inflation=USE_COVARIANCE_INFLATION,
            inflation_factor=INFLATION_FACTOR
        )

        # Update the state and covariance.
        current_state = state_forecast
        P_current = P_forecast
        current_other_states = new_other_states

        if VERBOSE and step_idx % 24 == 0:
            print(f"   EKF forecast: t={step_idx - 1} → t={step_idx}^f")
            print(f"     SH2O(1)^f: {float(current_state[idx_sh2o1]):.4f}")
            print(f"     P_diag(SH2O1): {float(P_current[idx_sh2o1, idx_sh2o1]):.6f}")

    # Data-assimilation workflow detail.
        current_month = int(date_months_win[0])
        in_assim_season = (1 <= current_month <= 12)

        # Data-assimilation workflow detail.
        has_observation = False
        if step_idx < len(obs_sh2o_surf) and not jnp.isnan(obs_sh2o_surf[step_idx]) and in_assim_season:
            has_observation = True
            num_assimilations += 1

            observation_value = float(obs_sh2o_surf[step_idx])

            # Data-assimilation workflow detail.
            state_analysis, P_analysis, K, innovation = ekf_analysis_step(
                state_forecast,
                P_forecast,
                observation_value,
                obs_error_sh2o_placeholder ** 2,  # Implementation detail.
                OBS_ERROR_SH2O_RELATIVE,
                localization_weights_sh2o,  # State-variable handling.
                localization_weights_smc,   # State-variable handling.
                NSOIL,
                SMCMAX_CORRECTED,
                use_relative_error=True
            )

            # State-variable handling.
            increment = state_analysis - state_forecast
            increment_STC = jnp.clip(increment[:NSOIL], -MAX_INCREMENT_STC, MAX_INCREMENT_STC)
            increment_SH2O = jnp.clip(increment[NSOIL:2 * NSOIL], -MAX_INCREMENT_SH2O, MAX_INCREMENT_SH2O)  #  the middle segment is SH2O
            increment_SMC = jnp.clip(increment[2 * NSOIL:3 * NSOIL], -MAX_INCREMENT_SMC, MAX_INCREMENT_SMC)  #  the final segment is SMC
            limited_increment = jnp.concatenate([increment_STC, increment_SH2O, increment_SMC])  # State-variable handling.

            # apply the limited increment
            current_state = state_forecast + limited_increment
            P_current = P_analysis

            # physical constraints
            def clip_state(state_vec):
                stc = jnp.clip(state_vec[:NSOIL], STATE_CLIP_STC_MIN, STATE_CLIP_STC_MAX)
                sh2o = jnp.clip(state_vec[NSOIL:2 * NSOIL], STATE_CLIP_SH2O_MIN, SMCMAX_CORRECTED)  #  the middle segment is SH2O
                smc = jnp.clip(state_vec[2 * NSOIL:3 * NSOIL], STATE_CLIP_SMC_MIN, SMCMAX_CORRECTED)  #  the final segment is SMC
                sh2o = jnp.minimum(sh2o, smc)
                return jnp.concatenate([stc, sh2o, smc])  #  return the correct order [STC, SH2O, SMC]

            current_state = clip_state(current_state)

            # Data-assimilation workflow detail.
            # recompute from the analysis statebackground-error covariance
            reset_variances = (jnp.abs(current_state) * relative_errors) ** 2
            reset_variances = jnp.maximum(reset_variances, 1e-8)
            P_current = jnp.diag(reset_variances)
            
            last_regeneration_step = step_idx  # update the reset record

            # debug information
            if step_idx % 24 == 0:
                obs_pred = observation_operator(state_forecast, NSOIL)
                if VERBOSE:
                    print(
                        f"   Assimilation at t={step_idx}: Obs={observation_value:.4f}, Forecast={float(obs_pred):.4f}, Innovation={float(innovation):.4f}")
                    print(f"    Analysis SH2O(1)^a: {float(current_state[idx_sh2o1]):.4f}")
                    # Implementation detail.
                    print(f"   K values: K[SH2O(1)]={float(K[NSOIL, 0]):.6f}, K[SH2O(2)]={float(K[NSOIL+1, 0]):.6f}")
                    print(f"   K values: K[SMC(1)]={float(K[2*NSOIL, 0]):.6f}, K[SMC(2)]={float(K[2*NSOIL+1, 0]):.6f}")
                    # Implementation detail.
                    print(f"   Layer 2 Increment: SH2O(2)={float(limited_increment[NSOIL+1]):.6f}, SMC(2)={float(limited_increment[2*NSOIL+1]):.6f}")
        else:
            # no assimilation; use the forecast result
            if VERBOSE and step_idx % 240 == 0:
                print(f"  ⏭  No assimilation at t={step_idx} (out of season or no obs)")

        # State-variable handling.
        # State-variable handling.
        current_std = jnp.sqrt(jnp.diag(P_current))  # standard deviations come from the diagonal of matrix P

        current_STC = current_state[:NSOIL]
        current_SH2O = current_state[NSOIL:2 * NSOIL]     #  the middle segment is SH2O
        current_SMC = current_state[2 * NSOIL:3 * NSOIL]  #  the final segment is SMC
        current_std_STC = current_std[:NSOIL]
        current_std_SH2O = current_std[NSOIL:2 * NSOIL]    #  the middle segment is SH2O
        current_std_SMC = current_std[2 * NSOIL:3 * NSOIL] #  the final segment is SMC

        state_dict = {
            'step_index': step_idx,  #  step_idxcorresponding timet_{step_idx}
            'has_observation': has_observation,
            'STC': current_STC,
            'SMC': current_SMC,
            'SH2O': current_SH2O,  # Implementation detail.
            'std_STC': current_std_STC,
            'std_SMC': current_std_SMC,
            'std_SH2O': current_std_SH2O,
            # newflux outputs
            'T1': current_other_states.get('T1', jnp.nan),
            'SHEAT': current_other_states.get('SHEAT', jnp.nan),
            'ETA': current_other_states.get('ETA', jnp.nan),
            'SSOIL': current_other_states.get('SSOIL', jnp.nan),
            # flux uncertainty(based on relative-error estimates)
            'std_T1': jnp.abs(current_other_states.get('T1', 0.0)) * 0.01,  # Implementation detail.
            'std_SHEAT': jnp.abs(current_other_states.get('SHEAT', 0.0)) * 0.10,  # Implementation detail.
            'std_ETA': jnp.abs(current_other_states.get('ETA', 0.0)) * 0.10,  # Implementation detail.
            'std_SSOIL': jnp.abs(current_other_states.get('SSOIL', 0.0)) * 0.10,  # Implementation detail.
        }
        all_analysis_states.append(state_dict)
        
        # Implementation detail.
        #if step_idx % 24 == 0:
            #print(f"   Step {step_idx}: SH2O(1)={float(current_SH2O[0]):.4f}, SH2O(2)={float(current_SH2O[1]):.4f}, SMC(1)={float(current_SMC[0]):.4f}, SMC(2)={float(current_SMC[1]):.4f}")

        # debug information
        if VERBOSE and step_idx % 24 == 0 and not has_observation:
            print(f"  No obs, State: {current_SH2O[0]:.4f}, Std: {current_std_SH2O[0]:.4f}")

    print(f"\nAssimilation complete! Total time: {total_cycle_time:.1f} seconds, Number of assimilations: {num_assimilations}")

    # --- 5. save results ---
    print("\n[5/5] Saving results...")

    try:
        # convert to NumPy arrays for saving
        all_analysis_states_np = tree_util.tree_map(
            lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
            all_analysis_states
        )

        # Implementation detail.
        # stateorder: [STC(4), SMC(4), SH2O(4)]
        num_states_out = len(all_analysis_states_np)
        output_freq = pd.to_timedelta(noahlsm_timestep, unit='s')

        # generate the date sequence
        dates_for_output = pd.date_range(
            start=startdate,
            periods=num_states_out,
            freq=output_freq
        )

        # convert to a two-dimensional array
        analysis_states_array = np.zeros((num_states_out, NSOIL * 3))
        for i, state_dict in enumerate(all_analysis_states_np):
            # [STC(4), SMC(4), SH2O(4)]
            analysis_states_array[i, 0:NSOIL] = state_dict['STC']
            analysis_states_array[i, NSOIL:NSOIL * 2] = state_dict['SMC']
            analysis_states_array[i, NSOIL * 2:NSOIL * 3] = state_dict['SH2O']

        results_dict = {
            'analysis_states': analysis_states_array,  # Implementation detail.
            'analysis_states_detailed': all_analysis_states_np,  # retain the original dictionary list for CSV output
            'analysis_covariances': None,  # Data-assimilation workflow detail.
            'metadata': {
                'nsoil': NSOIL,
                'obs_error_relative': OBS_ERROR_SH2O_RELATIVE,
                'background_error_sh2o_relative': BACKGROUND_ERROR_SH2O_RELATIVE,
                'model_error_ratio_relative': MODEL_ERROR_RATIO_RELATIVE,
                'num_assimilations': num_assimilations,
                'total_time': total_cycle_time,
                'use_inflation': USE_COVARIANCE_INFLATION,
                'inflation_factor': INFLATION_FACTOR if USE_COVARIANCE_INFLATION else None,
                'algorithm': 'EKF',
                'error_type': 'relative',
                'dates': dates_for_output.strftime('%Y-%m-%d %H:%M:%S').tolist()
            }
        }

        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"Results saved to {OUTPUT_FILE}")

        # Output handling.
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
                row[f'std_STC({i + 1})'] = round(float(state_dict['std_STC'][i]), 3)
                row[f'std_SMC({i + 1})'] = round(float(state_dict['std_SMC'][i]), 6)
                row[f'std_SH2O({i + 1})'] = round(float(state_dict['std_SH2O'][i]), 6)
            # newflux outputs
            row['T1'] = round(float(state_dict.get('T1', np.nan)), 3)
            row['SHEAT'] = round(float(state_dict.get('SHEAT', np.nan)), 3)
            row['ETA'] = round(float(state_dict.get('ETA', np.nan)), 3)
            row['SSOIL'] = round(float(state_dict.get('SSOIL', np.nan)), 3)
            row['std_T1'] = round(float(state_dict.get('std_T1', np.nan)), 3)
            row['std_SHEAT'] = round(float(state_dict.get('std_SHEAT', np.nan)), 3)
            row['std_ETA'] = round(float(state_dict.get('std_ETA', np.nan)), 3)
            row['std_SSOIL'] = round(float(state_dict.get('std_SSOIL', np.nan)), 3)
            df_list.append(row)

        df = pd.DataFrame(df_list)
        df.to_csv(csv_file, index=False)
        print(f"CSV results saved to {csv_file}")

    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()

    # print the exception-log summary
    exception_logger.print_summary()

    print("\n" + "=" * 80)
    print("=== EKF Assimilation Complete ===")
    print("=" * 80)

    # Output handling.
    return results_dict


if __name__ == "__main__":
    # Implementation detail.
    try:
        with open(FORCING_FILE, 'r') as f:
            content = f.read()
            ftime = float(
                [line.split('=')[1].strip() for line in content.splitlines() if 'Forcing_Timestep' in line][0])
            # Implementation detail.
            mtime_lines = [line for line in content.splitlines() if 'Noahlsm_Timestep' in line and '=' in line]
            if mtime_lines:
                mtime = float(mtime_lines[0].split('=')[1].strip())
            else:
                mtime = 3600.0  # Implementation detail.

            if ftime / mtime != STEPS_PER_FORCING_INTERVAL:
                print("=" * 60)
                print(f"Warning: forcing.txt settings do not match the code!")
                print(f"  Forcing_Timestep = {ftime}s")
                print(f"  Noahlsm_Timestep = {mtime}s")
                print(f"  Ratio = {ftime / mtime}")
                print(f"  STEPS_PER_FORCING_INTERVAL = {STEPS_PER_FORCING_INTERVAL}")
                print("Expected ratio should equal STEPS_PER_FORCING_INTERVAL")
                print("=" * 60)
            else:
                print(f"\u2713 Time step settings are correct: Forcing={ftime}s, Noah={mtime}s, Ratio={ftime / mtime}")
    except Exception as e:
        print(f"Unable to check forcing.txt: {e}")

    run_cycling_ekf()
