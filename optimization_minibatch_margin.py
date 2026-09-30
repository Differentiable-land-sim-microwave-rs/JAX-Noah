from __future__ import annotations

import pickle
import time
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd

from Noah_jax_simple import (
    month_d,
    open_forcing_file,
    run_noah_simulation_pure,
    soil_veg_gen_parm,
)


# ==========================================
# 1. Configuration
# ==========================================

WINDOW_SIZE_MODEL_STEPS = 480
WINDOWS_PER_BATCH = 5
BATCHES_PER_EPOCH = 40
MAX_BATCH_ATTEMPTS_PER_EPOCH = BATCHES_PER_EPOCH * 5
NUM_EPOCHS = 100
VALIDATION_INTERVAL_EPOCHS = 1
TRAIN_SPLIT = 0.60
USE_VALIDATION_SPLIT = False
RANDOM_SEED = 2023

STEPS_PER_FORCING = 3
WINDOW_LEN_FORCING = (WINDOW_SIZE_MODEL_STEPS // STEPS_PER_FORCING) + 1

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR
Forcing_DIR = PROJECT_DIR
PARAM_DIR = PROJECT_DIR / "parameter_new"
SOIL_PARAM_FILE = PARAM_DIR / "SOILPARM.TBL"

FORCING_FILE = Forcing_DIR / "wudaoliang-forcing_cmfd.txt"
OBS_SMAP_CSV_FILE = BASE_DIR / "wudaoliang-smap_apr01_oct31_valid.csv"

# Current SMAP csv is still on local time and needs BJT -> UTC.
SMAP_TIME_SHIFT_TO_UTC_HOURS = -8.0

LEARNING_RATES = {
    "BB": 2e-3,
    "log_SATDK": 1e-3,
    "SATPSI": 1e-3,
    "MAXSMC": 2e-3,
}

USE_SGDR = True
SGDR_FIRST_CYCLE_EPOCHS = 5
SGDR_CYCLE_MULT = 2.0
SGDR_NUM_CYCLES = 4
SGDR_MIN_LR_SCALE = 0.05


LOSS_WEIGHTS = {
    "SMAP": 1.0,
}

SMAP_VALID_RANGE = (0.02, 0.60)

# Strictly use the nearest already-happened SMAP observation before the window start.
# 96h is safer for sparse SMAP revisit while avoiding stale carry-over.
SMAP_WARM_MAX_GAP_HOURS = 96

INIT_PARAMS = None
# INIT_PARAMS = {
#     "BB": 2.79,
#     "SATDK": 4.87e-5,
#     "SATPSI": 0.1347,
#     "MAXSMC": 0.42,
# }

# Hard-coded from the current SOILPARM.TBL first 12 soil classes.
PARAM_BOUNDS_PHYSICAL = {
    "BB": (2.79, 11.55),
    "log_SATDK": (jnp.log(9.74e-7), jnp.log(1.41e-5)),
    "SATPSI": (0.036, 0.759),
    "MAXSMC": (0.339, 0.476),
}

PARAM_BOUNDS_EXTENDED = {
    "BB": (2.79, 11.55),
    "log_SATDK": (jnp.log(9.74e-7), jnp.log(1.41e-5)),
    "SATPSI": (0.036, 0.759),
    "MAXSMC": (0.339, 0.476),
}

# Keep the margin term as a regularizer instead of letting it dominate the
# satellite data term.
MARGIN_PENALTY_WEIGHT = 0.05

DIAG_PARAM_ORDER = (
    "BB_raw",
    "log_SATDK_raw",
    "SATPSI_raw",
    "MAXSMC_raw",
)

DIAG_PARAM_LABELS = {
    "BB_raw": "BB",
    "log_SATDK_raw": "logK",
    "SATPSI_raw": "PSI",
    "MAXSMC_raw": "MAXSMC",
}


# ==========================================
# 2. Parameter transforms
# ==========================================

def soft_clamp(x, lo, hi, scale=0.5):
    return lo + (hi - lo) * jax.nn.sigmoid(x * scale)


def inverse_soft_clamp(y, lo, hi, scale=0.5):
    p = (y - lo) / (hi - lo)
    p = jnp.clip(p, 0.01, 0.99)
    return jnp.log(p / (1.0 - p)) / scale


def soft_margin_penalty(val, lo_phys, hi_phys):
    penalty_low = jax.nn.relu(lo_phys - val) ** 2
    penalty_high = jax.nn.relu(val - hi_phys) ** 2
    return penalty_low + penalty_high


def decode_param_views(params):
    bb = soft_clamp(params["BB_raw"], *PARAM_BOUNDS_EXTENDED["BB"], scale=0.5)
    log_satdk = soft_clamp(params["log_SATDK_raw"], *PARAM_BOUNDS_EXTENDED["log_SATDK"], scale=0.5)
    satpsi = soft_clamp(params["SATPSI_raw"], *PARAM_BOUNDS_EXTENDED["SATPSI"], scale=0.5)
    maxsmc = soft_clamp(params["MAXSMC_raw"], *PARAM_BOUNDS_EXTENDED["MAXSMC"], scale=0.5)
    return {
        "BB": bb,
        "log_SATDK": log_satdk,
        "SATDK": jnp.exp(log_satdk),
        "SATPSI": satpsi,
        "MAXSMC": maxsmc,
    }


def abs_scalar_tree(tree):
    return jax.tree_util.tree_map(lambda leaf: jnp.abs(leaf), tree)


def nan_scalar_tree_like(tree):
    return jax.tree_util.tree_map(lambda leaf: jnp.full_like(leaf, jnp.nan), tree)


def tree_to_ordered_float_dict(tree):
    values = {}
    for key in DIAG_PARAM_ORDER:
        if key in tree:
            values[key] = float(np.asarray(tree[key]))
    return values


def average_ordered_scalar_dicts(diag_dicts):
    averaged = {}
    for key in DIAG_PARAM_ORDER:
        key_vals = np.array([diag.get(key, np.nan) for diag in diag_dicts], dtype=np.float64)
        averaged[key] = float(np.nanmean(key_vals)) if np.any(np.isfinite(key_vals)) else np.nan
    return averaged


def format_ordered_scalar_dict(label, values):
    parts = []
    for key in DIAG_PARAM_ORDER:
        display_name = DIAG_PARAM_LABELS[key]
        value = values.get(key, np.nan)
        if np.isfinite(value):
            parts.append(f"{display_name}={value:.2e}")
        else:
            parts.append(f"{display_name}=nan")
    return f" | {label}[{', '.join(parts)}]"


def compute_margin_penalty(params):
    decoded = decode_param_views(params)
    penalty_bb = soft_margin_penalty(decoded["BB"], *PARAM_BOUNDS_PHYSICAL["BB"])
    penalty_satdk = soft_margin_penalty(decoded["log_SATDK"], *PARAM_BOUNDS_PHYSICAL["log_SATDK"])
    penalty_satpsi = soft_margin_penalty(decoded["SATPSI"], *PARAM_BOUNDS_PHYSICAL["SATPSI"])
    penalty_maxsmc = soft_margin_penalty(decoded["MAXSMC"], *PARAM_BOUNDS_PHYSICAL["MAXSMC"])
    return penalty_bb + penalty_satdk + penalty_satpsi + penalty_maxsmc


def build_sgdr_schedule(base_lr):
    if base_lr <= 0.0:
        return 0.0

    cycle_decay_steps = max(int(round(SGDR_FIRST_CYCLE_EPOCHS * BATCHES_PER_EPOCH)), 1)
    cosine_kwargs = []
    for _cycle_idx in range(SGDR_NUM_CYCLES):
        cosine_kwargs.append(
            {
                "init_value": float(base_lr),
                "peak_value": float(base_lr),
                "warmup_steps": 0,
                "decay_steps": cycle_decay_steps,
                "end_value": float(base_lr * SGDR_MIN_LR_SCALE),
            }
        )
        cycle_decay_steps = max(int(round(cycle_decay_steps * SGDR_CYCLE_MULT)), 1)

    return optax.sgdr_schedule(cosine_kwargs)


def build_optimizer():
    learning_rates_raw = {
        "BB_raw": LEARNING_RATES["BB"],
        "log_SATDK_raw": LEARNING_RATES["log_SATDK"],
        "SATPSI_raw": LEARNING_RATES["SATPSI"],
        "MAXSMC_raw": LEARNING_RATES["MAXSMC"],
    }
    learning_rate_transforms = {}
    for param_name, base_lr in learning_rates_raw.items():
        if base_lr <= 0.0:
            learning_rate_transforms[param_name] = optax.set_to_zero()
        else:
            learning_rate = build_sgdr_schedule(base_lr) if USE_SGDR else base_lr
            learning_rate_transforms[param_name] = optax.adam(learning_rate)

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            learning_rate_transforms,
            lambda p: {k: k for k in p},
        ),
    )


def build_current_soil_params(optim_params, static_arrays, static_config_scalars):
    decoded = decode_param_views(optim_params)
    current_soil_params = static_arrays["base_soil_params_jax"].copy()
    soil_idx = static_config_scalars["soil_type_index_to_optimize"] - 1

    current_soil_params["BB"] = current_soil_params["BB"].at[soil_idx].set(decoded["BB"])
    current_soil_params["SATDK"] = current_soil_params["SATDK"].at[soil_idx].set(decoded["SATDK"])
    current_soil_params["SATPSI"] = current_soil_params["SATPSI"].at[soil_idx].set(decoded["SATPSI"])
    current_soil_params["MAXSMC"] = current_soil_params["MAXSMC"].at[soil_idx].set(decoded["MAXSMC"])
    return current_soil_params, decoded


def make_initial_params(base_soil_params, soil_idx):
    if INIT_PARAMS is not None:
        print("    Using custom initial parameters")
        maxsmc_init = INIT_PARAMS.get("MAXSMC", float(base_soil_params["MAXSMC"][soil_idx]))
        return {
            "BB_raw": inverse_soft_clamp(
                jnp.array(INIT_PARAMS["BB"], dtype=jnp.float32),
                *PARAM_BOUNDS_EXTENDED["BB"],
                scale=0.5,
            ),
            "log_SATDK_raw": inverse_soft_clamp(
                jnp.log(jnp.array(INIT_PARAMS["SATDK"], dtype=jnp.float32)),
                *PARAM_BOUNDS_EXTENDED["log_SATDK"],
                scale=0.5,
            ),
            "SATPSI_raw": inverse_soft_clamp(
                jnp.array(INIT_PARAMS["SATPSI"], dtype=jnp.float32),
                *PARAM_BOUNDS_EXTENDED["SATPSI"],
                scale=0.5,
            ),
            "MAXSMC_raw": inverse_soft_clamp(
                jnp.array(maxsmc_init, dtype=jnp.float32),
                *PARAM_BOUNDS_EXTENDED["MAXSMC"],
                scale=0.5,
            ),
        }

    print("    Using SOILPARM.TBL defaults as initialization")
    return {
        "BB_raw": inverse_soft_clamp(
            base_soil_params["BB"][soil_idx],
            *PARAM_BOUNDS_EXTENDED["BB"],
            scale=0.5,
        ),
        "log_SATDK_raw": inverse_soft_clamp(
            jnp.log(base_soil_params["SATDK"][soil_idx]),
            *PARAM_BOUNDS_EXTENDED["log_SATDK"],
            scale=0.5,
        ),
        "SATPSI_raw": inverse_soft_clamp(
            base_soil_params["SATPSI"][soil_idx],
            *PARAM_BOUNDS_EXTENDED["SATPSI"],
            scale=0.5,
        ),
        "MAXSMC_raw": inverse_soft_clamp(
            base_soil_params["MAXSMC"][soil_idx],
            *PARAM_BOUNDS_EXTENDED["MAXSMC"],
            scale=0.5,
        ),
    }


# ==========================================
# 3. Satellite loading and cleaning
# ==========================================

def _select_obs_column(df, preferred_cols, tag):
    for col in preferred_cols:
        if col in df.columns:
            return col

    fallback_cols = [c for c in df.columns if c != "Date"]
    if not fallback_cols:
        raise ValueError(f"{tag} file has no usable observation column")

    fallback = fallback_cols[0]
    print(f"    Warning: {tag} preferred columns {preferred_cols} missing, fallback to {fallback}")
    return fallback


def _build_previous_observation_arrays(series, max_gap_hours=None):
    values = series.to_numpy(dtype=np.float32)
    prev_values = np.full(values.shape, np.nan, dtype=np.float32)
    prev_age_hours = np.full(values.shape, np.nan, dtype=np.float32)

    last_value = np.nan
    last_idx = -1
    for i, value in enumerate(values):
        if last_idx >= 0:
            age = float(i - last_idx)
            if max_gap_hours is None or age <= max_gap_hours:
                prev_values[i] = last_value
                prev_age_hours[i] = age
        if np.isfinite(value):
            last_value = float(value)
            last_idx = i

    return prev_values, prev_age_hours


def _read_satellite_series(
    csv_path,
    full_idx,
    preferred_cols,
    valid_range,
    tag,
    shift_to_utc_hours,
    warm_start_max_gap_hours=None,
    build_previous_values=False,
):
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError(f"{tag} file {csv_path} is missing a Date column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    col = _select_obs_column(df, preferred_cols, tag)

    df["value"] = pd.to_numeric(df[col], errors="coerce")
    if shift_to_utc_hours != 0:
        df["Date"] = df["Date"] + pd.to_timedelta(shift_to_utc_hours, unit="h")
    df["Date"] = df["Date"].dt.floor("h")

    series = df.groupby("Date", as_index=True)["value"].mean().sort_index().reindex(full_idx)
    lo, hi = valid_range
    series = series.where((series >= lo) & (series <= hi), np.nan)

    prev_values = None
    prev_age_hours = None
    if build_previous_values:
        prev_values, prev_age_hours = _build_previous_observation_arrays(series, warm_start_max_gap_hours)

    valid_mask = series.notna()
    valid_count = int(valid_mask.sum())
    valid_hours = pd.Series(series.index[valid_mask]).dt.hour.value_counts().sort_index().to_dict()
    print(
        f"    {tag}: source={Path(csv_path).name}:{col}, shift_to_utc={shift_to_utc_hours:+.1f}h, "
        f"valid={valid_count}, valid_hours={valid_hours}"
    )

    return {
        "obs": series.to_numpy(dtype=np.float32),
        "prev_values": prev_values,
        "prev_age_hours": prev_age_hours,
        "column": col,
    }


# ==========================================
# 4. Model data assembly
# ==========================================

def load_aligned_data(forcing_file, smap_csv_path):
    (
        _Date_pd,
        forcing_data_jax,
        _,
        _,
        _,
        NSOIL,
        startdate,
        enddate,
        _loop_for_a_while,
        _,
        _,
        forcing_timestep,
        noahlsm_timestep,
        _,
        T1_init,
        STC_init,
        SMC_init,
        SH2O_init,
        STYPE_jax,
        SLDPTH_jax,
        CMC_init,
        SNOWH_init,
        SNEQV_init,
        TBOT,
        VEGTYP,
        SOILTYP_idx,
        SLOPETYP_idx,
        SNOALB,
        ZLVL,
        ZLVL_WIND,
        albedo_monthly,
        shdfac_monthly,
        z0brd_monthly,
        lai_monthly,
        _,
        _,
        SHDMIN,
        SHDMAX,
        USEMONALB,
        RDLAI2D,
        LLANDUSE,
        SBETA_OPTION,
        DF_OPTION,
        ROOT_OPTION,
        INF_OPTION,
        SOC_OPTION_KS,
        SOC_OPTION2_THERMAL,
        RIC_OPTION,
        BLIM_OPTION,
        CK_OPTION,
        IZ0TLND,
        sfcdif_option,
        SOC_jax,
    ) = open_forcing_file(str(forcing_file))

    full_idx = pd.date_range(start=startdate, end=enddate, freq="1h")

    print("  Loading satellite observations...")
    smap_data = _read_satellite_series(
        smap_csv_path,
        full_idx,
        preferred_cols=["SH2O(1)"],
        valid_range=SMAP_VALID_RANGE,
        tag="SMAP",
        shift_to_utc_hours=SMAP_TIME_SHIFT_TO_UTC_HOURS,
        warm_start_max_gap_hours=SMAP_WARM_MAX_GAP_HOURS,
        build_previous_values=True,
    )

    if LLANDUSE.upper() == "IGBP":
        veg_param_path = PARAM_DIR / "VEGPARM-IGBP.TBL"
    else:
        veg_param_path = PARAM_DIR / "VEGPARM-USGS.TBL"

    veg_parameter_df = pd.read_csv(
        veg_param_path,
        sep=r",\s*",
        engine="python",
        header=0,
        index_col=0,
        usecols=range(18),
        dtype=np.float32,
    )
    soil_parameter_df = pd.read_csv(
        SOIL_PARAM_FILE,
        sep=r",\s*",
        engine="python",
        header=0,
        index_col=0,
        usecols=range(11),
        dtype=np.float32,
    )

    base_soil_params_jax = {
        "BB": jnp.array(soil_parameter_df["BB"].to_numpy(), dtype=jnp.float32),
        "MAXSMC": jnp.array(soil_parameter_df["MAXSMC"].to_numpy(), dtype=jnp.float32),
        "SATDK": jnp.array(soil_parameter_df["SATDK"].to_numpy(), dtype=jnp.float32),
        "SATPSI": jnp.array(soil_parameter_df["SATPSI"].to_numpy(), dtype=jnp.float32),
        "QTZ": jnp.array(soil_parameter_df["QTZ"].to_numpy(), dtype=jnp.float32),
        "BLIMBX": jnp.full_like(jnp.array(soil_parameter_df["BB"].to_numpy(), dtype=jnp.float32), 4.0),
    }

    veg_params_dict_static = {
        "NROOT": int(veg_parameter_df.at[VEGTYP, "NROOT"]),
        "SNUP": float(veg_parameter_df.at[VEGTYP, "SNUP"]),
        "RS": float(veg_parameter_df.at[VEGTYP, "RS"]),
        "RGL": float(veg_parameter_df.at[VEGTYP, "RGL"]),
        "HS": float(veg_parameter_df.at[VEGTYP, "HS"]),
        "EMISSMIN": float(veg_parameter_df.at[VEGTYP, "EMISSMIN"]),
        "EMISSMAX": float(veg_parameter_df.at[VEGTYP, "EMISSMAX"]),
        "LAIMIN": float(veg_parameter_df.at[VEGTYP, "LAIMIN"]),
        "LAIMAX": float(veg_parameter_df.at[VEGTYP, "LAIMAX"]),
        "Z0MIN": float(veg_parameter_df.at[VEGTYP, "Z0MIN"]),
        "Z0MAX": float(veg_parameter_df.at[VEGTYP, "Z0MAX"]),
        "ALBEDOMIN": float(veg_parameter_df.at[VEGTYP, "ALBEDOMIN"]),
        "ALBEDOMAX": float(veg_parameter_df.at[VEGTYP, "ALBEDOMAX"]),
        "ROOTA": float(veg_parameter_df.at[VEGTYP, "ROOTA"]),
        "ROOTB": float(veg_parameter_df.at[VEGTYP, "ROOTB"]),
    }

    gen_params_dict = soil_veg_gen_parm(LLANDUSE)
    gen_params_dict_jax = {k: v for k, v in gen_params_dict.items() if isinstance(v, jnp.ndarray)}
    gen_params_dict_jax["SLOPE_DATA"] = jnp.array(gen_params_dict["SLOPE_DATA"])
    gen_params_dict_static = {
        k: v
        for k, v in gen_params_dict.items()
        if not isinstance(v, (jnp.ndarray, list))
    }

    ZSOIL_jax = jnp.zeros_like(SH2O_init)
    ZSOIL_jax = ZSOIL_jax.at[0].set(-SLDPTH_jax[0])
    for i in range(1, NSOIL):
        ZSOIL_jax = ZSOIL_jax.at[i].set(-SLDPTH_jax[i] + ZSOIL_jax[i - 1])

    date_months = jnp.array([d.month for d in full_idx], dtype=jnp.int32)
    date_days = jnp.array([d.day for d in full_idx], dtype=jnp.int32)
    date_years = jnp.array([d.year for d in full_idx], dtype=jnp.int32)
    steps_per_forcing = int(forcing_timestep / noahlsm_timestep)

    obs_arrays = {
        "smap_sh2o1": jnp.array(smap_data["obs"], dtype=jnp.float32),
        "smap_prev_sh2o1": jnp.array(smap_data["prev_values"], dtype=jnp.float32),
        "smap_prev_age_hours": jnp.array(smap_data["prev_age_hours"], dtype=jnp.float32),
    }

    static_arrays = {
        "forcing_data_jax": forcing_data_jax,
        "SLDPTH_jax": SLDPTH_jax,
        "ZSOIL_jax": ZSOIL_jax,
        "DT_jax": jnp.array(noahlsm_timestep, dtype=jnp.float32),
        "EMISSI_jax": jnp.array(0.96, dtype=jnp.float32),
        "ALBEDO_init": month_d(albedo_monthly, startdate),
        "Z0_init": month_d(z0brd_monthly, startdate),
        "Z0BRD_init": month_d(z0brd_monthly, startdate) if sfcdif_option == 1 else jnp.array(-1.0e36),
        "CZIL_jax": gen_params_dict_jax["CZIL_DATA"],
        "CH_init": jnp.array(1.0e-4),
        "CM_init": jnp.array(1.0e-4),
        "CMC_init": CMC_init,
        "T1_init": T1_init,
        "STC_init": STC_init,
        "SMC_init": SMC_init,
        "SH2O_init": SH2O_init,
        "SNOWH_init": SNOWH_init,
        "SNEQV_init": SNEQV_init,
        "TBOT_jax": jnp.array(TBOT),
        "ZLVL_jax": jnp.array(ZLVL),
        "ZLVL_WIND_jax": jnp.array(ZLVL_WIND),
        "SHDMIN_jax": jnp.array(SHDMIN),
        "SHDMAX_jax": jnp.array(SHDMAX),
        "SNOALB_jax": jnp.array(SNOALB),
        "STYPE_jax": STYPE_jax,
        "RDLAI2D_jax": jnp.array(RDLAI2D),
        "USEMONALB_jax": jnp.array(USEMONALB),
        "SOC_jax": SOC_jax,
        "albedo_monthly_idx": jnp.array(albedo_monthly[:12], dtype=jnp.float32),
        "shdfac_monthly_idx": jnp.array(shdfac_monthly[:12], dtype=jnp.float32),
        "lai_monthly_idx": jnp.array(lai_monthly[:12], dtype=jnp.float32),
        "XLAI_init_jax": jnp.where(
            jnp.array(RDLAI2D).astype(bool),
            month_d(lai_monthly, startdate),
            jnp.array(-1.0e36),
        ),
        "date_months": date_months,
        "date_days": date_days,
        "date_years": date_years,
        "base_soil_params_jax": base_soil_params_jax,
        "gen_params_dict_jax": gen_params_dict_jax,
    }

    static_config_scalars = {
        "NSOIL": NSOIL,
        "VEGTYP_jax": int(VEGTYP),
        "SLOPETYP_jax": int(SLOPETYP_idx),
        "NROOT_jax": int(veg_parameter_df.at[VEGTYP, "NROOT"]),
        "sfcdif_option_jax": int(sfcdif_option),
        "SBETA_OPTION_jax": int(SBETA_OPTION),
        "DF_OPTION_jax": int(DF_OPTION),
        "ROOT_OPTION_jax": int(ROOT_OPTION),
        "INF_OPTION_jax": int(INF_OPTION),
        "SOC_OPTION_KS_jax": int(SOC_OPTION_KS),
        "SOC_OPTION2_THERMAL_jax": int(SOC_OPTION2_THERMAL),
        "RIC_OPTION_jax": int(RIC_OPTION),
        "BLIM_OPTION_jax": int(BLIM_OPTION),
        "CK_OPTION_jax": int(CK_OPTION),
        "IZ0TLND_jax": int(IZ0TLND),
        "steps_per_forcing": steps_per_forcing,
        "soil_type_index_to_optimize": int(SOILTYP_idx),
    }

    return static_arrays, static_config_scalars, veg_params_dict_static, gen_params_dict_static, obs_arrays, full_idx


# ==========================================
# 5. Loss functions
# ==========================================

def compute_loss_norm_params(obs_arrays, n_train_pool):
    train_smap = np.array(obs_arrays["smap_sh2o1"][:n_train_pool], dtype=np.float32)

    smap_valid = train_smap[np.isfinite(train_smap) & (train_smap > 0.0)]

    smap_mean = float(np.mean(smap_valid)) if len(smap_valid) > 0 else 0.25
    smap_std = float(np.std(smap_valid)) if len(smap_valid) > 0 else 0.05
    smap_std = max(smap_std, 0.01)

    print("    Loss normalization from training period:")
    print(f"      SMAP SH2O(1): mean={smap_mean:.4f}, std={smap_std:.4f}, n={len(smap_valid)}")

    return {
        "SMAP_MEAN": smap_mean,
        "SMAP_STD": smap_std,
    }


def masked_normalized_mse_jax(sim, obs, std):
    mask = ~jnp.isnan(obs)
    mask_f = mask.astype(jnp.float32)
    count = jnp.sum(mask_f)
    residual = (sim - jnp.nan_to_num(obs)) / std
    loss = jnp.sum(mask_f * residual ** 2) / jnp.maximum(count, 1.0)
    return loss, count


def masked_normalized_mse_np(sim, obs, std):
    mask = np.isfinite(obs)
    if not np.any(mask):
        return np.nan, 0
    residual = (sim[mask] - obs[mask]) / std
    return float(np.mean(residual ** 2)), int(mask.sum())


def compute_satellite_loss_numpy(sim_sh2o1, obs_smap, loss_norm_params):
    smap_loss, smap_count = masked_normalized_mse_np(sim_sh2o1, obs_smap, loss_norm_params["SMAP_STD"])
    total_loss = LOSS_WEIGHTS["SMAP"] * smap_loss if np.isfinite(smap_loss) else np.nan
    return total_loss, {
        "smap_loss": smap_loss,
        "smap_count": smap_count,
    }


# ==========================================
# 6. Warm-start and simulation
# ==========================================

def _prepare_initial_states_jax(obs_arrays, static_arrays, start_obs_idx, maxsmc_safe, use_satellite_warm_start=True):
    new_sh2o_init = jnp.minimum(static_arrays["SH2O_init"], maxsmc_safe)
    new_smc_init = jnp.minimum(static_arrays["SMC_init"], maxsmc_safe)
    new_stc_init = static_arrays["STC_init"]
    new_t1_init = static_arrays["T1_init"]

    if use_satellite_warm_start:
        warm_sh2o1 = lax.dynamic_slice(obs_arrays["smap_prev_sh2o1"], (start_obs_idx,), (1,))[0]
        valid_sh2o1 = ~jnp.isnan(warm_sh2o1) & (warm_sh2o1 > 0.0)
        new_sh2o_init = new_sh2o_init.at[0].set(jnp.where(valid_sh2o1, warm_sh2o1, new_sh2o_init[0]))
        new_smc_init = new_smc_init.at[0].set(
            jnp.where(valid_sh2o1, jnp.maximum(new_smc_init[0], new_sh2o_init[0]), new_smc_init[0])
        )

    new_smc_init = jnp.minimum(new_smc_init, maxsmc_safe)
    new_sh2o_init = jnp.minimum(new_sh2o_init, new_smc_init)
    return new_t1_init, new_stc_init, new_smc_init, new_sh2o_init


def _prepare_initial_states_numpy(obs_arrays, static_arrays, val_start_idx, maxsmc_safe, use_satellite_warm_start=True):
    new_sh2o_init = np.minimum(np.array(static_arrays["SH2O_init"]), float(maxsmc_safe))
    new_smc_init = np.minimum(np.array(static_arrays["SMC_init"]), float(maxsmc_safe))
    new_stc_init = np.array(static_arrays["STC_init"])
    new_t1_init = float(static_arrays["T1_init"])

    if use_satellite_warm_start:
        warm_sh2o1 = float(obs_arrays["smap_prev_sh2o1"][val_start_idx])
        if np.isfinite(warm_sh2o1) and warm_sh2o1 > 0.0:
            new_sh2o_init[0] = warm_sh2o1
            new_smc_init[0] = max(new_smc_init[0], new_sh2o_init[0])

    new_smc_init = np.minimum(new_smc_init, float(maxsmc_safe))
    new_sh2o_init = np.minimum(new_sh2o_init, new_smc_init)
    return new_t1_init, new_stc_init, new_smc_init, new_sh2o_init


def run_window_simulation(
    optim_params,
    static_arrays,
    static_config_scalars,
    veg_params_dict_static,
    gen_params_dict_static,
    obs_arrays,
    start_obs_idx,
    use_satellite_warm_start=True,
):
    current_soil_params, decoded = build_current_soil_params(optim_params, static_arrays, static_config_scalars)
    maxsmc_safe = decoded["MAXSMC"]
    new_t1_init, new_stc_init, new_smc_init, new_sh2o_init = _prepare_initial_states_jax(
        obs_arrays,
        static_arrays,
        start_obs_idx,
        maxsmc_safe,
        use_satellite_warm_start=use_satellite_warm_start,
    )

    start_forcing_idx = start_obs_idx // STEPS_PER_FORCING
    forcing_slice = lax.dynamic_slice(
        static_arrays["forcing_data_jax"],
        (start_forcing_idx, 0),
        (WINDOW_LEN_FORCING, static_arrays["forcing_data_jax"].shape[1]),
    )
    date_m_slice = lax.dynamic_slice(static_arrays["date_months"], (start_obs_idx,), (WINDOW_SIZE_MODEL_STEPS,))
    date_d_slice = lax.dynamic_slice(static_arrays["date_days"], (start_obs_idx,), (WINDOW_SIZE_MODEL_STEPS,))
    date_y_slice = lax.dynamic_slice(static_arrays["date_years"], (start_obs_idx,), (WINDOW_SIZE_MODEL_STEPS,))
    full_gen_params = {**static_arrays["gen_params_dict_jax"], **gen_params_dict_static}

    return run_noah_simulation_pure(
        forcing_data_jax=forcing_slice,
        NSOIL=static_config_scalars["NSOIL"],
        SLDPTH_jax=static_arrays["SLDPTH_jax"],
        ZSOIL_jax=static_arrays["ZSOIL_jax"],
        DT_jax=static_arrays["DT_jax"],
        EMISSI_jax=static_arrays["EMISSI_jax"],
        ALBEDO_init=static_arrays["ALBEDO_init"],
        Z0_init=static_arrays["Z0_init"],
        Z0BRD_init=static_arrays["Z0BRD_init"],
        CZIL_jax=static_arrays["CZIL_jax"],
        CH_init=static_arrays["CH_init"],
        CM_init=static_arrays["CM_init"],
        sfcdif_option_jax=static_config_scalars["sfcdif_option_jax"],
        CMC_init=static_arrays["CMC_init"],
        T1_init=new_t1_init,
        STC_init=new_stc_init,
        SMC_init=new_smc_init,
        SH2O_init=new_sh2o_init,
        SNOWH_init=static_arrays["SNOWH_init"],
        SNEQV_init=static_arrays["SNEQV_init"],
        TBOT_jax=static_arrays["TBOT_jax"],
        ZLVL_jax=static_arrays["ZLVL_jax"],
        ZLVL_WIND_jax=static_arrays["ZLVL_WIND_jax"],
        SHDMIN_jax=static_arrays["SHDMIN_jax"],
        SHDMAX_jax=static_arrays["SHDMAX_jax"],
        SNOALB_jax=static_arrays["SNOALB_jax"],
        STYPE_jax=static_arrays["STYPE_jax"],
        VEGTYP_jax=static_config_scalars["VEGTYP_jax"],
        SLOPETYP_jax=static_config_scalars["SLOPETYP_jax"],
        RDLAI2D_jax=static_arrays["RDLAI2D_jax"],
        USEMONALB_jax=static_arrays["USEMONALB_jax"],
        SOC_jax=static_arrays["SOC_jax"],
        SBETA_OPTION_jax=static_config_scalars["SBETA_OPTION_jax"],
        DF_OPTION_jax=static_config_scalars["DF_OPTION_jax"],
        ROOT_OPTION_jax=static_config_scalars["ROOT_OPTION_jax"],
        INF_OPTION_jax=static_config_scalars["INF_OPTION_jax"],
        SOC_OPTION_KS_jax=static_config_scalars["SOC_OPTION_KS_jax"],
        SOC_OPTION2_THERMAL_jax=static_config_scalars["SOC_OPTION2_THERMAL_jax"],
        RIC_OPTION_jax=static_config_scalars["RIC_OPTION_jax"],
        BLIM_OPTION_jax=static_config_scalars["BLIM_OPTION_jax"],
        CK_OPTION_jax=static_config_scalars["CK_OPTION_jax"],
        IZ0TLND_jax=static_config_scalars["IZ0TLND_jax"],
        albedo_monthly_idx=static_arrays["albedo_monthly_idx"],
        shdfac_monthly_idx=static_arrays["shdfac_monthly_idx"],
        lai_monthly_idx=static_arrays["lai_monthly_idx"],
        XLAI_init_jax=static_arrays["XLAI_init_jax"],
        date_months=date_m_slice,
        date_days=date_d_slice,
        date_years=date_y_slice,
        total_steps=WINDOW_SIZE_MODEL_STEPS,
        forcing_data_len=WINDOW_LEN_FORCING,
        soil_params_jax=current_soil_params,
        veg_params_dict=veg_params_dict_static,
        gen_params_dict=full_gen_params,
        steps_per_forcing=STEPS_PER_FORCING,
        NROOT_jax=static_config_scalars["NROOT_jax"],
    )


def calculate_minibatch_loss(
    optim_params,
    static_arrays,
    static_config_scalars,
    veg_params_dict_static,
    gen_params_dict_static,
    obs_arrays,
    loss_norm_params,
    start_obs_idx,
):
    final_outputs = run_window_simulation(
        optim_params,
        static_arrays,
        static_config_scalars,
        veg_params_dict_static,
        gen_params_dict_static,
        obs_arrays,
        start_obs_idx,
        use_satellite_warm_start=True,
    )

    sim_sh2o1 = final_outputs["SH2O"][:, 0]
    obs_smap = lax.dynamic_slice(obs_arrays["smap_sh2o1"], (start_obs_idx,), (WINDOW_SIZE_MODEL_STEPS,))
    smap_loss, smap_count = masked_normalized_mse_jax(sim_sh2o1, obs_smap, loss_norm_params["SMAP_STD"])

    total_loss = jnp.array(0.0, dtype=jnp.float32)
    total_weight = jnp.array(0.0, dtype=jnp.float32)

    use_smap = smap_count > 0
    total_loss = total_loss + jnp.where(use_smap, LOSS_WEIGHTS["SMAP"] * smap_loss, 0.0)
    total_weight = total_weight + jnp.where(use_smap, LOSS_WEIGHTS["SMAP"], 0.0)

    data_loss = total_loss / jnp.maximum(total_weight, 1.0e-6)
    margin_loss = MARGIN_PENALTY_WEIGHT * compute_margin_penalty(optim_params)
    total_loss = data_loss + margin_loss
    return total_loss, data_loss, margin_loss


def run_full_sequence_validation(
    optim_params,
    static_arrays,
    static_config_scalars,
    veg_params_dict_static,
    gen_params_dict_static,
    obs_arrays,
    loss_norm_params,
    val_start_idx,
    val_length,
    use_satellite_warm_start=False,
):
    current_soil_params, decoded = build_current_soil_params(optim_params, static_arrays, static_config_scalars)
    maxsmc_safe = decoded["MAXSMC"]
    new_t1_init, new_stc_init, new_smc_init, new_sh2o_init = _prepare_initial_states_numpy(
        obs_arrays,
        static_arrays,
        val_start_idx,
        maxsmc_safe,
        use_satellite_warm_start=use_satellite_warm_start,
    )

    start_forcing_idx = val_start_idx // STEPS_PER_FORCING
    forcing_len_needed = (val_length // STEPS_PER_FORCING) + 2
    forcing_slice = static_arrays["forcing_data_jax"][start_forcing_idx : start_forcing_idx + forcing_len_needed]
    date_m_slice = static_arrays["date_months"][val_start_idx : val_start_idx + val_length]
    date_d_slice = static_arrays["date_days"][val_start_idx : val_start_idx + val_length]
    date_y_slice = static_arrays["date_years"][val_start_idx : val_start_idx + val_length]
    full_gen_params = {**static_arrays["gen_params_dict_jax"], **gen_params_dict_static}

    final_outputs = run_noah_simulation_pure(
        forcing_data_jax=forcing_slice,
        NSOIL=static_config_scalars["NSOIL"],
        SLDPTH_jax=static_arrays["SLDPTH_jax"],
        ZSOIL_jax=static_arrays["ZSOIL_jax"],
        DT_jax=static_arrays["DT_jax"],
        EMISSI_jax=static_arrays["EMISSI_jax"],
        ALBEDO_init=static_arrays["ALBEDO_init"],
        Z0_init=static_arrays["Z0_init"],
        Z0BRD_init=static_arrays["Z0BRD_init"],
        CZIL_jax=static_arrays["CZIL_jax"],
        CH_init=static_arrays["CH_init"],
        CM_init=static_arrays["CM_init"],
        sfcdif_option_jax=static_config_scalars["sfcdif_option_jax"],
        CMC_init=static_arrays["CMC_init"],
        T1_init=new_t1_init,
        STC_init=jnp.array(new_stc_init, dtype=jnp.float32),
        SMC_init=jnp.array(new_smc_init, dtype=jnp.float32),
        SH2O_init=jnp.array(new_sh2o_init, dtype=jnp.float32),
        SNOWH_init=static_arrays["SNOWH_init"],
        SNEQV_init=static_arrays["SNEQV_init"],
        TBOT_jax=static_arrays["TBOT_jax"],
        ZLVL_jax=static_arrays["ZLVL_jax"],
        ZLVL_WIND_jax=static_arrays["ZLVL_WIND_jax"],
        SHDMIN_jax=static_arrays["SHDMIN_jax"],
        SHDMAX_jax=static_arrays["SHDMAX_jax"],
        SNOALB_jax=static_arrays["SNOALB_jax"],
        STYPE_jax=static_arrays["STYPE_jax"],
        VEGTYP_jax=static_config_scalars["VEGTYP_jax"],
        SLOPETYP_jax=static_config_scalars["SLOPETYP_jax"],
        RDLAI2D_jax=static_arrays["RDLAI2D_jax"],
        USEMONALB_jax=static_arrays["USEMONALB_jax"],
        SOC_jax=static_arrays["SOC_jax"],
        SBETA_OPTION_jax=static_config_scalars["SBETA_OPTION_jax"],
        DF_OPTION_jax=static_config_scalars["DF_OPTION_jax"],
        ROOT_OPTION_jax=static_config_scalars["ROOT_OPTION_jax"],
        INF_OPTION_jax=static_config_scalars["INF_OPTION_jax"],
        SOC_OPTION_KS_jax=static_config_scalars["SOC_OPTION_KS_jax"],
        SOC_OPTION2_THERMAL_jax=static_config_scalars["SOC_OPTION2_THERMAL_jax"],
        RIC_OPTION_jax=static_config_scalars["RIC_OPTION_jax"],
        BLIM_OPTION_jax=static_config_scalars["BLIM_OPTION_jax"],
        CK_OPTION_jax=static_config_scalars["CK_OPTION_jax"],
        IZ0TLND_jax=static_config_scalars["IZ0TLND_jax"],
        albedo_monthly_idx=static_arrays["albedo_monthly_idx"],
        shdfac_monthly_idx=static_arrays["shdfac_monthly_idx"],
        lai_monthly_idx=static_arrays["lai_monthly_idx"],
        XLAI_init_jax=static_arrays["XLAI_init_jax"],
        date_months=date_m_slice,
        date_days=date_d_slice,
        date_years=date_y_slice,
        total_steps=val_length,
        forcing_data_len=len(forcing_slice),
        soil_params_jax=current_soil_params,
        veg_params_dict=veg_params_dict_static,
        gen_params_dict=full_gen_params,
        steps_per_forcing=STEPS_PER_FORCING,
        NROOT_jax=static_config_scalars["NROOT_jax"],
    )

    sim_sh2o1 = np.array(final_outputs["SH2O"][:, 0])
    obs_smap = np.array(obs_arrays["smap_sh2o1"][val_start_idx : val_start_idx + val_length])
    loss, components = compute_satellite_loss_numpy(sim_sh2o1, obs_smap, loss_norm_params)

    return {
        "loss": loss,
        "components": components,
        "sim_sh2o1": sim_sh2o1,
        "obs_smap": obs_smap,
    }


# ==========================================
# 7. Training utilities
# ==========================================

def build_training_start_indices(obs_arrays, n_train_pool):
    max_start_idx = n_train_pool - WINDOW_SIZE_MODEL_STEPS
    if max_start_idx <= 0:
        raise ValueError("Training period is shorter than the mini-batch window.")

    smap_valid = np.isfinite(np.array(obs_arrays["smap_sh2o1"][:n_train_pool], dtype=np.float32))
    cumulative = np.concatenate(([0], smap_valid.astype(np.int32).cumsum()))

    starts = np.arange(0, max_start_idx + 1, STEPS_PER_FORCING, dtype=np.int32)
    smap_counts = cumulative[starts + WINDOW_SIZE_MODEL_STEPS] - cumulative[starts]
    candidate_starts = starts[smap_counts > 0]

    if len(candidate_starts) == 0:
        raise ValueError("No training windows contain SMAP observations after cleaning/alignment.")

    print(
        f"    Training windows with at least one SMAP obs: "
        f"{len(candidate_starts)} / {len(starts)}"
    )
    return jnp.array(candidate_starts, dtype=jnp.int32)


def resolve_data_partitions(n_total_obs):
    if USE_VALIDATION_SPLIT:
        n_train_pool = int(n_total_obs * TRAIN_SPLIT)
        if not 0 < n_train_pool < n_total_obs:
            raise ValueError(
                f"When USE_VALIDATION_SPLIT=True, TRAIN_SPLIT must leave both train and validation samples. "
                f"Got TRAIN_SPLIT={TRAIN_SPLIT}, n_total_obs={n_total_obs}."
            )
        return {
            "validation_enabled": True,
            "n_train_pool": n_train_pool,
            "monitor_start_idx": n_train_pool,
            "monitor_length": n_total_obs - n_train_pool,
            "monitor_title": "Validation",
            "monitor_series_label": "Validation Loss (Full Sequence)",
            "monitor_scatter_title": "Validation Scatter: SH2O(1)",
        }

    return {
        "validation_enabled": False,
        "n_train_pool": n_total_obs,
        "monitor_start_idx": 0,
        "monitor_length": n_total_obs,
        "monitor_title": "Full-Train",
        "monitor_series_label": "Full-Sequence Training Loss",
        "monitor_scatter_title": "Full-Data Scatter: SH2O(1)",
    }


def make_minibatch_train_step(
    static_arrays,
    static_config_scalars,
    veg_params_dict_static,
    gen_params_dict_static,
    obs_arrays,
    loss_norm_params,
    optimizer,
):
    def loss_fn(params, start_idx):
        total_loss, data_loss, margin_loss = calculate_minibatch_loss(
            params,
            static_arrays,
            static_config_scalars,
            veg_params_dict_static,
            gen_params_dict_static,
            obs_arrays,
            loss_norm_params,
            start_idx,
        )
        return total_loss, (data_loss, margin_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def leaf_sample_finite_mask(leaf):
        finite = jnp.isfinite(leaf)
        if finite.ndim == 1:
            return finite
        return jnp.all(finite.reshape((finite.shape[0], -1)), axis=1)

    @jax.jit
    def train_step(params, opt_state, start_indices):
        (losses, (data_losses, margin_losses)), grads = jax.vmap(
            lambda start_idx: grad_fn(params, start_idx)
        )(start_indices)

        valid_mask = (
            jnp.isfinite(losses)
            & jnp.isfinite(data_losses)
            & jnp.isfinite(margin_losses)
        )

        grads_finite_mask = jnp.ones_like(valid_mask, dtype=bool)
        grad_finite_leaves = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(leaf_sample_finite_mask, grads)
        )
        for leaf_mask in grad_finite_leaves:
            grads_finite_mask = grads_finite_mask & leaf_mask
        valid_mask = valid_mask & grads_finite_mask

        valid_count = jnp.sum(valid_mask.astype(jnp.int32))
        valid_count_f = jnp.maximum(valid_count.astype(jnp.float32), 1.0)
        valid_mask_f = valid_mask.astype(jnp.float32)

        mean_loss = jnp.sum(jnp.where(valid_mask, losses, 0.0)) / valid_count_f
        mean_data_loss = jnp.sum(jnp.where(valid_mask, data_losses, 0.0)) / valid_count_f
        mean_margin_loss = jnp.sum(jnp.where(valid_mask, margin_losses, 0.0)) / valid_count_f

        def mean_valid_grad(leaf):
            reshape_dims = (valid_mask_f.shape[0],) + (1,) * (leaf.ndim - 1)
            mask = valid_mask_f.reshape(reshape_dims)
            return jnp.sum(leaf * mask, axis=0) / valid_count_f

        mean_grads = jax.tree_util.tree_map(mean_valid_grad, grads)
        mean_grad_abs = abs_scalar_tree(mean_grads)
        grad_norm = optax.global_norm(mean_grads)

        def do_update(_):
            updates, new_opt_state = optimizer.update(mean_grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            param_delta = jax.tree_util.tree_map(lambda new, old: new - old, new_params, params)
            param_delta_norm = optax.global_norm(param_delta)
            param_delta_abs = abs_scalar_tree(param_delta)
            return new_params, new_opt_state, param_delta_norm, param_delta_abs

        def skip_update(_):
            return params, opt_state, jnp.array(jnp.nan, dtype=jnp.float32), nan_scalar_tree_like(params)

        new_params, new_opt_state, param_delta_norm, param_delta_abs = lax.cond(
            valid_count > 0,
            do_update,
            skip_update,
            operand=None,
        )
        mean_loss = jnp.where(valid_count > 0, mean_loss, jnp.nan)
        mean_data_loss = jnp.where(valid_count > 0, mean_data_loss, jnp.nan)
        mean_margin_loss = jnp.where(valid_count > 0, mean_margin_loss, jnp.nan)
        grad_norm = jnp.where(valid_count > 0, grad_norm, jnp.nan)
        param_delta_norm = jnp.where(valid_count > 0, param_delta_norm, jnp.nan)
        mean_grad_abs = jax.tree_util.tree_map(
            lambda leaf: jnp.where(valid_count > 0, leaf, jnp.nan),
            mean_grad_abs,
        )
        return (
            new_params,
            new_opt_state,
            mean_loss,
            mean_data_loss,
            mean_margin_loss,
            valid_count,
            grad_norm,
            param_delta_norm,
            mean_grad_abs,
            param_delta_abs,
        )

    return train_step


def compute_rmse(pred, obs):
    mask = np.isfinite(pred) & np.isfinite(obs)
    if not np.any(mask):
        return np.nan, 0
    return float(np.sqrt(np.mean((pred[mask] - obs[mask]) ** 2))), int(mask.sum())


def print_param_summary(label, params):
    decoded = decode_param_views(params)
    print(
        f"    {label}: "
        f"BB={float(decoded['BB']):.4f}, "
        f"SATDK={float(decoded['SATDK']):.2e}, "
        f"SATPSI={float(decoded['SATPSI']):.4f}, "
        f"MAXSMC={float(decoded['MAXSMC']):.4f}"
    )


def save_diagnostic_plots(
    full_idx,
    n_train_pool,
    train_history,
    monitor_history,
    best_monitor_loss,
    full_before,
    full_after,
    validation_enabled,
    monitor_series_label,
    monitor_scatter_title,
):
    import matplotlib.dates as mdates

    fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(train_history)
    axes[0].set_title("Training Loss (Mini-batch)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    monitor_epochs, monitor_losses = zip(*monitor_history)
    axes[1].plot(monitor_epochs, monitor_losses, "o-", color="orange")
    axes[1].axhline(best_monitor_loss, color="red", linestyle="--", label=f"Best={best_monitor_loss:.4f}")
    axes[1].set_title(monitor_series_label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("minibatch_loss.png", dpi=150)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))

    axes2[0].plot(full_idx, full_before["sim_sh2o1"], color="green", alpha=0.6, label="Before")
    axes2[0].plot(full_idx, full_after["sim_sh2o1"], color="red", alpha=0.8, label="After")
    axes2[0].scatter(full_idx, full_after["obs_smap"], color="blue", s=8, alpha=0.6, label="SMAP")
    if validation_enabled:
        split_date = full_idx[n_train_pool]
        axes2[0].axvline(split_date, color="black", linestyle="--", linewidth=1.5)
    axes2[0].set_title("SH2O(1) vs SMAP")
    axes2[0].set_ylabel("SH2O(1)")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend(loc="upper right", fontsize=9)

    if validation_enabled:
        scatter_obs = full_after["obs_smap"][n_train_pool:]
        scatter_sim = full_after["sim_sh2o1"][n_train_pool:]
    else:
        scatter_obs = full_after["obs_smap"]
        scatter_sim = full_after["sim_sh2o1"]

    val_mask_smap = np.isfinite(scatter_obs)
    if np.any(val_mask_smap):
        obs = scatter_obs[val_mask_smap]
        sim = scatter_sim[val_mask_smap]
        axes2[1].scatter(obs, sim, alpha=0.35, s=10)
        min_v = min(np.nanmin(obs), np.nanmin(sim))
        max_v = max(np.nanmax(obs), np.nanmax(sim))
        axes2[1].plot([min_v, max_v], [min_v, max_v], "r--")
    axes2[1].set_title(monitor_scatter_title)
    axes2[1].set_xlabel("SMAP")
    axes2[1].set_ylabel("Model")
    axes2[1].grid(True, alpha=0.3)

    for ax in [axes2[0]]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig("satellite_comparison.png", dpi=150)
    plt.close(fig2)


# ==========================================
# 8. Main
# ==========================================

def main():
    print(">>> Random mini-batch training with SMAP-only loss")
    print(f"    Model step: 1 hour")
    print(f"    Window size: {WINDOW_SIZE_MODEL_STEPS} steps ({WINDOW_SIZE_MODEL_STEPS / 24:.1f} days)")
    print(f"    Windows per batch: {WINDOWS_PER_BATCH}")
    print(f"    Loss weight: SMAP={LOSS_WEIGHTS['SMAP']}")
    print(f"    Learning rates: {LEARNING_RATES}")
    if USE_SGDR:
        print(
            "    LR schedule: SGDR"
            f" (first_cycle={SGDR_FIRST_CYCLE_EPOCHS} epochs,"
            f" cycle_mult={SGDR_CYCLE_MULT},"
            f" num_cycles={SGDR_NUM_CYCLES},"
            f" min_lr_scale={SGDR_MIN_LR_SCALE})"
        )
    else:
        print("    LR schedule: constant")
    print(f"    Margin penalty weight: {MARGIN_PENALTY_WEIGHT}")
    print(f"    SMAP file: {OBS_SMAP_CSV_FILE}")
    print(f"    SMAP local->UTC shift: {SMAP_TIME_SHIFT_TO_UTC_HOURS:+.1f} h")
    print(f"    SMAP warm-start max gap: {SMAP_WARM_MAX_GAP_HOURS} h")
    print("    Ground observations are not loaded or used anywhere in this script.")

    try:
        (
            static_arrays,
            static_config_scalars,
            veg_params_dict_static,
            gen_params_dict_static,
            obs_arrays,
            full_idx,
        ) = load_aligned_data(FORCING_FILE, OBS_SMAP_CSV_FILE)
    except Exception as exc:
        print(f"Data loading failed: {exc}")
        import traceback

        traceback.print_exc()
        return

    actual_steps_per_forcing = static_config_scalars["steps_per_forcing"]
    if actual_steps_per_forcing != STEPS_PER_FORCING:
        raise ValueError(
            f"steps_per_forcing mismatch: data={actual_steps_per_forcing}, expected={STEPS_PER_FORCING}"
        )

    n_total_obs = len(obs_arrays["smap_sh2o1"])
    split_cfg = resolve_data_partitions(n_total_obs)
    validation_enabled = split_cfg["validation_enabled"]
    n_train_pool = split_cfg["n_train_pool"]
    monitor_start_idx = split_cfg["monitor_start_idx"]
    monitor_length = split_cfg["monitor_length"]
    monitor_title = split_cfg["monitor_title"]
    print(f"    Total timeline length: {n_total_obs}")
    if validation_enabled:
        print(f"    Training pool: 0 ~ {n_train_pool}")
        print(f"    Validation pool: {n_train_pool} ~ {n_total_obs}")
    else:
        print(f"    Training pool: 0 ~ {n_total_obs} (all data)")
        print("    Validation pool: disabled")

    print("\n2. Building SMAP-guided training windows...")
    candidate_start_indices = build_training_start_indices(obs_arrays, n_train_pool)
    if len(candidate_start_indices) < WINDOWS_PER_BATCH:
        raise ValueError(
            f"Not enough training windows ({len(candidate_start_indices)}) for WINDOWS_PER_BATCH={WINDOWS_PER_BATCH}."
        )

    print("\n3. Computing loss normalization...")
    loss_norm_params = compute_loss_norm_params(obs_arrays, n_train_pool)

    soil_idx = static_config_scalars["soil_type_index_to_optimize"] - 1
    base_params = static_arrays["base_soil_params_jax"]
    params = make_initial_params(base_params, soil_idx)
    initial_params = {k: jnp.array(v) for k, v in params.items()}

    print_param_summary("Initial parameters", params)
    print(f"    Initial margin penalty: {float(compute_margin_penalty(params)):.6f}")

    optimizer = build_optimizer()
    opt_state = optimizer.init(params)

    print("    JIT compiling train step...")
    train_step = make_minibatch_train_step(
        static_arrays,
        static_config_scalars,
        veg_params_dict_static,
        gen_params_dict_static,
        obs_arrays,
        loss_norm_params,
        optimizer,
    )

    def evaluate_monitor(p):
        full_result = run_full_sequence_validation(
            p,
            static_arrays,
            static_config_scalars,
            veg_params_dict_static,
            gen_params_dict_static,
            obs_arrays,
            loss_norm_params,
            monitor_start_idx,
            monitor_length,
            use_satellite_warm_start=True,
        )
        return full_result["loss"], full_result["components"]

    print(f"\n4. Computing initial {monitor_title.lower()} loss...")
    val_start_time = time.time()
    initial_monitor_loss, initial_components = evaluate_monitor(params)
    print(
        f"    Initial {monitor_title.lower()} loss: {initial_monitor_loss:.6f} "
        f"(SMAP={initial_components['smap_loss']}) "
        f"in {time.time() - val_start_time:.1f}s"
    )

    key = jax.random.PRNGKey(RANDOM_SEED)
    train_history = []
    monitor_history = [(0, initial_monitor_loss)]
    best_monitor_loss = initial_monitor_loss
    best_params = {k: np.array(v) for k, v in params.items()}
    best_epoch = 0

    print(
        f"\n5. Start training ({NUM_EPOCHS} epochs, "
        f"{BATCHES_PER_EPOCH} batches/epoch, {WINDOWS_PER_BATCH} windows/batch)..."
    )
    epoch_batch_capacity = int(len(candidate_start_indices) // WINDOWS_PER_BATCH)
    if epoch_batch_capacity <= 0:
        raise ValueError(
            f"Training window pool is too small for WINDOWS_PER_BATCH={WINDOWS_PER_BATCH}."
        )
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        total_losses = []
        data_losses = []
        margin_losses = []
        grad_norms = []
        param_delta_norms = []
        grad_abs_diags = []
        step_abs_diags = []
        dropped_batch_count = 0
        dropped_window_count = 0
        valid_window_count_total = 0
        attempt_count = 0
        consecutive_nan = 0
        last_valid_params = {k: jnp.array(v) for k, v in params.items()}
        last_valid_opt_state = opt_state
        epoch_pool_exhausted = False

        key, epoch_perm_key = jax.random.split(key)
        epoch_start_indices = jax.random.permutation(epoch_perm_key, candidate_start_indices)
        epoch_max_attempts = min(MAX_BATCH_ATTEMPTS_PER_EPOCH, epoch_batch_capacity)

        while len(total_losses) < BATCHES_PER_EPOCH and attempt_count < epoch_max_attempts:
            batch_offset = attempt_count * WINDOWS_PER_BATCH
            batch_start_indices = lax.dynamic_slice(
                epoch_start_indices,
                (batch_offset,),
                (WINDOWS_PER_BATCH,),
            )
            attempt_count += 1
            (
                new_params,
                new_opt_state,
                loss,
                data_loss,
                margin_loss,
                valid_window_count,
                grad_norm,
                param_delta_norm,
                grad_abs_diag,
                step_abs_diag,
            ) = train_step(params, opt_state, batch_start_indices)
            loss = loss.block_until_ready()
            data_loss = data_loss.block_until_ready()
            margin_loss = margin_loss.block_until_ready()
            valid_window_count = int(valid_window_count.block_until_ready())
            grad_norm = grad_norm.block_until_ready()
            param_delta_norm = param_delta_norm.block_until_ready()
            grad_abs_diag = jax.device_get(grad_abs_diag)
            step_abs_diag = jax.device_get(step_abs_diag)
            loss_value = float(loss)
            data_loss_value = float(data_loss)
            margin_loss_value = float(margin_loss)
            grad_norm_value = float(grad_norm)
            param_delta_norm_value = float(param_delta_norm)
            dropped_window_count += WINDOWS_PER_BATCH - valid_window_count

            if (
                valid_window_count <= 0
                or valid_window_count > WINDOWS_PER_BATCH
                or loss_value is None
                or data_loss_value is None
                or margin_loss_value is None
                or grad_norm_value is None
                or param_delta_norm_value is None
                or not np.isfinite(loss_value)
                or not np.isfinite(data_loss_value)
                or not np.isfinite(margin_loss_value)
                or not np.isfinite(grad_norm_value)
                or not np.isfinite(param_delta_norm_value)
            ):
                dropped_batch_count += 1
                consecutive_nan += 1
                if consecutive_nan >= 5:
                    params = last_valid_params
                    opt_state = last_valid_opt_state
                    consecutive_nan = 0
                continue

            has_nan_params = any(
                bool(jnp.any(jnp.isnan(v)) | jnp.any(jnp.isinf(v)))
                for v in new_params.values()
            )
            if has_nan_params:
                dropped_batch_count += 1
                consecutive_nan += 1
                if consecutive_nan >= 5:
                    params = last_valid_params
                    opt_state = last_valid_opt_state
                    consecutive_nan = 0
                continue

            params = new_params
            opt_state = new_opt_state
            total_losses.append(loss_value)
            data_losses.append(data_loss_value)
            margin_losses.append(margin_loss_value)
            grad_norms.append(grad_norm_value)
            param_delta_norms.append(param_delta_norm_value)
            grad_abs_diags.append(tree_to_ordered_float_dict(grad_abs_diag))
            step_abs_diags.append(tree_to_ordered_float_dict(step_abs_diag))
            valid_window_count_total += valid_window_count
            consecutive_nan = 0
            last_valid_params = {k: jnp.array(v) for k, v in params.items()}
            last_valid_opt_state = opt_state

        if len(total_losses) == 0:
            print(
                f"Epoch {epoch + 1:03d} | all batches invalid, skipping"
                f" | Attempts={attempt_count}"
                f" | DroppedBatches={dropped_batch_count}"
                f" | DroppedWindows={dropped_window_count}"
            )
            train_history.append(np.nan)
            continue

        avg_train_loss = float(np.mean(total_losses))
        avg_train_data_loss = float(np.mean(data_losses))
        avg_train_margin_loss = float(np.mean(margin_losses))
        avg_grad_norm = float(np.mean(grad_norms))
        avg_param_delta_norm = float(np.mean(param_delta_norms))
        avg_grad_abs_diag = average_ordered_scalar_dicts(grad_abs_diags)
        avg_step_abs_diag = average_ordered_scalar_dicts(step_abs_diags)
        train_history.append(avg_train_loss)

        monitor_str = ""
        if (epoch + 1) % VALIDATION_INTERVAL_EPOCHS == 0:
            monitor_loss, monitor_components = evaluate_monitor(params)
            monitor_history.append((epoch + 1, monitor_loss))
            if np.isfinite(monitor_loss) and monitor_loss < best_monitor_loss:
                best_monitor_loss = monitor_loss
                best_params = {k: np.array(v) for k, v in params.items()}
                best_epoch = epoch + 1
            monitor_str = (
                f" | {monitor_title}={monitor_loss:.6f}"
                f" (SMAP={monitor_components['smap_loss']})"
            )

        decoded = decode_param_views(params)
        margin_val = float(compute_margin_penalty(params))
        margin_term = MARGIN_PENALTY_WEIGHT * margin_val
        capped_str = ""
        if len(total_losses) < BATCHES_PER_EPOCH and attempt_count >= epoch_max_attempts:
            capped_str = " | Attempt cap reached"
        if len(total_losses) < BATCHES_PER_EPOCH and attempt_count >= epoch_batch_capacity:
            epoch_pool_exhausted = True
        batch_str = f" | ValidBatches={len(total_losses)}/{BATCHES_PER_EPOCH} | Attempts={attempt_count}"
        window_str = (
            f" | ValidWindows={valid_window_count_total}/{len(total_losses) * WINDOWS_PER_BATCH}"
        )
        dropped_batch_str = (
            f" | DroppedBatches={dropped_batch_count}" if dropped_batch_count > 0 else ""
        )
        dropped_window_str = (
            f" | DroppedWindows={dropped_window_count}" if dropped_window_count > 0 else ""
        )
        pool_str = " | EpochPoolExhausted" if epoch_pool_exhausted else ""
        print(
            f"Epoch {epoch + 1:03d} | TrainTotal={avg_train_loss:.6f}"
            f" | TrainData={avg_train_data_loss:.6f}"
            f" | TrainReg={avg_train_margin_loss:.6f}"
            f" | GradNorm={avg_grad_norm:.3e}"
            f" | ParamDeltaNorm={avg_param_delta_norm:.3e}"
            f"{format_ordered_scalar_dict('GradRawAbs', avg_grad_abs_diag)}"
            f"{format_ordered_scalar_dict('StepRawAbs', avg_step_abs_diag)}"
            f"{monitor_str}{batch_str}{window_str}{dropped_batch_str}{dropped_window_str}{capped_str}{pool_str}"
            f" | Time={time.time() - epoch_start:.2f}s"
            f" | BB={float(decoded['BB']):.4f}"
            f" | K={float(decoded['SATDK']):.2e}"
            f" | PSI={float(decoded['SATPSI']):.4f}"
            f" | MAXSMC={float(decoded['MAXSMC']):.4f}"
            f" | MarginRaw={margin_val:.4f}"
            f" | MarginTerm={margin_term:.4f}"
        )

    print("\n" + "=" * 60)
    print("Training finished")
    print(f"Best {monitor_title.lower()} loss: {best_monitor_loss:.6f} (Epoch {best_epoch})")
    print_param_summary("Best parameters", best_params)

    print("\n6. Running full-sequence diagnostics...")
    full_before = run_full_sequence_validation(
        initial_params,
        static_arrays,
        static_config_scalars,
        veg_params_dict_static,
        gen_params_dict_static,
        obs_arrays,
        loss_norm_params,
        0,
        n_total_obs,
        use_satellite_warm_start=False,
    )
    full_after = run_full_sequence_validation(
        best_params,
        static_arrays,
        static_config_scalars,
        veg_params_dict_static,
        gen_params_dict_static,
        obs_arrays,
        loss_norm_params,
        0,
        n_total_obs,
        use_satellite_warm_start=False,
    )

    if validation_enabled:
        train_slice = slice(0, n_train_pool)
        val_slice = slice(n_train_pool, n_total_obs)

        rmse_before_train_smap, n_train_smap = compute_rmse(full_before["sim_sh2o1"][train_slice], full_after["obs_smap"][train_slice])
        rmse_after_train_smap, _ = compute_rmse(full_after["sim_sh2o1"][train_slice], full_after["obs_smap"][train_slice])
        rmse_before_val_smap, n_val_smap = compute_rmse(full_before["sim_sh2o1"][val_slice], full_after["obs_smap"][val_slice])
        rmse_after_val_smap, _ = compute_rmse(full_after["sim_sh2o1"][val_slice], full_after["obs_smap"][val_slice])

        print("\n=== SMAP SH2O(1) RMSE ===")
        print(f"Train: before={rmse_before_train_smap:.4f}, after={rmse_after_train_smap:.4f}, n={n_train_smap}")
        print(f"Val:   before={rmse_before_val_smap:.4f}, after={rmse_after_val_smap:.4f}, n={n_val_smap}")
    else:
        all_slice = slice(0, n_total_obs)
        rmse_before_all_smap, n_all_smap = compute_rmse(full_before["sim_sh2o1"][all_slice], full_after["obs_smap"][all_slice])
        rmse_after_all_smap, _ = compute_rmse(full_after["sim_sh2o1"][all_slice], full_after["obs_smap"][all_slice])

        print("\n=== SMAP SH2O(1) RMSE (All Data) ===")
        print(f"All: before={rmse_before_all_smap:.4f}, after={rmse_after_all_smap:.4f}, n={n_all_smap}")

    save_diagnostic_plots(
        full_idx,
        n_train_pool,
        train_history,
        monitor_history,
        best_monitor_loss,
        full_before,
        full_after,
        validation_enabled,
        split_cfg["monitor_series_label"],
        split_cfg["monitor_scatter_title"],
    )

    fit_df = pd.DataFrame(
        {
            "Date": full_idx,
            "SMAP_SH2O1": full_after["obs_smap"],
            "Sim_SH2O1_Before": full_before["sim_sh2o1"],
            "Sim_SH2O1_After": full_after["sim_sh2o1"],
        }
    )
    fit_df.to_csv("satellite_fit_timeseries.csv", index=False)
    print("Saved satellite_fit_timeseries.csv")

    best_decoded = decode_param_views(best_params)
    optimized_params = {
        "BB": float(best_decoded["BB"]),
        "SATDK": float(best_decoded["SATDK"]),
        "SATPSI": float(best_decoded["SATPSI"]),
        "MAXSMC": float(best_decoded["MAXSMC"]),
    }
    with open("minibatch_margin_optimized_params.pkl", "wb") as f:
        pickle.dump(optimized_params, f)

    print("\nSaved optimized parameters to minibatch_margin_optimized_params.pkl")
    for key_name, value in optimized_params.items():
        if key_name == "SATDK":
            print(f"  {key_name}: {value:.4e}")
        else:
            print(f"  {key_name}: {value:.4f}")


if __name__ == "__main__":
    main()
