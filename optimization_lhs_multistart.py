from __future__ import annotations

import importlib.util
import json
import pickle
import time
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

import optimization_minibatch_margin as base

ROOT_DIR = Path(__file__).resolve().parent
DETERMINISTIC_ASSIM_SCRIPT = ROOT_DIR / "cycling_enkf_jax_relative_error.py"


# ==========================================
# 1. User configuration
# ==========================================

# Output files
OUTPUT_PREFIX = "lhs_multistart"
OUTPUT_DIR = ROOT_DIR / f"{OUTPUT_PREFIX}_outputs"
RUN_DEFAULT_DEENKF_AFTER_TRAINING = True

# Base-data overrides. Leave as None to inherit from optimization_minibatch_margin.py.
FORCING_FILE = None
OBS_SMAP_CSV_FILE = None
SMAP_TIME_SHIFT_TO_UTC_HOURS = None
SMAP_WARM_MAX_GAP_HOURS = None
USE_VALIDATION_SPLIT = None

# Base-training overrides. Leave as None to inherit from optimization_minibatch_margin.py.
# This keeps the multi-start script aligned with the single-start script by default.
TRAIN_SPLIT = None
RANDOM_SEED = None
NUM_EPOCHS = None
BATCHES_PER_EPOCH = None
WINDOWS_PER_BATCH = None
VALIDATION_INTERVAL_EPOCHS = None
MAX_BATCH_ATTEMPTS_PER_EPOCH = None
LEARNING_RATES = None
USE_SGDR = None
SGDR_FIRST_CYCLE_EPOCHS = None
SGDR_CYCLE_MULT = None
SGDR_NUM_CYCLES = None
SGDR_MIN_LR_SCALE = None
MARGIN_PENALTY_WEIGHT = None

# Multi-start / LHS settings
LHS_NUM_SAMPLES = 256
MULTISTART_TOP_K = 16
# Proxy screening mode switch:
#   "full_sequence_train": use the entire training period to score each LHS start.
#       This is slower but the ranking is more stable and usually better aligned with
#       the final objective, so it is suitable as the main experimental setting.
#   "fixed_windows": use PROXY_WINDOWS_PER_START evenly spaced training windows
#       for a fast proxy score. This is quicker and still covers the full training
#       period, but the ranking may differ from full-sequence performance.
# To switch back to the window-based proxy, simply change this line to:
#   PROXY_SCREENING_MODE = "fixed_windows"
PROXY_SCREENING_MODE = "fixed_windows"
PROXY_WINDOWS_PER_START = 32
ALWAYS_INCLUDE_DEFAULT_START = True

LHS_RANDOM_SEED = 2025
TRAINING_SEED_OFFSET = 1000


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def output_path(filename: str) -> Path:
    return ensure_output_dir() / filename


# ==========================================
# 2. Apply runtime overrides to the base optimizer
# ==========================================

def _inherit_or_override(override, current_value):
    return current_value if override is None else override


def _resolve_learning_rates():
    if LEARNING_RATES is None:
        return dict(base.LEARNING_RATES)
    merged = dict(base.LEARNING_RATES)
    merged.update(LEARNING_RATES)
    return merged


def apply_base_runtime_config():
    base.FORCING_FILE = _inherit_or_override(FORCING_FILE, base.FORCING_FILE)
    base.OBS_SMAP_CSV_FILE = _inherit_or_override(OBS_SMAP_CSV_FILE, base.OBS_SMAP_CSV_FILE)
    base.SMAP_TIME_SHIFT_TO_UTC_HOURS = _inherit_or_override(
        SMAP_TIME_SHIFT_TO_UTC_HOURS,
        base.SMAP_TIME_SHIFT_TO_UTC_HOURS,
    )
    base.SMAP_WARM_MAX_GAP_HOURS = _inherit_or_override(
        SMAP_WARM_MAX_GAP_HOURS,
        base.SMAP_WARM_MAX_GAP_HOURS,
    )
    base.USE_VALIDATION_SPLIT = _inherit_or_override(
        USE_VALIDATION_SPLIT,
        base.USE_VALIDATION_SPLIT,
    )

    resolved_train_split = _inherit_or_override(TRAIN_SPLIT, base.TRAIN_SPLIT)
    resolved_random_seed = _inherit_or_override(RANDOM_SEED, base.RANDOM_SEED)
    resolved_num_epochs = _inherit_or_override(NUM_EPOCHS, base.NUM_EPOCHS)
    resolved_batches_per_epoch = _inherit_or_override(BATCHES_PER_EPOCH, base.BATCHES_PER_EPOCH)
    resolved_windows_per_batch = _inherit_or_override(WINDOWS_PER_BATCH, base.WINDOWS_PER_BATCH)
    resolved_validation_interval = _inherit_or_override(
        VALIDATION_INTERVAL_EPOCHS,
        base.VALIDATION_INTERVAL_EPOCHS,
    )
    resolved_max_batch_attempts = _inherit_or_override(
        MAX_BATCH_ATTEMPTS_PER_EPOCH,
        resolved_batches_per_epoch * 5,
    )

    base.TRAIN_SPLIT = resolved_train_split
    base.RANDOM_SEED = resolved_random_seed
    base.NUM_EPOCHS = resolved_num_epochs
    base.BATCHES_PER_EPOCH = resolved_batches_per_epoch
    base.WINDOWS_PER_BATCH = resolved_windows_per_batch
    base.VALIDATION_INTERVAL_EPOCHS = resolved_validation_interval
    base.MAX_BATCH_ATTEMPTS_PER_EPOCH = resolved_max_batch_attempts
    base.LEARNING_RATES = _resolve_learning_rates()
    base.USE_SGDR = _inherit_or_override(USE_SGDR, base.USE_SGDR)
    base.SGDR_FIRST_CYCLE_EPOCHS = _inherit_or_override(
        SGDR_FIRST_CYCLE_EPOCHS,
        base.SGDR_FIRST_CYCLE_EPOCHS,
    )
    base.SGDR_CYCLE_MULT = _inherit_or_override(SGDR_CYCLE_MULT, base.SGDR_CYCLE_MULT)
    base.SGDR_NUM_CYCLES = _inherit_or_override(SGDR_NUM_CYCLES, base.SGDR_NUM_CYCLES)
    base.SGDR_MIN_LR_SCALE = _inherit_or_override(
        SGDR_MIN_LR_SCALE,
        base.SGDR_MIN_LR_SCALE,
    )
    base.MARGIN_PENALTY_WEIGHT = _inherit_or_override(
        MARGIN_PENALTY_WEIGHT,
        base.MARGIN_PENALTY_WEIGHT,
    )


apply_base_runtime_config()


DEENKF_OBS_ERROR_BY_STATION = {
    "maqu": 0.05,
    "naqu": 0.15,
    "sqh": 0.15,
    "wdl": 0.05,
}


def load_python_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_station_key(forcing_file, obs_csv_file) -> str | None:
    haystack = f"{forcing_file} {obs_csv_file}".lower()
    if "wudaoliang" in haystack or "wdl" in haystack:
        return "wdl"
    if "maqu" in haystack or "mq" in haystack:
        return "maqu"
    if "naqu" in haystack or "nq" in haystack:
        return "naqu"
    if "sqh" in haystack or "shengquhe" in haystack or "sq-" in haystack:
        return "sqh"
    return None


def infer_deenkf_obs_error(default_value: float) -> float:
    station_key = infer_station_key(base.FORCING_FILE, base.OBS_SMAP_CSV_FILE)
    if station_key is None:
        return float(default_value)
    return float(DEENKF_OBS_ERROR_BY_STATION.get(station_key, default_value))


def infer_soil_type_row(module, forcing_file: str) -> int | None:
    try:
        forcing_tuple = module.open_forcing_file(forcing_file)
        stype = forcing_tuple[18]
        return int(stype[0])
    except Exception:
        return None


def run_deterministic_assimilation_cases(
    output_prefix: str,
    default_reference_params: dict[str, float],
    best_soil_param_overrides: dict[str, float],
) -> list[dict]:
    if not DETERMINISTIC_ASSIM_SCRIPT.exists():
        raise FileNotFoundError(f"Deterministic assimilation script not found: {DETERMINISTIC_ASSIM_SCRIPT}")

    forcing_file = str(base.FORCING_FILE)
    obs_csv_file = str(base.OBS_SMAP_CSV_FILE)
    case_summaries = []

    run_specs = []
    if RUN_DEFAULT_DEENKF_AFTER_TRAINING:
        run_specs.append(
            {
                "label": "default",
                "soil_param_overrides": {},
                "reference_params": dict(default_reference_params),
            }
        )
    else:
        print("\n8. Skipping deterministic assimilation with default soil parameters.")

    run_specs.append(
        {
            "label": "best",
            "soil_param_overrides": dict(best_soil_param_overrides),
            "reference_params": dict(best_soil_param_overrides),
        }
    )

    for run_spec in run_specs:
        label = run_spec["label"]
        output_file = output_path(f"{output_prefix}_{label}_deenkf_analysis.pkl")
        exception_log_file = output_path(f"{output_prefix}_{label}_deenkf_exceptions.log")
        print(f"\n8. Running deterministic assimilation with {label} soil parameters...")

        try:
            module = load_python_file(f"lhs_multistart_deenkf_{label}", DETERMINISTIC_ASSIM_SCRIPT)
            obs_error_relative = infer_deenkf_obs_error(float(module.OBS_ERROR_SH2O_RELATIVE))

            module.FORCING_FILE = forcing_file
            module.OBS_CSV_FILE = obs_csv_file
            module.OBS_ERROR_SH2O_RELATIVE = obs_error_relative
            module.OUTPUT_FILE = str(output_file)
            module.EXCEPTION_LOG_FILE = str(exception_log_file)
            module.SOIL_PARAM_OVERRIDES = dict(run_spec["soil_param_overrides"])
            module.VERBOSE = False
            module.exception_logger = module.EnKFExceptionLogger(module.EXCEPTION_LOG_FILE)

            soil_type_row = infer_soil_type_row(module, forcing_file)
            results = module.run_cycling_deterministic()
            csv_file = output_file.with_suffix(".csv")

            summary = {
                "label": label,
                "status": "ok",
                "script_path": str(DETERMINISTIC_ASSIM_SCRIPT),
                "forcing_file": forcing_file,
                "obs_csv_file": obs_csv_file,
                "obs_error_sh2o_relative": obs_error_relative,
                "soil_type_row": soil_type_row,
                "soil_param_overrides": dict(run_spec["soil_param_overrides"]),
                "reference_params": dict(run_spec["reference_params"]),
                "output_file": str(output_file),
                "csv_file": str(csv_file),
                "exception_log_file": str(exception_log_file),
                "output_exists": output_file.exists(),
                "csv_exists": csv_file.exists(),
                "num_assimilations": results["metadata"].get("num_assimilations"),
                "n_output_steps": len(results["metadata"].get("dates", [])),
            }
        except Exception as exc:
            summary = {
                "label": label,
                "status": "failed",
                "script_path": str(DETERMINISTIC_ASSIM_SCRIPT),
                "forcing_file": forcing_file,
                "obs_csv_file": obs_csv_file,
                "soil_param_overrides": dict(run_spec["soil_param_overrides"]),
                "reference_params": dict(run_spec["reference_params"]),
                "output_file": str(output_file),
                "exception_log_file": str(exception_log_file),
                "error": repr(exc),
            }
            print(f"    Deterministic assimilation failed for {label}: {exc}")

        case_summaries.append(summary)

    summary_path = output_path(f"{output_prefix}_deenkf_run_summary.json")
    summary_path.write_text(
        json.dumps(case_summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {summary_path}")
    return case_summaries


# ==========================================
# 3. LHS helpers
# ==========================================

RAW_KEY_BY_PARAM = {
    "BB": "BB_raw",
    "log_SATDK": "log_SATDK_raw",
    "SATPSI": "SATPSI_raw",
    "MAXSMC": "MAXSMC_raw",
}


def resolve_proxy_screening_mode() -> str:
    normalized = str(PROXY_SCREENING_MODE).strip().lower()
    alias_map = {
        "full": "full_sequence_train",
        "full_sequence": "full_sequence_train",
        "full_sequence_train": "full_sequence_train",
        "train_full_sequence": "full_sequence_train",
        "windows": "fixed_windows",
        "fixed_windows": "fixed_windows",
        "window_proxy": "fixed_windows",
    }
    if normalized not in alias_map:
        valid_modes = ", ".join(sorted(set(alias_map.values())))
        raise ValueError(
            f"Unsupported PROXY_SCREENING_MODE={PROXY_SCREENING_MODE!r}. "
            f"Expected one of: {valid_modes}."
        )
    return alias_map[normalized]


def latin_hypercube_unit(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n_samples + 1, dtype=np.float32)
    lhs = np.zeros((n_samples, n_dim), dtype=np.float32)
    for dim_idx in range(n_dim):
        points = edges[:-1] + rng.random(n_samples, dtype=np.float32) * (1.0 / n_samples)
        rng.shuffle(points)
        lhs[:, dim_idx] = points
    return lhs


def get_sampled_param_names() -> list[str]:
    return ["BB", "log_SATDK", "SATPSI", "MAXSMC"]


def make_default_raw_params(base_params, soil_idx: int) -> dict:
    return {
        "BB_raw": base.inverse_soft_clamp(
            base_params["BB"][soil_idx],
            *base.PARAM_BOUNDS_EXTENDED["BB"],
            scale=0.5,
        ),
        "log_SATDK_raw": base.inverse_soft_clamp(
            jnp.log(base_params["SATDK"][soil_idx]),
            *base.PARAM_BOUNDS_EXTENDED["log_SATDK"],
            scale=0.5,
        ),
        "SATPSI_raw": base.inverse_soft_clamp(
            base_params["SATPSI"][soil_idx],
            *base.PARAM_BOUNDS_EXTENDED["SATPSI"],
            scale=0.5,
        ),
        "MAXSMC_raw": base.inverse_soft_clamp(
            base_params["MAXSMC"][soil_idx],
            *base.PARAM_BOUNDS_EXTENDED["MAXSMC"],
            scale=0.5,
        ),
    }


def build_default_initialization(base_params, soil_idx: int) -> dict:
    params = make_default_raw_params(base_params, soil_idx)
    return {
        "label": "default",
        "source": "soilparm_default",
        "params": {k: jnp.array(v) for k, v in params.items()},
    }


def build_lhs_initializations(base_params, soil_idx: int) -> list[dict]:
    rng = np.random.default_rng(LHS_RANDOM_SEED)
    param_names = get_sampled_param_names()
    lhs_unit = latin_hypercube_unit(LHS_NUM_SAMPLES, len(param_names), rng)

    initializations = []
    for sample_idx in range(LHS_NUM_SAMPLES):
        sampled_params = make_default_raw_params(base_params, soil_idx)
        sampled_physical = {}

        for dim_idx, param_name in enumerate(param_names):
            lo, hi = base.PARAM_BOUNDS_PHYSICAL[param_name]
            sampled_val = float(lo + lhs_unit[sample_idx, dim_idx] * (hi - lo))
            sampled_physical[param_name] = sampled_val
            raw_key = RAW_KEY_BY_PARAM[param_name]
            sampled_params[raw_key] = base.inverse_soft_clamp(
                jnp.array(sampled_val, dtype=jnp.float32),
                *base.PARAM_BOUNDS_EXTENDED[param_name],
                scale=0.5,
            )

        initializations.append(
            {
                "label": f"lhs_{sample_idx + 1:03d}",
                "source": "lhs",
                "sampled_physical": sampled_physical,
                "params": {k: jnp.array(v) for k, v in sampled_params.items()},
            }
        )

    return initializations


def build_proxy_start_indices(candidate_start_indices) -> jnp.ndarray:
    candidate_np = np.array(candidate_start_indices, dtype=np.int32)
    n_proxy = min(PROXY_WINDOWS_PER_START, len(candidate_np))
    if n_proxy <= 0:
        raise ValueError("No candidate training windows are available for proxy evaluation.")

    positions = np.linspace(0, len(candidate_np) - 1, num=n_proxy)
    picked = np.unique(np.round(positions).astype(np.int32))
    if len(picked) < n_proxy:
        all_ids = np.arange(len(candidate_np), dtype=np.int32)
        remaining = all_ids[~np.isin(all_ids, picked)]
        picked = np.concatenate([picked, remaining[: n_proxy - len(picked)]])
    picked.sort()
    return jnp.array(candidate_np[picked], dtype=jnp.int32)


# ==========================================
# 4. Proxy screening
# ==========================================

def stack_param_dicts(param_dicts: list[dict]):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *param_dicts)


def get_param_tree_at(params_batched, idx: int):
    return jax.tree_util.tree_map(lambda leaf: leaf[idx], params_batched)


def tree_select(mask_1d, true_tree, false_tree):
    def select_leaf(true_leaf, false_leaf):
        reshape_dims = (mask_1d.shape[0],) + (1,) * (true_leaf.ndim - 1)
        mask = mask_1d.reshape(reshape_dims)
        return jnp.where(mask, true_leaf, false_leaf)

    return jax.tree_util.tree_map(select_leaf, true_tree, false_tree)


def tree_any_nonfinite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    invalid_mask = jnp.zeros((leaves[0].shape[0],), dtype=bool)
    for leaf in leaves:
        leaf_invalid = ~jnp.isfinite(leaf)
        if leaf.ndim > 1:
            leaf_invalid = jnp.any(leaf_invalid.reshape((leaf.shape[0], -1)), axis=1)
        invalid_mask = invalid_mask | leaf_invalid
    return invalid_mask


def make_proxy_loss_evaluator(
    static_arrays,
    static_config_scalars,
    veg_params_dict_static,
    gen_params_dict_static,
    obs_arrays,
    loss_norm_params,
):
    def eval_one_window(params, start_idx):
        return base.calculate_minibatch_loss(
            params,
            static_arrays,
            static_config_scalars,
            veg_params_dict_static,
            gen_params_dict_static,
            obs_arrays,
            loss_norm_params,
            start_idx,
        )

    @jax.jit
    def eval_many(params_batched, proxy_start_indices):
        return jax.vmap(
            lambda params: jax.vmap(
                lambda start_idx: eval_one_window(params, start_idx)
            )(proxy_start_indices)
        )(params_batched)

    return eval_many


def score_initializations_on_fixed_windows(initializations, proxy_start_indices, proxy_eval_fn):
    proxy_start_np = np.array(proxy_start_indices, dtype=np.int32)
    score_rows = []

    print("\n4. Proxy screening on fixed training windows...")
    print(f"    Proxy windows per start: {len(proxy_start_np)}")
    print(f"    LHS samples: {LHS_NUM_SAMPLES}, requested restarts: {MULTISTART_TOP_K}")

    params_batched = stack_param_dicts([entry["params"] for entry in initializations])
    total_losses, data_losses, reg_losses = proxy_eval_fn(params_batched, proxy_start_indices)
    total_losses = np.array(total_losses)
    data_losses = np.array(data_losses)
    reg_losses = np.array(reg_losses)

    valid_mask = (
        np.isfinite(total_losses)
        & np.isfinite(data_losses)
        & np.isfinite(reg_losses)
    )
    valid_count = valid_mask.sum(axis=1)
    safe_count = np.maximum(valid_count, 1)
    mean_total = np.where(valid_count > 0, np.sum(np.where(valid_mask, total_losses, 0.0), axis=1) / safe_count, np.inf)
    mean_data = np.where(valid_count > 0, np.sum(np.where(valid_mask, data_losses, 0.0), axis=1) / safe_count, np.inf)
    mean_reg = np.where(valid_count > 0, np.sum(np.where(valid_mask, reg_losses, 0.0), axis=1) / safe_count, np.inf)
    decoded_batched = base.decode_param_views(params_batched)

    for init_idx, init_entry in enumerate(initializations, start=1):
        mean_total_i = float(mean_total[init_idx - 1])
        mean_data_i = float(mean_data[init_idx - 1])
        mean_reg_i = float(mean_reg[init_idx - 1])
        valid_count_i = int(valid_count[init_idx - 1])

        score_rows.append(
            {
                "label": init_entry["label"],
                "source": init_entry["source"],
                "screening_mode": "fixed_windows",
                "proxy_total": mean_total_i,
                "proxy_data": mean_data_i,
                "proxy_reg": mean_reg_i,
                "valid_proxy_windows": valid_count_i,
                "BB": float(decoded_batched["BB"][init_idx - 1]),
                "SATDK": float(decoded_batched["SATDK"][init_idx - 1]),
                "SATPSI": float(decoded_batched["SATPSI"][init_idx - 1]),
                "MAXSMC": float(decoded_batched["MAXSMC"][init_idx - 1]),
            }
        )

        print(
            f"    [{init_idx:02d}/{len(initializations):02d}] {init_entry['label']}"
            f" | ProxyTotal={mean_total_i:.6f}"
            f" | ProxyData={mean_data_i:.6f}"
            f" | ProxyReg={mean_reg_i:.6f}"
            f" | ValidProxy={valid_count_i}/{len(proxy_start_np)}"
        )

    score_df = pd.DataFrame(score_rows).sort_values(
        by=["proxy_total", "proxy_data", "label"],
        ascending=[True, True, True],
    )
    score_path = output_path(f"{OUTPUT_PREFIX}_proxy_scores.csv")
    score_df.to_csv(score_path, index=False)
    print(f"Saved {score_path}")
    return score_df


def score_initializations_on_full_sequence(
    initializations,
    evaluate_full_training_sequence_single,
    train_sequence_length: int,
):
    score_rows = []

    print("\n4. Proxy screening on the full training sequence...")
    print(
        f"    Training sequence length: {train_sequence_length} steps "
        f"({train_sequence_length / 24:.1f} days)"
    )
    print(f"    LHS samples: {LHS_NUM_SAMPLES}, requested restarts: {MULTISTART_TOP_K}")

    params_batched = stack_param_dicts([entry["params"] for entry in initializations])
    total_losses, components = evaluate_full_monitor_many(
        params_batched,
        evaluate_full_training_sequence_single,
    )
    decoded_batched = base.decode_param_views(params_batched)

    for init_idx, init_entry in enumerate(initializations, start=1):
        total_loss_i = float(total_losses[init_idx - 1])
        is_valid_i = int(np.isfinite(total_loss_i))
        component_i = components[init_idx - 1]
        smap_loss_i = float(component_i["smap_loss"])
        smap_count_i = int(component_i["smap_count"])

        score_rows.append(
            {
                "label": init_entry["label"],
                "source": init_entry["source"],
                "screening_mode": "full_sequence_train",
                "proxy_total": total_loss_i,
                "proxy_data": total_loss_i,
                "proxy_reg": 0.0,
                "valid_proxy_windows": is_valid_i,
                "proxy_sequence_steps": int(train_sequence_length),
                "proxy_sequence_days": float(train_sequence_length / 24.0),
                "proxy_smap_loss": smap_loss_i,
                "proxy_smap_count": smap_count_i,
                "BB": float(decoded_batched["BB"][init_idx - 1]),
                "SATDK": float(decoded_batched["SATDK"][init_idx - 1]),
                "SATPSI": float(decoded_batched["SATPSI"][init_idx - 1]),
                "MAXSMC": float(decoded_batched["MAXSMC"][init_idx - 1]),
            }
        )

        print(
            f"    [{init_idx:02d}/{len(initializations):02d}] {init_entry['label']}"
            f" | ProxyTotal={total_loss_i:.6f}"
            f" | SMAP={smap_loss_i:.6f}"
            f" | SMAP_n={smap_count_i}"
        )

    score_df = pd.DataFrame(score_rows).sort_values(
        by=["proxy_total", "proxy_data", "label"],
        ascending=[True, True, True],
    )
    score_path = output_path(f"{OUTPUT_PREFIX}_proxy_scores.csv")
    score_df.to_csv(score_path, index=False)
    print(f"Saved {score_path}")
    return score_df


def select_restart_entries(initializations, score_df):
    init_map = {entry["label"]: entry for entry in initializations}
    ranked_labels = [label for label in score_df["label"].tolist() if np.isfinite(score_df.loc[score_df["label"] == label, "proxy_total"].iloc[0])]

    selected_labels = []
    if ALWAYS_INCLUDE_DEFAULT_START and "default" in init_map:
        selected_labels.append("default")

    for label in ranked_labels:
        if label not in selected_labels:
            selected_labels.append(label)
        if len(selected_labels) >= MULTISTART_TOP_K:
            break

    selected_entries = [init_map[label] for label in selected_labels]

    print("\n5. Selected restart seeds...")
    for order_idx, label in enumerate(selected_labels, start=1):
        row = score_df.loc[score_df["label"] == label].iloc[0]
        print(
            f"    {order_idx}. {label}"
            f" | ProxyTotal={row['proxy_total']:.6f}"
            f" | ProxyData={row['proxy_data']:.6f}"
            f" | ProxyReg={row['proxy_reg']:.6f}"
        )

    return selected_entries


# ==========================================
# 5. Training helpers
# ==========================================

def build_optimizer():
    return base.build_optimizer()


def make_batched_epoch_runner(single_restart_train_step, candidate_start_indices):
    epoch_batch_capacity = int(len(candidate_start_indices) // base.WINDOWS_PER_BATCH)
    epoch_batch_count = min(base.BATCHES_PER_EPOCH, epoch_batch_capacity)
    if epoch_batch_count <= 0:
        raise ValueError(
            f"Training window pool is too small for WINDOWS_PER_BATCH={base.WINDOWS_PER_BATCH}."
        )

    batched_train_step = jax.jit(jax.vmap(single_restart_train_step, in_axes=(0, 0, 0)))
    epoch_pool_exhausted = epoch_batch_capacity < base.BATCHES_PER_EPOCH

    @jax.jit
    def run_one_epoch(params_batched, opt_state_batched, key):
        n_restarts = params_batched["BB_raw"].shape[0]
        key, perm_master = jax.random.split(key)
        perm_keys = jax.random.split(perm_master, n_restarts)
        epoch_start_indices = jax.vmap(
            lambda subkey: jax.random.permutation(subkey, candidate_start_indices)
        )(perm_keys)
        used_start_indices = epoch_start_indices[:, : epoch_batch_count * base.WINDOWS_PER_BATCH]
        batch_start_indices_all = used_start_indices.reshape(
            (n_restarts, epoch_batch_count, base.WINDOWS_PER_BATCH)
        )
        batch_start_indices_all = jnp.swapaxes(batch_start_indices_all, 0, 1)

        def scan_step(carry, batch_start_indices):
            params_now, opt_state_now = carry
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
            ) = batched_train_step(
                params_now,
                opt_state_now,
                batch_start_indices,
            )

            finite_mask = (
                (valid_window_count > 0)
                & (valid_window_count <= base.WINDOWS_PER_BATCH)
                & jnp.isfinite(loss)
                & jnp.isfinite(data_loss)
                & jnp.isfinite(margin_loss)
                & jnp.isfinite(grad_norm)
                & jnp.isfinite(param_delta_norm)
            )
            param_invalid_mask = tree_any_nonfinite(new_params)
            accept_mask = finite_mask & (~param_invalid_mask)

            params_next = tree_select(accept_mask, new_params, params_now)
            opt_state_next = tree_select(accept_mask, new_opt_state, opt_state_now)

            valid_window_count_clipped = jnp.clip(valid_window_count, 0, base.WINDOWS_PER_BATCH)
            metrics = {
                "accepted": accept_mask,
                "loss": jnp.where(accept_mask, loss, jnp.nan),
                "data_loss": jnp.where(accept_mask, data_loss, jnp.nan),
                "margin_loss": jnp.where(accept_mask, margin_loss, jnp.nan),
                "grad_norm": jnp.where(accept_mask, grad_norm, jnp.nan),
                "param_delta_norm": jnp.where(accept_mask, param_delta_norm, jnp.nan),
                "grad_abs_diag": jax.tree_util.tree_map(
                    lambda leaf: jnp.where(accept_mask, leaf, jnp.nan),
                    grad_abs_diag,
                ),
                "step_abs_diag": jax.tree_util.tree_map(
                    lambda leaf: jnp.where(accept_mask, leaf, jnp.nan),
                    step_abs_diag,
                ),
                "valid_window_count": jnp.where(accept_mask, valid_window_count_clipped, 0),
                "dropped_window_count": base.WINDOWS_PER_BATCH - valid_window_count_clipped,
            }
            return (params_next, opt_state_next), metrics

        (params_next, opt_state_next), metrics = lax.scan(
            scan_step,
            (params_batched, opt_state_batched),
            batch_start_indices_all,
        )
        return params_next, opt_state_next, key, metrics

    return run_one_epoch, epoch_batch_count, epoch_pool_exhausted


def summarize_epoch_metrics(epoch_metrics_np):
    accepted = epoch_metrics_np["accepted"].astype(bool)
    valid_batches = accepted.sum(axis=0)
    safe_valid_batches = np.maximum(valid_batches, 1)
    mean_total = np.where(
        valid_batches > 0,
        np.nansum(epoch_metrics_np["loss"], axis=0) / safe_valid_batches,
        np.nan,
    )
    mean_data = np.where(
        valid_batches > 0,
        np.nansum(epoch_metrics_np["data_loss"], axis=0) / safe_valid_batches,
        np.nan,
    )
    mean_reg = np.where(
        valid_batches > 0,
        np.nansum(epoch_metrics_np["margin_loss"], axis=0) / safe_valid_batches,
        np.nan,
    )
    mean_grad_norm = np.where(
        valid_batches > 0,
        np.nansum(epoch_metrics_np["grad_norm"], axis=0) / safe_valid_batches,
        np.nan,
    )
    mean_param_delta_norm = np.where(
        valid_batches > 0,
        np.nansum(epoch_metrics_np["param_delta_norm"], axis=0) / safe_valid_batches,
        np.nan,
    )
    mean_grad_abs_diag = jax.tree_util.tree_map(
        lambda arr: np.where(
            valid_batches > 0,
            np.nansum(arr, axis=0) / safe_valid_batches,
            np.nan,
        ),
        epoch_metrics_np["grad_abs_diag"],
    )
    mean_step_abs_diag = jax.tree_util.tree_map(
        lambda arr: np.where(
            valid_batches > 0,
            np.nansum(arr, axis=0) / safe_valid_batches,
            np.nan,
        ),
        epoch_metrics_np["step_abs_diag"],
    )
    valid_windows = np.sum(epoch_metrics_np["valid_window_count"], axis=0)
    dropped_windows = np.sum(epoch_metrics_np["dropped_window_count"], axis=0)
    return {
        "valid_batches": valid_batches,
        "mean_total": mean_total,
        "mean_data": mean_data,
        "mean_reg": mean_reg,
        "mean_grad_norm": mean_grad_norm,
        "mean_param_delta_norm": mean_param_delta_norm,
        "mean_grad_abs_diag": mean_grad_abs_diag,
        "mean_step_abs_diag": mean_step_abs_diag,
        "valid_windows": valid_windows,
        "dropped_windows": dropped_windows,
    }


def evaluate_full_monitor_many(params_batched, evaluate_full_monitor_single):
    n_restarts = int(params_batched["BB_raw"].shape[0])
    monitor_losses = []
    monitor_components = []
    for restart_idx in range(n_restarts):
        params_i = get_param_tree_at(params_batched, restart_idx)
        loss_i, components_i = evaluate_full_monitor_single(params_i)
        monitor_losses.append(float(loss_i))
        monitor_components.append(components_i)
    return np.array(monitor_losses, dtype=np.float64), monitor_components


def train_one_restart(
    run_label: str,
    run_seed: int,
    init_params,
    train_step,
    optimizer,
    candidate_start_indices,
    evaluate_full_monitor,
):
    print("\n" + "-" * 60)
    print(f"Restart {run_label}")

    params = {k: jnp.array(v) for k, v in init_params.items()}
    initial_params = {k: jnp.array(v) for k, v in init_params.items()}
    opt_state = optimizer.init(params)

    base.print_param_summary("Initial parameters", params)
    print(
        f"    Initial margin penalty: "
        f"{float(base.compute_margin_penalty(params)):.6f}"
    )

    monitor_start_time = time.time()
    initial_monitor_loss, initial_components = evaluate_full_monitor(params)
    print(
        f"    Initial monitor loss: {initial_monitor_loss:.6f}"
        f" (SMAP={initial_components['smap_loss']})"
        f" in {time.time() - monitor_start_time:.1f}s"
    )

    key = jax.random.PRNGKey(run_seed)
    train_history = []
    monitor_history = [(0, initial_monitor_loss)]
    best_monitor_loss = initial_monitor_loss
    best_params = {k: np.array(v) for k, v in params.items()}
    best_epoch = 0

    epoch_batch_capacity = int(len(candidate_start_indices) // base.WINDOWS_PER_BATCH)
    if epoch_batch_capacity <= 0:
        raise ValueError(
            f"Training window pool is too small for WINDOWS_PER_BATCH={base.WINDOWS_PER_BATCH}."
        )

    for epoch in range(base.NUM_EPOCHS):
        epoch_start = time.time()
        total_losses = []
        data_losses = []
        margin_losses = []
        grad_norms = []
        param_delta_norms = []
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
        epoch_max_attempts = min(base.MAX_BATCH_ATTEMPTS_PER_EPOCH, epoch_batch_capacity)

        while len(total_losses) < base.BATCHES_PER_EPOCH and attempt_count < epoch_max_attempts:
            batch_offset = attempt_count * base.WINDOWS_PER_BATCH
            batch_start_indices = lax.dynamic_slice(
                epoch_start_indices,
                (batch_offset,),
                (base.WINDOWS_PER_BATCH,),
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
            ) = train_step(params, opt_state, batch_start_indices)
            loss = loss.block_until_ready()
            data_loss = data_loss.block_until_ready()
            margin_loss = margin_loss.block_until_ready()
            valid_window_count = int(valid_window_count.block_until_ready())
            grad_norm = grad_norm.block_until_ready()
            param_delta_norm = param_delta_norm.block_until_ready()
            loss_value = float(loss)
            data_loss_value = float(data_loss)
            margin_loss_value = float(margin_loss)
            grad_norm_value = float(grad_norm)
            param_delta_norm_value = float(param_delta_norm)
            dropped_window_count += base.WINDOWS_PER_BATCH - valid_window_count

            if (
                valid_window_count <= 0
                or valid_window_count > base.WINDOWS_PER_BATCH
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
            valid_window_count_total += valid_window_count
            consecutive_nan = 0
            last_valid_params = {k: jnp.array(v) for k, v in params.items()}
            last_valid_opt_state = opt_state

        if len(total_losses) == 0:
            print(
                f"[{run_label}] Epoch {epoch + 1:03d} | all batches invalid, skipping"
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
        train_history.append(avg_train_loss)
        monitor_str = ""
        if (epoch + 1) % base.VALIDATION_INTERVAL_EPOCHS == 0:
            monitor_loss, monitor_components = evaluate_full_monitor(params)
            monitor_history.append((epoch + 1, monitor_loss))
            if np.isfinite(monitor_loss) and monitor_loss < best_monitor_loss:
                best_monitor_loss = monitor_loss
                best_params = {k: np.array(v) for k, v in params.items()}
                best_epoch = epoch + 1
            monitor_str = (
                f" | Monitor={monitor_loss:.6f}"
                f" (SMAP={monitor_components['smap_loss']})"
            )

        decoded = base.decode_param_views(params)
        margin_val = float(base.compute_margin_penalty(params))
        margin_term = base.MARGIN_PENALTY_WEIGHT * margin_val
        capped_str = ""
        if len(total_losses) < base.BATCHES_PER_EPOCH and attempt_count >= epoch_max_attempts:
            capped_str = " | Attempt cap reached"
        if len(total_losses) < base.BATCHES_PER_EPOCH and attempt_count >= epoch_batch_capacity:
            epoch_pool_exhausted = True
        batch_str = (
            f" | ValidBatches={len(total_losses)}/{base.BATCHES_PER_EPOCH}"
            f" | Attempts={attempt_count}"
        )
        window_str = (
            f" | ValidWindows={valid_window_count_total}/{len(total_losses) * base.WINDOWS_PER_BATCH}"
        )
        dropped_batch_str = (
            f" | DroppedBatches={dropped_batch_count}" if dropped_batch_count > 0 else ""
        )
        dropped_window_str = (
            f" | DroppedWindows={dropped_window_count}" if dropped_window_count > 0 else ""
        )
        pool_str = " | EpochPoolExhausted" if epoch_pool_exhausted else ""

        print(
            f"[{run_label}] Epoch {epoch + 1:03d} | TrainTotal={avg_train_loss:.6f}"
            f" | TrainData={avg_train_data_loss:.6f}"
            f" | TrainReg={avg_train_margin_loss:.6f}"
            f" | GradNorm={avg_grad_norm:.3e}"
            f" | ParamDeltaNorm={avg_param_delta_norm:.3e}"
            f"{monitor_str}{batch_str}{window_str}{dropped_batch_str}{dropped_window_str}{capped_str}{pool_str}"
            f" | Time={time.time() - epoch_start:.2f}s"
            f" | BB={float(decoded['BB']):.4f}"
            f" | K={float(decoded['SATDK']):.2e}"
            f" | PSI={float(decoded['SATPSI']):.4f}"
            f" | MAXSMC={float(decoded['MAXSMC']):.4f}"
            f" | MarginRaw={margin_val:.4f}"
            f" | MarginTerm={margin_term:.4f}"
        )

    print(f"[{run_label}] Best monitor loss: {best_monitor_loss:.6f} (Epoch {best_epoch})")
    base.print_param_summary(f"{run_label} best parameters", best_params)

    return {
        "label": run_label,
        "initial_params": initial_params,
        "best_params": best_params,
        "best_monitor_loss": best_monitor_loss,
        "best_epoch": best_epoch,
        "train_history": train_history,
        "monitor_history": monitor_history,
    }


# ==========================================
# 6. Main
# ==========================================

def main():
    proxy_screening_mode = resolve_proxy_screening_mode()

    print(">>> LHS multi-start optimization with SMAP-only loss")
    print("    Strategy: Latin hypercube sampling -> proxy screening -> multi-start gradient training")
    print("    Acceleration: fixed-window proxy screening uses jax.vmap; restart training uses jax.vmap + jax.lax.scan.")
    print(f"    Output directory: {ensure_output_dir()}")
    print(f"    Base script: {Path(base.__file__).name}")
    print(f"    Window size: {base.WINDOW_SIZE_MODEL_STEPS} steps ({base.WINDOW_SIZE_MODEL_STEPS / 24:.1f} days)")
    print(f"    Batch setup: {base.BATCHES_PER_EPOCH} batches/epoch, {base.WINDOWS_PER_BATCH} windows/batch")
    print(f"    Learning rates: {base.LEARNING_RATES}")
    print(f"    Margin penalty weight: {base.MARGIN_PENALTY_WEIGHT}")
    print(f"    LHS samples: {LHS_NUM_SAMPLES}")
    print(f"    Restarts to train: {MULTISTART_TOP_K}")
    print(f"    Proxy screening mode: {proxy_screening_mode}")
    if proxy_screening_mode == "fixed_windows":
        print(f"    Proxy windows per start: {PROXY_WINDOWS_PER_START}")

    (
        static_arrays,
        static_config_scalars,
        veg_params_dict_static,
        gen_params_dict_static,
        obs_arrays,
        full_idx,
    ) = base.load_aligned_data(base.FORCING_FILE, base.OBS_SMAP_CSV_FILE)

    n_total_obs = len(obs_arrays["smap_sh2o1"])
    split_cfg = base.resolve_data_partitions(n_total_obs)
    validation_enabled = split_cfg["validation_enabled"]
    n_train_pool = split_cfg["n_train_pool"]
    monitor_start_idx = split_cfg["monitor_start_idx"]
    monitor_length = split_cfg["monitor_length"]
    monitor_title = split_cfg["monitor_title"]
    monitor_mode = "validation" if validation_enabled else "full_train"

    print(f"    Total timeline length: {n_total_obs}")
    if validation_enabled:
        print(f"    Training pool: 0 ~ {n_train_pool}")
        print(f"    Validation pool: {n_train_pool} ~ {n_total_obs}")
        if proxy_screening_mode != "fixed_windows":
            print(f"    Proxy screening span: training full sequence ({base.TRAIN_SPLIT:.0%} split)")
    else:
        print(f"    Training pool: 0 ~ {n_total_obs} (all data)")
        print("    Validation pool: disabled")
        print("    Note: held-out validation is disabled; monitoring uses the full training sequence.")
        if proxy_screening_mode != "fixed_windows":
            print("    Proxy screening span: full sequence (all data)")

    print("\n2. Building SMAP-guided training windows...")
    candidate_start_indices = base.build_training_start_indices(obs_arrays, n_train_pool)
    if len(candidate_start_indices) < base.WINDOWS_PER_BATCH:
        raise ValueError(
            f"Not enough training windows ({len(candidate_start_indices)}) for WINDOWS_PER_BATCH={base.WINDOWS_PER_BATCH}."
        )

    print("\n3. Computing loss normalization...")
    loss_norm_params = base.compute_loss_norm_params(obs_arrays, n_train_pool)

    soil_idx = static_config_scalars["soil_type_index_to_optimize"] - 1
    base_params = static_arrays["base_soil_params_jax"]

    default_init = build_default_initialization(base_params, soil_idx)
    lhs_inits = build_lhs_initializations(base_params, soil_idx)
    all_initializations = [default_init] + lhs_inits

    if proxy_screening_mode == "fixed_windows":
        proxy_start_indices = build_proxy_start_indices(candidate_start_indices)
        proxy_eval_fn = make_proxy_loss_evaluator(
            static_arrays,
            static_config_scalars,
            veg_params_dict_static,
            gen_params_dict_static,
            obs_arrays,
            loss_norm_params,
        )
        score_df = score_initializations_on_fixed_windows(
            all_initializations,
            proxy_start_indices,
            proxy_eval_fn,
        )
    else:
        def evaluate_full_training_sequence(p):
            full_result = base.run_full_sequence_validation(
                p,
                static_arrays,
                static_config_scalars,
                veg_params_dict_static,
                gen_params_dict_static,
                obs_arrays,
                loss_norm_params,
                0,
                n_train_pool,
                use_satellite_warm_start=False,
            )
            return full_result["loss"], full_result["components"]

        score_df = score_initializations_on_full_sequence(
            all_initializations,
            evaluate_full_training_sequence,
            train_sequence_length=n_train_pool,
        )
    selected_entries = select_restart_entries(all_initializations, score_df)

    optimizer = build_optimizer()
    single_restart_train_step = base.make_minibatch_train_step(
        static_arrays,
        static_config_scalars,
        veg_params_dict_static,
        gen_params_dict_static,
        obs_arrays,
        loss_norm_params,
        optimizer,
    )
    run_one_epoch_batched, epoch_batch_count, epoch_pool_exhausted = make_batched_epoch_runner(
        single_restart_train_step,
        candidate_start_indices,
    )

    def evaluate_full_monitor(p):
        full_result = base.run_full_sequence_validation(
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

    selected_labels = [entry["label"] for entry in selected_entries]
    params_batched = stack_param_dicts([entry["params"] for entry in selected_entries])
    initial_params_batched = params_batched
    opt_state_batched = jax.vmap(optimizer.init)(params_batched)

    print("\n6. Start multi-start training...")
    print(f"    Parallel restarts: {len(selected_entries)}")
    print(f"    Batched epoch updates: {epoch_batch_count}")

    initial_monitor_losses, initial_monitor_components = evaluate_full_monitor_many(
        params_batched,
        evaluate_full_monitor,
    )
    best_monitor_losses = initial_monitor_losses.copy()
    best_epochs = np.zeros(len(selected_entries), dtype=np.int32)
    best_params_batched = params_batched
    train_histories = [[] for _ in selected_entries]
    monitor_histories = [[(0, float(initial_monitor_losses[i]))] for i in range(len(selected_entries))]

    for restart_idx, label in enumerate(selected_labels):
        print("\n" + "-" * 60)
        print(f"Restart {label}")
        base.print_param_summary("Initial parameters", get_param_tree_at(params_batched, restart_idx))
        print(
            f"    Initial {monitor_title.lower()} loss: {initial_monitor_losses[restart_idx]:.6f}"
            f" (SMAP={initial_monitor_components[restart_idx]['smap_loss']})"
        )

    train_key = jax.random.PRNGKey(base.RANDOM_SEED + TRAINING_SEED_OFFSET)

    for epoch in range(base.NUM_EPOCHS):
        epoch_start = time.time()
        params_batched, opt_state_batched, train_key, epoch_metrics = run_one_epoch_batched(
            params_batched,
            opt_state_batched,
            train_key,
        )
        epoch_metrics_np = jax.device_get(epoch_metrics)
        epoch_summary = summarize_epoch_metrics(epoch_metrics_np)

        monitor_losses = None
        monitor_components = None
        if (epoch + 1) % base.VALIDATION_INTERVAL_EPOCHS == 0:
            monitor_losses, monitor_components = evaluate_full_monitor_many(
                params_batched,
                evaluate_full_monitor,
            )
            improved_mask = np.isfinite(monitor_losses) & (monitor_losses < best_monitor_losses)
            best_monitor_losses = np.where(improved_mask, monitor_losses, best_monitor_losses)
            best_epochs = np.where(improved_mask, epoch + 1, best_epochs)
            best_params_batched = tree_select(jnp.array(improved_mask), params_batched, best_params_batched)

        for restart_idx, label in enumerate(selected_labels):
            valid_batches_i = int(epoch_summary["valid_batches"][restart_idx])
            attempts_i = epoch_batch_count
            valid_windows_i = int(epoch_summary["valid_windows"][restart_idx])
            dropped_windows_i = int(epoch_summary["dropped_windows"][restart_idx])
            dropped_batches_i = attempts_i - valid_batches_i

            if valid_batches_i <= 0:
                print(
                    f"[{label}] Epoch {epoch + 1:03d} | all batches invalid, skipping"
                    f" | Attempts={attempts_i}"
                    f" | DroppedBatches={dropped_batches_i}"
                    f" | DroppedWindows={dropped_windows_i}"
                )
                train_histories[restart_idx].append(np.nan)
                continue

            avg_train_total_i = float(epoch_summary["mean_total"][restart_idx])
            avg_train_data_i = float(epoch_summary["mean_data"][restart_idx])
            avg_train_reg_i = float(epoch_summary["mean_reg"][restart_idx])
            avg_grad_norm_i = float(epoch_summary["mean_grad_norm"][restart_idx])
            avg_param_delta_norm_i = float(epoch_summary["mean_param_delta_norm"][restart_idx])
            avg_grad_abs_diag_i = base.tree_to_ordered_float_dict(
                get_param_tree_at(epoch_summary["mean_grad_abs_diag"], restart_idx)
            )
            avg_step_abs_diag_i = base.tree_to_ordered_float_dict(
                get_param_tree_at(epoch_summary["mean_step_abs_diag"], restart_idx)
            )
            train_histories[restart_idx].append(avg_train_total_i)

            monitor_str = ""
            if monitor_losses is not None and monitor_components is not None:
                monitor_histories[restart_idx].append((epoch + 1, float(monitor_losses[restart_idx])))
                monitor_str = (
                    f" | {monitor_title}={float(monitor_losses[restart_idx]):.6f}"
                    f" (SMAP={monitor_components[restart_idx]['smap_loss']})"
                )

            params_i = get_param_tree_at(params_batched, restart_idx)
            decoded_i = base.decode_param_views(params_i)
            margin_val_i = float(base.compute_margin_penalty(params_i))
            margin_term_i = base.MARGIN_PENALTY_WEIGHT * margin_val_i
            pool_str = " | EpochPoolExhausted" if epoch_pool_exhausted else ""
            print(
                f"[{label}] Epoch {epoch + 1:03d} | TrainTotal={avg_train_total_i:.6f}"
                f" | TrainData={avg_train_data_i:.6f}"
                f" | TrainReg={avg_train_reg_i:.6f}"
                f" | GradNorm={avg_grad_norm_i:.3e}"
                f" | ParamDeltaNorm={avg_param_delta_norm_i:.3e}"
                f"{base.format_ordered_scalar_dict('GradRawAbs', avg_grad_abs_diag_i)}"
                f"{base.format_ordered_scalar_dict('StepRawAbs', avg_step_abs_diag_i)}"
                f"{monitor_str}"
                f" | ValidBatches={valid_batches_i}/{base.BATCHES_PER_EPOCH}"
                f" | Attempts={attempts_i}"
                f" | ValidWindows={valid_windows_i}/{attempts_i * base.WINDOWS_PER_BATCH}"
                f" | DroppedBatches={dropped_batches_i}"
                f" | DroppedWindows={dropped_windows_i}"
                f"{pool_str}"
                f" | Time={time.time() - epoch_start:.2f}s"
                f" | BB={float(decoded_i['BB']):.4f}"
                f" | K={float(decoded_i['SATDK']):.2e}"
                f" | PSI={float(decoded_i['SATPSI']):.4f}"
                f" | MAXSMC={float(decoded_i['MAXSMC']):.4f}"
                f" | MarginRaw={margin_val_i:.4f}"
                f" | MarginTerm={margin_term_i:.4f}"
            )

    run_results = []
    for restart_idx, label in enumerate(selected_labels):
        best_params_i = get_param_tree_at(best_params_batched, restart_idx)
        run_results.append(
            {
                "label": label,
                "initial_params": get_param_tree_at(initial_params_batched, restart_idx),
                "best_params": best_params_i,
                "best_monitor_loss": float(best_monitor_losses[restart_idx]),
                "monitor_mode": monitor_mode,
                "best_epoch": int(best_epochs[restart_idx]),
                "train_history": train_histories[restart_idx],
                "monitor_history": monitor_histories[restart_idx],
            }
        )
        print(f"[{label}] Best {monitor_title.lower()} loss: {best_monitor_losses[restart_idx]:.6f} (Epoch {best_epochs[restart_idx]})")
        base.print_param_summary(f"{label} best parameters", best_params_i)

    best_run = min(run_results, key=lambda item: item["best_monitor_loss"])
    print("\n" + "=" * 60)
    print(f"Overall best restart: {best_run['label']}")
    print(f"Overall best {monitor_title.lower()} loss: {best_run['best_monitor_loss']:.6f}")
    base.print_param_summary("Overall best parameters", best_run["best_params"])

    restart_rows = []
    for result in run_results:
        proxy_row = score_df.loc[score_df["label"] == result["label"]].iloc[0]
        restart_rows.append(
            {
                "label": result["label"],
                "screening_mode": proxy_row["screening_mode"],
                "proxy_total": proxy_row["proxy_total"],
                "proxy_data": proxy_row["proxy_data"],
                "proxy_reg": proxy_row["proxy_reg"],
                "best_monitor_loss": result["best_monitor_loss"],
                "monitor_mode": result["monitor_mode"],
                "best_epoch": result["best_epoch"],
            }
        )
    restart_df = pd.DataFrame(restart_rows).sort_values(by=["best_monitor_loss", "proxy_total", "label"])
    restart_path = output_path(f"{OUTPUT_PREFIX}_restart_summary.csv")
    restart_df.to_csv(restart_path, index=False)
    print(f"Saved {restart_path}")

    # Implementation detail.

    # ==============================================================
    # Output handling.
    # ==============================================================
    print("\n[Saving Full Training History...]")
    history_rows = []
    for result in run_results:
        label = result["label"]
        train_hist = result["train_history"]
        # monitor_history has the structure  [(epoch, loss), ...], convert to a dictionary for matching
        monitor_hist_dict = dict(result["monitor_history"])

        for epoch_idx, t_loss in enumerate(train_hist):
            epoch = epoch_idx + 1
            monitor_loss = monitor_hist_dict.get(epoch, np.nan)  # Numerical-stability safeguard.
            history_rows.append({
                "Restart_Label": label,
                "Epoch": epoch,
                "Train_Loss": t_loss,
                "Monitor_Loss": monitor_loss
            })

    # Output handling.
    history_df = pd.DataFrame(history_rows)
    history_path = output_path(f"{OUTPUT_PREFIX}_training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Saved {history_path}")

    # Model-parameter handling.
    # Model-parameter handling.
    full_records_path = output_path(f"{OUTPUT_PREFIX}_full_records.pkl")
    with full_records_path.open("wb") as f:
        pickle.dump(run_results, f)
    print(f"Saved {full_records_path}")
    # ==============================================================

    print("\n7. Running full-sequence diagnostics for the overall best restart...")
    full_before = base.run_full_sequence_validation(
        default_init["params"],
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
    full_after = base.run_full_sequence_validation(
        best_run["best_params"],
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

        rmse_before_train_smap, n_train_smap = base.compute_rmse(
            full_before["sim_sh2o1"][train_slice],
            full_after["obs_smap"][train_slice],
        )
        rmse_after_train_smap, _ = base.compute_rmse(
            full_after["sim_sh2o1"][train_slice],
            full_after["obs_smap"][train_slice],
        )
        rmse_before_val_smap, n_val_smap = base.compute_rmse(
            full_before["sim_sh2o1"][val_slice],
            full_after["obs_smap"][val_slice],
        )
        rmse_after_val_smap, _ = base.compute_rmse(
            full_after["sim_sh2o1"][val_slice],
            full_after["obs_smap"][val_slice],
        )

        print("\n=== SMAP SH2O(1) RMSE ===")
        print(f"Train: before={rmse_before_train_smap:.4f}, after={rmse_after_train_smap:.4f}, n={n_train_smap}")
        print(f"Val:   before={rmse_before_val_smap:.4f}, after={rmse_after_val_smap:.4f}, n={n_val_smap}")
    else:
        all_slice = slice(0, n_total_obs)
        rmse_before_all_smap, n_all_smap = base.compute_rmse(
            full_before["sim_sh2o1"][all_slice],
            full_after["obs_smap"][all_slice],
        )
        rmse_after_all_smap, _ = base.compute_rmse(
            full_after["sim_sh2o1"][all_slice],
            full_after["obs_smap"][all_slice],
        )

        print("\n=== SMAP SH2O(1) RMSE (All Data) ===")
        print(f"All: before={rmse_before_all_smap:.4f}, after={rmse_after_all_smap:.4f}, n={n_all_smap}")

    fit_df = pd.DataFrame(
        {
            "Date": full_idx,
            "SMAP_SH2O1": full_after["obs_smap"],
            "Sim_SH2O1_Default": full_before["sim_sh2o1"],
            "Sim_SH2O1_Best": full_after["sim_sh2o1"],
        }
    )
    fit_path = output_path(f"{OUTPUT_PREFIX}_satellite_fit_timeseries.csv")
    fit_df.to_csv(fit_path, index=False)
    print(f"Saved {fit_path}")

    default_decoded = base.decode_param_views(default_init["params"])
    best_decoded = base.decode_param_views(best_run["best_params"])
    optimized_params = {
        "BB": float(best_decoded["BB"]),
        "SATDK": float(best_decoded["SATDK"]),
        "SATPSI": float(best_decoded["SATPSI"]),
        "MAXSMC": float(best_decoded["MAXSMC"]),
        "best_restart": best_run["label"],
        "best_monitor_loss": float(best_run["best_monitor_loss"]),
        "monitor_mode": monitor_mode,
    }
    pkl_path = output_path(f"{OUTPUT_PREFIX}_optimized_params.pkl")
    with pkl_path.open("wb") as f:
        pickle.dump(optimized_params, f)
    print(f"Saved {pkl_path}")

    default_reference_params = {
        "BB": float(default_decoded["BB"]),
        "SATDK": float(default_decoded["SATDK"]),
        "SATPSI": float(default_decoded["SATPSI"]),
        "MAXSMC": float(default_decoded["MAXSMC"]),
    }
    best_soil_param_overrides = {
        "BB": float(best_decoded["BB"]),
        "SATDK": float(best_decoded["SATDK"]),
        "SATPSI": float(best_decoded["SATPSI"]),
        "MAXSMC": float(best_decoded["MAXSMC"]),
    }
    deenkf_summaries = run_deterministic_assimilation_cases(
        output_prefix=OUTPUT_PREFIX,
        default_reference_params=default_reference_params,
        best_soil_param_overrides=best_soil_param_overrides,
    )
    print(json.dumps(deenkf_summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
