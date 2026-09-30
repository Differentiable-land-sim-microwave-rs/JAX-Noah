"""Technical documentation."""

import jax.numpy as jnp
import jax
from typing import Tuple, Any  # new Any

from Module_model_constants_jax_simple import EMISSI_S, CP, TFREEZ

jax.config.update("jax_enable_x64", False)

# ============================================================================
# Numerical-stability safeguard.
# ============================================================================

def safe_divide(numerator, denominator, epsilon=1e-9, default_value=0.0):
    """Technical documentation."""
    safe_denom = jnp.where(jnp.abs(denominator) < epsilon, 
                           jnp.sign(denominator) * epsilon, 
                           denominator)
    result = numerator / safe_denom
    # Output handling.
    result = jnp.where(jnp.isfinite(result), result, default_value)
    return result

def clip_state_variables(STC, SMC, SH2O, SMCMAX):
    """Technical documentation."""
    # temperaturerange: 200-350K
    STC = jnp.clip(STC, 200.0, 350.0)
    
    # SMC range: 0.01 to SMCMAX
    SMC = jnp.clip(SMC, 0.01, SMCMAX)
    
    # State-variable handling.
    SH2O = jnp.clip(SH2O, 0.01, SMC)
    
    # State-variable handling.
    SMC = jnp.maximum(SMC, SH2O)
    
    return STC, SMC, SH2O

def check_and_fix_nan(x, default_value=0.0, name="variable"):
    """Technical documentation."""
    # Jacobian and gradient handling.
    # Numerical-stability safeguard.
    return x

# ============================================================================
# Numerical-stability safeguard.
# ============================================================================

# Output handling.
# --- (REMOVED) ---
# grads = {} (Removed)
# now_date = None (Removed)
# k_time = None (Removed)
# veg_param_path = ... (Removed)
# soil_param_path = ... (Removed)
# veg_parameter = pd.read_csv(...) (Removed)
# soil_parameter = pd.read_csv(...) (Removed)
# --- (END REMOVED) ---


def SFLX(FFROZP, DT, SLDPTH, ZSOIL, LWDN, SOLDN, SOLNET, SFCPRS, PRCP, SFCTMP, Q2, TH2,
         Q2SAT, DQSDT2, VEGTYP: int, SLOPETYP: int, SHDFAC, SHDMIN, SHDMAX,
         ALB, SNOALB, TBOT, CMC, T1, STC, SMC, SH2O, STYPE, SNOWH, SNEQV, CH,
         PC, XLAI, RDLAI2D, USEMONALB, SNOTIME1, RIBB,
         # --- (UPDATED) ---
         nowdate: Any,  # Implementation detail.
         ktime: int,
         SOC,
         # --- (UPDATED) ---
         # Implementation detail.
         soil_params_dict: dict,  # <-- UPDATED: usedictionary
         veg_params_dict: dict,
         gen_params_dict: dict,
         # --- (END UPDATED) ---
         SBETA_OPTION, DF_OPTION, ROOT_OPTION, INF_OPTION, SOC_OPTION_KS,
         SOC_OPTION2_THERMAL, RIC_OPTION, BLIM_OPTION, CK_OPTION, RLMO,
         NROOT_jax: int):
    # --- (REMOVED) ---
    # Implementation detail.
    # global now_date, k_time
    # now_date = nowdate
    # k_time = ktime
    # --- (END REMOVED) ---

    RD = 287.04
    CP = 1004.6
    LVH2O = 2.501E+6
    LSUBS = 2.83E+6


    (CFACTR, CMCMAX, RSMAX, TOPT, REFKDT, KDT, SBETA, SHDFAC, RSMIN, RGL, HS, ZBOT, FRZX, PSISAT, SLOPE, SNUP, SALP,
     BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ, FXEXP,
     RTDIS, NROOT, CZIL, LAIMIN, LAIMAX, EMISSMIN, EMISSMAX, ALBEDOMIN, ALBEDOMAX, Z0MIN,
     Z0MAX, CSOIL, PTU, LVCOEF) = REDPRM(
        soil_params_dict, veg_params_dict, gen_params_dict,  # <-- UPDATED
        VEGTYP, STYPE[0], SLOPETYP, SLDPTH, ZSOIL, SHDFAC, ROOT_OPTION,
        NROOT_jax
    )


#Revised by Zheng and Zhang
    (PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, KDT, FRZX, F1) = SOCH(
        gen_params_dict,  # <-- NEW
        SOC_OPTION_KS, SOC[0],
        PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY,
        KDT, FRZX, REFKDT, F1)
# end

    condition_1 = SHDFAC >= SHDMAX
    condition_2 = SHDFAC <= SHDMIN
    INTERP_FRACTION = jnp.clip((SHDFAC - SHDMIN) / jnp.maximum(SHDMAX - SHDMIN, 1e-9), 0.0, 1.0)
    EMBRD = jnp.where(condition_1, EMISSMAX, jnp.where(condition_2, EMISSMIN, (
            1.0 - INTERP_FRACTION) * EMISSMIN + INTERP_FRACTION * EMISSMAX))

    XLAI_at_max = jnp.where(jnp.logical_not(RDLAI2D), LAIMAX, XLAI)
    XLAI_at_min = jnp.where(jnp.logical_not(RDLAI2D), LAIMIN, XLAI)
    XLAI_interp = jnp.where(jnp.logical_not(RDLAI2D),
                            (1.0 - INTERP_FRACTION) * LAIMIN + INTERP_FRACTION * LAIMAX,
                            XLAI)
    XLAI = jnp.where(condition_1, XLAI_at_max,
                     jnp.where(condition_2, XLAI_at_min, XLAI_interp))

    ALB_at_max = jnp.where(jnp.logical_not(USEMONALB), ALBEDOMIN, ALB)
    ALB_at_min = jnp.where(jnp.logical_not(USEMONALB), ALBEDOMAX, ALB)
    ALB_interp = jnp.where(jnp.logical_not(USEMONALB),
                           (1.0 - INTERP_FRACTION) * ALBEDOMAX + INTERP_FRACTION * ALBEDOMIN,
                           ALB)
    ALB = jnp.where(condition_1, ALB_at_max,
                    jnp.where(condition_2, ALB_at_min, ALB_interp))

    Z0BRD = jnp.where(condition_1, Z0MAX,
                      jnp.where(condition_2, Z0MIN,
                                (1.0 - INTERP_FRACTION) * Z0MIN + INTERP_FRACTION * Z0MAX))

    is_no_snow = SNEQV <= 1.0e-7
    SNEQV = jnp.where(is_no_snow, jnp.array(0.0), SNEQV)
    # Numerical-stability safeguard.
    safe_snowh = SNOWH + 1e-9
    SNDENS = jnp.where(is_no_snow, jnp.array(0.0), SNEQV / safe_snowh)
    SNOWH = jnp.where(is_no_snow, jnp.array(0.0), SNOWH)
    SNCOND = jnp.where(is_no_snow, jnp.array(1.0), CSNOW(SNDENS))
    SNOWNG = jnp.where((PRCP > 0.0) & (FFROZP > 0.5), jnp.array(True), jnp.array(False))
    FRZGRA = jnp.where((PRCP > 0.0) & (FFROZP <= 0.5) & (T1 <= TFREEZ), jnp.array(True), jnp.array(False))

    is_snow_or_frozen = SNOWNG | FRZGRA
    SN_NEW = jnp.where(is_snow_or_frozen, PRCP * DT * 0.001, jnp.array(0.0))
    SNEQV = jnp.where(is_snow_or_frozen, SNEQV + SN_NEW, SNEQV)
    PRCPF = jnp.where(is_snow_or_frozen, jnp.array(0.0), PRCP)

    SNDENS_new, SNOWH_new = SNOW_NEW(SFCTMP, SN_NEW, SNOWH, SNDENS)
    SNDENS = jnp.where(is_snow_or_frozen, SNDENS_new, SNDENS)
    SNOWH = jnp.where(is_snow_or_frozen, SNOWH_new, SNOWH)
    SNCOND = jnp.where(is_snow_or_frozen, CSNOW(SNDENS), SNCOND)

    DSOIL = -(0.5 * ZSOIL[0])

#Revise By Zheng and zhang
    FSSOC = SOC * 2700 / (SOC * 2700 + (1 - SOC) * 1300 + 1e-9)
    if SOC_OPTION2_THERMAL == 1:
        DF1 = TDFCND_SOC(SMC[0], QUARTZ, SMCMAX, SH2O[0], FSSOC[0])
    else:
        DF1 = TDFCND(SMC[0], QUARTZ, SMCMAX, SH2O[0], STYPE[0])
#end SOC_OPTION2_THERMAL

#Revised by Zheng and Zhang
    # --- Implementation details ---
    safe_exp_arg_default = jnp.clip(SBETA * SHDFAC, -20.0, 20.0)
    safe_exp_arg_unstable = jnp.clip(-1.0 * SHDFAC, -20.0, 20.0)
    factor_default = jnp.exp(safe_exp_arg_default)
    factor_unstable = jnp.exp(safe_exp_arg_unstable)
    # --- endsafeguard ---
    is_unstable_condition = (SBETA_OPTION == 1) & (RLMO <= 0)
    final_exp_factor = jnp.where(is_unstable_condition, factor_unstable, factor_default)
    DF1 = DF1 * final_exp_factor
#end SBETA_OPTION

    has_snow = SNEQV > 0.0
    SNCOVR_no_snow = jnp.array(0.0)
    ALBEDO_no_snow = ALB
    EMISSI_no_snow = EMBRD
    SSOIL_no_snow = DF1 * (T1 - STC[0]) / (DSOIL + 1e-9)

    SNCOVR_with_snow = SNFRAC(SNEQV, SNUP, SALP)
    SNCOVR_with_snow = jnp.clip(SNCOVR_with_snow, 0.0, 0.98)
    ALBEDO_with_snow, EMISSI_with_snow, SNOTIME1_with_snow = ALCALC(ALB, SNOALB, EMBRD, SNCOVR_with_snow, DT, SNOWNG,
                                                                    SNOTIME1, LVCOEF)
    DF1_with_snow = jnp.where(SNCOVR_with_snow > 0.97, SNCOND, DF1)
    DTOT = SNOWH + DSOIL + 1e-9
    DF1A = SNOWH / DTOT * SNCOND + DSOIL / DTOT * DF1_with_snow
    DF1_with_snow_final = DF1A * SNCOVR_with_snow + DF1_with_snow * (1.0 - SNCOVR_with_snow)
    SSOIL_with_snow = DF1_with_snow_final * (T1 - STC[0]) / DTOT

    SNCOVR = jnp.where(has_snow, SNCOVR_with_snow, SNCOVR_no_snow)
    ALBEDO = jnp.where(has_snow, ALBEDO_with_snow, ALBEDO_no_snow)
    EMISSI = jnp.where(has_snow, EMISSI_with_snow, EMISSI_no_snow)
    SNOTIME1 = jnp.where(has_snow, SNOTIME1_with_snow, SNOTIME1)
    DF1_final = jnp.where(has_snow, DF1_with_snow_final, DF1)
    SSOIL = jnp.where(has_snow, SSOIL_with_snow, SSOIL_no_snow)

    Z0 = jnp.where(SNCOVR > 0., SNOWZ0(SNCOVR, Z0BRD, SNOWH), Z0BRD)

    SOLNET_updated = SOLDN * (1.0 - ALBEDO)
    FDOWN = SOLNET_updated + LWDN
    T2V = SFCTMP * (1.0 + 0.61 * Q2)
    EPSCA, ETP, RCH, RR, FLX2, T24 = PENMAN(SFCTMP, SFCPRS, CH, T2V, TH2, PRCP, FDOWN, SSOIL, Q2, Q2SAT, SNOWNG,
                                            FRZGRA, DQSDT2, EMISSI, SNCOVR)

    has_vegetation = SHDFAC > 0.

    RC_no_veg = jnp.array(0.0)
    RCS_no_veg = jnp.array(0.0)
    RCT_no_veg = jnp.array(0.0)
    RCQ_no_veg = jnp.array(0.0)
    RCSOIL_no_veg = jnp.array(0.0)
    PC_no_veg = PC


    PC_with_veg, RC_with_veg, RCS_with_veg, RCT_with_veg, RCQ_with_veg, RCSOIL_with_veg = CANRES(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        SOLDN, CH, SFCTMP, Q2, SFCPRS, SH2O, ZSOIL,
        STYPE, RSMIN, Q2SAT, DQSDT2,
        TOPT, RSMAX, RGL, HS, XLAI, EMISSI, NROOT, SOC, SOC_OPTION_KS, RTDIS)

    PC = jnp.where(has_vegetation, PC_with_veg, PC_no_veg)
    RC = jnp.where(has_vegetation, RC_with_veg, RC_no_veg)
    RCS = jnp.where(has_vegetation, RCS_with_veg, RCS_no_veg)
    RCT = jnp.where(has_vegetation, RCT_with_veg, RCT_no_veg)
    RCQ = jnp.where(has_vegetation, RCQ_with_veg, RCQ_no_veg)
    RCSOIL = jnp.where(has_vegetation, RCSOIL_with_veg, RCSOIL_no_veg)

    def no_snow_branch(_):
        results = NOPAC(
            soil_params_dict, gen_params_dict,  # <-- UPDATED
            STYPE, ETP, PRCP, SMC, SMCMAX, SMCDRY, CMC, CMCMAX, DT, SHDFAC, SBETA,
            SFCTMP, T24, TH2, FDOWN, EMISSI, STC, EPSCA, PC, RCH, RR, CFACTR, SH2O, SLOPE, KDT, FRZX, ZSOIL, TBOT,
            ZBOT, NROOT, RTDIS, QUARTZ, FXEXP, CSOIL, DF_OPTION, INF_OPTION, SOC_OPTION2_THERMAL, SOC,
            RIC_OPTION, PSISAT,
            BEXP, SOC_OPTION_KS, BLIM_OPTION, CK_OPTION, SBETA_OPTION, RLMO, ktime)
        STC_n, SMC_n, SH2O_n, DEW_n, DRIP_n, EC_kin_n, EDIR_kin_n, ETA_mass_n, ET_kin_n, ETT_kin_n, \
            RUNOFF1_n, RUNOFF2_n, RUNOFF3_n, SSOIL_n, CMC_n, BETA_n, T1_n, FLX1_n, FLX3_n = results

        ETNS_mass_n = jnp.array(0.0)
        ESNOW_mass_n = jnp.array(0.0)
        SNOMLT_n = jnp.array(0.0)
        ETA_KINEMATIC_n = ETA_mass_n
        ETA_W_n = ETA_mass_n * LVH2O
        ETP_n = ETP
        SNEQV_n = SNEQV
        SNOWH_n = SNOWH
        SNCOVR_n = SNCOVR
        SNDENS_n = SNDENS

        return (STC_n, SMC_n, SH2O_n, DEW_n, DRIP_n, EC_kin_n, EDIR_kin_n, ET_kin_n, ETT_kin_n,
                ETNS_mass_n, ESNOW_mass_n, RUNOFF1_n, RUNOFF2_n, RUNOFF3_n, SSOIL_n, CMC_n, BETA_n, T1_n,
                FLX1_n, FLX3_n, SNOMLT_n, ETA_KINEMATIC_n, ETA_W_n, ETP_n, SNEQV_n, SNOWH_n, SNCOVR_n, SNDENS_n)

    def with_snow_branch(_):
        # --- (UPDATED) ---
        # Implementation detail.
        results = SNOPAC(
            soil_params_dict, gen_params_dict,  # <-- UPDATED
            STYPE, ETP, PRCP, PRCPF, SNOWNG, SMC,
            SMCMAX, SMCDRY, CMC, CMCMAX, DT, DF1_final, T1, SFCTMP, T24,
            TH2, FDOWN, STC, PC, RCH, RR,
            CFACTR, SNCOVR, SNEQV, SNDENS, SNOWH,
            SH2O, SLOPE, KDT, FRZX, ZSOIL, TBOT, ZBOT,
            SHDFAC, NROOT, RTDIS, FXEXP,
            CSOIL, FLX2, EMISSI, RIBB, DF_OPTION, INF_OPTION, SOC_OPTION2_THERMAL, SOC,
            RIC_OPTION, PSISAT,
            BEXP, SOC_OPTION_KS, BLIM_OPTION, CK_OPTION, ktime, QUARTZ)
        STC_s, SMC_s, SH2O_s, DEW_s, DRIP_s, ETP_s, EC_kin_s, EDIR_kin_s, ET_kin_s, ETT_kin_s, ETNS_mass_s, ESNOW_mass_s, \
            FLX1_s, FLX3_s, RUNOFF1_s, RUNOFF2_s, RUNOFF3_s, SSOIL_s, SNOMLT_s, CMC_s, BETA_s, SNEQV_s, SNOWH_s, \
            SNCOVR_s, SNDENS_s, T1_s, ETANRG_s = results

        ETA_KINEMATIC_s = ESNOW_mass_s + ETNS_mass_s
        ETA_W_s = ETANRG_s

        return (STC_s, SMC_s, SH2O_s, DEW_s, DRIP_s, EC_kin_s, EDIR_kin_s, ET_kin_s, ETT_kin_s,
                ETNS_mass_s, ESNOW_mass_s, RUNOFF1_s, RUNOFF2_s, RUNOFF3_s, SSOIL_s, CMC_s, BETA_s, T1_s,
                FLX1_s, FLX3_s, SNOMLT_s, ETA_KINEMATIC_s, ETA_W_s, ETP_s, SNEQV_s, SNOWH_s, SNCOVR_s, SNDENS_s)

    has_snow_cond = SNEQV > 0.0
    (STC, SMC, SH2O, DEW, DRIP, EC_kin, EDIR_kin, ET_kin, ETT_kin,
     ETNS_mass, ESNOW_mass, RUNOFF1, RUNOFF2, RUNOFF3, SSOIL, CMC, BETA, T1,
     FLX1, FLX3, SNOMLT, ETA_KINEMATIC, ETA_W, ETP, SNEQV, SNOWH, SNCOVR, SNDENS) = jax.lax.cond(
        has_snow_cond, with_snow_branch, no_snow_branch, None)

    # State-variable handling.
    # State-variable handling.
    SH2O = jnp.minimum(SH2O, SMC)

    ETP_W = ETP * ((1. - SNCOVR) * LVH2O + SNCOVR * LSUBS)
    ETA_W = jnp.where(ETP > 0.0, ETA_W, ETP_W)

    # (*** v1.1 numericalsafeguard ***) use safe_divide preventdivision by zero
    Q1 = Q2 + safe_divide(ETA_KINEMATIC * CP, RCH, epsilon=1e-9, default_value=0.0)
    SHEAT = - safe_divide(CH * CP * SFCPRS, RD * T2V, epsilon=1e-9, default_value=0.0) * (TH2 - T1)

    SSOIL_out = -SSOIL

    EC_mass = EC_kin * 1000.
    EDIR_mass = EDIR_kin * 1000.
    ET_mass = ET_kin * 1000.
    ETT_mass = ETT_kin * 1000.

    SOILM = -SMC[0] * ZSOIL[0]
    for k in range(1, len(SMC)):
        SOILM += SMC[k] * (ZSOIL[k - 1] - ZSOIL[k])

    # --- (UPDATED) ---
    # Implementation detail.
    _, _, _, _, SMCMAX_layers, SMCWLT_layers, _, _, _, _, _ = REDSTP(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        STYPE, jnp.arange(len(SMC)), SOC, SOC_OPTION_KS,
        required_grad=False)

    SOILWM = -(SMCMAX_layers[0] - SMCWLT_layers[0]) * ZSOIL[0]
    SOILWW = -(SMC[0] - SMCWLT_layers[0]) * ZSOIL[0]

    # (*** v1.1 numericalsafeguard ***) use safe_divide preventdivision by zero
    SMAV = safe_divide(SMC - SMCWLT_layers, SMCMAX_layers - SMCWLT_layers, epsilon=1e-9, default_value=0.0)

    for k in range(1, NROOT):
        SOILWM += (SMCMAX_layers[k] - SMCWLT_layers[k]) * (ZSOIL[k - 1] - ZSOIL[k])
        SOILWW += (SMC[k] - SMCWLT_layers[k]) * (ZSOIL[k - 1] - ZSOIL[k])

    # (*** v1.1 numericalsafeguard ***) use safe_divide preventdivision by zero
    SOILW = jnp.where(SOILWM < 1.0E-6, jnp.array(0.0), safe_divide(SOILWW, SOILWM, epsilon=1e-9, default_value=0.0))

    # (*** v1.1 numericalsafeguard ***) clip state variables to physical ranges before returning
    STC, SMC, SH2O = clip_state_variables(STC, SMC, SH2O, SMCMAX_layers)
    T1 = jnp.clip(T1, 200.0, 350.0)  # Implementation detail.
    
    # Numerical-stability safeguard.
    STC = check_and_fix_nan(STC, default_value=273.15, name="STC")
    SMC = check_and_fix_nan(SMC, default_value=0.2, name="SMC")
    SH2O = check_and_fix_nan(SH2O, default_value=0.2, name="SH2O")
    T1 = check_and_fix_nan(T1, default_value=273.15, name="T1")
    CMC = check_and_fix_nan(CMC, default_value=0.0, name="CMC")
    SNOWH = check_and_fix_nan(SNOWH, default_value=0.0, name="SNOWH")
    SNEQV = check_and_fix_nan(SNEQV, default_value=0.0, name="SNEQV")

    return (
        CMC, T1, STC, SMC, SH2O,
        SNOWH, SNEQV, SNOTIME1, ALBEDO, CH,
        ETA_W, SHEAT, ETA_KINEMATIC, FDOWN, EC_mass, EDIR_mass,
        ET_mass,
        ETT_mass, ESNOW_mass, DRIP, DEW, BETA, ETP,
        SSOIL_out,
        FLX1, FLX2, FLX3, SNOMLT, SNCOVR, RUNOFF1,
        RUNOFF2, RUNOFF3, RC, PC, RSMIN, XLAI,
        RCS, RCT, RCQ, RCSOIL,
        SOILW, SOILM, Q1, SMAV,
        Z0, Z0BRD, EMISSI
    )


# --- Implementation details ---
# soil_params_dict structure: {'BB': jnp.array, 'MAXSMC': jnp.array, 'SATDK': jnp.array, 'SATPSI': jnp.array, 'QTZ': jnp.array}
# Model-parameter handling.


# --- (REMOVED) ---
# Implementation detail.
# Model-parameter handling.
# --- (END REMOVED) ---


# --- (REMOVED) ---
# Implementation detail.
# Soil_Parameter = SoilParamJAX()
# --- (END REMOVED) ---


# --- (REMOVED) ---
# Implementation detail.
# def register_hooks(var, name): ...
# --- (END REMOVED) ---


# --- (REMOVED) ---
# Model-parameter handling.
# gen_parameters = { ... }
# --- (END REMOVED) ---


def REDSTP(soil_params_dict: dict, gen_params_dict: dict,  # <-- UPDATED: usedictionary
           STPNUM: jnp.ndarray, target: jnp.ndarray,
           SOC: jnp.ndarray = None, SOC_OPTION_KS: int = 0, required_grad: bool = True
           ) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Technical documentation."""

    # --- (UPDATED) ---
    # Implementation detail.
    index = STPNUM - 1
    BEXP = soil_params_dict['BB'][index]
    SMCMAX = soil_params_dict['MAXSMC'][index]
    DKSAT = soil_params_dict['SATDK'][index]
    PSISAT = soil_params_dict['SATPSI'][index]
    QUARTZ = soil_params_dict['QTZ'][index]

    F1 = jnp.log10(jnp.maximum(PSISAT, 1e-9)) + BEXP * jnp.log10(jnp.maximum(SMCMAX, 1e-9)) + 2.0
    REFSMC1 = SMCMAX * jnp.power(jnp.maximum(5.79E-9 / jnp.maximum(DKSAT, 1e-9), 1e-9), 1 / (2 * BEXP + 3 + 1e-9))
    SMCREF = REFSMC1 + 1. / 3. * (SMCMAX - REFSMC1)
    WLTSMC1 = SMCMAX * jnp.power(jnp.maximum(200. / jnp.maximum(PSISAT, 1e-9), 1e-9), -1. / (BEXP + 1e-9))
    SMCWLT = 0.5 * WLTSMC1
    DWSAT = BEXP * DKSAT * PSISAT / jnp.maximum(SMCMAX, 1e-9)
    SMCDRY = SMCWLT
    SMCMAX_base = SMCMAX

#Revised by Zheng and Zhang
    if SOC is not None:
        # --- (UPDATED) ---
        # read from the passed dictionary
        REFKDT = gen_params_dict['REFKDT_DATA']
        (PSISAT_soc, BEXP_soc, DKSAT_soc, DWSAT_soc, SMCMAX_soc, SMCWLT_soc, SMCREF_soc, SMCDRY_soc,
         KDT_soc, FRZX_soc, F1_soc) = SOCH(
            gen_params_dict,  # Implementation detail.
            SOC_OPTION_KS, SOC,
            PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY,
            None, None, REFKDT, F1
        )

        PSISAT = PSISAT_soc
        BEXP = BEXP_soc
        DKSAT = DKSAT_soc
        DWSAT = DWSAT_soc
        SMCMAX = SMCMAX_soc
        SMCWLT = SMCWLT_soc
        SMCREF = SMCREF_soc
        SMCDRY = SMCDRY_soc
        F1 = F1_soc
# end SOC_OPTION_KS

    return PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ, SMCMAX_base


def DEVAP(ETP1, SMC, SHDFAC, SMCMAX, SMCDRY, FXEXP) -> jnp.ndarray:

    SRATIO = (SMC - SMCDRY) / jnp.maximum(SMCMAX - SMCDRY, 1e-9)
    SRATIO = jnp.clip(SRATIO, 0, None)

    FX = jnp.where(SRATIO > 0.0, jnp.clip(jnp.power(jnp.maximum(SRATIO, 1e-9), FXEXP), 0.0, 1.0), jnp.array(0.0))

    return FX * (1.0 - SHDFAC) * ETP1


def TRANSP(STYPE, ETP1, SMC, CMC, SHDFAC, CMCMAX, PC, CFACTR, NROOT: int,
           RTDIS, SMCWLT, SMCREF) -> jnp.ndarray:

    exponent_part = jnp.clip(CMC / jnp.maximum(CMCMAX, 1e-9), 1e-9, None)
    ETP1A = jnp.where(CMC != 0.0,
                      SHDFAC * PC * ETP1 * (1.0 - exponent_part ** CFACTR),
                      SHDFAC * PC * ETP1)
    ET = jnp.zeros_like(SMC)
    SMCWLT_root = SMCWLT[:NROOT]
    SMCREF_root = SMCREF[:NROOT]
    GX = jnp.clip((SMC[:NROOT] - SMCWLT_root) / jnp.maximum(SMCREF_root - SMCWLT_root, 1e-9), 0.0, 1.0)
    SGX = jnp.sum(GX)
    SGX = SGX / jnp.maximum(NROOT, 1)
    RTX = RTDIS[:NROOT] + GX[:NROOT] - SGX
    GX = GX * jnp.clip(RTX, 0.0)
    DENOM = jnp.sum(GX)
    DENOM = jnp.where(DENOM <= 0.0, jnp.array(1), DENOM)
    ET = ET.at[:NROOT].set(ETP1A * GX / DENOM)

    return ET


def EVAPO(soil_params_dict: dict, gen_params_dict: dict,  # <-- UPDATED: usedictionary
          SMC, CMC, ETP1, DT, SH2O, SMCMAX, PC, STYPE, SHDFAC, CMCMAX,
          SMCDRY, CFACTR, NROOT: int, RTDIS, FXEXP, SOC, SOC_OPTION_KS):
    """Technical documentation."""
    condition1 = ETP1 > 0.0
    condition2 = SHDFAC < 1.0
    condition3 = SHDFAC > 0.0
    condition4 = CMC > 0.0

    # --- (UPDATED) ---
    PSISAT_arr, BEXP_arr, DKSAT_arr, DWSAT_arr, SMCMAX_arr, SMCWLT_arr, SMCREF_arr, SMCDRY_arr, F1_arr, QUARTZ_arr, _ = REDSTP(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        STYPE, jnp.arange(len(STYPE)), SOC, SOC_OPTION_KS, required_grad=True)

    EDIR = jnp.where(condition1 & condition2,
                     DEVAP(ETP1, SMC[0], SHDFAC, SMCMAX_arr[0], SMCDRY_arr[0], FXEXP),
                     jnp.array(0))

    ET = jnp.where(condition1 & condition3,
                   TRANSP(STYPE, ETP1, SH2O, CMC, SHDFAC, CMCMAX, PC, CFACTR, NROOT, RTDIS, SMCWLT_arr, SMCREF_arr),
                   jnp.zeros_like(SH2O))

    ETT = jnp.sum(ET)

    exponent_part = jnp.clip(CMC / jnp.maximum(CMCMAX, 1e-9), 1e-9, None)
    EC = jnp.where(condition1 & condition3 & condition4,
                   SHDFAC * (jnp.power(jnp.maximum(exponent_part, 1e-9), CFACTR)) * ETP1,
                   jnp.array(0.0))

    CMC2MS = CMC / jnp.maximum(DT, 1e-9)
    EC = jnp.minimum(CMC2MS, EC)

    ETA1 = EDIR + ETT + EC

    return ETA1, EDIR, ETT, EC, ET


def FAC2MIT(SMCMAX: jnp.ndarray) -> jnp.ndarray:
    EPS = 1E-6
    FLIMIT = jnp.full_like(SMCMAX, 0.90)
    condition = jnp.abs(SMCMAX - 0.395) < EPS
    FLIMIT = jnp.where(condition, jnp.array(0.59), FLIMIT)
    condition = (jnp.abs(SMCMAX - 0.434) < EPS) | (jnp.abs(SMCMAX - 0.404) < EPS)
    FLIMIT = jnp.where(condition, jnp.array(0.85), FLIMIT)
    condition = (jnp.abs(SMCMAX - 0.465) < EPS) | (jnp.abs(SMCMAX - 0.406) < EPS)
    FLIMIT = jnp.where(condition, jnp.array(0.86), FLIMIT)
    condition = (jnp.abs(SMCMAX - 0.476) < EPS) | (jnp.abs(SMCMAX - 0.439) < EPS)
    FLIMIT = jnp.where(condition, jnp.array(0.74), FLIMIT)
    condition = (jnp.abs(SMCMAX - 0.200) < EPS) | (jnp.abs(SMCMAX - 0.464) < EPS)
    FLIMIT = jnp.where(condition, jnp.array(0.80), FLIMIT)
    return FLIMIT


def CSNOW(DSNOW):
    UNIT = 0.11631
    C = 0.328 * 10 ** (2.25 * DSNOW)
    return 2.0 * UNIT * C


def SNFRAC(SNEQV, SNUP, SALP):
    RSNOW = SNEQV / jnp.maximum(SNUP, 1e-9)
    SNCOVR = jnp.where(SNEQV < SNUP,
                       1 - (jnp.exp(- SALP * RSNOW) - RSNOW * jnp.exp(- SALP)),
                       jnp.array(1.0))
    return SNCOVR


def ALCALC(ALB, SNOALB, EMBRD, SNCOVR, DT, SNOWNG, SNOTIME1, LVCOEF):
    SNACCA = 0.94
    SNACCB = 0.58
    EMISSI = EMBRD + SNCOVR * (EMISSI_S - EMBRD)
    SNOALB1 = SNOALB + LVCOEF * (0.85 - SNOALB)
    SNOTIME1 = jnp.where(SNOWNG, jnp.array(0.0), SNOTIME1 + DT)
    SNOALB2 = jnp.where(SNOWNG,
                        SNOALB1,
                        SNOALB1 * (SNACCA ** ((SNOTIME1 / 86400.0) ** SNACCB)))
    SNOALB2 = jnp.maximum(SNOALB2, ALB)
    ALBEDO = ALB + SNCOVR * (SNOALB2 - ALB)
    ALBEDO = jnp.clip(ALBEDO, None, SNOALB2)
    return ALBEDO, EMISSI, SNOTIME1


def TDFCND(SMC, QZ, SMCMAX, SH2O, NSOILTYPE):
    THKICE = 2.2
    THKW = 0.57
    THKO = 2.0
    THKQTZ = 7.7

    SATRATIO = SMC / jnp.maximum(SMCMAX, 1e-9)

    THKS = (THKQTZ ** QZ) * (THKO ** (1.0 - QZ))
    THKS = jnp.clip(THKS, 1e-9, None)

    XUNFROZ = SH2O / jnp.maximum(SMC, 1e-9)
    XU = XUNFROZ * SMCMAX

    THKSAT = THKS ** (1.0 - SMCMAX) * THKICE ** (SMCMAX - XU) * THKW ** XU
    THKDRY = (0.135 * (1.0 - SMCMAX) * 2700.0 + 64.7) / jnp.maximum(2700.0 - 0.947 * (1.0 - SMCMAX) * 2700.0, 1e-9)

    # --- JAX gradientfix ---
    # Jacobian and gradient handling.
    SATRATIO_safe_log = jnp.clip(SATRATIO, 1e-9, None)
    log_branch = jnp.log10(SATRATIO_safe_log) + 1.0

    AKE = jnp.where((SH2O + 0.0005) < SMC,
                    SATRATIO,
                    jnp.where(SATRATIO > 0.1,
                              log_branch,  # use the safe branch
                              jnp.array(0.0)))
    # --- fixend ---

    DF = AKE * (THKSAT - THKDRY) + THKDRY

    return DF


# Revised By Zhang
def TDFCND_SOC(SMC, QZ, SMCMAX, SH2O, FSSOC):
    THKICE = 2.2
    THKW = 0.57
    THKO = 2.0
    THKQTZ = 7.7
    THKSOC = 0.25

    SATRATIO = SMC / jnp.maximum(SMCMAX, 1e-9)

    THKS = (THKSOC ** FSSOC) * \
           (THKQTZ ** (QZ * (1.0 - FSSOC))) * \
           (THKO ** ((1.0 - QZ) * (1.0 - FSSOC)))

    THKS = jnp.clip(THKS, 1e-9)

    XUNFROZ = SH2O / jnp.maximum(SMC, 1e-9)
    XU = XUNFROZ * SMCMAX

    THKSAT = THKS ** (1.0 - SMCMAX) * THKICE ** (SMCMAX - XU) * THKW ** XU

    THKDRY = (0.135 * (1.0 - SMCMAX) * 2700.0 + 64.7) / (2700.0 - 0.947 * (1.0 - SMCMAX) * 2700.0 + 1e-9)

    # --- JAX gradientfix ---
    # Jacobian and gradient handling.
    SATRATIO_safe_log = jnp.clip(SATRATIO, 1e-9, None)
    log_branch = jnp.log10(SATRATIO_safe_log) + 1.0

    AKE = jnp.where((SH2O + 0.0005) < SMC,
                    SATRATIO,
                    jnp.where(SATRATIO > 0.1,
                              log_branch,  # use the safe branch
                              jnp.array(0.0)))
    # --- fixend ---

    DF = AKE * (THKSAT - THKDRY) + THKDRY

    return DF
# end SOC_OPTION2_THERMAL


def SNOW_NEW(TEMP, NEWSN, SNOWH, SNDENS):
    SNOWHC = SNOWH * 100.
    NEWSNC = NEWSN * 100.
    TEMPC = TEMP - 273.15
    DSNEW = jnp.where(TEMPC <= -15.,
                      jnp.array(0.05),
                      0.05 + 0.0017 * (TEMPC + 15.) ** 1.5)
    HNEWC = NEWSNC / jnp.maximum(DSNEW, 1e-9)
    safe_denominator = SNOWHC + HNEWC + 1e-9
    SNDENS = jnp.where(SNOWHC + HNEWC < 1.0E-3,
                       jnp.maximum(DSNEW, SNDENS),
                       (SNOWHC * SNDENS + HNEWC * DSNEW) / safe_denominator)
    SNOWHC = SNOWHC + HNEWC
    SNOWH = SNOWHC * 0.01
    return SNDENS, SNOWH


def SNOWZ0(SNCOVR, Z0BRD, SNOWH):
    Z0S = 0.001
    BURIAL = 7 * Z0BRD - SNOWH
    Z0EFF = jnp.where(BURIAL < 0.0007, Z0S, BURIAL / 7.0)
    return (1. - SNCOVR) * Z0BRD + SNCOVR * Z0EFF


def WDFCND(SH2OA, SMCMAX, BEXP, DKSAT, DWSAT, SICEMAX):
    # *** correction: raise the SH2OA lower bound to prevent Jacobian explosion ***
    # when SH2OA is very small, FACTR2^EXPONofthe derivative becomes very large
    # set a reasonable lower bound(0.01 = 1%volumetric water content)
    SH2OA_safe = jnp.maximum(SH2OA, 0.01)  # Numerical-stability safeguard.
    
    FACTR1 = 0.05 / jnp.maximum(SMCMAX, 1e-9)
    FACTR2 = SH2OA_safe / jnp.maximum(SMCMAX, 1e-9)
    FACTR1 = jnp.minimum(FACTR1, FACTR2)
    FACTR2 = jnp.maximum(FACTR2, 0.01)  # from1e-9increase to 0.01
    FACTR1 = jnp.maximum(FACTR1, 0.01)  # from1e-9increase to 0.01
    EXPON = BEXP + 2.0
    WDF = DWSAT * jnp.power(FACTR2, EXPON)
    VKWGT = jnp.where(SICEMAX > 0.0,
                      1.0 / (1.0 + (500.0 * SICEMAX) ** 3.0),
                      1.0)
    WDF = jnp.where(SICEMAX > 0.0,
                    VKWGT * WDF + (1.0 - VKWGT) * DWSAT * jnp.power(FACTR1, EXPON),
                    WDF)
    EXPON = (2.0 * BEXP) + 3.0
    WCND = DKSAT * jnp.power(FACTR2, EXPON)
    return WDF, WCND


def TMPAVG(TUP, TM, TDN, ZSOIL):
    T0 = 273.15
    DZ = jnp.zeros_like(TUP)
    DZ = DZ.at[0].set(-ZSOIL[0])
    DZ = DZ.at[1:].set(ZSOIL[:-1] - ZSOIL[1:])
    DZH = DZ * 0.5
    X0 = (T0 - TM) * DZH / (TDN - TM + 1e-9)
    XUP_1 = (T0 - TUP) * DZH / (TM - TUP + 1e-9)
    XUP_2 = DZH - (T0 - TUP) * DZH / (TM - TUP + 1e-9)
    XDN_1 = DZH - (T0 - TM) * DZH / (TDN - TM + 1e-9)
    XDN_2 = (T0 - TM) * DZH / (TDN - TM + 1e-9)
    TAVG = jnp.where(
        TUP < T0,
        jnp.where(
            TM < T0,
            jnp.where(
                TDN < T0,
                (TUP + 2.0 * TM + TDN) / 4.0,
                0.5 * (TUP * DZH + TM * (DZH + X0) + T0 * (2.0 * DZH - X0)) / jnp.maximum(DZ, 1e-9)
            ),
            jnp.where(
                TDN < T0,
                0.5 * (TUP * XUP_1 + T0 * (2.0 * DZ - XUP_1 - XDN_2) + TDN * XDN_2) / jnp.maximum(DZ, 1e-9),
                0.5 * (TUP * XUP_1 + T0 * (2.0 * DZ - XUP_1)) / jnp.maximum(DZ, 1e-9)
            )
        ),
        jnp.where(
            TM < T0,
            jnp.where(
                TDN < T0,
                0.5 * (T0 * (DZ - XUP_2) + TM * (DZH + XUP_2) + TDN * DZH) / jnp.maximum(DZ, 1e-9),
                0.5 * (T0 * (2.0 * DZ - XUP_2 - XDN_2) + TM * (XUP_2 + XDN_2)) / jnp.maximum(DZ, 1e-9)
            ),
            jnp.where(
                TDN < T0,
                (T0 * (DZ - XDN_1) + 0.5 * (T0 + TDN) * XDN_1) / jnp.maximum(DZ, 1e-9),
                (TUP + 2.0 * TM + TDN) / 4.0
            )
        )
    )
    return TAVG


def TBND(TU, TB, ZSOIL, ZBOT):
    ZUP = jnp.concatenate([jnp.array([0.0]), ZSOIL[:-1]])
    ZB_bottom = 2.0 * ZBOT - ZSOIL[-1]
    ZB = jnp.concatenate([ZSOIL[1:], jnp.array([ZB_bottom])])
    Z_MID = ZSOIL
    delta_F = (ZUP - Z_MID) / (ZUP - ZB + 1e-9)
    return TU + (TB - TU) * delta_F


def FRH2O_tensor(TKELV, SMC, SH2O, SMCMAX, BEXP, PSIS, BLIM_OPTION, CK_OPTION, BLIMBX):
    # This JAX version is modified to replicate the conditional logic
    # of the provided NoahPy implementation, preventing NaN gradients
    # by masking operations that are invalid when TKELV > T0.

    epsilon = 1e-9  # Small epsilon for numerical stability

#Revised by Zheng and Zhang
    if CK_OPTION == 0:
        CK = jnp.full_like(SH2O, 8.0)
    else:
        CK = jnp.full_like(SH2O, 0.0)  # CK_OPTION=1 (Flerchinger) not fully implemented in PyTorch version
#end CK_OPTION

# Revised by Zheng and Zhang
    if BLIM_OPTION == 0:
        BLIM = jnp.full_like(SH2O, 5.5)
    else:
        BLIM = jnp.asarray(BLIMBX, dtype=SH2O.dtype)
#end BLIM_OPTION

    ERROR = jnp.full_like(SH2O, 0.005)
    HLICE = jnp.full_like(SH2O, 3.335E5)
    GS = jnp.full_like(SH2O, 9.81)
    T0 = jnp.full_like(SH2O, 273.15)

    BX = jnp.where(BEXP > BLIM, BLIM, BEXP)
    if BLIM_OPTION == 1:
        BX = jnp.asarray(BLIMBX, dtype=SH2O.dtype)

    KCOUNT = jnp.zeros_like(SH2O)

    # Use SH2O if >= 0.02, otherwise use 0.02
    SWL = jnp.where(SH2O < 0.02, SMC - 0.02, SMC - SH2O)
    SWL = jnp.clip(SWL, 0.0)

    # We unroll the loop (as in the original JAX code)
    for _ in range(10):
        # This is the "active" mask:
        is_active = (KCOUNT == 0) & (TKELV <= (T0 - 1.E-3))

        # --- 1. Calculate DF (the log terms) ---
        # We must provide safe inputs (1.0) to log() if is_active is False
        # Argument 1: (PSIS * ...)
        denom_smc_safe = jnp.clip(SMC - SWL, 0.02, None)
        arg1_safe = (PSIS * GS / (HLICE + epsilon)) * ((1. + CK * SWL) ** 2.) * jnp.power(jnp.maximum(SMCMAX / (denom_smc_safe + epsilon), epsilon), BX)
        # If not active, use 1.0 (log(1.0) = 0.0)
        arg1 = jnp.where(is_active, jnp.clip(arg1_safe, epsilon, None), 1.0)

        # Argument 2: -(TKELV - T0) / TKELV
        # Use TKELV + epsilon to avoid division by zero if TKELV is near 0
        arg2_safe = -(TKELV - T0) / jnp.maximum(TKELV, epsilon)
        # If not active, use 1.0 (log(1.0) = 0.0)
        arg2 = jnp.where(is_active, jnp.clip(arg2_safe, epsilon, None), 1.0)

        # DF_new is now safe. If not active, DF_new = log(1.0) - log(1.0) = 0.0
        DF_new = jnp.log(arg1) - jnp.log(arg2)
        DF = jnp.where(is_active, DF_new, 0.0)  # Match PyTorch 'DF' default

        # --- 2. Calculate DENOM ---
        DENOM_safe = 2. * CK / (1. + CK * SWL + epsilon) + BX / (denom_smc_safe + epsilon)
        # If not active, use 1.0 (safe denominator)
        DENOM = jnp.where(KCOUNT == 0, jnp.clip(DENOM_safe, epsilon, None), 1.0)
        DENOM = jnp.where(is_active, DENOM, 1.0)  # Extra protection

        # --- 3. Calculate SWLK ---
        # If not active, SWLK = SWL - 0.0 / 1.0 = SWL (no change)
        SWLK = SWL - DF / DENOM

        # Clipping (always applied)
        SWLK = jnp.where(SWLK > (SMC - 0.02), SMC - 0.02, SWLK)
        SWLK = jnp.clip(SWLK, 0.0)

        DSWL = jnp.abs(SWLK - SWL)

        # --- 4. Update SWL and KCOUNT ---
        # Only update SWL if KCOUNT was 0 (as in PyTorch)
        SWL = jnp.where(KCOUNT == 0, SWLK, SWL)

        # Only update KCOUNT if KCOUNT was 0 (as in PyTorch)
        KCOUNT = jnp.where((KCOUNT == 0) & (DSWL <= ERROR), 1.0, KCOUNT)
    # --- End of loop ---

    FREE = SMC - SWL

    condition_fleur = (KCOUNT == 0)

    # --- 5. Calculate Flerchinger Fallback (FK) ---
    # We must provide safe inputs for the exponentiation
    # Base of the exponent: (-(TKELV - T0) / TKELV)
    base_safe_arg = -(TKELV - T0) / (TKELV + epsilon)

    # We only calculate FK if it's *both* unconverged AND frozen.
    is_fallback_active = condition_fleur & (TKELV <= (T0 - 1.E-3))

    # If not active, use 1.0 (safe base for exponent)
    base_arg = jnp.where(is_fallback_active, jnp.clip(base_safe_arg, epsilon, None), 1.0)

    exponent = -1 / (BX + epsilon)

    # If not active, FK_calc = (const * 1.0) ** exponent = safe value
    FK_calc = jnp.power(jnp.maximum(((HLICE / (GS * jnp.clip(-PSIS, epsilon, None) + epsilon)) * base_arg), epsilon), exponent) * SMCMAX
    FK_calc = jnp.maximum(FK_calc, 0.02)

    FREE_Fleur = jnp.minimum(FK_calc, SMC)

    # Apply fallback only if the "active" condition was met
    FREE = jnp.where(is_fallback_active, FREE_Fleur, FREE)

    # This is the (forward-pass) selection
    condition_T0 = TKELV > (T0 - 1.E-3)
    FREE = jnp.where(condition_T0, SMC, FREE)

    return FREE


def SNOWPACK(ESD, DTSEC, SNOWH, SNDENS, TSNOW, TSOIL):
    C1 = 0.01
    C2 = 21.0
    ESDC = ESD * 100.
    DTHR = DTSEC / 3600.
    TSNOWC = TSNOW - 273.15
    TSOILC = TSOIL - 273.15
    TAVGC = 0.5 * (TSNOWC + TSOILC)
    ESDCX = jnp.maximum(ESDC, 1.E-2)
    BFAC = DTHR * C1 * jnp.exp(0.08 * TAVGC - C2 * SNDENS)
    PEXP = jnp.array(0.0)
    for J in range(4, 0, -1):
        PEXP = (1.0 + PEXP) * BFAC * ESDCX / (J + 1.0)
    PEXP = PEXP + 1.
    DSX = SNDENS * PEXP
    DSX = jnp.clip(DSX, 0.05, 0.4)

    # liquid-water densification formula during snowmelt
    # Implementation detail.
    densification = DSX * (1. - 0.13 * DTHR / 24.) + 0.13 * DTHR / 24.

    SNDENS = jnp.where(TSNOWC >= 0,
                       jnp.clip(densification, None, 0.4),  # use the corrected formula
                       DSX)
    SNOWHC = ESDC / jnp.maximum(SNDENS, 1e-9)
    SNOWH = SNOWHC * 0.01
    return SNOWH, SNDENS


def SNKSRC(TAVG, SMC, SH2O, ZSOIL, SMCMAX, PSISAT, BEXP, DT, QTOT, BLIM_OPTION, CK_OPTION, BLIMBX):
    DH2O = 1.0000E3
    HLICE = 3.3350E5
    DZ = jnp.zeros(ZSOIL.shape)
    DZ = DZ.at[0].set(-ZSOIL[0])
    DZ = DZ.at[1:].set(ZSOIL[:-1] - ZSOIL[1:])
    FREE = FRH2O_tensor(TAVG, SMC, SH2O, SMCMAX, BEXP, PSISAT, BLIM_OPTION, CK_OPTION, BLIMBX)
    FREE = FREE - SH2O + SH2O
    #the following section adds numerical safeguards
    # before modification: 
    # XH2O = SH2O + QTOT * DT / (DH2O * HLICE * DZ)
    # after modification: 
    safe_DZ = DZ + 1e-9  # prevent DZ from being zero
    XH2O = SH2O + QTOT * DT / (DH2O * HLICE * safe_DZ + 1e-9)
    # before modification: 
    # condition_freeze = (XH2O < SH2O) & (XH2O < FREE)
    # after modification: 
    condition_freeze = (jax.lax.stop_gradient(XH2O) < jax.lax.stop_gradient(SH2O)) & \
                       (jax.lax.stop_gradient(XH2O) < jnp.maximum(jax.lax.stop_gradient(FREE), 0.0))
    XH2O = jnp.where(condition_freeze, jnp.minimum(FREE, SH2O), XH2O)
    # before modification: 
    # condition_thaw = (XH2O > SH2O) & (XH2O > FREE)
    # after modification: 
    condition_thaw = (jax.lax.stop_gradient(XH2O) > jax.lax.stop_gradient(SH2O)) & \
                     (jax.lax.stop_gradient(XH2O) > jnp.maximum(jax.lax.stop_gradient(FREE), 0.0))
    XH2O = jnp.where(condition_thaw, jnp.maximum(FREE, SH2O), XH2O)
    XH2O = jnp.clip(XH2O, 0.0, SMC)
    # before modification: 
    # TSNSR = -DH2O * HLICE * DZ * (XH2O - SH2O) / DT
    # after modification: 
    safe_DT = DT + 1e-9  # prevent DT from being zero
    TSNSR = -DH2O * HLICE * DZ * (XH2O - SH2O) / safe_DT
    #end end numerical safeguards
    return TSNSR, XH2O


def PENMAN(SFCTMP, SFCPRS, CH, T2V, TH2, PRCP, FDOWN, SSOIL, Q2, Q2SAT, SNOWNG, FRZGRA, DQSDT2,
           EMISSI_IN, SNCOVR):
    CP = 1004.6
    CPH2O = 4.218E+3
    CPICE = 2.106E+3
    ELCP = 2.4888E+3
    LSUBF = 3.335E+5
    LSUBC = 2.501000E+6
    SIGMA = 5.67E-8
    LSUBS = 2.83E+6
    RD = 287.04
    EMISSI = EMISSI_IN
    ELCP1 = (1.0 - SNCOVR) * ELCP + SNCOVR * ELCP * LSUBS / LSUBC
    LVS = (1.0 - SNCOVR) * LSUBC + SNCOVR * LSUBS
    FLX2 = jnp.array(0.0)
    DELTA = ELCP1 * DQSDT2
    T24 = SFCTMP * SFCTMP * SFCTMP * SFCTMP
    RR = EMISSI * T24 * 6.48E-8 / (SFCPRS * CH + 1e-9) + 1.0
    RHO = SFCPRS / (RD * T2V + 1e-9)
    RCH = RHO * CP * CH
    con = (~SNOWNG) & (PRCP > 0.0)
    RR = jnp.where(con, RR + CPH2O * PRCP / (RCH + 1e-9), RR + CPICE * PRCP / (RCH + 1e-9))
    FNET = FDOWN - EMISSI * SIGMA * T24 - SSOIL
    FLX2 = jnp.where(FRZGRA, -LSUBF * PRCP, FLX2)
    FNET = jnp.where(FRZGRA, FNET - FLX2, FNET)
    RAD = FNET / (RCH + 1e-9) + TH2 - SFCTMP
    A = ELCP1 * (Q2SAT - Q2)
    EPSCA = (A * RR + RAD * DELTA) / (DELTA + RR + 1e-9)
    ETP = EPSCA * RCH / (LVS + 1e-9)
    return EPSCA, ETP, RCH, RR, FLX2, T24

#numerical safeguards were added to this function
def CANRES(soil_params_dict: dict, gen_params_dict: dict,  # <-- UPDATED: usedictionary
           SOLAR, CH, SFCTMP, Q2, SFCPRS, SMC, ZSOIL, STYPE, RSMIN, Q2SAT, DQSDT2,
           TOPT, RSMAX, RGL, HS, XLAI, EMISSI, NROOT: int, SOC, SOC_OPTION_KS, RTDIS):
    """Technical documentation."""
    CP = 1004.6
    RD = 287.04
    SIGMA = 5.67E-8
    SLV = 2.501000E6

    FF = 0.55 * 2.0 * SOLAR / jnp.maximum(RGL * XLAI, 1e-9)
    RCS = (FF + RSMIN / jnp.maximum(RSMAX, 1e-9)) / (1.0 + FF + 1e-9)
    RCS = jnp.maximum(RCS, 0.0001)

    RCT = 1.0 - 0.0016 * ((TOPT - SFCTMP) ** 2.0)
    RCT = jnp.maximum(RCT, 0.0001)

    RCQ = 1.0 / (1.0 + HS * (Q2SAT - Q2))
    RCQ = jnp.maximum(RCQ, 0.01)

    PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ, _ = REDSTP(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        STYPE[:NROOT], jnp.arange(NROOT), SOC[:NROOT], SOC_OPTION_KS, required_grad=True)

    GX = jnp.clip((SMC[:NROOT] - SMCWLT) / jnp.maximum(SMCREF - SMCWLT, 1e-9), 0.0, 1.0)

    PART = RTDIS[:NROOT] * GX
    RCSOIL = jnp.sum(PART)
    RCSOIL = jnp.maximum(RCSOIL, 0.0001)

    RC = RSMIN / (XLAI * RCS * RCT * RCQ * RCSOIL + 1e-9)

    RR = (4. * EMISSI * SIGMA * RD / CP) * (SFCTMP ** 4.) / (SFCPRS * CH + 1e-9) + 1.0
    DELTA = (SLV / CP) * DQSDT2

    PC = (RR + DELTA) / (RR * (1. + RC * CH) + DELTA + 1e-9)

    return PC, RC, RCS, RCT, RCQ, RCSOIL

# Revised by Zheng and Zhang
def SOCH(gen_params_dict: dict,  # <-- NEW
         SOC_OPTION_KS, SOC,
         PSISAT_base, BEXP_base, DKSAT_base, DWSAT_base, SMCMAX_base, SMCWLT_base, SMCREF_base, SMCDRY_base,
         KDT_base, FRZX_base, REFKDT, F1_base):
    """Technical documentation."""
    SMCMAXs = 0.83
    PSISATs = 0.0101
    BEXPs = 12.0

    FTSOC = SOC * 2700 * (1 - SMCMAX_base) / (SOC * 2700 * (1 - SMCMAX_base) + (1 - SOC) * 130 + 1e-9)

    SMCMAX_new = (1 - FTSOC) * SMCMAX_base + FTSOC * SMCMAXs
    PSISAT_new = (1 - FTSOC) * PSISAT_base + FTSOC * PSISATs
    BEXP_new = (1 - FTSOC) * BEXP_base + FTSOC * BEXPs
    F1_new = jnp.log10(jnp.maximum(PSISAT_new, 1e-9)) + BEXP_new * jnp.log10(jnp.maximum(SMCMAX_new, 1e-9)) + 2.0

#Revised by Zheng and Zhang
    DKSAT_base_s = jnp.full_like(DKSAT_base, 1.0E-4)
    DKSAT_linear = (1 - FTSOC) * DKSAT_base + FTSOC * DKSAT_base_s

    CCCs = 1930.0 / 1000.0 / 3600.0
    SMC33 = SMCMAX_new * jnp.power(jnp.maximum(3.366 / jnp.maximum(PSISAT_new, 1e-9), 1e-9), (-1.0 / (BEXP_new + 1e-9)))
    DKSAT_kcarman = CCCs * jnp.power(jnp.maximum(SMCMAX_new - SMC33, 1e-9), (3.0 - 1.0 / (BEXP_new + 1e-9)))

    # Implementation detail.
    # Jacobian and gradient handling.
    # Model-parameter handling.
    # Jacobian and gradient handling.
    # Jacobian and gradient handling.
    # Implementation detail.
    # Jacobian and gradient handling.
    # DKSAT_ste = DKSAT_kcarman + (DKSAT_base - jax.lax.stop_gradient(DKSAT_base))
    # Implementation detail.
    # Jacobian and gradient handling.
    DKSAT_ste = DKSAT_kcarman + (DKSAT_base - jax.lax.stop_gradient(DKSAT_base))
    
    DKSAT_final = jnp.where(SOC_OPTION_KS == 1, DKSAT_ste, DKSAT_linear)
#end SOC_OPTION_KS

    DWSAT_new = BEXP_new * DKSAT_final * PSISAT_new / jnp.maximum(SMCMAX_new, 1e-9)

    # read from the passed dictionary
    SMLOW = gen_params_dict['SMLOW_DATA']
    SMHIGH = gen_params_dict['SMHIGH_DATA']

    SMCREF_new = SMCMAX_new * jnp.power(jnp.maximum(3.366 / jnp.maximum(PSISAT_new, 1e-9), 1e-9), (-1.0 / (BEXP_new + 1e-9)))

    WLTSMC1 = SMCMAX_new * jnp.power(jnp.maximum(200.0 / jnp.maximum(PSISAT_new, 1e-9), 1e-9), (-1.0 / (BEXP_new + 1e-9)))
    SMCWLT_new = WLTSMC1 - SMLOW * WLTSMC1
    SMCDRY_new = SMCWLT_new

    # read from the passed dictionary
    REFDK = gen_params_dict['REFDK_DATA']
    FRZK = gen_params_dict['FRZK_DATA']

    KDT_new = REFKDT * DKSAT_final / jnp.maximum(REFDK, 1e-9)
    FRZX_new = FRZK * (SMCMAX_new / jnp.maximum(SMCREF_new, 1e-9)) * (0.412 / 0.468)

    return (PSISAT_new, BEXP_new, DKSAT_final, DWSAT_new, SMCMAX_new, SMCWLT_new, SMCREF_new, SMCDRY_new,
            KDT_new, FRZX_new, F1_new)



def NOPAC(soil_params_dict: dict, gen_params_dict: dict,  # <-- UPDATED: usedictionary
          STYPE, ETP, PRCP, SMC, SMCMAX, SMCDRY, CMC, CMCMAX, DT, SHDFAC, SBETA,
          SFCTMP, T24, TH2, FDOWN, EMISSI, STC, EPSCA, PC, RCH, RR, CFACTR, SH2O, SLOPE, KDT, FRZFACT, ZSOIL, TBOT,
          ZBOT, NROOT: int, RTDIS, QUARTZ, FXEXP, CSOIL, DF_OPTION, INF_OPTION, SOC_OPTION2_THERMAL, SOC,
          RIC_OPTION,
          PSISAT, BEXP, SOC_OPTION_KS, BLIM_OPTION, CK_OPTION, SBETA_OPTION, RLMO, ktime: int):
    """Technical documentation."""
    CPH2O = 4.218E+3
    SIGMA = 5.67E-8

    PRCP1 = PRCP * 0.001
    ETP1 = ETP * 0.001
    # --- (UPDATED) ---
    ETA1, EDIR1, ETT1, EC1, ET1 = EVAPO(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        SMC, CMC, ETP1, DT, SH2O, SMCMAX, PC, STYPE, SHDFAC, CMCMAX,
        SMCDRY, CFACTR, NROOT, RTDIS, FXEXP, SOC, SOC_OPTION_KS)

    DEW = jnp.where(ETP > 0, jnp.array(0.0), -ETP1)
    PRCP1 = jnp.where(ETP > 0, PRCP * 0.001, PRCP1 + DEW)

    # --- (UPDATED) ---
    SH2O, SMC, RUNOFF1, RUNOFF2, RUNOFF3, CMC, DRIP = SMFLX(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        STYPE, SMC, CMC, DT, PRCP1, ZSOIL, SH2O, SLOPE,
        KDT, FRZFACT, SHDFAC, CMCMAX, EDIR1, EC1, ET1, INF_OPTION,
        NROOT, RIC_OPTION, SOC, SOC_OPTION_KS)
    ETA = jnp.where(ETP > 0, ETA1 * 1000, ETP)
    BETA = jnp.where(ETP > 0, ETA / jnp.maximum(ETP, 1e-9), jnp.where(ETP < 0.0, 1, 0))

# Revised by Zheng and Zhang
    FSSOC = SOC * 2700 / (SOC * 2700 + (1 - SOC) * 1300 + 1e-9)
    if SOC_OPTION2_THERMAL == 1:
        DF1_raw = TDFCND_SOC(SMC[0], QUARTZ, SMCMAX, SH2O[0], FSSOC[0])
    else:
        DF1_raw = TDFCND(SMC[0], QUARTZ, SMCMAX, SH2O[0], STYPE[0])
# end SOC_OPTION2_THERMAL

    DF1_before_SBETA = DF1_raw

# Revised by Zheng and Zhang
    factor_default = jnp.exp(SBETA * SHDFAC)
    factor_unstable = jnp.exp(-1.0 * SHDFAC)
    is_unstable_day = (SBETA_OPTION == 1) & (RLMO <= 0)
    final_factor = jnp.where(is_unstable_day, factor_unstable, factor_default)
    DF1_with_SBETA = DF1_raw * final_factor
# end SBETA_OPTION

# Revised by Zheng and Zhang
    DF1_for_SHFLX_SSOIL = DF1_with_SBETA
    DF1_for_DF2 = jnp.where(DF_OPTION == 0, DF1_with_SBETA, DF1_before_SBETA)
# end DF_OPTION

    YYNUM = FDOWN - EMISSI * SIGMA * T24
    YY = SFCTMP + (YYNUM / jnp.maximum(RCH, 1e-9) + TH2 - SFCTMP - BETA * EPSCA) / jnp.maximum(RR, 1e-9)
    # (*** fix ***) protect the absolute value of the denominator, rather than the denominator itself
    denom_zz1 = -0.5 * ZSOIL[0] * RCH * RR
    ZZ1 = DF1_for_SHFLX_SSOIL / jnp.where(jnp.abs(denom_zz1) > 1e-9, denom_zz1, jnp.sign(denom_zz1) * 1e-9) + 1.0

    # --- (UPDATED) ---
    STC, T1, SSOIL, SH2O = SHFLX(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        STYPE, STC, SMC, DT, YY, ZZ1, ZSOIL, TBOT, ZBOT, SH2O, DF1_for_SHFLX_SSOIL,
        DF1_for_DF2, CSOIL, DF_OPTION, SOC_OPTION2_THERMAL, SOC, SOC_OPTION_KS, BLIM_OPTION,
        CK_OPTION, ktime)
    FLX1 = CPH2O * PRCP * (T1 - SFCTMP)
    FLX3 = jnp.array(0.0)
    return STC, SMC, SH2O, DEW, DRIP, EC1, EDIR1, ETA, ET1, ETT1, RUNOFF1, RUNOFF2, RUNOFF3, SSOIL, CMC, BETA, T1, FLX1, FLX3


def SNOPAC(soil_params_dict: dict, gen_params_dict: dict,  # <-- UPDATED: usedictionary
           STYPE, ETP, PRCP, PRCPF, SNOWNG, SMC, SMCMAX, SMCDRY, CMC, CMCMAX, DT, DF1,
           T1, SFCTMP, T24, TH2, FDOWN, STC, PC, RCH, RR, CFACTR, SNCOVR,
           ESD: jnp.ndarray, SNDENS, SNOWH, SH2O, SLOPE, KDT, FRZFACT, ZSOIL, TBOT, ZBOT, SHDFAC, NROOT: int, RTDIS,
           FXEXP,
           CSOIL, FLX2, EMISSI, RIBB, DF_OPTION, INF_OPTION, SOC_OPTION2_THERMAL, SOC, RIC_OPTION, PSISAT,
           BEXP, SOC_OPTION_KS, BLIM_OPTION, CK_OPTION, ktime, QUARTZ):

    """Technical documentation."""

    NSOIL = SH2O.shape[0]
    SIGMA = 5.67E-8
    LSUBC = 2.501000E+6
    CPICE = 2.106E+3
    CPH2O = 4.218E+3
    LSUBF = 3.335E+5
    LSUBS = 2.83E+6
    TFREEZ = 273.15
    ESDMIN = 1.E-6
    SNOEXP = 2.0

    EDIR1 = jnp.array(0.0)
    DEW = jnp.array(0.0)
    ETT1 = jnp.array(0.0)
    ET1 = jnp.zeros(NSOIL)
    EC1 = jnp.array(0.0)
    PRCP1 = PRCPF * 0.001
    BETA = jnp.array(1.0)
    ETNS1 = jnp.array(0.0)
    ESNOW = jnp.array(0.0)

    is_negative_etp = ETP <= 0
    adjust_etp = (RIBB >= 0.1) & (FDOWN > 150.0)
    ETP_adjusted = jnp.where(adjust_etp,
                             (jnp.minimum(ETP * (1.0 - RIBB), jnp.array(0.0)) * SNCOVR / 0.980 + ETP * (
                                     0.980 - SNCOVR)) / 0.980,
                             ETP)
    ETP = jnp.where(is_negative_etp, ETP_adjusted, ETP)

    BETA_neg = jnp.where(ETP == 0, jnp.array(0.0), BETA)
    ETP1_neg = ETP * 0.001
    DEW_neg = -ETP1_neg
    ESNOW2_neg = ETP1_neg * DT
    ETANRG_neg = ETP * ((1. - SNCOVR) * LSUBC + SNCOVR * LSUBS)
    ETNS1_neg = jnp.array(0.0)
    EDIR1_neg = jnp.array(0.0)
    ET1_neg = jnp.zeros(NSOIL)
    EC1_neg = jnp.array(0.0)
    ETT1_neg = jnp.array(0.0)
    ESNOW_neg = jnp.array(0.0)

    ETP1_pos = ETP * 0.001

    has_bare_ground = SNCOVR < 1.0
    # --- (UPDATED) ---
    ETNS1_calc, EDIR1_calc, ETT1_calc, EC1_calc, ET1_calc = EVAPO(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        SMC, CMC, ETP1_pos, DT, SH2O, SMCMAX, PC, STYPE, SHDFAC,
        CMCMAX, SMCDRY, CFACTR, NROOT, RTDIS, FXEXP, SOC, SOC_OPTION_KS)

    EDIR1_pos = jnp.where(has_bare_ground, EDIR1_calc * (1. - SNCOVR), jnp.array(0.0))
    ET1_pos = jnp.where(has_bare_ground, ET1_calc * (1. - SNCOVR), jnp.zeros_like(ET1_calc))
    EC1_pos = jnp.where(has_bare_ground, EC1_calc * (1. - SNCOVR), jnp.array(0.0))
    ETT1_pos = jnp.where(has_bare_ground, ETT1_calc * (1. - SNCOVR), jnp.array(0.0))
    ETNS1_pos = jnp.where(has_bare_ground, ETNS1_calc * (1. - SNCOVR), jnp.array(0.0))

    ESNOW_pos = ETP * SNCOVR
    ESNOW1_pos = ESNOW_pos * 0.001
    ESNOW2_pos = ESNOW1_pos * DT
    ETNS_W_pos = ETNS1_pos * 1000. * LSUBC
    ETANRG_pos = ESNOW_pos * LSUBS + ETNS_W_pos
    DEW_pos = jnp.array(0.0)
    BETA = jnp.where(is_negative_etp, BETA_neg, BETA)
    DEW = jnp.where(is_negative_etp, DEW_neg, DEW_pos)
    ESNOW2 = jnp.where(is_negative_etp, ESNOW2_neg, ESNOW2_pos)
    ETANRG = jnp.where(is_negative_etp, ETANRG_neg, ETANRG_pos)
    ETNS1 = jnp.where(is_negative_etp, ETNS1_neg, ETNS1_pos)
    EDIR1 = jnp.where(is_negative_etp, EDIR1_neg, EDIR1_pos)
    ET1 = jnp.where(is_negative_etp, ET1_neg, ET1_pos)
    EC1 = jnp.where(is_negative_etp, EC1_neg, EC1_pos)
    ETT1 = jnp.where(is_negative_etp, ETT1_neg, ETT1_pos)
    ESNOW = jnp.where(is_negative_etp, ESNOW_neg, ESNOW_pos)

    FLX1 = jnp.where(SNOWNG, CPICE * PRCP * (T1 - SFCTMP),
                     jnp.where(PRCP > 0.0, CPH2O * PRCP * (T1 - SFCTMP), 0))

    DSOIL = -(0.5 * ZSOIL[0])
    DTOT = SNOWH + DSOIL + 1e-9
    DENOM = 1.0 + DF1 / jnp.maximum(DTOT * RR * RCH, 1e-9)
    T12A = ((FDOWN - FLX1 - FLX2 - EMISSI * SIGMA * T24) / jnp.maximum(RCH, 1e-9) + TH2 - SFCTMP - ETANRG / jnp.maximum(RCH, 1e-9)) / jnp.maximum(RR, 1e-9)
    T12B = DF1 * STC[0] / jnp.maximum(DTOT * RR * RCH, 1e-9)
    T12 = (SFCTMP + T12A + T12B) / jnp.maximum(DENOM, 1e-9)

    is_frozen = T12 <= TFREEZ

    T1_frozen = T12
    ssoil_frozen = DF1 * (T1_frozen - STC[0]) / jnp.maximum(DTOT, 1e-9)
    ESD_frozen = jnp.maximum(jnp.array(0.0), ESD - ESNOW2)
    FLX3_frozen = jnp.array(0.0)
    EX_frozen = jnp.array(0.0)
    SNOMLT_frozen = jnp.array(0.0)
    PRCP1_frozen = PRCP1
    BETA_frozen = BETA

    T1_thaw = TFREEZ * jnp.power(jnp.maximum(SNCOVR, 1e-9), SNOEXP) + T12 * (1.0 - jnp.power(jnp.maximum(SNCOVR, 1e-9), SNOEXP))
    BETA_thaw = jnp.array(1.0)
    ssoil_thaw = DF1 * (T1_thaw - STC[0]) / jnp.maximum(DTOT, 1e-9)

    is_snow_depleted = (ESD - ESNOW2) <= ESDMIN

    ESD_depleted = jnp.array(0.0)
    EX_depleted = jnp.array(0.0)
    SNOMLT_depleted = jnp.array(0.0)
    FLX3_depleted = jnp.array(0.0)

    ESD_not_depleted = ESD - ESNOW2
    SEH = RCH * (T1_thaw - TH2)
    T14 = jnp.power(T1_thaw, 4)
    FLX3_calc = jnp.maximum(FDOWN - FLX1 - FLX2 - EMISSI * SIGMA * T14 - ssoil_thaw - SEH - ETANRG, 0.0)
    # (*** v1.1 numericalsafeguard ***) use safe_divide preventdivision by zero
    EX_calc = safe_divide(FLX3_calc * 0.001, LSUBF, epsilon=1e-9, default_value=0.0)
    SNOMLT_calc = EX_calc * DT

    is_enough_snow = (ESD_not_depleted - SNOMLT_calc) >= ESDMIN
    ESD_enough = ESD_not_depleted - SNOMLT_calc
    EX_enough = EX_calc
    FLX3_enough = FLX3_calc
    SNOMLT_enough = SNOMLT_calc

    # (*** v1.1 numericalsafeguard ***) use safe_divide preventdivision by zero
    EX_not_enough = safe_divide(ESD_not_depleted, DT, epsilon=1e-9, default_value=0.0)
    FLX3_not_enough = EX_not_enough * 1000.0 * LSUBF
    SNOMLT_not_enough = ESD_not_depleted
    ESD_not_enough = jnp.array(0.0)

    ESD_not_depleted_final = jnp.where(is_enough_snow, ESD_enough, ESD_not_enough)
    EX_not_depleted_final = jnp.where(is_enough_snow, EX_enough, EX_not_enough)
    FLX3_not_depleted_final = jnp.where(is_enough_snow, FLX3_enough, FLX3_not_enough)
    SNOMLT_not_depleted_final = jnp.where(is_enough_snow, SNOMLT_enough, SNOMLT_not_enough)

    ESD_thaw = jnp.where(is_snow_depleted, ESD_depleted, ESD_not_depleted_final)
    EX_thaw = jnp.where(is_snow_depleted, EX_depleted, EX_not_depleted_final)
    FLX3_thaw = jnp.where(is_snow_depleted, FLX3_depleted, FLX3_not_depleted_final)
    SNOMLT_thaw = jnp.where(is_snow_depleted, SNOMLT_depleted, SNOMLT_not_depleted_final)
    PRCP1_thaw = PRCP1 + EX_thaw

    T1 = jnp.where(is_frozen, T1_frozen, T1_thaw)
    ssoil_out = jnp.where(is_frozen, ssoil_frozen, ssoil_thaw)
    ESD = jnp.where(is_frozen, ESD_frozen, ESD_thaw)
    FLX3 = jnp.where(is_frozen, FLX3_frozen, FLX3_thaw)
    EX = jnp.where(is_frozen, EX_frozen, EX_thaw)
    SNOMLT = jnp.where(is_frozen, SNOMLT_frozen, SNOMLT_thaw)
    PRCP1 = jnp.where(is_frozen, PRCP1_frozen, PRCP1_thaw)
    BETA = jnp.where(is_frozen, BETA_frozen, BETA_thaw)

# Implementation detail.
# Revised by Zheng and Zhang
    FSSOC = SOC * 2700 / (SOC * 2700 + (1 - SOC) * 1300 + 1e-9)
    DF1_before_SBETA = jax.lax.cond(SOC_OPTION2_THERMAL == 1,
                                    lambda _: TDFCND_SOC(SMC[0], QUARTZ, SMCMAX, SH2O[0], FSSOC[0]),
                                    lambda _: TDFCND(SMC[0], QUARTZ, SMCMAX, SH2O[0], STYPE[0]),
                                    None)
    DF1_for_DF2 = jnp.where(DF_OPTION == 0, DF1, DF1_before_SBETA)
# end SOC_OPTION2_THERMAL and DF_OPTION

    # --- (UPDATED) ---
    SH2O, SMC, RUNOFF1, RUNOFF2, RUNOFF3, CMC, DRIP = SMFLX(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        STYPE, SMC, CMC, DT, PRCP1, ZSOIL, SH2O, SLOPE,
        KDT, FRZFACT, SHDFAC, CMCMAX, EDIR1, EC1, ET1, INF_OPTION,
        NROOT, RIC_OPTION, SOC, SOC_OPTION_KS)


    ZZ1 = jnp.array(1.0)
    YY = STC[0] - 0.5 * ssoil_out * ZSOIL[0] * ZZ1 / jnp.maximum(DF1, 1e-9)
    # --- (UPDATED) ---
    STC, T1_dummy, ssoil_dummy, SH2O = SHFLX(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        STYPE, STC, SMC, DT, YY, ZZ1, ZSOIL, TBOT, ZBOT, SH2O, DF1, DF1_for_DF2,
        CSOIL, DF_OPTION, SOC_OPTION2_THERMAL,
        SOC, SOC_OPTION_KS, BLIM_OPTION, CK_OPTION, ktime)

    def has_snow_branch(_):
        SNOWH_out, SNDENS_out = SNOWPACK(ESD, DT, SNOWH, SNDENS, T1, YY)
        return ESD, SNOWH_out, SNDENS_out, SNCOVR

    def no_snow_branch(_):
        return jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)

    ESD, SNOWH, SNDENS, SNCOVR = jax.lax.cond(ESD > 0, has_snow_branch, no_snow_branch, None)

    ETNS = ETNS1 * 1000.0

    return (
        STC, SMC, SH2O, DEW, DRIP, ETP, EC1, EDIR1, ET1, ETT1, ETNS, ESNOW, FLX1, FLX3, RUNOFF1, RUNOFF2, RUNOFF3,
        ssoil_out,
        SNOMLT, CMC, BETA, ESD, SNOWH, SNCOVR, SNDENS, T1, ETANRG)


def SMFLX(soil_params_dict: dict, gen_params_dict: dict,  # <-- UPDATED: usedictionary
          STYPE, SMC, CMC, DT, PRCP1, ZSOIL, SH2O, SLOPE, KDT, FRZFACT,
          SHDFAC, CMCMAX, EDIR, EC, ET, INF_OPTION, NROOT, RIC_OPTION, SOC, SOC_OPTION_KS):
    """Technical documentation."""
    NSOIL = SMC.shape[0]
    RHSCT = SHDFAC * PRCP1 - EC
    TRHSCT = DT * RHSCT
    EXCESS = CMC + TRHSCT
    DRIP = jnp.where(EXCESS > CMCMAX, EXCESS - CMCMAX, jnp.array(0.0))

    # (*** v1.1 numericalsafeguard ***) use safe_divide preventdivision by zero
    PCPDRP = (1. - SHDFAC) * PRCP1 + safe_divide(DRIP, DT, epsilon=1e-9, default_value=0.0)
    SICE = (SMC - SH2O)
    # (*** v1.1 numericalsafeguard ***) ice content must not be negative
    SICE = jnp.maximum(SICE, 0.0)
    # --- (UPDATED) ---
    PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ, SMCMAX_BASE = REDSTP(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        STYPE,
        jnp.arange(NSOIL), SOC,
        SOC_OPTION_KS,
        required_grad=True)

#Revise by Zheng and Zhang
    # (*** v1.1 numericalsafeguard ***) use safe_divide preventdivision by zero
    FAC2 = jnp.max(safe_divide(SH2O, SMCMAX_BASE, epsilon=1e-9, default_value=0.0))
    SMCMAX_BASE = SMCMAX_BASE[0]
    SMCMAX_cfg = SMCMAX[0]  # State-variable handling.
    FLIMIT = FAC2MIT(SMCMAX_BASE)
#end SOC_OPTION2_THERMAL

    def high_runoff_branch(_):
        # --- (UPDATED) ---
        RUNOFF1_fg, RUNOFF2_fg, RUNOFF3_fg, SH2OFG, SMCFG, DUMMY = SRT(
            soil_params_dict, gen_params_dict,  # <-- UPDATED
            STYPE, EDIR, ET, SH2O, SH2O, PCPDRP, ZSOIL, DT, SLOPE,
            KDT, FRZFACT, SICE, CMC, CMCMAX, RHSCT, INF_OPTION, NROOT, RIC_OPTION, SOC, SOC_OPTION_KS)
        SH2OA = 0.5 * (SH2O + SH2OFG)
        # --- (UPDATED) ---
        RUNOFF1_final, RUNOFF2_final, RUNOFF3_final, SH2OOUT_final, SMCOUT_final, CMC_final = SRT(
            soil_params_dict, gen_params_dict,  # <-- UPDATED
            STYPE, EDIR, ET, SH2O, SH2OA, PCPDRP, ZSOIL, DT, SLOPE,
            KDT, FRZFACT, SICE, CMC, CMCMAX, RHSCT, INF_OPTION, NROOT, RIC_OPTION, SOC, SOC_OPTION_KS)
        return RUNOFF1_final, RUNOFF2_final, RUNOFF3_final, SH2OOUT_final, SMCOUT_final, CMC_final

    def low_runoff_branch(_):
        # --- (UPDATED) ---
        RUNOFF1_final, RUNOFF2_final, RUNOFF3_final, SH2OOUT_final, SMCOUT_final, CMC_final = SRT(
            soil_params_dict, gen_params_dict,  # <-- UPDATED
            STYPE, EDIR, ET, SH2O, SH2O, PCPDRP, ZSOIL, DT, SLOPE,
            KDT, FRZFACT, SICE, CMC, CMCMAX, RHSCT, INF_OPTION, NROOT, RIC_OPTION, SOC, SOC_OPTION_KS)
        return RUNOFF1_final, RUNOFF2_final, RUNOFF3_final, SH2OOUT_final, SMCOUT_final, CMC_final

    needs_iteration = ((PCPDRP * DT) > (0.0001 * 1000.0 * (- ZSOIL[0]) * SMCMAX_cfg)) | (FAC2 > FLIMIT)

    RUNOFF1, RUNOFF2, RUNOFF3, SH2OOUT, SMCOUT, CMC = jax.lax.cond(
        needs_iteration, high_runoff_branch, low_runoff_branch, None)

    return SH2OOUT, SMCOUT, RUNOFF1, RUNOFF2, RUNOFF3, CMC, DRIP

# Revise by Zhang
def solve_tridiagonal_jax(a, b, c, d):
    # Implementation detail.
    CO_Matrix = jnp.diag(a[1:], -1) + jnp.diag(b) + jnp.diag(c[:-1], 1)
    from jax.scipy import linalg
    x = linalg.solve(CO_Matrix, d, assume_a='gen')
    return x
# end use JAX linalg.solve to solve the tridiagonal matrix instead of the Thomas algorithm in NoahA

def SHFLX(soil_params_dict: dict, gen_params_dict: dict,  # <-- UPDATED: usedictionary
          STYPE, STC, SMC, DT, YY, ZZ1, ZSOIL, TBOT, ZBOT, SH2O, DF1_SSOIL, DF1_DF2, CSOIL, DF_OPTION,
          SOC_OPTION2_THERMAL, SOC, SOC_OPTION_KS, BLIM_OPTION, CK_OPTION, ktime=None):
    """Technical documentation."""
    NSOIL = SH2O.shape[0]
    T0 = 273.15
    CAIR = 1004.0
    CICE = 2.106e6
    CH2O = 4.2e6
    CSOIL_LOC = CSOIL

    # --- (UPDATED) ---
    PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ, _ = REDSTP(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        STYPE, jnp.arange(NSOIL), SOC,
        SOC_OPTION_KS,
        required_grad=True)

#Revised by Zheng and Zhang
    FSSOC = SOC * 2700 / (SOC * 2700 + (1 - SOC) * 1300 + 1e-9)
    if SOC_OPTION2_THERMAL == 1:
        DF1N = TDFCND_SOC(SMC, QUARTZ, SMCMAX, SH2O, FSSOC)
    else:
        DF1N = TDFCND(SMC, QUARTZ, SMCMAX, SH2O, STYPE)
    BLIMBX = soil_params_dict['BLIMBX'][STYPE - 1]
    SICE = SMC - SH2O
    DF2 = jnp.concatenate([DF1_DF2.reshape(1), DF1N[1:]])
# end SOC_OPTION2_THERMAL and DF_OPTION

    DENOM2 = jnp.zeros(NSOIL)
    DDZ2 = jnp.zeros(NSOIL - 1)
    AI = jnp.zeros(NSOIL)
    CI = jnp.zeros(NSOIL)
    BI = jnp.zeros(NSOIL)
    RHSTS = jnp.zeros(NSOIL)
    delta_STC = jnp.zeros(NSOIL)
    delta_STC = delta_STC.at[:-1].set(STC[:-1] - STC[1:])
    delta_STC = delta_STC.at[-1].set(STC[-1] - TBOT)

# Revised by Zheng and Zhang
    HCPCT_no_soc = SH2O * CH2O + (1.0 - SMCMAX) * CSOIL_LOC + (SMCMAX - SMC) * CAIR + SICE * CICE
    CSOC = 2.5E6
    FSSOC = SOC * 2700 / (SOC * 2700 + (1 - SOC) * 1300 + 1e-9)
    HCPCT_soc = SH2O * CH2O + \
                (1.0 - SMCMAX) * (1.0 - FSSOC) * CSOIL_LOC + \
                (1.0 - SMCMAX) * FSSOC * CSOC + \
                (SMCMAX - SMC) * CAIR + \
                (SMC - SH2O) * CICE
    HCPCT = jnp.where(SOC_OPTION2_THERMAL == 1, HCPCT_soc, HCPCT_no_soc)
    
    # *** fixnumericalunstable: set a lower bound for heat capacity ***
    # Implementation detail.
    # Jacobian and gradient handling.
    # set a reasonable lower bound to prevent numerical instability, while preserving physical plausibility
    # Implementation detail.
    # Jacobian and gradient handling.
    # Jacobian and gradient handling.
    MIN_HCPCT = 1.5e6  # Implementation detail.
    HCPCT = jnp.maximum(HCPCT, MIN_HCPCT)
# end SOC_OPTION2_THERMAL

    DENOM2 = DENOM2.at[0].set(-ZSOIL[0])
    DENOM2 = DENOM2.at[1:].set(ZSOIL[:-1] - ZSOIL[1:])
    DDZ2 = DDZ2.at[0].set((0 - ZSOIL[1]) * 0.5)
    DDZ2 = DDZ2.at[1:].set((ZSOIL[:-2] - ZSOIL[2:]) * 0.5)
    AI = AI.at[1:].set(-DF2[:-1] / jnp.maximum(DENOM2[1:] * DDZ2 * HCPCT[1:], 1e-9))
# Revised by Zheng and Zhang
    CI = CI.at[:-1].set(-DF2[:-1] / jnp.maximum(DENOM2[:-1] * DDZ2 * HCPCT[:-1], 1e-9))
# end DF_OPTION
    # Numerical-stability safeguard.
    denom_bi = 0.5 * ZSOIL[0] * ZSOIL[0] * HCPCT[0] * ZZ1
    BI = BI.at[0].set(-CI[0] + DF1_SSOIL / jnp.where(jnp.abs(denom_bi) > 1e-9, denom_bi, jnp.sign(denom_bi) * 1e-9))

    BI = BI.at[1:].set(-(AI[1:] + CI[1:]))
    # Numerical-stability safeguard.
    denom_ssoil = 0.5 * ZSOIL[0] * ZZ1
    SSOIL = DF1_SSOIL * (STC[0] - YY) / jnp.where(jnp.abs(denom_ssoil) > 1e-9, denom_ssoil, jnp.sign(denom_ssoil) * 1e-9)
#Revised by Zheng and Zhang
    RHSTS = RHSTS.at[0].set((SSOIL - delta_STC[0] * DF2[0] / jnp.maximum(DDZ2[0], 1e-9)) / jnp.maximum(DENOM2[0] * HCPCT[0], 1e-9))
#end DF_OPTION
    RHSTS = RHSTS.at[1:-1].set(-AI[1:-1] * delta_STC[:-2] + CI[1:-1] * delta_STC[1:-1])
    RHSTS = RHSTS.at[-1].set(-AI[-1] * delta_STC[-2] - (DF2[-1] * delta_STC[-1]) / jnp.maximum(
            DENOM2[-1] * HCPCT[-1] * (.5 * (ZSOIL[-2] + ZSOIL[-1]) - ZBOT), 1e-9))
    QTOT = RHSTS * DENOM2 * HCPCT

    TU_TBND = STC
    ZB_bottom = 2.0 * ZBOT - ZSOIL[-1]
    TB_TBND = jnp.concatenate([STC[1:], jnp.array([TBOT])])
    TDN = TBND(TU_TBND, TB_TBND, ZSOIL, ZBOT)

    TSURF = (YY + (ZZ1 - 1) * STC[0]) / jnp.maximum(ZZ1, 1e-9)
    TUP_Vector = jnp.concatenate([jnp.array([TSURF]), TDN[:-1]])
    TAVG = TMPAVG(TUP_Vector, STC, TDN, ZSOIL)

    TSNSR, SH2O_New = SNKSRC(
        TAVG, SMC, SH2O, ZSOIL, SMCMAX, PSISAT, BEXP, DT, QTOT, BLIM_OPTION, CK_OPTION, BLIMBX
    )

    condition = (SICE > 0) | (STC < T0) | (TUP_Vector < T0) | (TDN < T0)
    TSNSR = jnp.where(condition, TSNSR, 0)
    SH2O_New = jnp.where(condition, SH2O_New, SH2O)
    RHSTS = RHSTS + TSNSR / jnp.maximum(DENOM2 * HCPCT, 1e-9)

    RHSTS = RHSTS.astype(jnp.float32)
    AI = AI.astype(jnp.float32)
    BI = BI.astype(jnp.float32)
    CI = CI.astype(jnp.float32)
    DT = jnp.asarray(DT, dtype=jnp.float32)

    RHSTS = RHSTS * DT
    AI = AI * DT
    BI = 1 + BI * DT
    CI = CI * DT

    P = solve_tridiagonal_jax(AI, BI, CI, RHSTS)
    
    # --- Implementation details ---
    # Numerical-stability safeguard.
    P = jnp.clip(P, -20.0, 20.0)
    # --- (fixend) ---
    
    STC = STC + P

    T1 = (YY + (ZZ1 - 1.0) * STC[0]) / jnp.maximum(ZZ1, 1e-9)
    # Numerical-stability safeguard.
    denom_ssoil_final = 0.5 * ZSOIL[0]
    SSOIL = DF1_SSOIL * (STC[0] - T1) / jnp.where(jnp.abs(denom_ssoil_final) > 1e-9, denom_ssoil_final, jnp.sign(denom_ssoil_final) * 1e-9)

    return STC, T1, SSOIL, SH2O_New


def REDPRM(soil_params_dict: dict, veg_params_dict: dict, gen_params_dict: dict,  # <-- UPDATED: usedictionary
           VEGTYP: int, SOILTYP: int, SLOPETYP: int, SLDPTH, ZSOIL, SHDFAC, ROOT_OPTION: int,
           NROOT_jax: int):
    """Technical documentation."""

    # --- (UPDATED) ---
    # from gen_params_dict read
    CSOIL = gen_params_dict['CSOIL_DATA']

    # --- (UPDATED) ---
    # from soil_params_dict read directly, the index starts at 1, so  -1
    index = SOILTYP - 1
    BEXP = soil_params_dict['BB'][index]
    SMCMAX = soil_params_dict['MAXSMC'][index]
    DKSAT = soil_params_dict['SATDK'][index]
    PSISAT = soil_params_dict['SATPSI'][index]
    QUARTZ = soil_params_dict['QTZ'][index]

    F1 = jnp.log10(jnp.maximum(PSISAT, 1e-9)) + BEXP * jnp.log10(jnp.maximum(SMCMAX, 1e-9)) + 2.0
    REFSMC1 = SMCMAX * jnp.power(jnp.maximum(5.79E-9 / jnp.maximum(DKSAT, 1e-9), 1e-9), (1 / (2 * BEXP + 3 + 1e-9)))
    SMCREF = REFSMC1 + 1. / 3. * (SMCMAX - REFSMC1)
    WLTSMC1 = SMCMAX * jnp.power(jnp.maximum(200. / jnp.maximum(PSISAT, 1e-9), 1e-9), (-1. / (BEXP + 1e-9)))
    SMCWLT = 0.5 * WLTSMC1
    DWSAT = BEXP * DKSAT * PSISAT / jnp.maximum(SMCMAX, 1e-9)
    SMCDRY = SMCWLT

    # --- (UPDATED) ---
    # from gen_params_dict read
    ZBOT = gen_params_dict['ZBOT_DATA']
    SALP = gen_params_dict['SALP_DATA']
    SBETA = gen_params_dict['SBETA_DATA']
    REFDK = gen_params_dict['REFDK_DATA']
    FRZK = gen_params_dict['FRZK_DATA']
    FXEXP = gen_params_dict['FXEXP_DATA']
    REFKDT = gen_params_dict['REFKDT_DATA']
    PTU = jnp.array(0.0)
    KDT = REFKDT * DKSAT / jnp.maximum(REFDK, 1e-9)
    CZIL = gen_params_dict['CZIL_DATA']
    SLOPE = gen_params_dict['SLOPE_DATA'][int(SLOPETYP) - 1]
    LVCOEF = gen_params_dict['LVCOEF_DATA']

    FRZFACT = (SMCMAX / jnp.maximum(SMCREF, 1e-9)) * (0.412 / 0.468)
    FRZX = FRZK * FRZFACT

    # --- (UPDATED) ---
    # from gen_params_dict read
    TOPT = gen_params_dict['TOPT_DATA']
    CMCMAX = gen_params_dict['CMCMAX_DATA']
    CFACTR = gen_params_dict['CFACTR_DATA']
    RSMAX = gen_params_dict['RSMAX_DATA']

    # --- (UPDATED) ---
    # from veg_params_dict read
    NROOT = NROOT_jax
    if NROOT < 1:
        NROOT = 1
    SNUP = jnp.array(veg_params_dict['SNUP'])
    RSMIN = jnp.array(veg_params_dict['RS'])
    RGL = jnp.array(veg_params_dict['RGL'])
    HS = jnp.array(veg_params_dict['HS'])
    EMISSMIN = jnp.array(veg_params_dict['EMISSMIN'])
    EMISSMAX = jnp.array(veg_params_dict['EMISSMAX'])
    LAIMIN = jnp.array(veg_params_dict['LAIMIN'])
    LAIMAX = jnp.array(veg_params_dict['LAIMAX'])
    Z0MIN = jnp.array(veg_params_dict['Z0MIN'])
    Z0MAX = jnp.array(veg_params_dict['Z0MAX'])
    ALBEDOMIN = jnp.array(veg_params_dict['ALBEDOMIN'])
    ALBEDOMAX = jnp.array(veg_params_dict['ALBEDOMAX'])

    # Keep BARE as a JAX scalar so REDPRM remains compatible with jit/vmap tracing.
    bare_vegtyp = jnp.asarray(gen_params_dict["BARE"], dtype=jnp.int32)
    SHDFAC = jnp.where(VEGTYP == bare_vegtyp, jnp.array(0.0), SHDFAC)

#Revised by Zheng and Zhang
    rt_uniform = - SLDPTH[:NROOT] / ZSOIL[NROOT - 1]
    positive_depths = -ZSOIL[:NROOT]
    max_root_depth = positive_depths[-1]
    ROOTA = jnp.array(veg_params_dict['ROOTA'])
    ROOTB = jnp.array(veg_params_dict['ROOTB'])
    denominator = 1.0 - 0.5 * (jnp.exp(-ROOTA * max_root_depth) + jnp.exp(-ROOTB * max_root_depth))
    cumulative_root_fraction = (1.0 - 0.5 * (
            jnp.exp(-ROOTA * positive_depths) + jnp.exp(-ROOTB * positive_depths))) / (denominator + 1e-9)
    rt_exp = jnp.zeros(NROOT)
    rt_exp = rt_exp.at[0].set(cumulative_root_fraction[0])
    rt_exp = rt_exp.at[1:].set(cumulative_root_fraction[1:] - cumulative_root_fraction[:-1])

    RTDIS = jax.lax.cond(ROOT_OPTION == 1,
                         lambda: rt_exp,
                         lambda: rt_uniform)
#end ROOT_OPTION

    return (CFACTR, CMCMAX, RSMAX, TOPT, REFKDT, KDT, SBETA, SHDFAC, RSMIN, RGL, HS, ZBOT,
            FRZX, PSISAT, SLOPE, SNUP, SALP, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ, FXEXP,
            RTDIS, NROOT, CZIL, LAIMIN, LAIMAX, EMISSMIN, EMISSMAX, ALBEDOMIN, ALBEDOMAX, Z0MIN,
            Z0MAX, CSOIL, PTU, LVCOEF)


def SRT(soil_params_dict: dict, gen_params_dict: dict,  # <-- UPDATED: usedictionary
        STYPE, EDIR, ET, SH2O, SH2OA, PCPDRP, ZSOIL, DT, SLOPE, KDT, FRZX, SICE, CMC, CMCMAX, RHSCT, INF_OPTION,
        NROOT, RIC_OPTION, SOC, SOC_OPTION_KS):
    """Technical documentation."""
    NSOIL = SH2O.shape[0]
    RUNOFF2 = jnp.array(0.0)

    DENOM2 = jnp.zeros_like(SH2O)
    DENOM2 = DENOM2.at[0].set(-ZSOIL[0])
    DENOM2 = DENOM2.at[1:].set(ZSOIL[:-1] - ZSOIL[1:])

    # --- (UPDATED) ---
    PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ, _ = REDSTP(
        soil_params_dict, gen_params_dict,  # <-- UPDATED
        STYPE[:NSOIL],
        jnp.arange(NSOIL),
        SOC[:NSOIL], SOC_OPTION_KS,
        required_grad=True)

    SMCAV = SMCMAX - SMCWLT

    DMAX = DENOM2 * SMCAV * (1 - (SH2OA + SICE - SMCWLT) / jnp.maximum(SMCAV, 1e-9))

    DD = jnp.sum(DMAX)
    DDT = DD * (1.0 - jnp.exp(- KDT * DT / 86400.0))
    PX = jnp.maximum(PCPDRP * DT, 0.0)

    soil_thickness = DENOM2
    ice_per_layer = soil_thickness * SICE

# Revised by Zheng and Zhang
    dice_all_layers = jnp.sum(ice_per_layer)
    dice_root_layers = jnp.sum(ice_per_layer[:NROOT])
    DICE = jnp.where(INF_OPTION == 0, dice_all_layers, dice_root_layers)
# end

    CVFRZ = 3.0
    THRESHOLD = 1.0E-2
    fcr_if_false = jnp.ones_like(DICE)
    DICE_safe = DICE + 1e-9
    ACRT = CVFRZ * FRZX / DICE_safe
    SUM = 1.0 + ACRT + (ACRT ** 2) / 2.0
    fcr_if_true = 1.0 - jnp.exp(-ACRT) * SUM
    condition = DICE > THRESHOLD
    FCR = jnp.where(condition, fcr_if_true, fcr_if_false)

    SICEMAX = jnp.max(SICE)
    WDF_all, WCND_all = WDFCND(SH2OA, SMCMAX, BEXP, DKSAT, DWSAT, SICEMAX)
    WDF_all = jnp.maximum(WDF_all, 1e-12)  # or 1e-9
    WCND_all = jnp.maximum(WCND_all, 1e-12)  # Implementation detail.
    INFMAX = jnp.clip(FCR * (PX * (DDT / (PX + DDT))) / DT, WCND_all[0], PX / DT)
    RUNOFF1 = jnp.where(PCPDRP > INFMAX, PCPDRP - INFMAX, jnp.array(0.0))
    PDDUM = jnp.where(PCPDRP > INFMAX, INFMAX, PCPDRP)

    DDZ2 = jnp.zeros(NSOIL - 1)
    AI = jnp.zeros(NSOIL)
    CI = jnp.zeros(NSOIL)
    RHSTT = jnp.zeros(NSOIL)
    delta_H2O = jnp.zeros(NSOIL)
    delta_H2O = delta_H2O.at[:-1].set(SH2O[:-1] - SH2O[1:])

#Revised by Zheng and Zhang
    if RIC_OPTION == 1:
        SMX2 = 0.5 * (SH2OA + SH2O)
        PSI = PSISAT * jnp.power(jnp.maximum(SMX2 / jnp.maximum(SMCMAX, 1e-9), 1e-9), -BEXP)

        def smxc_body(k, SMXC_state):
            DZ_K = jnp.where(k == 0,
                             -0.5 * ZSOIL[0],
                             -0.5 * (ZSOIL[k] - ZSOIL[k - 1]))
            DZ_K1 = -0.5 * (ZSOIL[k + 1] - ZSOIL[k])
            numerator = ((WCND_all[k] - WCND_all[k + 1]) +
                         WCND_all[k] * PSI[k] / jnp.maximum(DZ_K, 1e-9) +
                         WCND_all[k + 1] * PSI[k + 1] / jnp.maximum(DZ_K1, 1e-9))
            denominator = WCND_all[k] / jnp.maximum(DZ_K, 1e-9) + WCND_all[k + 1] / jnp.maximum(DZ_K1, 1e-9)
            PSIC = numerator / jnp.maximum(denominator, 1e-9)
            SMXa = SMCMAX[k] * jnp.power(jnp.maximum(PSIC / jnp.maximum(PSISAT[k], 1e-9), 1e-9), -1.0 / (BEXP[k] + 1e-9))
            SMXb = SMCMAX[k + 1] * jnp.power(jnp.maximum(PSIC / jnp.maximum(PSISAT[k + 1], 1e-9), 1e-9), -1.0 / (BEXP[k + 1] + 1e-9))
            return SMXC_state.at[k].set(SMXa - SMXb)

        SMXC = jax.lax.fori_loop(0, NSOIL - 1, smxc_body, jnp.zeros(NSOIL - 1))
        delta_H2O = delta_H2O.at[:-1].set(delta_H2O[:-1] - SMXC / 2.0)
#end RIC_OPTION

    DDZ2 = DDZ2.at[0].set((0 - ZSOIL[1]) * 0.5)
    DDZ2 = DDZ2.at[1:].set((ZSOIL[:-2] - ZSOIL[2:]) * 0.5)

    # Numerical-stability safeguard.
    AI = AI.at[1:].set(-WDF_all[:-1] / jnp.maximum(DDZ2 * DENOM2[1:], 1e-9))
    CI = CI.at[:-1].set(-WDF_all[:-1] / jnp.maximum(DDZ2 * DENOM2[:-1], 1e-9))
    BI = -(AI + CI)

    # Numerical-stability safeguard.
    denom_rhstt0 = -DENOM2[0]
    RHSTT = RHSTT.at[0].set(CI[0] * delta_H2O[0] + (WCND_all[0] - PDDUM + EDIR + ET[0]) / jnp.where(jnp.abs(denom_rhstt0) > 1e-9, denom_rhstt0, jnp.sign(denom_rhstt0) * 1e-9))

    if NSOIL > 2:
        # Numerical-stability safeguard.
        denom_rhstt_mid = -DENOM2[1:-1]
        RHSTT = RHSTT.at[1:-1].set(-AI[1:-1] * delta_H2O[:-2] + CI[1:-1] * delta_H2O[1:-1] + \
                                   (WCND_all[1:-1] - WCND_all[:-2] + ET[1:-1]) / jnp.where(jnp.abs(denom_rhstt_mid) > 1e-9, denom_rhstt_mid, jnp.sign(denom_rhstt_mid) * 1e-9))

    RUNOFF2 = SLOPE * WCND_all[-1]
    k = NSOIL - 1
    rhstt_numerator = (RUNOFF2 - WCND_all[k - 1] + ET[k])
    rhstt_denominator = -DENOM2[k]
    RHSTT = RHSTT.at[k].set(-AI[k] * delta_H2O[k - 1] + rhstt_numerator / rhstt_denominator)

    RHSTT = RHSTT * DT
    AI = AI * DT
    BI = 1 + BI * DT
    CI = CI * DT

    P = solve_tridiagonal_jax(AI, BI, CI, RHSTT)

    PLUS = SMCMAX - (SH2O + P + SICE)

    def excess_water_branch(_):
        DDZ = jnp.zeros(NSOIL)
        DDZ = DDZ.at[0].set(-ZSOIL[0])
        DDZ = DDZ.at[1:].set(ZSOIL[:-1] - ZSOIL[1:])
        WPLUS = jnp.zeros(NSOIL + 1)

        def wplus_body(K, WPLUS_state):
            excess = ((SH2O[K - 1] + P[K - 1] + SICE[K - 1] + WPLUS_state[K - 1] / DDZ[K - 1]) - SMCMAX[K - 1]) * DDZ[
                K - 1]
            return WPLUS_state.at[K].set(jnp.maximum(excess, 0.0))

        WPLUS = jax.lax.fori_loop(1, NSOIL + 1, wplus_body, WPLUS)

        SH2OOUT_ex = jnp.clip(SH2O + P + WPLUS[:-1] / DDZ, 0.0, SMCMAX - SICE)
        SMC_ex = jnp.clip(SH2O + P + WPLUS[:-1] / DDZ + SICE, 0.02, SMCMAX)
        RUNOFF3_ex = WPLUS[-1]
        return SH2OOUT_ex, SMC_ex, RUNOFF3_ex

    def normal_branch(_):
        SH2OOUT_norm = jnp.clip(SH2O + P, 0.0, SMCMAX - SICE)
        SMC_norm = jnp.clip(SH2O + P + SICE, 0.02, SMCMAX)
        RUNOFF3_norm = jnp.array(0.0)
        return SH2OOUT_norm, SMC_norm, RUNOFF3_norm

    has_excess = jnp.any(PLUS < 0)
    SH2OOUT, SMC, RUNOFF3 = jax.lax.cond(has_excess, excess_water_branch, normal_branch, None)

    CMC = CMC + DT * RHSCT
    CMC = jnp.where(CMC < 1E-20, jnp.array(0.0), CMC)
    CMC = jnp.clip(CMC, 0.0, CMCMAX)

    return RUNOFF1, RUNOFF2, RUNOFF3, SH2OOUT, SMC, CMC
