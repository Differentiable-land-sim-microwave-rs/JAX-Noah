"""Technical documentation."""

import jax.numpy as jnp
import jax
from Module_model_constants_jax_simple import *

jax.config.update("jax_enable_x64", False)

ITRMX = jnp.array(5, dtype=jnp.int32)
EXCML = jnp.array(0.0001)
EXCMS = jnp.array(0.0001)
VKARMAN = jnp.array(0.4)
ZTFC = jnp.array(1.0)
ELOCP = jnp.array(2.72e6) / CP
EPSU2 = jnp.array(1.0e-6)
EPSUST = jnp.array(1.0e-9)
SQVISC = jnp.array(258.2)
RIC = jnp.array(0.505)
EPSZT = jnp.array(1.0e-28)
RD = jnp.array(287.0)

KZTM = 10001
KZTM2 = KZTM - 2

WWST = jnp.array(1.2)
WWST2 = WWST * WWST

# --- Implementation details ---

# ============================================================
# Implementation detail.
# mathematically equivalent to the original method, with identical forward results
# ============================================================
def differentiable_table_interp(table, RZ, KZTM2_val):
    """
    differentiable lookup-table linear interpolation

    Original method:
        K = jnp.floor(RZ).astype(jnp.int32)
        RDZT = RZ - K.astype(jnp.float32)
        K = jnp.clip(K, 0, KZTM2)
        result = (table[K + 1] - table[K]) * RDZT + table[K]

    Equivalent transformation:
        result = table[K] + (table[K+1] - table[K]) * RDZT
               = table[K] * (1 - RDZT) + table[K+1] * RDZT
               = table[K] * w_low + table[K+1] * w_high

    Mathematically, the two forms are fully equivalent, with identical forward results.
    """
    idx_low = jnp.floor(RZ).astype(jnp.int32)
    idx_low = jnp.clip(idx_low, 0, KZTM2_val)
    idx_high = idx_low + 1

    # weight calculation
    w_high = RZ - jnp.floor(RZ)  # Implementation detail.
    w_low = 1.0 - w_high

    # linear interpolation
    result = table[idx_low] * w_low + table[idx_high] * w_high
    return result

def MYJSFCINIT():
    """Technical documentation."""

    PIHF = jnp.pi / 2.
    EPS = jnp.array(1.0e-6)
    ZTMIN1 = jnp.array(-5.0)
    ZTMIN2 = jnp.array(-5.0)

    ZTMAX1 = jnp.array(1.0)
    ZTMAX2_calc = jnp.array(1.0)
    ZRNG1 = ZTMAX1 - ZTMIN1
    ZRNG2 = ZTMAX2_calc - ZTMIN2
    DZETA1 = ZRNG1 / (KZTM - 1)
    DZETA2_calc = ZRNG2 / (KZTM - 1)
    ZETA1 = ZTMIN1
    ZETA2 = ZTMIN2

    PSIM2_local = jnp.zeros(KZTM)
    PSIH2_local = jnp.zeros(KZTM)

    for K in range(1, KZTM + 1):
        # --- (FIX) use jnp.where ---
        condition = ZETA2 < 0.
        # Implementation detail.
        X = jnp.sqrt(jnp.sqrt(jnp.maximum(1.0e-10, 1. - 16. * ZETA2)))
        psim2_unstable = -2. * jnp.log((X + 1.) / 2.) - jnp.log((X * X + 1.) / 2.) + 2. * jnp.arctan(X) - PIHF
        psih2_unstable = -2. * jnp.log((X * X + 1.) / 2.)
        # Implementation detail.
        stable_func = 0.7 * ZETA2 + 0.75 * ZETA2 * (6. - 0.35 * ZETA2) * jnp.exp(-0.35 * ZETA2)
        psim2_stable = stable_func
        psih2_stable = stable_func

        PSIM2_local = PSIM2_local.at[K - 1].set(jnp.where(condition, psim2_unstable, psim2_stable))
        PSIH2_local = PSIH2_local.at[K - 1].set(jnp.where(condition, psih2_unstable, psih2_stable))
        # --- (END FIX) ---

        if K == KZTM:
            ZTMAX1 = ZETA1
            ZTMAX2_calc = ZETA2

        ZETA1 = ZETA1 + DZETA1
        ZETA2 = ZETA2 + DZETA2_calc

    ZTMAX1 -= EPS
    ZTMAX2_calc -= EPS

    return PSIM2_local, PSIH2_local, ZTMAX2_calc, DZETA2_calc, ZTMIN2


def SFCDIF_MYJ(ZSL, ZSL_WIND, Z0, Z0BASE, SFCPRS, TZ0, TLOW, QZ0, QLOW, SFCSPD, CZIL, AKMS, AKHS, IZ0TLND,
               PSIM2_param, PSIH2_param, ZTMAX2_param, DZETA2_param, ZTMIN2_param):

    # Implementation detail.
    PSIM2 = PSIM2_param
    PSIH2 = PSIH2_param
    ZTMAX2 = ZTMAX2_param
    DZETA2 = DZETA2_param
    ZTMIN2 = ZTMIN2_param

    THLOW = TLOW * (P0 / SFCPRS) ** RCP
    THZ0 = TZ0 * (P0 / SFCPRS) ** RCP
    THELOW = THLOW

    CXCHL = EXCML / ZSL
    BTGX = G / THLOW
    ELFC = VKARMAN * BTGX
    BTGH = BTGX * 1000.

    THM = (THELOW + THZ0) * 0.5
    TEM = (TLOW + TZ0) * 0.5
    A = THM * P608
    B = (ELOCP / TEM - 1. - P608) * THM


    CWMLOW = jnp.array(0.0)
    DTHV = ((THELOW - THZ0) * ((QLOW + QZ0 + CWMLOW) * (0.5 * P608) + 1.) +
            (QLOW - QZ0 + CWMLOW) * A + CWMLOW * B)

    DU2 = jnp.maximum(SFCSPD * SFCSPD, EPSU2)
    RIB = BTGX * DTHV * ZSL_WIND * ZSL_WIND / DU2 / ZSL

    ZU = Z0
    ZT = ZU * ZTFC
    ZSLU = ZSL_WIND + ZU

    RZSU = jnp.array(ZSLU / ZU)
    RLOGU = jnp.log(RZSU)
    ZSLT = ZSL + ZU

    # (*** mild numerical safeguard iz0tlnd=1 ***)
    # prevent extreme anomalies without over-constraining physical values
    Z0_safe = jnp.clip(Z0, 1e-6, 10.0)  # Implementation detail.
    # Implementation detail.
    exponent = jnp.clip(-0.40 * (Z0_safe / 0.07), -20.0, 5.0)
    CZIL_LOCAL = jnp.power(10.0, exponent)
    CZIL_LOCAL = jnp.clip(CZIL_LOCAL, 1e-8, 1000.0)
    ZILFC = jnp.where(IZ0TLND == 0,
                      -CZIL * VKARMAN * SQVISC,
                      -CZIL_LOCAL * VKARMAN * SQVISC)

    CZETMAX = 10.
    # Numerical-stability safeguard.
    RIB_RIC_ratio = RIB / jnp.maximum(jnp.abs(RIC), 1e-12)
    # limit the squared ratio, avoid extremes while retaining dynamic range
    RIB_RIC_ratio_sq = jnp.clip(RIB_RIC_ratio * RIB_RIC_ratio, 0.0, 1000.0)
    ZZIL = jnp.where(DTHV > 0,
                     jnp.where(RIB < RIC,
                               ZILFC * (1.0 + RIB_RIC_ratio_sq * CZETMAX),
                               ZILFC * (1.0 + CZETMAX)),
                     ZILFC)
    # Implementation detail.
    ZZIL = jnp.clip(ZZIL, -1000.0, 1000.0)

    # Jacobian and gradient handling.
    wstar_base = jnp.maximum(jnp.abs(BTGH * AKHS * DTHV), 1e-15)
    WSTAR2 = jnp.where(BTGH * AKHS * DTHV != 0.0,
                       WWST2 * jnp.power(wstar_base, 2.0 / 3.0),
                       jnp.array(0.0))

    USTAR = jnp.maximum(jnp.sqrt(AKMS * jnp.sqrt(DU2 + WSTAR2)), EPSUST)

    ITRMX_int = 5
    for ITR in range(0, ITRMX_int): # Implementation detail.
        # Model-parameter handling.
        exp_arg = jnp.clip(ZZIL * jnp.sqrt(jnp.maximum(USTAR * Z0BASE, 1e-12)), -50.0, 50.0)
        ZT = jnp.maximum(jnp.exp(exp_arg) * Z0BASE, EPSZT)
        # only prevent division by zero, without clipping
        RZST = ZSLT / jnp.maximum(ZT, 1e-12)
        RLOGT = jnp.log(jnp.maximum(RZST, 1e-12))

        # Numerical-stability safeguard.
        USTAR_cubed = jnp.maximum(USTAR ** 3, 1e-15)
        RLMO = ELFC * AKHS * DTHV / USTAR_cubed
        # without clippingRLMO, allow the iteration to converge naturally

        ZETALU = ZSLU * RLMO
        ZETALT = ZSLT * RLMO
        ZETAU = ZU * RLMO
        ZETAT = ZT * RLMO

        ZETALU = jnp.clip(ZETALU, ZTMIN2, ZTMAX2)
        ZETALT = jnp.clip(ZETALT, ZTMIN2, ZTMAX2)
        ZETAU = jnp.clip(ZETAU, ZTMIN2 / RZSU, ZTMAX2 / RZSU)
        ZETAT = jnp.clip(ZETAT, ZTMIN2 / RZST, ZTMAX2 / RZST)

        # ============================================================
        # momentum stability-function interpolation (PSIM)
        # ============================================================
        RZ = (ZETAU - ZTMIN2) / DZETA2
        # --- original code (comment retained) ---
        #K = jnp.floor(RZ).astype(jnp.int32)
        #RDZT = RZ - K.astype(jnp.float32)
        #K = jnp.clip(K, 0, KZTM2)
        #PSMZ = (PSIM2[K + 1] - PSIM2[K]) * RDZT + PSIM2[K]
        # --- Implementation details ---
        PSMZ = differentiable_table_interp(PSIM2, RZ, KZTM2)

        RZ = (ZETALU - ZTMIN2) / DZETA2
        # --- original code (comment retained) ---
        #K = jnp.floor(RZ).astype(jnp.int32)
        #RDZT = RZ - K.astype(jnp.float32)
        #K = jnp.clip(K, 0, KZTM2)
        #PSMZL = (PSIM2[K + 1] - PSIM2[K]) * RDZT + PSIM2[K]
        # --- Implementation details ---
        PSMZL = differentiable_table_interp(PSIM2, RZ, KZTM2)

        SIMM = PSMZL - PSMZ + RLOGU

        # ============================================================
        # thermal stability-function interpolation (PSIH)
        # ============================================================
        RZ = (ZETAT - ZTMIN2) / DZETA2
        # --- original code (comment retained) ---
        #K = jnp.floor(RZ).astype(jnp.int32)
        #RDZT = RZ - K.astype(jnp.float32)
        #K = jnp.clip(K, 0, KZTM2)
        #PSHZ = (PSIH2[K + 1] - PSIH2[K]) * RDZT + PSIH2[K]
        # --- Implementation details ---
        PSHZ = differentiable_table_interp(PSIH2, RZ, KZTM2)

        RZ = (ZETALT - ZTMIN2) / DZETA2
        # --- original code (comment retained) ---
        #K = jnp.floor(RZ).astype(jnp.int32)
        #RDZT = RZ - K.astype(jnp.float32)
        #K = jnp.clip(K, 0, KZTM2)
        #PSHZL = (PSIH2[K + 1] - PSIH2[K]) * RDZT + PSIH2[K]
        # --- Implementation details ---
        PSHZL = differentiable_table_interp(PSIH2, RZ, KZTM2)

        SIMH = PSHZL - PSHZ + RLOGT

        USTARK = USTAR * VKARMAN

        # (*** Safety recommendation ***)
        # Numerical-stability safeguard.
        # Implementation detail.
        safe_denom_m = jnp.where(jnp.abs(SIMM) < 1e-9, jnp.sign(SIMM + 1e-20) * 1e-9, SIMM)
        safe_denom_h = jnp.where(jnp.abs(SIMH) < 1e-9, jnp.sign(SIMH + 1e-20) * 1e-9, SIMH)

        # (*** Core fix ***) your version: 
        AKMS = jnp.maximum(USTARK / safe_denom_m, CXCHL)
        AKHS = jnp.maximum(USTARK / safe_denom_h, CXCHL)

        # Numerical-stability safeguard.
        wstar2_base = jnp.maximum(jnp.abs(BTGH * AKHS * DTHV), 1e-15)
        WSTAR2 = jnp.where(
            DTHV <= 0.0,
            WWST2 * jnp.power(wstar2_base, 2.0 / 3.0),
            jnp.array(0.0)
        )

        USTAR = jnp.maximum(jnp.sqrt(AKMS * jnp.sqrt(DU2 + WSTAR2)), EPSUST)

    return RIB, AKMS, AKHS, RLMO


def SFCDIF_MYJ_Y08(z0m, zm, zh, wspd1, tsfc, tair, qair, psfc):
    # Implementation detail.
    excm = jnp.array(0.001)
    aa = jnp.array(0.007)
    p0 = jnp.array(1.0e5)

    wspd = jnp.maximum(wspd1, jnp.array(0.01))

    rhoair = psfc / (RD * tair * (1 + 0.61 * qair))

    ptair = tair * (psfc / (psfc - rhoair * 9.81 * zh)) ** RCP
    ptsfc = tsfc

    pt1 = ptair

    c_u = 0.4 / jnp.log(zm / z0m)
    c_pt = 0.4 / jnp.log(zh / z0m)

    tstr = c_pt * (pt1 - ptsfc)
    ustr = c_u * wspd

    lmo = ptair * ustr ** 2 / (0.4 * 9.81 * tstr)

    nu = 1.328e-5 * (p0 / psfc) * (pt1 / 273.15) ** 1.754

    # Implementation detail.
    for i in range(3):
        z0h = z0mz0h(zh, z0m, nu, ustr, tstr)
        c_u, c_pt, ribbb = flxpar(zm, zh, z0m, z0h, wspd, ptsfc, pt1)

        ustr = c_u * wspd
        tstr = c_pt * (pt1 - ptsfc)

    c_pt = jnp.where(jnp.abs(ptair - ptsfc) < 0.001, c_pt, tstr / (ptair - ptsfc))

    ra = 1 / (ustr * c_pt)

    chh = 1.0 / ra

    chh = jnp.maximum(chh, excm * (1.0 / zm))

    return ribbb, chh, lmo


def z0mz0h(zh, z0m, nu, ustr, tstr):
    # Implementation detail.
    a = jnp.array(70.0)
    b = jnp.array(-7.2)

    z0h = a * nu / ustr * jnp.exp(b * jnp.sqrt(ustr) * jnp.sqrt(jnp.sqrt(jnp.abs(-tstr))))
    z0h = jnp.minimum(zh / 10, jnp.maximum(z0h, jnp.array(1.0E-10)))

    return z0h


def flxpar(zm, zh, z0m, z0h, wspd, ptsfc, pt1):
    # Implementation detail.
    lmo, ribb1 = MOlength(zm, zh, z0m, z0h, wspd, ptsfc, pt1)

    c_u, c_pt = CuCpt(lmo, z0m, z0h, zm, zh)

    return c_u, c_pt, ribb1


def MOlength(zm, zh, z0m, z0h, wspd, ptsfc, pt1):
    # Implementation detail.
    g = jnp.array(9.81)
    prantl01 = jnp.array(1.0)
    prantl02 = jnp.array(0.95)
    betah = jnp.array(8.0)
    betam = jnp.array(5.3)
    gammah = jnp.array(11.6)
    gammam = jnp.array(19.0)

    bulkri = (g / pt1) * (pt1 - ptsfc) * (zm - z0m) / (wspd ** 2)

    # --- (FIX) use jnp.where and jnp.minimum/maximum ---
    # Implementation detail.
    bulkri_unstable = jnp.maximum(bulkri, jnp.array(-10.0))
    d_unstable = bulkri_unstable / prantl02
    numerator_unstable = d_unstable * ((jnp.log(zm / z0m)) ** 2 / jnp.log(zh / z0h)) * (1 / (zm - z0m))
    a_unstable = jnp.log(-d_unstable)
    b_unstable = jnp.log(jnp.log(zm / z0m))
    c_unstable = jnp.log(jnp.log(zh / z0h))
    p_unstable = 0.03728 - 0.093143 * a_unstable - 0.24069 * b_unstable + 0.30616 * c_unstable + \
                 0.017131 * a_unstable ** 2 + 0.037666 * a_unstable * b_unstable - 0.084598 * b_unstable ** 2 - 0.016498 * a_unstable * c_unstable + \
                 0.1828 * b_unstable * c_unstable - 0.12587 * c_unstable ** 2
    p_unstable = jnp.maximum(jnp.array(0.0), p_unstable)
    coef_unstable = d_unstable * gammam ** 2 / 8 / gammah * (zm - z0m) / (zh - z0h)
    # Implementation detail.
    # Numerical-stability safeguard.
    denominator_unstable = (1 - coef_unstable * p_unstable)
    # Numerical-stability safeguard.
    lmo_inv_unstable = numerator_unstable / (denominator_unstable + 1e-9)

    # Implementation detail.
    bulkri_stable_clip = prantl01 * betah * (1 - z0h / zh) / betam ** 2 / (1 - z0m / zm) - 0.05
    # Implementation detail.
    bulkri_stable = jnp.minimum(bulkri, jnp.array(0.2))
    bulkri_stable = jnp.minimum(bulkri_stable, bulkri_stable_clip)

    d_stable = bulkri_stable / prantl01
    a_stable = d_stable * betam ** 2 * (zm - z0m) - betah * (zh - z0h)
    b_stable = 2 * d_stable * betam * jnp.log(zm / z0m) - jnp.log(zh / z0h)
    c_stable = d_stable * jnp.log(zm / z0m) ** 2 / (zm - z0m)
    # Implementation detail.
    discriminant = b_stable ** 2 - 4 * a_stable * c_stable
    lmo_inv_stable = (-b_stable - jnp.sqrt(jnp.maximum(0., discriminant))) / (2 * a_stable + 1e-9) # Numerical-stability safeguard.

    # Implementation detail.
    lmo_inv = jnp.where(bulkri < 0.0, lmo_inv_unstable, lmo_inv_stable)

    # --- Implementation details ---
    lmo = jnp.where( (lmo_inv > 0) & (lmo_inv < 1.0e-6), 1.0e-6, lmo_inv)
    lmo = jnp.where( (lmo < 0) & (lmo > -1.0e-6), -1.0e-6, lmo)

    lmo = 1 / (lmo + 1e-9) # Numerical-stability safeguard.
    # --------------------------------------------------

    # Implementation detail.
    return lmo, bulkri


def CuCpt(lmo, zm1, zh1, zm2, zh2):
    # Implementation detail.
    kv = 0.4
    gammam = 19.0
    gammah = 11.6
    prantl01 = 1.0
    prantl02 = 0.95
    betam = 5.3
    betah = 8.0

    # --- (FIX) use jnp.where ---
    # Implementation detail.
    xx2_unstable = jnp.sqrt(jnp.sqrt(1 - gammam * zm2 / lmo))
    xx1_unstable = jnp.sqrt(jnp.sqrt(1 - gammam * zm1 / lmo))
    psim_unstable = 2 * jnp.log((1 + xx2_unstable) / (1 + xx1_unstable)) + jnp.log((1 + xx2_unstable ** 2) / (1 + xx1_unstable ** 2)) \
           - 2 * jnp.arctan(xx2_unstable) + 2 * jnp.arctan(xx1_unstable)

    yy2_unstable = jnp.sqrt(1 - gammah * zh2 / lmo)
    yy1_unstable = jnp.sqrt(1 - gammah * zh1 / lmo)
    psih_unstable = 2 * jnp.log((1 + yy2_unstable) / (1 + yy1_unstable))

    uprf_unstable = jnp.maximum(jnp.log(zm2 / zm1) - psim_unstable, 0.50 * jnp.log(zm2 / zm1))
    ptprf_unstable = jnp.maximum(jnp.log(zh2 / zh1) - psih_unstable, 0.33 * jnp.log(zm2 / zm1))

    c_u_unstable = kv / (uprf_unstable + 1e-9)
    c_pt_unstable = kv / (ptprf_unstable * prantl02 + 1e-9)

    # Implementation detail.
    psim_stable = -betam * (zm2 - zm1) / lmo
    psih_stable = -betah * (zh2 - zh1) / lmo
    # Implementation detail.
    psim_stable = jnp.maximum(-betam, psim_stable)
    psih_stable = jnp.maximum(-betah, psih_stable)

    uprf_stable = jnp.minimum(jnp.log(zm2 / zm1) - psim_stable, 2.0 * jnp.log(zm2 / zm1))
    ptprf_stable = jnp.minimum(jnp.log(zh2 / zh1) - psih_stable, 2.0 * jnp.log(zm2 / zm1))

    c_u_stable = kv / (uprf_stable + 1e-9)
    c_pt_stable = kv / (ptprf_stable * prantl01 + 1e-9)

    # Implementation detail.
    c_u = jnp.where(lmo < 0, c_u_unstable, c_u_stable)
    c_pt = jnp.where(lmo < 0, c_pt_unstable, c_pt_stable)
    # --------------------------

    return c_u, c_pt