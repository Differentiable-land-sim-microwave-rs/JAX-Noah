
import os
from datetime import datetime
from functools import partial 
import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np 
import pandas as pd


from Module_sf_noahlsm_jax_simple import *
from Module_sfcdif_wrf_jax_simple import *

jax.config.update("jax_enable_x64", False)


def open_forcing_file(forcing_file_path):

    parameters = {
        "startdate": datetime.now(),
        "enddate": datetime.now(),
        "loop_for_a_while": 0,
        "output_dir": "",
        "Latitude": None,
        "Longitude": None,
        "Forcing_Timestep": 0,
        "Noahlsm_Timestep": 0,
        "Sea_ice_point": False,
        "Soil_layer_thickness": [],
        "Soil_Temperature": [],
        "Soil_Moisture": [],
        "Soil_Liquid": [],
        "Soil_htype": [],
        "SOC": [],
        "Skin_Temperature": 0,
        "Canopy_water": 0,
        "Snow_depth": 0,
        "Snow_equivalent": 0,
        "Deep_Soil_Temperature": 0,
        "Landuse_dataset": "",
        "Soil_type_index": 0,
        "Vegetation_type_index": 0,
        "Urban_veg_category": 0,
        "glacial_veg_category": 0,
        "Slope_type_index": 0,
        "Max_snow_albedo": 0,
        "Air_temperature_level": 0,
        "Wind_level": 0,
        "Green_Vegetation_Min": 0,
        "Green_Vegetation_Max": 0,
        "Usemonalb": False,
        "Rdlai2d": False,
        "sfcdif_option": 0,
        'SBETA_OPTION': 0,
        'DF_OPTION': 0,
        'ROOT_OPTION': 0,
        'INF_OPTION': 0,
        'SOC_OPTION_KS': 0,
        'SOC_OPTION2_THERMAL': 0,
        'RIC_OPTION': 0,
        'BLIM_OPTION': 0,
        'CK_OPTION': 0,
        "iz0tlnd": 0,
        "Albedo_monthly": [],
        "Shdfac_monthly": [],
        "lai_monthly": [],
        "Z0brd_monthly": []
    }

    with open(forcing_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.__contains__('='):
                key, value = line.strip().split('=')
                key = ''.join(key.split())

                if parameters.__contains__(key):
                    if isinstance(parameters.get(key), list):
                        value = [float(word) for word in value.split()]
                        parameters[key] = value
                    elif isinstance(parameters.get(key), datetime):
                        value = ''.join(value.split())
                        value = value.strip('"')
                        parameters[key] = pd.to_datetime(value, format="%Y%m%d%H%M")
                    elif isinstance(parameters.get(key), str):
                        value = value.strip().strip('"')
                        parameters[key] = value
                    elif isinstance(parameters.get(key), bool):
                        val_str = value.strip().upper().strip('')
                        if val_str == '.TRUE.':
                            parameters[key] = True
                        elif val_str == '.FALSE.':
                            parameters[key] = False
                    else:
                        parameters[key] = float(value.strip())
            elif line.strip().__eq__('/'):
                break

    output_dir = parameters['output_dir']
    forcing_filename = os.path.basename(forcing_file_path)
    infotext = None
    NSOIL = len(parameters['Soil_layer_thickness'])
    startdate = parameters['startdate']
    enddate = parameters['enddate']
    loop_for_a_while = parameters['loop_for_a_while']
    latitude = parameters['Latitude']
    longitude = parameters['Longitude']
    forcing_timestep = parameters['Forcing_Timestep']
    noahlsm_timestep = parameters['Noahlsm_Timestep']
    ice = parameters['Sea_ice_point']

    T1 = jnp.array(parameters['Skin_Temperature'])
    STC = jnp.array(parameters['Soil_Temperature'])
    SMC = jnp.array(parameters['Soil_Moisture'])
    SH2O = jnp.array(parameters['Soil_Liquid'])
    soil_type_val = int(parameters['Soil_type_index'])
    STYPE = jnp.full((NSOIL,), soil_type_val, dtype=jnp.int32)
    SOC = jnp.array(parameters['SOC'])
    SLDPTH = jnp.array(parameters['Soil_layer_thickness'])
    CMC = jnp.array(parameters['Canopy_water'])
    SNOWH = jnp.array(parameters['Snow_depth'])
    SNEQV = jnp.array(parameters['Snow_equivalent'])
    TBOT = jnp.array(parameters['Deep_Soil_Temperature'])

    VEGTYP = int(parameters['Vegetation_type_index'])
    SOILTYP = parameters['Soil_type_index']
    SLOPETYP = parameters['Slope_type_index']
    SNOALB = jnp.array(parameters['Max_snow_albedo'])
    ZLVL = jnp.array(parameters['Air_temperature_level'])
    ZLVL_WIND = jnp.array(parameters['Wind_level'])

    albedo_monthly = parameters['Albedo_monthly']
    shdfac_monthly = parameters['Shdfac_monthly']
    z0brd_monthly = parameters['Z0brd_monthly']
    lai_monthly = parameters['lai_monthly']

    use_urban_module = False
    ISURBAN = False

    SHDMIN = jnp.array(parameters['Green_Vegetation_Min'])
    SHDMAX = jnp.array(parameters['Green_Vegetation_Max'])
    USEMONALB = jnp.array(parameters['Usemonalb'])
    RDLAI2D = jnp.array(parameters['Rdlai2d'])
    LLANDUSE = parameters['Landuse_dataset']

    SBETA_OPTION = jnp.array(parameters['SBETA_OPTION'])
    DF_OPTION = jnp.array(parameters['DF_OPTION'])
    ROOT_OPTION = jnp.array(parameters['ROOT_OPTION'])
    INF_OPTION = jnp.array(parameters['INF_OPTION'])
    SOC_OPTION_KS = jnp.array(parameters['SOC_OPTION_KS'])
    SOC_OPTION2_THERMAL = jnp.array(parameters['SOC_OPTION2_THERMAL'])
    RIC_OPTION = jnp.array(parameters['RIC_OPTION'])
    BLIM_OPTION = jnp.array(parameters['BLIM_OPTION'])
    CK_OPTION = jnp.array(parameters['CK_OPTION'])
    IZ0TLND = jnp.array(parameters['iz0tlnd'])
    sfcdif_option = parameters['sfcdif_option']

    forcing_columns_name = ['Year', 'Month', 'Day', 'Hour', 'minutes', 'windspeed', 'winddir', 'temperature',
                            'humidity', 'pressure', 'shortwave', 'longwave', 'precipitation']
    x_target_columns = ['windspeed', 'winddir', 'temperature', 'humidity', 'pressure', 'shortwave', 'longwave',
                        'precipitation']

    forcing_data = pd.read_csv(forcing_file_path, sep=r'\s+', names=forcing_columns_name, header=None,
                               skiprows=55)

    forcing_data['Date'] = pd.to_datetime(forcing_data[['Year', 'Month', 'Day', 'Hour']])
    forcing_data.set_index('Date', inplace=True)
    forcing_data = forcing_data[x_target_columns]

    condition = (forcing_data.index >= startdate) & (forcing_data.index <= enddate)
    forcing_data = forcing_data[condition]
    Date = forcing_data.index

    forcing_data = jnp.array(forcing_data.to_numpy(), dtype=jnp.float32)

    return (Date, forcing_data, output_dir, forcing_filename, infotext, NSOIL, startdate, enddate, loop_for_a_while, latitude,
            longitude,
            forcing_timestep, noahlsm_timestep, ice, T1, STC, SMC, SH2O, STYPE,
            SLDPTH, CMC, SNOWH, SNEQV, TBOT, VEGTYP, SOILTYP, SLOPETYP, SNOALB, ZLVL, ZLVL_WIND,
            albedo_monthly, shdfac_monthly, z0brd_monthly, lai_monthly, use_urban_module, ISURBAN,
            SHDMIN, SHDMAX, USEMONALB, RDLAI2D, LLANDUSE, SBETA_OPTION, DF_OPTION, ROOT_OPTION, INF_OPTION, SOC_OPTION_KS,
            SOC_OPTION2_THERMAL, RIC_OPTION, BLIM_OPTION, CK_OPTION, IZ0TLND, sfcdif_option, SOC)


def month_d(a12, nowdate):

    nowy = nowdate.year
    nowm = nowdate.month
    nowd = nowdate.day
    ndays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if nowm == 2:
        if nowy % 4 == 0 and (nowy % 100 != 0 or nowy % 400 == 0):
            ndays[1] = 29

    prevm, postm, factor = 0, 0, 0
    if nowd == 15:
        return jnp.array(a12[nowm - 1])
    elif nowd < 15:
        prevm = nowm - 1 if nowm > 1 else 12
        postm = nowm
        factor = (ndays[prevm - 1] - 15 + nowd) / ndays[prevm - 1]
    elif nowd > 15:
        prevm = nowm
        postm = nowm + 1 if nowm < 12 else 1
        factor = (nowd - 15) / ndays[prevm - 1]

    val_prev = jnp.array(a12[prevm - 1])
    val_post = jnp.array(a12[postm - 1])
    return val_prev * (1.0 - factor) + val_post * factor


def resolve_bare_vegtype_index(landuse_dataset: str) -> int:
    landuse = str(landuse_dataset).strip().upper()
    if landuse == "IGBP":
        return 16
    return 19


def soil_veg_gen_parm(landuse_dataset: str = "USGS"):

    gen_parameters = {
        'SBETA_DATA': jnp.array(-2.0),
        'FXEXP_DATA': jnp.array(2.0),
        'CSOIL_DATA': jnp.array(2.00E+6),
        'SALP_DATA': jnp.array(2.6),
        'REFDK_DATA': jnp.array(2.0E-6),
        'REFKDT_DATA': jnp.array(3.0),
        'FRZK_DATA': jnp.array(0.15),
        'ZBOT_DATA': jnp.array(-8.0),
        'CZIL_DATA': jnp.array(0.1),
        'SMLOW_DATA': jnp.array(0.5),
        'SMHIGH_DATA': jnp.array(3.0),
        'LVCOEF_DATA': jnp.array(0.5),
        'TOPT_DATA': jnp.array(298.0),
        'CMCMAX_DATA': jnp.array(0.5E-3),
        'CFACTR_DATA': jnp.array(0.5),
        'RSMAX_DATA': jnp.array(5000.0),
        'BARE': resolve_bare_vegtype_index(landuse_dataset),
        'NATURAL': jnp.array(5),
        'SLOPE_DATA': [0.1, 0.6, 1.0, 0.35, 0.55, 0.8, 0.63, 0.0, 0.0]
    }
    return gen_parameters




def CALTMP(T1, SFCTMP, SFCPRS, ZLVL, Q2):
    """Technical documentation."""
    # Implementation detail.
    RD = 287.04; TH2 = SFCTMP + (0.0098 * ZLVL); T1V = T1 * (1.0 + 0.61 * Q2)
    TH2V = TH2 * (1.0 + 0.61 * Q2); T2V = SFCTMP * (1.0 + 0.61 * Q2)
    RHO = SFCPRS / (RD * T2V)
    return TH2, T1V, TH2V, T2V, RHO


def CALHUM(SFCTMP, SFCPRS):
    """Technical documentation."""
    # Implementation detail.
    A2 = 17.67; A3 = 273.15; A4 = 29.65; ELWV = 2.501e6
    A23M4 = A2 * (A3 - A4); E0 = 611.0; RV = 461.0; EPSILON = 0.622
    ES = E0 * jnp.exp(ELWV / RV * (1. / A3 - 1. / SFCTMP))
    Q2SAT = EPSILON * ES / (SFCPRS - (1 - EPSILON) * ES)
    DQSDT2 = Q2SAT * A23M4 / (SFCTMP - A4) ** 2
    return Q2SAT, DQSDT2


# --- (MODIFIED v1.3) ---
def create_step_function(forcing_data_jax, ZLVL_jax, ZLVL_WIND_jax, DT_jax, EMISSI_jax, Z0_jax, Z0BRD_jax,
                        CZIL_jax, CM_jax, SLDPTH_jax, ZSOIL_jax, SHDMIN_jax, SHDMAX_jax, SNOALB_jax,
                        TBOT_jax, STYPE_jax, VEGTYP_jax, SLOPETYP_jax, RDLAI2D_jax, USEMONALB_jax, SOC_jax,
                        SBETA_OPTION_jax, DF_OPTION_jax, ROOT_OPTION_jax, INF_OPTION_jax, SOC_OPTION_KS_jax,
                        SOC_OPTION2_THERMAL_jax, RIC_OPTION_jax, BLIM_OPTION_jax, CK_OPTION_jax, IZ0TLND_jax, sfcdif_option_jax,
                        albedo_monthly_idx, shdfac_monthly_idx, lai_monthly_idx, startdate_index, XLAI_init,
                        PSIM2_jax, PSIH2_jax, ZTMAX2_jax, DZETA2_jax, ZTMIN2_jax,
                        soil_params_dict,  # <-- UPDATED: usedictionary
                        veg_params_dict,
                        gen_params_dict,
                        steps_per_forcing: int,
                        date_months, date_days, date_years,
                        NROOT_jax: int,
                        # --- (NEW v1.6) ---
                        lai_factors_jax=None  # <-- LAIdynamic correction factor
                        # --- (END NEW) ---
                        ):
    """Technical documentation."""
    from Module_sf_noahlsm_jax_simple import SFLX

    # --- (NEW v1.6) LAIdynamic correction factorsettings ---
    use_lai_factors = lai_factors_jax is not None
    # --- (END NEW) ---

    def step_function(carry, idx):
        # --- Implementation details ---
        TRUNCATION_LENGTH = 720
        def detach_carry(c): return jax.tree_util.tree_map(jax.lax.stop_gradient, c)
        def identity_carry(c): return c
        is_truncation_step = (idx > 0) & (idx % TRUNCATION_LENGTH == 0)
        carry = jax.lax.cond(is_truncation_step, detach_carry, identity_carry, carry)
        # --- (end) ---

        (CMC, T1, STC, SMC, SH2O, SNOWH, SNEQV, SNOTIME1, ALBEDO, CH, CM, PC, XLAI, Q1,
         storage_m_prev, SFCTMP_prev, forcing_data_len, Z0, Z0BRD, EMISSI_jax) = carry

        STBOLT = jnp.array(5.67E-8, dtype=jnp.float32)

        # --- NEW FORTRAN-REPLICA INTERPOLATION LOGIC ---
        # This logic replicates the linear interpolation of forcing data found in the Fortran model.
        
        # Calculate indices for the 'before' and 'after' forcing data points for interpolation.
        forcing_idx_before = idx // steps_per_forcing
        forcing_idx_after = forcing_idx_before + 1

        # Clamp the 'after' index to the last valid index to avoid out-of-bounds errors.
        # When at the very last model step of the simulation, 'after' will equal 'before',
        # effectively resulting in no interpolation, using the last data point directly.
        forcing_idx_after = jnp.minimum(forcing_idx_after, forcing_data_len - 1)

        # Apply modulo for long simulations that loop over the forcing data.
        forcing_idx_before = forcing_idx_before % forcing_data_len
        forcing_idx_after = forcing_idx_after % forcing_data_len

        # Calculate interpolation weights. 'weight_before' corresponds to the Fortran 'fraction'.
        time_substep = idx % steps_per_forcing
        weight_after = time_substep / steps_per_forcing
        weight_before = 1.0 - weight_after

        # Fetch the 'before' and 'after' forcing data arrays.
        forcing_before = jnp.take(forcing_data_jax, forcing_idx_before, axis=0)
        forcing_after = jnp.take(forcing_data_jax, forcing_idx_after, axis=0)

        # Perform linear interpolation for all variables.
        specified_row = forcing_before * weight_before + forcing_after * weight_after

        # Per Fortran logic, precipitation is not interpolated but carried forward from the 'before' step.
        # The precipitation variable is at index 7 in the forcing data array.
        prcp_before = forcing_before[7]
        specified_row = specified_row.at[7].set(prcp_before)
        
        # Monthly varying parameters (like albedo, LAI) are also treated as step functions,
        # changing only at the beginning of a new forcing interval.
        current_k_time = forcing_idx_before
        # --- END OF NEW LOGIC ---

        SFCSPD = specified_row[0]
        WDIR = specified_row[1]
        SFCTMP = specified_row[2]
        RHF = specified_row[3] * 0.01
        SFCPRS = specified_row[4] * 100
        SOLDN = specified_row[5]
        LONGWAVE = specified_row[6]
        PRCP = specified_row[7]

        # --- Implementation details ---
        SFCU = -SFCSPD * jnp.sin(WDIR * jnp.pi / 180.0)
        SFCV = -SFCSPD * jnp.cos(WDIR * jnp.pi / 180.0)
        TO = 273.15; CPV = 1870.0; RV = 461.5; CW = 4187.0; ESO = 611.2; eps = 0.622
        LW = 2.501e6 - (CW - CPV) * (SFCTMP - TO)
        svp = ESO * jnp.exp(LW * (1.0 / TO - 1.0 / SFCTMP) / RV)
        QS = eps * svp / (SFCPRS - (1.0 - eps) * svp)
        E = (SFCPRS * svp * RHF) / (SFCPRS - svp * (1.0 - RHF))
        SPECHUMD = (eps * E) / (SFCPRS - (1.0 - eps) * E)
        Q2 = jnp.clip(SPECHUMD, jnp.array(0.1E-5, dtype=jnp.float32), QS * 0.99)
        # State-variable handling.
        Q1_from_carry = Q1  # inupdateQ1beforesave
        # State-variable handling.
        Q1 = jnp.where(idx == 0, Q2, Q1)
        FFROZP = jnp.where((PRCP > 0) & (SFCTMP < 273.15), 1, 0)
        TH2, T1V, TH2V, T2V, RHO = CALTMP(T1, SFCTMP, SFCPRS, ZLVL_jax, Q2)
        Q2SAT, DQSDT2 = CALHUM(SFCTMP, SFCPRS)
        # --- (end) ---

        # --- Implementation details ---
        def interpolate_monthly_params(param_monthly_vec, step_idx):
            nowm = date_months[step_idx]; nowd = date_days[step_idx]; nowy = date_years[step_idx]
            ndays = jnp.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=jnp.int32)
            is_leap_year = (nowy % 4 == 0) & ((nowy % 100 != 0) | (nowy % 400 == 0))
            is_february = (nowm == 2)
            ndays_adjusted = ndays.at[1].set(jnp.where(is_leap_year & is_february, 29, 28))
            is_day_15 = (nowd == 15); is_before_15 = (nowd < 15); is_after_15 = (nowd > 15)
            prevm_before = jnp.where(nowm > 1, nowm - 1, 12); postm_before = nowm
            prev_month_ndays = ndays_adjusted[(prevm_before - 1)]
            factor_before = (prev_month_ndays.astype(jnp.float32) - 15.0 + nowd.astype(jnp.float32)) / prev_month_ndays.astype(jnp.float32)
            prevm_after = nowm; postm_after = jnp.where(nowm < 12, nowm + 1, 1)
            this_month_ndays = ndays_adjusted[(prevm_after - 1)]
            factor_after = (nowd.astype(jnp.float32) - 15.0) / this_month_ndays.astype(jnp.float32)
            prevm = jnp.where(is_before_15, prevm_before, jnp.where(is_after_15, prevm_after, nowm))
            postm = jnp.where(is_before_15, postm_before, jnp.where(is_after_15, postm_after, nowm))
            factor = jnp.where(is_before_15, factor_before, jnp.where(is_after_15, factor_after, 0.0))
            prevm_idx = (prevm - 1); postm_idx = (postm - 1)
            result = param_monthly_vec[prevm_idx] * (1.0 - factor) + param_monthly_vec[postm_idx] * factor
            nowm_idx = (nowm - 1)
            return jnp.where(is_day_15, param_monthly_vec[nowm_idx], result)
        # --- (end) ---

        ALB = interpolate_monthly_params(albedo_monthly_idx, current_k_time)
        XLAI_base = interpolate_monthly_params(lai_monthly_idx, current_k_time)
        # --- (NEW v1.6) applyLAIdynamic correction factor ---
        if use_lai_factors:
            XLAI = XLAI_base * lai_factors_jax[idx]
        else:
            XLAI = XLAI_base
        # --- (END NEW) ---
        SHDFAC = interpolate_monthly_params(shdfac_monthly_idx, current_k_time)

        # --- Implementation details ---
        # State-variable handling.
        # State-variable handling.
        # State-variable handling.
        # State-variable handling.
        # Implementation detail.
        is_ekf_inherited = jnp.abs(Q1_from_carry) > 1e-10
        Q1_for_sfcdif = jnp.where(idx == 0,
                                   jnp.where(is_ekf_inherited, Q1_from_carry, Q2),  # State-variable handling.
                                   Q1)  # State-variable handling.

        # Define branches for jax.lax.cond
        def sfcdif_true_branch_inner(operands):
            # Unpack operands
            zlvl, zlvl_wind, z0, z0brd, sfcprs, t1, sfctmp, q1, q2, sfcspd, czil, cm_in, ch_in, iz0tlnd, \
            psim2, psih2, ztmax2, dzeta2, ztmin2 = operands
            # This corresponds to sfcdif_option_jax == 1
            ribb, cm_out, ch_out, rlmo = SFCDIF_MYJ(
                zlvl, zlvl_wind, z0, z0brd, sfcprs, t1, sfctmp, q1, q2, sfcspd, czil, cm_in, ch_in, iz0tlnd,
                psim2, psih2, ztmax2, dzeta2, ztmin2)
            return ribb, cm_out, ch_out, rlmo

        def sfcdif_false_branch_inner(operands):
            # Unpack operands
            zlvl, zlvl_wind, z0, z0brd, sfcprs, t1, sfctmp, q1, q2, sfcspd, czil, cm_in, ch_in, iz0tlnd, \
            _, _, _, _, _ = operands # Ignored sfcdif table vars
            # This corresponds to sfcdif_option_jax != 1
            ribb, ch_out, rlmo = SFCDIF_MYJ_Y08(z0, zlvl_wind, zlvl, sfcspd, t1, sfctmp, q2, sfcprs)
            cm_out = cm_in
            return ribb, cm_out, ch_out, rlmo

        # Package the operands for the conditional function
        sfcdif_operands = (ZLVL_jax, ZLVL_WIND_jax, Z0, Z0BRD, SFCPRS, T1, SFCTMP, Q1_for_sfcdif, Q2, SFCSPD, CZIL_jax, CM, CH, IZ0TLND_jax, \
                           PSIM2_jax, PSIH2_jax, ZTMAX2_jax, DZETA2_jax, ZTMIN2_jax)

        RIBB, CM_out, CH_out, RLMO = jax.lax.cond(
            sfcdif_option_jax == 1,
            sfcdif_true_branch_inner,
            sfcdif_false_branch_inner,
            sfcdif_operands
        )

        CH = CH_out
        CM = CM_out
        # --- (end) ---

        SOLNET = SOLDN * (1.0 - ALBEDO)
        LWDN = LONGWAVE * EMISSI_jax

        # --- Implementation details ---
        (CMC_out, T1_out, STC_out, SMC_out, SH2O_out,
         SNOWH_out, SNEQV_out, SNOTIME1_out, ALBEDO_out, CH_out_arg,
         ETA, SHEAT, ETA_KINEMATIC, FDOWN, EC_mass, EDIR_mass,
         ET_mass, ETT_mass, ESNOW_mass, DRIP, DEW, BETA, ETP,
         SSOIL_out_,
         FLX1, FLX2, FLX3, SNOMLT, SNCOVR, RUNOFF1,
         RUNOFF2, RUNOFF3, RC_out, PC_out, RSMIN_out, XLAI_out_arg,
         RCS_out, RCT_out, RCQ_out, RCSOIL_out,
         SOILW_out, SOILM_out, Q1_out_arg, SMAV_out,
         Z0_out_arg, Z0BRD_out_arg, EMISSI_out_arg) = SFLX(
            FFROZP, DT_jax, SLDPTH_jax, ZSOIL_jax, LWDN, SOLDN, SOLNET, SFCPRS, PRCP, SFCTMP, Q2, TH2,
            Q2SAT, DQSDT2, VEGTYP_jax, SLOPETYP_jax, SHDFAC, SHDMIN_jax, SHDMAX_jax,
            ALB, SNOALB_jax, TBOT_jax, CMC, T1, STC, SMC, SH2O, STYPE_jax, SNOWH, SNEQV, CH,
            PC, # State-variable handling.
            XLAI, RDLAI2D_jax, USEMONALB_jax, SNOTIME1, RIBB,
            None, idx, SOC_jax,
            soil_params_dict, veg_params_dict, gen_params_dict,
            SBETA_OPTION_jax, DF_OPTION_jax, ROOT_OPTION_jax, INF_OPTION_jax, SOC_OPTION_KS_jax,
            SOC_OPTION2_THERMAL_jax, RIC_OPTION_jax, BLIM_OPTION_jax, CK_OPTION_jax, RLMO,
            NROOT_jax)
        # --- (end) ---

        T1_final = T1_out

        # --- Implementation details ---
        LWUP = EMISSI_out_arg * STBOLT * (T1_out ** 4)
        R_in = FDOWN
        R_out = LWUP + SHEAT + ETA - SSOIL_out_ + FLX3 + FLX2 + FLX1
        energy_residual_step = R_in - R_out

        precip_m_step = (PRCP * DT_jax) / 1000.0
        et_m_step = (ETA_KINEMATIC * DT_jax) / 1000.0
        runoff_m_step = (RUNOFF1 + RUNOFF2) * DT_jax + RUNOFF3
        storage_m_curr_phys = CMC_out + SNEQV_out + jnp.sum(SOILM_out)
        dS_m_step = storage_m_curr_phys - storage_m_prev
        water_residual_step = precip_m_step - et_m_step - runoff_m_step - dS_m_step

        # applyphysical constraints, prevent state variables from leaving their valid ranges
        # temperature constraint (200K - 350K)
        STC_out_constrained = jnp.clip(STC_out, 200.0, 350.0)
        
        # get soil-parameter constraints (read directly from the dictionary)
        MAXSMC_local = soil_params_dict['MAXSMC'][STYPE_jax - 1]  # <-- UPDATED

        FTSOC = SOC_jax * 2700 * (1 - MAXSMC_local) / (SOC_jax * 2700 * (1 - MAXSMC_local) + (1 - SOC_jax) * 130 + 1e-9)
        SMCMAX_CORRECTED = (1 - FTSOC) * MAXSMC_local + FTSOC * 0.83
        # SMC constraint (0.02 - MAXSMC)
        SMC_out_constrained = jnp.clip(SMC_out, 0.02, SMCMAX_CORRECTED)
        
        # SH2O constraint (0.01 - SMC)
        # State-variable handling.
        SH2O_out_constrained = jnp.clip(SH2O_out, 0.01, SMC_out_constrained)

        storage_m_prev_for_next_step = storage_m_curr_phys

        # --- Implementation details ---
        outputs = {
            'STC_Kelvin': STC_out_constrained, # <--- Renamed
            'SH2O': SH2O_out_constrained,
            'SMC': SMC_out_constrained,
            'CMC': CMC_out, 'SNEQV': SNEQV_out, 'SNOWH': SNOWH_out, 'SHEAT': SHEAT,
            'ETA': ETA, 'SSOIL': SSOIL_out_, 'FDOWN': FDOWN, 'T1_final': T1_final,
            'EMISSI': EMISSI_out_arg, 'FLX1': FLX1, 'FLX2': FLX2, 'FLX3': FLX3,
            'Energy_Residual': energy_residual_step, 'Water_Residual': water_residual_step * 1000.0,
            'EC_mass': EC_mass, 'EDIR_mass': EDIR_mass, 'ET_mass': ET_mass, 'ETT_mass': ETT_mass,
            'ESNOW_mass': ESNOW_mass, 'DRIP': DRIP, 'DEW': DEW, 'BETA': BETA, 'ETP': ETP,
            'SNOMLT': SNOMLT, 'SNCOVR': SNCOVR, 'RUNOFF1': RUNOFF1, 'RUNOFF2': RUNOFF2,
            'RUNOFF3': RUNOFF3, 'RC': RC_out, 'RSMIN': RSMIN_out, 'RCS': RCS_out,
            'RCT': RCT_out, 'RCQ': RCQ_out, 'RCSOIL': RCSOIL_out, 'SOILW': SOILW_out,
            'SOILM': SOILM_out, 'SMAV': SMAV_out, 'XLAI_out': XLAI_out_arg,
            'Z0_out': Z0_out_arg, 'Z0BRD_out': Z0BRD_out_arg, 'ETA_KINEMATIC': ETA_KINEMATIC,
            'PRCP': PRCP, 'SOLDN': SOLDN, 'LONGWAVE': LONGWAVE, 'LWDN': LWDN, 'ALBEDO': ALBEDO_out,
            'Q1': Q1_out_arg,  # <--- (NEW v1.5) add Q1 to the output, usingsupport inheritance in EKF
            # State-variable handling.
            'CH': CH_out, 'CM': CM_out, 'PC': PC_out, 'SNOTIME1': SNOTIME1_out,
            'SFCTMP': SFCTMP, 'storage_m_prev': storage_m_prev_for_next_step
        }

        # --- Implementation details ---
        new_carry = (CMC_out, T1_out, STC_out_constrained, SMC_out_constrained, SH2O_out_constrained,
                     SNOWH_out, SNEQV_out, SNOTIME1_out, ALBEDO_out, CH_out, CM_out, PC_out,
                     XLAI_out_arg, Q1_out_arg,
                     storage_m_prev_for_next_step, # Output handling.
                     SFCTMP, forcing_data_len, Z0_out_arg, Z0BRD_out_arg, EMISSI_out_arg)

        return new_carry, outputs

    return step_function


# --- (MODIFIED v1.1) ---
@partial(jax.jit, static_argnames=(
    'NSOIL', 'sfcdif_option_jax', 'VEGTYP_jax', 'SLOPETYP_jax',
    'SBETA_OPTION_jax', 'DF_OPTION_jax', 'ROOT_OPTION_jax',
    'INF_OPTION_jax', 'SOC_OPTION_KS_jax',
    'SOC_OPTION2_THERMAL_jax', 'RIC_OPTION_jax',
    'BLIM_OPTION_jax', 'CK_OPTION_jax', 'IZ0TLND_jax',
    'total_steps', 'forcing_data_len', 'steps_per_forcing',
    'NROOT_jax'
))
def run_noah_simulation_pure(
    forcing_data_jax,
    NSOIL,
    SLDPTH_jax,
    ZSOIL_jax,
    DT_jax,
    EMISSI_jax,
    ALBEDO_init,
    Z0_init,
    Z0BRD_init,
    CZIL_jax,
    CH_init,
    CM_init,
    sfcdif_option_jax: int,
    # Initial State
    CMC_init, T1_init, STC_init, SMC_init, SH2O_init,
    SNOWH_init, SNEQV_init, TBOT_jax,
    # Static Config
    ZLVL_jax, ZLVL_WIND_jax, SHDMIN_jax, SHDMAX_jax, SNOALB_jax,
    STYPE_jax, VEGTYP_jax: int, SLOPETYP_jax: int, RDLAI2D_jax, USEMONALB_jax, SOC_jax,
    # Options (Static)
    SBETA_OPTION_jax: int, DF_OPTION_jax: int, ROOT_OPTION_jax: int, INF_OPTION_jax: int,
    SOC_OPTION_KS_jax: int, SOC_OPTION2_THERMAL_jax: int, RIC_OPTION_jax: int,
    BLIM_OPTION_jax: int, CK_OPTION_jax: int, IZ0TLND_jax: int,
    # Monthly data
    albedo_monthly_idx, shdfac_monthly_idx, lai_monthly_idx,
    XLAI_init_jax,
    # Date arrays
    date_months, date_days, date_years,
    # Loop config
    total_steps: int,
    forcing_data_len: int,
    # Parameter objects
    soil_params_jax: dict,
    veg_params_dict: dict,
    gen_params_dict: dict,
    steps_per_forcing: int,
    NROOT_jax: int,
    # --- (NEW v1.5) ---
    Q1_init=None,  # <-- Q1initial value(optional, defaults to Nonewhenuse0.0)
    # --- (END NEW) ---
    # --- (NEW v1.6) ---
    lai_factors=None  # <-- LAIdynamic correction factor(optional, with shape (total_steps,))
    # --- (END NEW) ---
):
    """Technical documentation."""
    badval = jnp.array(-1.E36, dtype=jnp.float32)

    # --- Implementation details ---
    # Jacobian and gradient handling.
    # Jacobian and gradient handling.
    STC_init = jnp.clip(STC_init, 200.0, 350.0)     # temperaturerange (K)
    SMC_init = jnp.clip(SMC_init, 0.01, 0.6)        # soilwater contentrange
    SH2O_init = jnp.clip(SH2O_init, 0.01, 0.6)      # liquid waterrange
    CMC_init = jnp.clip(CMC_init, 0.0, 0.01)        # Implementation detail.
    T1_init = jnp.clip(T1_init, 200.0, 350.0)       # Implementation detail.
    SNOWH_init = jnp.clip(SNOWH_init, 0.0, 10.0)    # Implementation detail.
    SNEQV_init = jnp.clip(SNEQV_init, 0.0, 5.0)     # Implementation detail.
    
    # State-variable handling.
    SH2O_init = jnp.minimum(SH2O_init, SMC_init)
    # --- (fixend) ---

    # --- (UPDATED) ---
    # Model-parameter handling.
    # soil_params_jax structure:
    # {'BB': ..., 'MAXSMC': ..., 'SATDK': ..., 'SATPSI': ..., 'QTZ': ..., 'BLIMBX': ...}
    # Model-parameter handling.
    # --- (END UPDATED) ---

    # --- Implementation details ---
    def _init_sfcdif_true():
        return MYJSFCINIT()

    def _init_sfcdif_false():
        # Return dummy values with correct shapes and dtypes
        # to match the true branch for JIT compatibility.
        # KZTM is imported from Module_sfcdif_wrf_jax_simple
        psim2 = jnp.zeros(KZTM, dtype=jnp.float32)
        psih2 = jnp.zeros(KZTM, dtype=jnp.float32)
        ztmax2 = jnp.array(0.0, dtype=jnp.float32)
        dzeta2 = jnp.array(0.0, dtype=jnp.float32)
        ztmin2 = jnp.array(0.0, dtype=jnp.float32)
        return psim2, psih2, ztmax2, dzeta2, ztmin2

    PSIM2_jax, PSIH2_jax, ZTMAX2_jax, DZETA2_jax, ZTMIN2_jax = jax.lax.cond(
        sfcdif_option_jax == 1,
        _init_sfcdif_true,
        _init_sfcdif_false,
    )
    # --- (end) ---

    # --- Implementation details ---
    # (*** modifiedfor JAX-style calculate ***)
    DZ = jnp.diff(jnp.concatenate([jnp.array([0.0]), -ZSOIL_jax]))
    initial_SOILM = jnp.sum(SMC_init * DZ)
    initial_storage_m = CMC_init + SNEQV_init + initial_SOILM
    # --- (end) ---

    # --- Implementation details ---
    XLAI_init = XLAI_init_jax
    PC_init = jnp.array(badval, dtype=jnp.float32)
    SNOTIME1_init = jnp.array(0.0, dtype=jnp.float32)
    # --- Implementation details ---
    if Q1_init is None:
        Q1_init = jnp.array(0.0, dtype=jnp.float32)  # Implementation detail.
    else:
        Q1_init = jnp.asarray(Q1_init, dtype=jnp.float32)  # State-variable handling.
    # --- (end) ---

    # --- Implementation details ---
    initial_carry = (CMC_init.astype(jnp.float32), T1_init.astype(jnp.float32),
                     STC_init.astype(jnp.float32), SMC_init.astype(jnp.float32), SH2O_init.astype(jnp.float32),
                     SNOWH_init.astype(jnp.float32), SNEQV_init.astype(jnp.float32),
                     SNOTIME1_init, ALBEDO_init.astype(jnp.float32), CH_init.astype(jnp.float32), CM_init.astype(jnp.float32),
                     PC_init, XLAI_init.astype(jnp.float32), Q1_init, initial_storage_m.astype(jnp.float32),
                     jnp.array(0.0, dtype=jnp.float32), # SFCTMP_prev
                     forcing_data_len,
                     Z0_init.astype(jnp.float32), Z0BRD_init.astype(jnp.float32), EMISSI_jax.astype(jnp.float32))
    # --- (end) ---

    # --- Implementation details ---
    step_func = create_step_function(
        forcing_data_jax, ZLVL_jax, ZLVL_WIND_jax, DT_jax, EMISSI_jax, Z0_init, Z0BRD_init,
        CZIL_jax, CM_init, SLDPTH_jax, ZSOIL_jax, SHDMIN_jax, SHDMAX_jax, SNOALB_jax,
        TBOT_jax, STYPE_jax, VEGTYP_jax, SLOPETYP_jax, RDLAI2D_jax, USEMONALB_jax, SOC_jax,
        SBETA_OPTION_jax, DF_OPTION_jax, ROOT_OPTION_jax, INF_OPTION_jax, SOC_OPTION_KS_jax,
        SOC_OPTION2_THERMAL_jax, RIC_OPTION_jax, BLIM_OPTION_jax, CK_OPTION_jax, IZ0TLND_jax, sfcdif_option_jax,
        albedo_monthly_idx, shdfac_monthly_idx, lai_monthly_idx, 0, XLAI_init,
        PSIM2_jax, PSIH2_jax, ZTMAX2_jax, DZETA2_jax, ZTMIN2_jax,
        soil_params_jax, veg_params_dict, gen_params_dict,  # <-- UPDATED: pass the dictionary directly
        steps_per_forcing,
        date_months, date_days, date_years,
        NROOT_jax,
        # --- (NEW v1.6) ---
        lai_factors  # Implementation detail.
        # --- (END NEW) ---
    )
    # --- (end) ---

    # --- Implementation details ---
    final_carry, all_outputs = jax.lax.scan(step_func, initial_carry, jnp.arange(total_steps))
    # --- (end) ---

    # --- Implementation details ---
    # (*** modified ***) no longer truncate outputs, return complete results for all time steps
    # Output handling.
    final_outputs = all_outputs
    # --- (end) ---

    return final_outputs


# --- (REFACTORED) MAIN FUNCTION (MODIFIED v1.2) ---
def noah_main(file_name, trained_parameter_dict=None, output_flag=False, output_filename=None):
    """Technical documentation."""
    jax.clear_caches()
    badval = -1.E36

    # --- Implementation details ---
    (Date, forcing_data, output_dir, forcing_filename, infotext, NSOIL, startdate, enddate, loop_for_a_while,
     latitude, longitude, forcing_timestep, noahlsm_timestep, ice, T1, STC, SMC, SH2O, STYPE,
     SLDPTH, CMC, SNOWH, SNEQV, TBOT, VEGTYP, SOILTYP, SLOPETYP, SNOALB, ZLVL, ZLVL_WIND,
     albedo_monthly, shdfac_monthly, z0brd_monthly, lai_monthly, use_urban_module, ISURBAN,
     SHDMIN, SHDMAX, USEMONALB, RDLAI2D, LLANDUSE, SBETA_OPTION, DF_OPTION, ROOT_OPTION, INF_OPTION, SOC_OPTION_KS,
     SOC_OPTION2_THERMAL, RIC_OPTION, BLIM_OPTION, CK_OPTION, IZ0TLND, sfcdif_option, SOC) = open_forcing_file(file_name)

    # select the VEGPARM table for Landuse_dataset
    if LLANDUSE.upper() == "USGS":
        veg_param_path = "parameter_new/VEGPARM-USGS.TBL"
    elif LLANDUSE.upper() == "IGBP":
        veg_param_path = "parameter_new/VEGPARM-IGBP.TBL"
    soil_param_path = "parameter_new/SOILPARM.TBL"
    try:
        veg_parameter_df = pd.read_csv(veg_param_path, sep=r',\s*', engine='python', header=0, index_col=0, usecols=range(18), dtype=np.float32)
        soil_parameter_df = pd.read_csv(soil_param_path, sep=r',\s*', engine='python', header=0, index_col=0, usecols=range(11), dtype=np.float32)
    except FileNotFoundError as e:
        print(f"錯誤：找不到參數檔案：{e}"); return

    soil_params_jax = {
        k: jnp.array(soil_parameter_df[k].to_numpy(), dtype=jnp.float32)
        for k in ['BB', 'MAXSMC', 'SATDK', 'SATPSI', 'QTZ']
    }
    soil_params_jax['BLIMBX'] = jnp.full_like(soil_params_jax['BB'], 4.0)
    if trained_parameter_dict is not None:
        for key, value in trained_parameter_dict.items():
            if key in soil_params_jax: soil_params_jax[key] = value

    veg_params_dict = {
        'NROOT': int(veg_parameter_df.at[VEGTYP, 'NROOT']),
        'SNUP': jnp.array(veg_parameter_df.at[VEGTYP, 'SNUP'], dtype=jnp.float32),
        'RS': jnp.array(veg_parameter_df.at[VEGTYP, 'RS'], dtype=jnp.float32),
        'RGL': jnp.array(veg_parameter_df.at[VEGTYP, 'RGL'], dtype=jnp.float32),
        'HS': jnp.array(veg_parameter_df.at[VEGTYP, 'HS'], dtype=jnp.float32),
        'EMISSMIN': jnp.array(veg_parameter_df.at[VEGTYP, 'EMISSMIN'], dtype=jnp.float32),
        'EMISSMAX': jnp.array(veg_parameter_df.at[VEGTYP, 'EMISSMAX'], dtype=jnp.float32),
        'LAIMIN': jnp.array(veg_parameter_df.at[VEGTYP, 'LAIMIN'], dtype=jnp.float32),
        'LAIMAX': jnp.array(veg_parameter_df.at[VEGTYP, 'LAIMAX'], dtype=jnp.float32),
        'Z0MIN': jnp.array(veg_parameter_df.at[VEGTYP, 'Z0MIN'], dtype=jnp.float32),
        'Z0MAX': jnp.array(veg_parameter_df.at[VEGTYP, 'Z0MAX'], dtype=jnp.float32),
        'ALBEDOMIN': jnp.array(veg_parameter_df.at[VEGTYP, 'ALBEDOMIN'], dtype=jnp.float32),
        'ALBEDOMAX': jnp.array(veg_parameter_df.at[VEGTYP, 'ALBEDOMAX'], dtype=jnp.float32),
        'ROOTA': jnp.array(veg_parameter_df.at[VEGTYP, 'ROOTA'], dtype=jnp.float32),
        'ROOTB': jnp.array(veg_parameter_df.at[VEGTYP, 'ROOTB'], dtype=jnp.float32),
    }

    gen_params_dict = soil_veg_gen_parm(LLANDUSE)
    gen_params_dict['SLOPE_DATA'] = jnp.array(gen_params_dict['SLOPE_DATA'], dtype=jnp.float32)
    # --- (end) ---

    # --- Implementation details ---
    ZSOIL = jnp.zeros_like(SH2O, dtype=jnp.float32); ZSOIL = ZSOIL.at[0].set(-SLDPTH[0])
    for i in range(1, NSOIL): ZSOIL = ZSOIL.at[i].set(-SLDPTH[i] + ZSOIL[i - 1])
    DT = jnp.array(noahlsm_timestep, dtype=jnp.float32)
    EMISSI = jnp.array(0.96, dtype=jnp.float32)
    ALBEDO = month_d(albedo_monthly, startdate)
    Z0 = month_d(z0brd_monthly, startdate)
    Z0BRD = Z0 if sfcdif_option == 1 else jnp.array(badval, dtype=jnp.float32)
    CZIL = gen_params_dict['CZIL_DATA']
    CH = jnp.array(1.E-4, dtype=jnp.float32)
    CM = jnp.array(1.E-4, dtype=jnp.float32)
    Date_filtered = Date[Date >= startdate]
    date_months = jnp.array([d.month for d in Date_filtered], dtype=jnp.int32)
    date_days = jnp.array([d.day for d in Date_filtered], dtype=jnp.int32)
    date_years = jnp.array([d.year for d in Date_filtered], dtype=jnp.int32)
    forcing_data_len = len(forcing_data)

    steps_per_forcing = 1
    if noahlsm_timestep > 0 and forcing_timestep > noahlsm_timestep:
        if forcing_timestep % noahlsm_timestep != 0:
            print("Warning: Forcing_Timestep is not a multiple of Noahlsm_Timestep. This may lead to unexpected behavior.")
        steps_per_forcing = int(forcing_timestep // noahlsm_timestep)

    total_steps = (int(loop_for_a_while * forcing_data_len) + forcing_data_len) * steps_per_forcing

    num_output_points = forcing_data_len * steps_per_forcing
    output_freq = pd.to_timedelta(noahlsm_timestep, unit='s')
    all_output_dates = pd.date_range(start=startdate, periods=num_output_points, freq=output_freq)
    Date_for_output = all_output_dates[all_output_dates <= enddate]

    albedo_monthly_idx = jnp.array(albedo_monthly[:12], dtype=jnp.float32)
    shdfac_monthly_idx = jnp.array(shdfac_monthly[:12], dtype=jnp.float32)
    lai_monthly_idx = jnp.array(lai_monthly[:12], dtype=jnp.float32)
    XLAI_init = jnp.where(RDLAI2D.astype(bool), month_d(lai_monthly, startdate), jnp.array(badval, dtype=jnp.float32))
    # --- (end) ---

    # --- Implementation details ---
    final_outputs = run_noah_simulation_pure(
        forcing_data_jax=forcing_data, NSOIL=NSOIL, SLDPTH_jax=SLDPTH.astype(jnp.float32), ZSOIL_jax=ZSOIL, DT_jax=DT,
        EMISSI_jax=EMISSI, ALBEDO_init=ALBEDO.astype(jnp.float32), Z0_init=Z0.astype(jnp.float32), Z0BRD_init=Z0BRD, CZIL_jax=CZIL.astype(jnp.float32),
        CH_init=CH, CM_init=CM, sfcdif_option_jax=int(sfcdif_option),
        CMC_init=CMC.astype(jnp.float32), T1_init=T1.astype(jnp.float32),
        STC_init=STC.astype(jnp.float32), SMC_init=SMC.astype(jnp.float32), SH2O_init=SH2O.astype(jnp.float32),
        SNOWH_init=SNOWH.astype(jnp.float32), SNEQV_init=SNEQV.astype(jnp.float32),
        TBOT_jax=TBOT.astype(jnp.float32),
        ZLVL_jax=ZLVL.astype(jnp.float32), ZLVL_WIND_jax=ZLVL_WIND.astype(jnp.float32),
        SHDMIN_jax=SHDMIN.astype(jnp.float32), SHDMAX_jax=SHDMAX.astype(jnp.float32),
        SNOALB_jax=SNOALB.astype(jnp.float32), STYPE_jax=STYPE, VEGTYP_jax=int(VEGTYP), SLOPETYP_jax=int(SLOPETYP),
        RDLAI2D_jax=RDLAI2D.astype(jnp.bool_), USEMONALB_jax=USEMONALB.astype(jnp.bool_),
        SOC_jax=SOC.astype(jnp.float32),
        SBETA_OPTION_jax=int(SBETA_OPTION), DF_OPTION_jax=int(DF_OPTION), ROOT_OPTION_jax=int(ROOT_OPTION),
        INF_OPTION_jax=int(INF_OPTION), SOC_OPTION_KS_jax=int(SOC_OPTION_KS),
        SOC_OPTION2_THERMAL_jax=int(SOC_OPTION2_THERMAL), RIC_OPTION_jax=int(RIC_OPTION),
        BLIM_OPTION_jax=int(BLIM_OPTION), CK_OPTION_jax=int(CK_OPTION), IZ0TLND_jax=int(IZ0TLND),
        albedo_monthly_idx=albedo_monthly_idx, shdfac_monthly_idx=shdfac_monthly_idx, lai_monthly_idx=lai_monthly_idx,
        XLAI_init_jax=XLAI_init,
        date_months=date_months, date_days=date_days, date_years=date_years,
        total_steps=total_steps, forcing_data_len=forcing_data_len,
        soil_params_jax=soil_params_jax, veg_params_dict=veg_params_dict, gen_params_dict=gen_params_dict,
        steps_per_forcing=steps_per_forcing,
        NROOT_jax=int(veg_params_dict['NROOT'])
    )
    # --- (end) ---

    # --- Implementation details ---
    if output_flag:
        SH2O_columns = [f'SH2O({i + 1})' for i in range(NSOIL)]
        STC_columns = [f'STC_Kelvin({i + 1})' for i in range(NSOIL)]
        SMC_columns = [f'SMC({i + 1})' for i in range(NSOIL)]
        out_columns = (STC_columns + SH2O_columns + SMC_columns +
                      ["SNEQV", "SNOWH", "SHEAT", "ETA", "SSOIL",
                       "Water_Balance_Residual_mm", "Energy_Balance_Residual_Wm2",
                       "FDOWN", "T1_final", "EMISSI", "FLX1", "FLX2", "FLX3",
                       "RUNOFF1", "RUNOFF2", "RUNOFF3", "PRCP", "SOLDN", "LONGWAVE","LWDN", "ALBEDO"])
        out_tensors = []

        stc_data = final_outputs['STC_Kelvin']
        for i in range(NSOIL): out_tensors.append(stc_data[:, i:i+1])

        sh2o_data = final_outputs['SH2O']
        for i in range(NSOIL): out_tensors.append(sh2o_data[:, i:i+1])
        smc_data = final_outputs['SMC']
        for i in range(NSOIL): out_tensors.append(smc_data[:, i:i+1])

        out_tensors.append(np.array(final_outputs['SNEQV']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['SNOWH']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['SHEAT']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['ETA']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['SSOIL']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['Water_Residual']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['Energy_Residual']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['FDOWN']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['T1_final']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['EMISSI']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['FLX1']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['FLX2']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['FLX3']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['RUNOFF1']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['RUNOFF2']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['RUNOFF3']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['PRCP']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['SOLDN']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['LONGWAVE']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['LWDN']).reshape(-1, 1))
        out_tensors.append(np.array(final_outputs['ALBEDO']).reshape(-1, 1))

        out = np.concatenate(out_tensors, axis=1)
        # supportcustom output filename
        if output_filename is not None:
            output_path = output_filename
        else:
            output_path = os.path.join(output_dir, "Noah_output_jax.txt")

        if out.shape[0] > len(Date_for_output):
            out = out[:len(Date_for_output)]

        if len(Date_for_output) != out.shape[0]:
            print(f"Warning: Length mismatch between dates ({len(Date_for_output)}) and output data ({out.shape[0]}). Truncating dates.")
            if len(Date_for_output) > out.shape[0]:
                 Date_for_output = Date_for_output[:out.shape[0]]

        # Output handling.
        if len(Date_for_output) == out.shape[0]:
            output_df = pd.DataFrame(out, columns=out_columns, index=Date_for_output)
            output_df.to_csv(output_path, index=True, index_label="Date", sep='\t', float_format='%.15f')
            print(f"output file written in: {output_path}")
        else:
            print("Error: Final date index length does not match output data length. File not written.")

    # --- Implementation details ---
    stc_result = final_outputs['STC_Kelvin']
    sh2o_result = final_outputs['SH2O']
    if output_flag:
        return Date_for_output, stc_result, sh2o_result, final_outputs
    else:
        return Date_for_output, stc_result, sh2o_result
    # --- (end) ---
