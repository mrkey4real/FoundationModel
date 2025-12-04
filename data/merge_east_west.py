from pathlib import Path

import numpy as np
import pandas as pd


def merge_two_weather() -> None:
    here = Path(__file__).resolve().parent

    east_file = here / "merged_east_labview.csv"
    west_file = here / "merged_west_labview.csv"

    east_df = pd.read_csv(east_file, index_col=0, parse_dates=True)
    west_df = pd.read_csv(west_file, index_col=0, parse_dates=True)

    east_columns = [
        "Air Temperature [degC]",
        "Relative Humidity [%]",
        "Relative Air Pressure [kPa]",
        "Wind Direction [deg]",
        "Dew Point [degC]",
        "Wind Chill Temperature [degC]",
        "Wind Speed [m/s]",
    ]

    west_columns = [
        "Total [W/m2]",
        "Total Irradiance [W/m2]",
        "Diffuse [W/m2]",
        "Diffuse Irradiance [W/m2]"
    ]

    east_available = [c for c in east_columns if c in east_df.columns]
    west_available = [c for c in west_columns if c in west_df.columns]

    if len(east_available) == 0 and len(west_available) == 0:
        raise ValueError("No requested columns found in either East or West data.")

    east_sel = east_df[east_available] if len(east_available) > 0 else pd.DataFrame(index=east_df.index)
    west_sel = west_df[west_available] if len(west_available) > 0 else pd.DataFrame(index=west_df.index)

    merged = pd.concat([east_sel, west_sel], axis=1)
    merged.sort_index(inplace=True)

    # Merge columns with different names into unified columns
    if "Total Irradiance [W/m2]" in merged.columns:
        if "Total [W/m2]" not in merged.columns:
            merged["Total [W/m2]"] = merged["Total Irradiance [W/m2]"]
        else:
            merged["Total [W/m2]"] = merged["Total [W/m2]"].fillna(merged["Total Irradiance [W/m2]"])
        merged.drop(columns=["Total Irradiance [W/m2]"], inplace=True)

    if "Diffuse Irradiance [W/m2]" in merged.columns:
        if "Diffuse [W/m2]" not in merged.columns:
            merged["Diffuse [W/m2]"] = merged["Diffuse Irradiance [W/m2]"]
        else:
            merged["Diffuse [W/m2]"] = merged["Diffuse [W/m2]"].fillna(merged["Diffuse Irradiance [W/m2]"])
        merged.drop(columns=["Diffuse Irradiance [W/m2]"], inplace=True)

    out_file = here / "two_merged_weather.csv"
    merged.to_csv(out_file, index=True)


def _find_time_col(cols):
    candidates = [
        "Time",
        "time",
        "datetime",
        "DateTime",
        "timestamp",
        "date",
        "Date",
    ]
    for k in candidates:
        if k in cols:
            return k
    for c in cols:
        if "time" in str(c).lower():
            return c
    raise ValueError("No datetime column found.")


def _parse_to_local_minute(s, tz_local: str):
    t = pd.to_datetime(s, errors="coerce")
    tz_attr = getattr(t.dt, "tz", None)
    if tz_attr is not None:
        t = t.dt.tz_convert(tz_local).dt.tz_localize(None)
    else:
        t = t.dt.tz_localize(None)
    return t.dt.floor("min")


def fill_with_open_meteo(base_csv: str,
                         om_csv: str,
                         out_csv: str,
                         prefix: str = "OM_",
                         tz_local: str = "America/Chicago",
                         year: int | None = None):
    base = pd.read_csv(base_csv)
    om = pd.read_csv(om_csv)

    bt = _find_time_col(base.columns)
    ot = _find_time_col(om.columns)

    base_time = _parse_to_local_minute(base[bt], tz_local)
    base = (base.loc[~base_time.isna()]
                 .assign(Time=base_time)
                 .drop_duplicates("Time")
                 .sort_values("Time")
                 .set_index("Time"))

    om_time = _parse_to_local_minute(om[ot], tz_local)
    om = (om.loc[~om_time.isna()]
               .assign(Time=om_time)
               .drop_duplicates("Time")
               .sort_values("Time")
               .set_index("Time"))

    if om.shape[0] == 0:
        out = base.reset_index()
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        return out

    om = om.apply(pd.to_numeric, errors="ignore")
    num_cols = om.select_dtypes(include="number").columns.tolist()
    if len(num_cols) == 0:
        om_1min = pd.DataFrame(index=base.index)
    else:
        om_1min = (om[num_cols]
                   .resample("1min").asfreq()
                   .interpolate(method="time")
                   .ffill()
                   .bfill())

    # --- Build annual 1-minute index from Jan 1 00:00 to next Jan 1 00:00 ---
    if year is None:
        years = []
        if base.index.size > 0:
            years.append(base.index.min().year)
        if om.index.size > 0:
            years.append(om.index.min().year)
        if len(years) == 0:
            raise ValueError("Cannot infer target year for annual index.")
        year = min(years)

    start = pd.Timestamp(year=year, month=1, day=1, hour=0, minute=0)
    end_exclusive = pd.Timestamp(year=year + 1, month=1, day=1, hour=0, minute=0)
    annual_index = pd.date_range(start, end_exclusive - pd.Timedelta(minutes=1), freq="1min")

    base = base.reindex(annual_index)
    om_1min = om_1min.reindex(annual_index)

    def _find_first_in(cols, candidates):
        for c in cols:
            s = str(c).lower().replace(" ", "").replace("_", "")
            for k in candidates:
                if k in s:
                    return c
        return None

    # --- Direct mapping ---
    raw_cols = list(om.columns)
    c_temp = _find_first_in(raw_cols, ["temperature2m", "airtemperature", "temp", "t2m"])
    c_rh = _find_first_in(raw_cols, ["relativehumidity2m", "humidity"])
    c_press = _find_first_in(raw_cols, ["surfacepressure", "pressuremsl", "mslpressure", "sealevelpressure"]) 
    c_wdir = _find_first_in(raw_cols, ["winddirection10m", "winddir"])
    c_dew = _find_first_in(raw_cols, ["dewpoint2m", "dewpoint"]) 
    c_chill = _find_first_in(raw_cols, ["apparenttemperature", "windchill"]) 
    c_wspd = _find_first_in(raw_cols, ["windspeed10m", "windspeed_10m", "windspeed", "wind_speed10m", "wind_speed_10m"]) 
    c_dif = _find_first_in(raw_cols, ["diffuseradiation", "dhi", "diffuseirradiance"]) 
    c_dir = _find_first_in(raw_cols, ["directradiation", "beamradiation", "dirrad"]) 
    c_ghi = _find_first_in(raw_cols, ["shortwaveradiation", "globalradiation", "globalirradiance", "ghi"]) 
    c_dni = _find_first_in(raw_cols, ["directnormalirradiance", "directnormal", "dni", "direct_normal_irradiance", "dirnorm", "dirnor"]) 
    c_rain = _find_first_in(raw_cols, ["rain", "rainsum", "rain_sum", "precipitation", "precip", "precipitationsum", "precipitation_sum"]) 

    source_cols = [c for c in [c_temp, c_rh, c_press, c_wdir, c_dew, c_chill, c_wspd, c_dif, c_dir, c_ghi, c_dni, c_rain] if c is not None]
    if len(source_cols) > 0:
        om_sel = om[source_cols].apply(pd.to_numeric, errors="coerce")
        om_1min = (om_sel
                   .resample("1min").asfreq()
                   .interpolate(method="time")
                   .ffill()
                   .bfill())
        om_1min = om_1min.reindex(base.index)
    else:
        om_1min = pd.DataFrame(index=base.index)

    # --- Prepare OM series with unit normalization ---
    if c_temp is not None and c_temp in om_1min.columns:
        s = om_1min[c_temp]
        base["Air Temperature [degC]"] = base["Air Temperature [degC]"] if "Air Temperature [degC]" in base.columns else np.nan
        base["Air Temperature [degC]"] = base["Air Temperature [degC]"].combine_first(s)

    if c_rh is not None and c_rh in om_1min.columns:
        s = om_1min[c_rh]
        base["Relative Humidity [%]"] = base["Relative Humidity [%]"] if "Relative Humidity [%]" in base.columns else np.nan
        base["Relative Humidity [%]"] = base["Relative Humidity [%]"].combine_first(s)

    if c_press is not None and c_press in om_1min.columns:
        s = om_1min[c_press] / 10.0
        base["Relative Air Pressure [kPa]"] = base["Relative Air Pressure [kPa]"] if "Relative Air Pressure [kPa]" in base.columns else np.nan
        base["Relative Air Pressure [kPa]"] = base["Relative Air Pressure [kPa]"].combine_first(s)

    if c_wdir is not None and c_wdir in om_1min.columns:
        s = om_1min[c_wdir]
        base["Wind Direction [deg]"] = base["Wind Direction [deg]"] if "Wind Direction [deg]" in base.columns else np.nan
        base["Wind Direction [deg]"] = base["Wind Direction [deg]"].combine_first(s)

    if c_wspd is not None and c_wspd in om_1min.columns:
        s = om_1min[c_wspd]
        name = str(c_wspd).lower()
        if ("windspeed" in name and "wind_speed" not in name) or "km" in name or "kph" in name:
            s = s * 0.2777777778
        base["Wind Speed [m/s]"] = base["Wind Speed [m/s]"] if "Wind Speed [m/s]" in base.columns else np.nan
        base["Wind Speed [m/s]"] = base["Wind Speed [m/s]"].combine_first(s)

    # Dew point: prefer direct column; if missing, compute via Magnus
    dew_series = None
    if c_dew is not None and c_dew in om_1min.columns:
        dew_series = om_1min[c_dew]
    else:
        if (c_temp is not None and c_rh is not None) and (c_temp in om_1min.columns and c_rh in om_1min.columns):
            T = om_1min[c_temp]
            RH = om_1min[c_rh]
            a = 17.27
            b = 237.7
            gamma = (a * T / (b + T)) + np.log(RH / 100.0)
            dew_series = (b * gamma) / (a - gamma)
    if dew_series is not None:
        base["Dew Point [degC]"] = base["Dew Point [degC]"] if "Dew Point [degC]" in base.columns else np.nan
        base["Dew Point [degC]"] = base["Dew Point [degC]"].combine_first(dew_series)

    # Wind chill: prefer apparent temperature; else compute when feasible
    chill_series = None
    if c_chill is not None and c_chill in om_1min.columns:
        chill_series = om_1min[c_chill]
    else:
        if (c_temp is not None and c_wspd is not None) and (c_temp in om_1min.columns and c_wspd in om_1min.columns):
            T = om_1min[c_temp]
            Vms = om_1min[c_wspd]
            name = str(c_wspd).lower()
            if ("windspeed" in name and "wind_speed" not in name) or "km" in name or "kph" in name:
                Vkmh = Vms
            else:
                Vkmh = Vms * 3.6
            chill_series = 13.12 + 0.6215 * T - 11.37 * (Vkmh ** 0.16) + 0.3965 * T * (Vkmh ** 0.16)
    if chill_series is not None:
        base["Wind Chill Temperature [degC]"] = base["Wind Chill Temperature [degC]"] if "Wind Chill Temperature [degC]" in base.columns else np.nan
        base["Wind Chill Temperature [degC]"] = base["Wind Chill Temperature [degC]"].combine_first(chill_series)

    # Radiation: Diffuse, Direct, GHI -> fill Diffuse, compute Total
    diffuse_series = om_1min[c_dif] if (c_dif is not None and c_dif in om_1min.columns) else None
    if diffuse_series is None and (c_ghi is not None and c_dir is not None) and (c_ghi in om_1min.columns and c_dir in om_1min.columns):
        diffuse_series = om_1min[c_ghi] - om_1min[c_dir]
    if diffuse_series is not None:
        base["Diffuse [W/m2]"] = base["Diffuse [W/m2]"] if "Diffuse [W/m2]" in base.columns else np.nan
        base["Diffuse [W/m2]"] = base["Diffuse [W/m2]"].combine_first(diffuse_series)

    total_series = None
    if (c_dir is not None and c_dif is not None) and (c_dir in om_1min.columns and c_dif in om_1min.columns):
        total_series = om_1min[c_dir] + om_1min[c_dif]
    elif (c_ghi is not None) and (c_ghi in om_1min.columns):
        total_series = om_1min[c_ghi]
    elif (c_dir is not None) and (c_dir in om_1min.columns):
        total_series = om_1min[c_dir]
    elif (c_dif is not None) and (c_dif in om_1min.columns):
        total_series = om_1min[c_dif]
    if total_series is not None:
        base["Total [W/m2]"] = base["Total [W/m2]"] if "Total [W/m2]" in base.columns else np.nan
        base["Total [W/m2]"] = base["Total [W/m2]"].combine_first(total_series)

    # --- OM extras: DNI and rain ---
    dni_series = None
    if (c_dni is not None) and (c_dni in om_1min.columns):
        dni_series = om_1min[c_dni]
    elif (c_dir is not None) and (c_dir in om_1min.columns):
        dni_series = om_1min[c_dir]
    if dni_series is not None:
        base["OM_DNI(W/m2)"] = dni_series.clip(lower=0)

    if (c_rain is not None) and (c_rain in om_1min.columns):
        base["OM_rain"] = om_1min[c_rain].clip(lower=0)

    # --- Solar QC using Open-Meteo as reference ---
    om_ghi_ref = om_1min[c_ghi] if (c_ghi is not None and c_ghi in om_1min.columns) else None
    om_dir_ref = om_1min[c_dir] if (c_dir is not None and c_dir in om_1min.columns) else None
    om_dif_ref = om_1min[c_dif] if (c_dif is not None and c_dif in om_1min.columns) else None
    om_dni_ref = om_1min[c_dni] if (c_dni is not None and c_dni in om_1min.columns) else om_dir_ref
    om_dhi_ref = om_dif_ref if (om_dif_ref is not None) else (om_ghi_ref - om_dir_ref if (om_ghi_ref is not None and om_dir_ref is not None) else None)

    night_mask = (om_ghi_ref <= 1.0) if (om_ghi_ref is not None) else None

    MAX_GHI = 1200.0
    MAX_DNI = 1100.0
    MAX_DHI = 600.0

    # Total (GHI)
    if "Total [W/m2]" in base.columns:
        s = pd.to_numeric(base["Total [W/m2]"], errors="coerce")
        if night_mask is not None:
            s.loc[night_mask] = 0.0
        if om_ghi_ref is not None and night_mask is not None:
            day_mask = ~night_mask
            abnormal = day_mask & ((s < 0) | (s > MAX_GHI) | s.isna())
            s.loc[abnormal] = om_ghi_ref.loc[abnormal]
        s = s.clip(lower=0, upper=MAX_GHI)
        base["Total [W/m2]"] = s

    # Diffuse (DHI)
    if "Diffuse [W/m2]" in base.columns:
        s = pd.to_numeric(base["Diffuse [W/m2]"], errors="coerce")
        if night_mask is not None:
            s.loc[night_mask] = 0.0
        if om_dhi_ref is not None and night_mask is not None:
            day_mask = ~night_mask
            abnormal = day_mask & ((s < 0) | (s > MAX_DHI) | s.isna())
            s.loc[abnormal] = om_dhi_ref.loc[abnormal]
        s = s.clip(lower=0, upper=MAX_DHI)
        if "Total [W/m2]" in base.columns:
            s = np.minimum(s, pd.to_numeric(base["Total [W/m2]"], errors="coerce"))
        base["Diffuse [W/m2]"] = s

    # DNI (store as OM_DNI(W/m2))
    if ("OM_DNI(W/m2)" in base.columns) or (om_dni_ref is not None):
        s = pd.to_numeric(base["OM_DNI(W/m2)"] , errors="coerce") if "OM_DNI(W/m2)" in base.columns else pd.Series(index=base.index, dtype=float)
        if night_mask is not None:
            s.loc[night_mask] = 0.0
        if om_dni_ref is not None and night_mask is not None:
            day_mask = ~night_mask
            abnormal = day_mask & ((s < 0) | (s > MAX_DNI) | s.isna())
            s.loc[abnormal] = om_dni_ref.loc[abnormal]
        s = s.clip(lower=0, upper=MAX_DNI)
        base["OM_DNI(W/m2)"] = s

    # --- Pressure final fill: time interpolation, then nearest future (bfill), then ffill ---
    if "Relative Air Pressure [kPa]" in base.columns:
        p = base["Relative Air Pressure [kPa]"]
        if p.isna().any():
            p2 = p.interpolate(method="time", limit_direction="both")
            p2 = p2.bfill().ffill()
            base["Relative Air Pressure [kPa]"] = p2

    # --- Enforce value constraints ---
    # Pressure > 0: set non-positive to NaN and re-fill by interpolation + nearest
    if "Relative Air Pressure [kPa]" in base.columns:
        p = base["Relative Air Pressure [kPa]"]
        p = p.mask(p <= 0, np.nan)
        p = p.interpolate(method="time", limit_direction="both").bfill().ffill()
        base["Relative Air Pressure [kPa]"] = p

    # Non-negative columns
    for col in ["Wind Speed [m/s]", "Total [W/m2]", "Diffuse [W/m2]"]:
        if col in base.columns:
            base[col] = base[col].clip(lower=0)

    # Relative Humidity within [0, 100]
    if "Relative Humidity [%]" in base.columns:
        base["Relative Humidity [%]"] = base["Relative Humidity [%]"].clip(lower=0, upper=100)

    # Wind direction wrap to [0, 360)
    if "Wind Direction [deg]" in base.columns:
        base["Wind Direction [deg]"] = np.mod(base["Wind Direction [deg]"], 360.0)

    out = base.reset_index().rename(columns={"index": "Time"})
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out

def write_epw_blocks(
    src="./final_merged_weather.csv",
    out_dir="./",
):
    SRC = Path(src)
    OUTDIR = Path(out_dir); OUTDIR.mkdir(parents=True, exist_ok=True)

    # ----- explicit column mapping (按你的列名，不猜测) -----
    MAP = {
        "time": "Time",
        "drybulb_C": "Air Temperature [degC]",
        "relhum_pct": "Relative Humidity [%]",
        "press_kPa": "Relative Air Pressure [kPa]",
        "ghi_Wm2": "Total [W/m2]",
        "dhi_Wm2": "Diffuse [W/m2]",
        "dni_Wm2": "OM_DNI(W/m2)",
        "wind_dir_deg": "Wind Direction [deg]",
        "wind_spd_mps": "Wind Speed [m/s]",
        # optional
        "dewp_C": "Dew Point [degC]",      # 如无此列，将写 EPW 缺测标记
        "precip_mm": "OM_rain",            # 如无此列，不造数据，写缺测标记
    }

    # ----- read & index -----
    df = pd.read_csv(SRC, parse_dates=[MAP["time"]])
    df = df.sort_values(MAP["time"]).set_index(MAP["time"])

    # 小工具：构造“小时-期末”索引
    def hour_ending_index(dt_min, dt_max):
        start = (dt_min.floor("h") + pd.Timedelta(hours=1)) if (dt_min != dt_min.floor("H")) else (dt_min + pd.Timedelta(hours=1))
        end   = dt_max.floor("h") + pd.Timedelta(hours=1)
        return pd.date_range(start, end, freq="h")

    hr_index = hour_ending_index(df.index.min(), df.index.max())

    # 聚合器（不插值）
    def rmean(col):
        s = pd.to_numeric(df[col], errors="coerce")
        return s.resample("h", label="right", closed="right").mean().reindex(hr_index)
    def rsum(col):
        s = pd.to_numeric(df[col], errors="coerce")
        return s.resample("h", label="right", closed="right").sum().reindex(hr_index)

    # 必需字段（只要有一个小时缺，就断开）
    drybulb = rmean(MAP["drybulb_C"])
    relhum  = rmean(MAP["relhum_pct"]).clip(0, 100)
    press   = rmean(MAP["press_kPa"]) * 1000.0          # kPa -> Pa
    ghi     = rmean(MAP["ghi_Wm2"])              # W/m2 -> Wh/m2
    dhi     = rmean(MAP["dhi_Wm2"])
    dni     = rmean(MAP["dni_Wm2"])
    wdir    = (rmean(MAP["wind_dir_deg"]) % 360.0)
    wspd    = rmean(MAP["wind_spd_mps"])

    # 可选字段（若无列，就用 EPW 缺测标记）
    dewp    = rmean(MAP["dewp_C"]) if MAP["dewp_C"] in df.columns else pd.Series(pd.NA, index=hr_index)
    precip  = rsum(MAP["precip_mm"]) if MAP["precip_mm"] in df.columns else pd.Series(pd.NA, index=hr_index)

    # 组合并形成“有效小时”掩码
    H = pd.DataFrame({
        "drybulb": drybulb,
        "relhum": relhum,
        "press": press,
        "ghi": ghi,
        "dni": dni,
        "dhi": dhi,
        "wdir": wdir,
        "wspd": wspd,
        "dewp": dewp,
        "precip": precip,
    })

    required = ["drybulb","relhum","press","ghi","dni","dhi","wdir","wspd"]
    valid_mask = H[required].notna().all(axis=1)

    # 仅保留有效小时，并计算对应的“有效日”和“EPW 小时(1..24)”
    if not valid_mask.any():
        raise RuntimeError("没有任何完整小时可用于导出。请检查必需列。")

    Hv = H.loc[valid_mask].copy()
    idx = Hv.index
    eff_date = []
    epw_hour = []
    for ts in idx:
        if ts.hour == 0:
            eff_date.append((ts - pd.Timedelta(days=1)).date())
            epw_hour.append(24)
        else:
            eff_date.append(ts.date())
            epw_hour.append(ts.hour)
    Hv["_eff_date"] = eff_date
    Hv["_epw_hour"] = epw_hour

    # 找到“完整天”（小时包含 1..24）
    hours_by_day = Hv.groupby("_eff_date")["_epw_hour"].agg(lambda s: set(s))
    complete_days = sorted([d for d, hs in hours_by_day.items() if all(h in hs for h in range(1, 25))])
    if len(complete_days) == 0:
        raise RuntimeError("没有任何完整天(24小时齐全)可用于导出。")

    # 将完整天切分为连续日期区间
    runs = []
    run = [complete_days[0]]
    for d in complete_days[1:]:
        if (pd.Timestamp(d) - pd.Timestamp(run[-1])).days == 1:
            run.append(d)
        else:
            runs.append(run)
            run = [d]
    runs.append(run)

    # ---- EPW 固定字段（保持简洁） ----
    lat, lon, tz, elev = 30.628, -96.334, -6.0, 96.0
    VIS_km, CEIL_m = 30, 77777
    PW_mm, AOD_mil = 20, 0.999  # AOD 缺测标记 (官方规范)
    SNOW_cm, DAYS_SNOW, ALBEDO = 0, 99, 0.2

    # EPW 缺测标记（官方建议值）
    M_T = 99.9        # temperature类
    M_P = 999999      # pressure Pa
    M_RAD = 9999      # radiation Wh/m2
    M_WDIR = 999      # wind dir
    M_WSPD = 999.0    # wind spd (官方规范)
    M_RAIN = 999.0    # precipitation mm (官方规范)
    ETR_H = M_RAD
    ETR_N = M_RAD
    H_IR  = M_RAD
    ILL   = 0
    written = []

    for k, days in enumerate(runs, start=1):
        start_day = pd.Timestamp(days[0])
        end_day = pd.Timestamp(days[-1])
        name = f"smart_home_{start_day.date()}_to_{end_day.date()}.epw"
        outp = OUTDIR / name

        header = [
            f"LOCATION,College Station,TX,USA,UserCSV,0,{lat:.3f},{lon:.3f},{tz:.1f},{elev:.1f}",
            "DESIGN CONDITIONS,0",
            "TYPICAL/EXTREME PERIODS,0",
            "GROUND TEMPERATURES,0",
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
            "COMMENTS 1,Generated strictly from source (no interpolation, no fabricated hours).",
            "COMMENTS 2,Hourly = minute-mean/sum → hour-ending EPW units; only hours with all required fields kept.",
            f"DATA PERIODS,1,1,Data,{start_day.day_name()},{start_day.month}/{start_day.day},{end_day.month}/{end_day.day}"
        ]

        with outp.open("w", newline="") as f:
            for line in header: f.write(line + "\n")
            for d in days:
                for hh in range(1, 25):
                    sel = Hv[(Hv["_eff_date"] == d) & (Hv["_epw_hour"] == hh)]
                    if sel.empty:
                        continue
                    row = sel.iloc[0]

                    Y, M, Dd = pd.Timestamp(d).year, pd.Timestamp(d).month, pd.Timestamp(d).day
                    Hh, mm = hh, 60

                    dry = round(float(row["drybulb"]), 1)
                    dwp = (round(float(row["dewp"]), 1) if pd.notna(row["dewp"]) else M_T)
                    rh  = int(round(float(row["relhum"])))
                    prs = int(round(float(row["press"])) ) if pd.notna(row["press"]) else M_P
                    GHI = int(round(float(row["ghi"])) ) if pd.notna(row["ghi"]) else M_RAD
                    DNI = int(round(float(row["dni"])) ) if pd.notna(row["dni"]) else M_RAD
                    DHI = int(round(float(row["dhi"])) ) if pd.notna(row["dhi"]) else M_RAD
                    wdr = int(round(float(row["wdir"])) ) if pd.notna(row["wdir"]) else M_WDIR
                    wsp = round(float(row["wspd"]), 1) if pd.notna(row["wspd"]) else M_WSPD
                    rmm = (round(float(row["precip"]), 2) if pd.notna(row["precip"]) else M_RAIN)

                    cols = [
                        Y, M, Dd, Hh, mm, 0,
                        dry, dwp, rh, prs,
                        ETR_H, ETR_N, H_IR,
                        GHI, DNI, DHI,
                        ILL, ILL, ILL, ILL,
                        wdr, wsp,
                        0, 0,
                        VIS_km, CEIL_m,
                        0, 0,
                        PW_mm, AOD_mil,
                        SNOW_cm, DAYS_SNOW,
                        ALBEDO,
                        rmm,
                        1
                    ]
                    f.write(",".join(map(str, cols)) + "\n")

        written.append(outp.as_posix())

    return written



#%%
merge_two_weather()

fill_with_open_meteo(base_csv = "./two_merged_weather.csv",
                     om_csv = "./openmeteo_data_2025.csv",
                     out_csv = "./final_merged_weather.csv",
                     prefix = "OM_",
                     tz_local = "America/Chicago",
                     year=2025)

write_epw_blocks()
