# dryer_kpi_monthly_final.py
"""
Lindner Dryer KPI Calculation Module
UPDATED: Support for Trockner A and Trockner B filtering
Column: "Trock-" with values "A" or "B"
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Trockner column configuration
TROCKNER_COLUMN = "Trock-"  # Exact column name in Hordenwagenverfolgung

CONFIG = {
    "energy_sheet": 0,
    "wagon_sheet": "Hordenwagenverfolgung",
    "wagon_header_row": 6,
    "gas_to_kwh": 11.5,
    "zones_seq": ["Z1", "Z2", "Z3", "Z4", "Z5"],
    "num_thermal_zones": 4,
    "trockner_column": TROCKNER_COLUMN,
}

ZONE_ENERGY_MAPPING = {
    "Z2": "Zone 2",
    "Z3": "Zone 3",
    "Z4": "Zone 4",
    "Z5": "Zone 5",
}

SUSPENSION_KG = 330
PLATES_PER_WAGON = 234

PRODUCT_SPECIFICATIONS = {
    "L28": {
        "product_type": "L",
        "final_thickness_mm": 28,
        "press_measurement_mm": 4,
        "pressed_thickness_mm": 32,
        "edge_length_mm": 602,
        "volume_m3": 0.0116,
        "suspension_kg": 330,
        "slope": -0.060,
        "intercept": 107.7,
        "formula": "-0.060x + 107.7",
        "water_per_mm_g": 88,
        "water_per_plate_kg": 2.81,
    },
    "L30": {
        "product_type": "L",
        "final_thickness_mm": 30,
        "press_measurement_mm": 4,
        "pressed_thickness_mm": 34,
        "edge_length_mm": 602,
        "volume_m3": 0.0123,
        "suspension_kg": 330,
        "slope": -0.042,
        "intercept": 102.3,
        "formula": "-0.042x + 102.3",
        "water_per_mm_g": 88,
        "water_per_plate_kg": 3.01,
    },
    "L32": {
        "product_type": "L",
        "final_thickness_mm": 32,
        "press_measurement_mm": 5,
        "pressed_thickness_mm": 37,
        "edge_length_mm": 602,
        "volume_m3": 0.0134,
        "suspension_kg": 330,
        "slope": -0.049,
        "intercept": 107.1,
        "formula": "-0.049x + 107.1 (interpolated)",
        "water_per_mm_g": 90,
        "water_per_plate_kg": 3.33,
        "interpolated": True,
    },
    "L34": {
        "product_type": "L",
        "final_thickness_mm": 34,
        "press_measurement_mm": 5,
        "pressed_thickness_mm": 39,
        "edge_length_mm": 602,
        "volume_m3": 0.0141,
        "suspension_kg": 330,
        "slope": -0.056,
        "intercept": 111.9,
        "formula": "-0.056x + 111.9",
        "water_per_mm_g": 93,
        "water_per_plate_kg": 3.64,
    },
    "L36": {
        "product_type": "L",
        "final_thickness_mm": 36,
        "press_measurement_mm": 6,
        "pressed_thickness_mm": 42,
        "edge_length_mm": 602,
        "volume_m3": 0.0152,
        "suspension_kg": 330,
        "slope": -0.025,
        "intercept": 76.6,
        "formula": "-0.025x + 76.6",
        "water_per_mm_g": 68,
        "water_per_plate_kg": 2.87,
    },
    "L38": {
        "product_type": "L",
        "final_thickness_mm": 38,
        "press_measurement_mm": 7,
        "pressed_thickness_mm": 45,
        "edge_length_mm": 602,
        "volume_m3": 0.0163,
        "suspension_kg": 330,
        "slope": -0.058,
        "intercept": 110.5,
        "formula": "-0.058x + 110.5",
        "water_per_mm_g": 91,
        "water_per_plate_kg": 4.11,
    },
    "L40": {
        "product_type": "L",
        "final_thickness_mm": 40,
        "press_measurement_mm": 8,
        "pressed_thickness_mm": 48,
        "edge_length_mm": 602,
        "volume_m3": 0.0174,
        "suspension_kg": 330,
        "slope": -0.033,
        "intercept": 92.4,
        "formula": "-0.033x + 92.4 (interpolated)",
        "water_per_mm_g": 82,
        "water_per_plate_kg": 3.94,
        "interpolated": True,
    },
    "L42": {
        "product_type": "L",
        "final_thickness_mm": 42,
        "press_measurement_mm": 8,
        "pressed_thickness_mm": 50,
        "edge_length_mm": 602,
        "volume_m3": 0.0181,
        "suspension_kg": 330,
        "slope": -0.007,
        "intercept": 74.2,
        "formula": "-0.007x + 74.2",
        "water_per_mm_g": 72,
        "water_per_plate_kg": 3.59,
    },
    "L44": {
        "product_type": "L",
        "final_thickness_mm": 44,
        "press_measurement_mm": 9,
        "pressed_thickness_mm": 53,
        "edge_length_mm": 602,
        "volume_m3": 0.0192,
        "suspension_kg": 330,
        "slope": -0.011,
        "intercept": 77.4,
        "formula": "-0.011x + 77.4",
        "water_per_mm_g": 74,
        "water_per_plate_kg": 3.91,
    },
    "N40": {
        "product_type": "N",
        "final_thickness_mm": 40,
        "press_measurement_mm": 10,
        "pressed_thickness_mm": 50,
        "edge_length_mm": 602,
        "volume_m3": 0.0181,
        "suspension_kg": 330,
        "slope": -0.103,
        "intercept": 102.5,
        "formula": "-0.103x + 102.5",
        "water_per_mm_g": 69,
        "water_per_plate_kg": 3.43,
    },
    "N44": {
        "product_type": "N",
        "final_thickness_mm": 44,
        "press_measurement_mm": 11,
        "pressed_thickness_mm": 55,
        "edge_length_mm": 602,
        "volume_m3": 0.0199,
        "suspension_kg": 330,
        "slope": -0.017,
        "intercept": 78.5,
        "formula": "-0.017x + 78.5",
        "water_per_mm_g": 73,
        "water_per_plate_kg": 4.01,
    },
    "Y44": {
        "product_type": "Y",
        "final_thickness_mm": 44,
        "press_measurement_mm": 12,
        "pressed_thickness_mm": 56,
        "edge_length_mm": 602,
        "volume_m3": 0.0203,
        "suspension_kg": 330,
        "slope": -0.157,
        "intercept": 200.0,
        "formula": "-0.157x + 200.0",
        "water_per_mm_g": 148,
        "water_per_plate_kg": 8.30,
    },
}

for product, spec in PRODUCT_SPECIFICATIONS.items():
    spec["water_per_m3_kg"] = spec["water_per_plate_kg"] / spec["volume_m3"]

WATER_PER_M3_KG = {
    product: spec["water_per_m3_kg"]
    for product, spec in PRODUCT_SPECIFICATIONS.items()
}


def safe_divide(numerator, denominator, default=0.0):
    """Safe division that handles zero and NaN values."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            (denominator != 0) & (~np.isnan(denominator)) & (np.isfinite(denominator)),
            numerator / denominator,
            default
        )
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)


def parse_duration_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"\bh\b", "hours", regex=True)
    s = s.str.replace(r"\bmin\b", "minutes", regex=True)
    s = s.str.replace(r"\bst\b", "seconds", regex=True)
    s = s.replace({r"^\s*$": np.nan, r"^-$": np.nan}, regex=True)
    td = pd.to_timedelta(s, errors="coerce")
    mask_nat = td.isna() & s.notna()
    if mask_nat.any():
        dt = pd.to_datetime(s[mask_nat], errors="coerce")
        td.loc[mask_nat] = dt - pd.Timestamp("1900-01-01")
    return td


def parse_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Parse hourly energy data."""
    logger.info("Parsing energy data...")
    df = df.copy()

    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], errors="coerce")
    df = df[df["Zeitstempel"].notna()].copy()

    df["Month"] = df["Zeitstempel"].dt.month
    df["Year"] = df["Zeitstempel"].dt.year

    # Calculate thermal energy per zone
    total_thermal = 0
    for z_key, z_name in ZONE_ENERGY_MAPPING.items():
        gas_col = f"Gasmenge, {z_name} [m³]"
        thermal_col = f"E_thermal_{z_name}_kWh"
        
        if gas_col in df.columns:
            gas_values = pd.to_numeric(df[gas_col], errors='coerce').fillna(0)
            df[thermal_col] = gas_values * CONFIG["gas_to_kwh"]
            zone_total = df[thermal_col].sum()
            total_thermal += zone_total
            logger.info(f"  {z_key}: {gas_values.sum():,.0f} m³ gas → {zone_total:,.0f} kWh thermal")
        else:
            df[thermal_col] = 0.0
            logger.warning(f"  {z_key}: Gas column not found")

    thermal_cols = [f"E_thermal_{z_name}_kWh" for z_name in ZONE_ENERGY_MAPPING.values()]
    existing_thermal_cols = [c for c in thermal_cols if c in df.columns]
    df["E_thermal_total_kWh"] = df[existing_thermal_cols].sum(axis=1)

    if "Energieverbrauch, elektr. [kWh]" in df.columns:
        df["E_el_kWh"] = pd.to_numeric(
            df["Energieverbrauch, elektr. [kWh]"], errors='coerce'
        ).fillna(0.0)
        total_electrical = df["E_el_kWh"].sum()
        logger.info(f"  Electrical: {total_electrical:,.0f} kWh total")
    else:
        df["E_el_kWh"] = 0.0
        total_electrical = 0
        logger.warning("  Electrical column not found")

    if total_electrical > 0 and total_thermal > 0:
        ratio = total_thermal / total_electrical
        logger.info(f"  ✓ Raw data ratio: Thermal/Electrical = {ratio:.1f}x")
    
    df["E_start"] = df["Zeitstempel"]
    df["E_end"] = df["Zeitstempel"] + pd.Timedelta(hours=1)

    logger.info(f"Parsed {len(df)} energy records")
    logger.info(f"  Total Thermal: {total_thermal:,.0f} kWh | Total Electrical: {total_electrical:,.0f} kWh")
    
    return df


def filter_wagons_by_trockner(df: pd.DataFrame, trockner: str) -> pd.DataFrame:
    """
    Filter wagon data by Trockner using the 'Trock-' column.
    
    Args:
        df: Wagon DataFrame (already parsed with clean column names)
        trockner: 'A' or 'B'
    
    Returns:
        Filtered DataFrame containing only wagons for the specified Trockner
    """
    logger.info(f"Filtering wagons for Trockner {trockner}...")
    
    # Find the Trock- column (handle potential variations in column name after cleaning)
    trockner_col = None
    for col in df.columns:
        col_clean = str(col).strip()
        if col_clean == TROCKNER_COLUMN or col_clean.startswith("Trock"):
            trockner_col = col
            break
    
    if trockner_col is None:
        logger.error(f"Column '{TROCKNER_COLUMN}' not found in wagon data!")
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Trockner column '{TROCKNER_COLUMN}' not found. Available columns: {list(df.columns)[:10]}")
    
    logger.info(f"  → Using column: '{trockner_col}'")
    
    # Get unique values before filtering
    unique_values = df[trockner_col].dropna().astype(str).str.strip().str.upper().unique()
    logger.info(f"  → Values in column: {unique_values}")
    
    # Count before filtering
    original_count = len(df)
    
    # Show distribution
    value_counts = df[trockner_col].astype(str).str.strip().str.upper().value_counts()
    logger.info(f"  → Distribution before filter:")
    for val, count in value_counts.items():
        logger.info(f"      {val}: {count} wagons")
    
    # Filter for selected Trockner
    trockner_upper = trockner.upper().strip()
    mask = df[trockner_col].astype(str).str.strip().str.upper() == trockner_upper
    
    filtered_df = df[mask].copy()
    filtered_count = len(filtered_df)
    
    logger.info(f"  → Filtered: {original_count} → {filtered_count} wagons (Trockner {trockner})")
    
    if filtered_count == 0:
        raise ValueError(
            f"No wagons found for Trockner {trockner}. "
            f"Available values in '{trockner_col}': {list(unique_values)}"
        )
    
    return filtered_df


def parse_wagon(df: pd.DataFrame, trockner: str = None) -> pd.DataFrame:
    """
    Parse wagon data with Trockner filtering.
    
    Args:
        df: Raw wagon DataFrame
        trockner: 'A' or 'B' to filter, or None for all wagons
    """
    logger.info("Parsing wagon data...")
    df = df.copy()

    # Clean column names (remove newlines, strip whitespace)
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    
    logger.info(f"  → Columns after cleaning: {list(df.columns)[:15]}...")

    # ===== FILTER BY TROCKNER FIRST =====
    if trockner:
        df = filter_wagons_by_trockner(df, trockner)

    # Parse timestamps
    if "Pressdat. + Zeit" in df.columns:
        df["t0"] = pd.to_datetime(df["Pressdat. + Zeit"], errors="coerce")
    else:
        date = df.get("Pressen-Datum", pd.Series()).astype(str)
        time = df.get("Press-Zeit", pd.Series()).astype(str)
        df["t0"] = pd.to_datetime(date + " " + time, errors="coerce")

    # Find wagon number column
    for col in df.columns:
        if str(col).startswith("WG-"):
            df = df.rename(columns={col: "WG_Nr"})
            break

    if "Produkt" in df.columns:
        df["Produkt"] = df["Produkt"].astype(str).str.strip()
    else:
        df["Produkt"] = "Unknown"

    # Calculate volume per wagon
    def get_wagon_volume(product):
        if product in PRODUCT_SPECIFICATIONS:
            spec = PRODUCT_SPECIFICATIONS[product]
            edge_m = spec["edge_length_mm"] / 1000.0
            thick_m = spec["final_thickness_mm"] / 1000.0
            volume_per_plate = edge_m * edge_m * thick_m
            volume_per_wagon = PLATES_PER_WAGON * volume_per_plate
            return volume_per_wagon
        else:
            logger.warning(f"Unknown product: {product}, using default volume")
            return 3.0
    
    df["m3"] = df["Produkt"].apply(get_wagon_volume)
    
    # Log summary
    trockner_label = f"TROCKNER {trockner}" if trockner else "ALL TROCKNER"
    logger.info(f"\n{'='*60}")
    logger.info(f"WAGON & VOLUME SUMMARY - {trockner_label}")
    logger.info(f"({PLATES_PER_WAGON} plates/wagon)")
    logger.info(f"{'='*60}")
    
    product_summary = df.groupby("Produkt").agg({
        "WG_Nr": "count" if "WG_Nr" in df.columns else "size",
        "m3": "first"
    })
    if "WG_Nr" in product_summary.columns:
        product_summary = product_summary.rename(columns={"WG_Nr": "Wagon_Count"})
    else:
        product_summary["Wagon_Count"] = df.groupby("Produkt").size()
    product_summary = product_summary.rename(columns={"m3": "Volume_per_Wagon"})
    
    product_summary["Total_Volume"] = product_summary["Wagon_Count"] * product_summary["Volume_per_Wagon"]
    product_summary = product_summary.sort_values("Total_Volume", ascending=False)
    
    for prod, row in product_summary.iterrows():
        logger.info(
            f"{prod:6s}: {row['Wagon_Count']:5.0f} wagons × {row['Volume_per_Wagon']:5.2f} m³/wagon = {row['Total_Volume']:8.1f} m³"
        )
    
    total_wagons = product_summary["Wagon_Count"].sum()
    total_volume = product_summary["Total_Volume"].sum()
    avg_volume = total_volume / total_wagons if total_wagons > 0 else 0
    
    logger.info(f"{'='*60}")
    logger.info(f"TOTAL: {total_wagons:,.0f} wagons = {total_volume:,.1f} m³ (avg {avg_volume:.2f} m³/wagon)")
    logger.info(f"{'='*60}\n")

    # Parse zone entry times
    for z in ("Z2", "Z3", "Z4", "Z5"):
        col = f"In {z}"
        df[f"{z}_in"] = (
            pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if col in df.columns else pd.NaT
        )

    df["Z1_in"] = df["t0"]

    if "Entnahme-Zeit" in df.columns:
        df["Entnahme"] = pd.to_datetime(df["Entnahme-Zeit"], errors="coerce", dayfirst=True)
    else:
        df["Entnahme"] = pd.NaT

    pairs = [
        ("Z1", "Z2_in", "t0"),
        ("Z2", "Z3_in", "Z2_in"),
        ("Z3", "Z4_in", "Z3_in"),
        ("Z4", "Z5_in", "Z4_in"),
        ("Z5", "Entnahme", "Z5_in"),
    ]

    for z, later, earlier in pairs:
        hours = (df[later] - df[earlier]).dt.total_seconds() / 3600
        df[f"{z}_dur_calc"] = hours

    for z in CONFIG["zones_seq"]:
        txt = f"Zeit in {z}"
        calc = f"{z}_dur_calc"
        out = f"{z}_dur"
        df[out] = pd.to_timedelta(df[calc], unit="h")
        if txt in df.columns:
            parsed = parse_duration_series(df[txt])
            mask = parsed.isna() | (parsed.dt.total_seconds() < 3600)
            df[out] = parsed.where(~mask, df[out])

    df["Month"] = df["t0"].dt.month
    df["Year"] = df["t0"].dt.year
    df = df[df["t0"].notna()].copy()
    
    # Store which Trockner this data is for
    df["Trockner"] = trockner if trockner else "ALL"
    
    logger.info(f"Parsed {len(df)} wagon records for Trockner {trockner or 'ALL'}")
    return df


def build_intervals(row: pd.Series) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    intervals = []
    prev_end = None

    for z in CONFIG["zones_seq"]:
        z_in = row.get(f"{z}_in", pd.NaT)
        if pd.isna(z_in):
            z_in = prev_end or row["t0"]

        z_dur = row.get(f"{z}_dur", pd.NaT)
        if pd.isna(z_dur):
            continue

        z_out = z_in + z_dur
        if z_out > z_in:
            intervals.append((z, z_in, z_out))
            prev_end = z_out

    return intervals


def explode_intervals(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Exploding wagon data into zone intervals...")
    rows = []
    for _, r in df.iterrows():
        for z, s, e in build_intervals(r):
            rows.append({
                "WG_Nr": r.get("WG_Nr", ""),
                "Produkt": r["Produkt"],
                "m3": r["m3"],
                "Zone": z,
                "P_start": s,
                "P_end": e,
                "Month": r["Month"],
                "Year": r["Year"],
                "Trockner": r.get("Trockner", "Unknown"),
            })
    
    result = pd.DataFrame(rows)
    logger.info(f"Created {len(result)} zone intervals")
    return result


def allocate_energy(e: pd.DataFrame, ivals: pd.DataFrame) -> pd.DataFrame:
    """Allocate energy correctly preserving thermal/electrical ratio."""
    logger.info("Allocating energy to products...")
    
    results = []

    for z_key, z_name in ZONE_ENERGY_MAPPING.items():
        thermal_col = f"E_thermal_{z_name}_kWh"

        if thermal_col not in e.columns:
            logger.warning(f"Thermal column {thermal_col} not found")
            continue

        e_zone = e[e[thermal_col] > 0].copy()
        iv_zone = ivals[ivals["Zone"] == z_key].copy()

        if e_zone.empty or iv_zone.empty:
            logger.info(f"Skipping {z_key}: no data (energy: {len(e_zone)}, intervals: {len(iv_zone)})")
            continue

        logger.info(f"Processing {z_key}: {len(e_zone)} hours × {len(iv_zone)} intervals")

        chunk = 1000
        zone_res = []

        for i in range(0, len(iv_zone), chunk):
            part = iv_zone.iloc[i:i+chunk]
            
            e_temp = e_zone.copy()
            e_temp["_key"] = 1
            p_temp = part.copy()
            p_temp["_key"] = 1

            merged = e_temp.merge(p_temp, on="_key", suffixes=("_e", "_p"))
            merged.drop("_key", axis=1, inplace=True)

            merged = merged[
                (merged["P_end"] > merged["E_start"]) &
                (merged["P_start"] < merged["E_end"])
            ]

            if merged.empty:
                continue

            merged["latest_start"] = merged[["E_start", "P_start"]].max(axis=1)
            merged["earliest_end"] = merged[["E_end", "P_end"]].min(axis=1)
            
            merged["Overlap_h"] = (
                (merged["earliest_end"] - merged["latest_start"])
                .dt.total_seconds() / 3600
            ).clip(lower=0, upper=1)
            
            merged = merged[merged["Overlap_h"] > 0].copy()

            if merged.empty:
                continue

            merged["E_hour_key"] = merged["E_start"].dt.strftime("%Y-%m-%d %H:00")
            
            hour_total_overlap = merged.groupby("E_hour_key")["Overlap_h"].transform("sum")
            
            merged["Hour_share"] = safe_divide(merged["Overlap_h"], hour_total_overlap)
            
            merged["Energy_thermal_kWh"] = merged[thermal_col] * merged["Hour_share"]
            
            merged["Zone"] = z_key
            
            if "Month_e" in merged.columns:
                month_col = "Month_e"
            elif "Month_p" in merged.columns:
                month_col = "Month_p"
            else:
                merged["Month"] = merged["E_start"].dt.month
                month_col = "Month"

            result = merged[[
                month_col, "Produkt", "m3", "Overlap_h", "Hour_share",
                "Energy_thermal_kWh", "Zone", "E_hour_key", "E_el_kWh"
            ]].copy()
            result = result.rename(columns={month_col: "Month"})

            zone_res.append(result)

        if zone_res:
            zone_df = pd.concat(zone_res, ignore_index=True)
            results.append(zone_df)

    if not results:
        logger.warning("No allocation results!")
        return pd.DataFrame(columns=[
            "Month", "Produkt", "Zone", "Energy_thermal_kWh", 
            "Energy_electrical_kWh", "Energy_share_kWh", "Overlap_h", "Hour_share", "m3"
        ])

    final = pd.concat(results, ignore_index=True)
    
    hour_totals = final.groupby("E_hour_key").agg({
        "Overlap_h": "sum",
        "E_el_kWh": "first"
    }).reset_index()
    hour_totals = hour_totals.rename(columns={"Overlap_h": "total_overlap_all_zones"})
    
    final = final.merge(hour_totals[["E_hour_key", "total_overlap_all_zones"]], on="E_hour_key", how="left")
    
    final["Global_share"] = safe_divide(final["Overlap_h"], final["total_overlap_all_zones"])
    
    final["Energy_electrical_kWh"] = final["E_el_kWh"] * final["Global_share"]
    
    final["Energy_share_kWh"] = final["Energy_thermal_kWh"] + final["Energy_electrical_kWh"]
    
    final = final.drop(columns=["E_hour_key", "E_el_kWh", "total_overlap_all_zones", "Global_share"], errors='ignore')
    
    final = final[
        (final["Energy_thermal_kWh"] >= 0) &
        (final["Energy_electrical_kWh"] >= 0) &
        (final["m3"] > 0)
    ].copy()
    
    total_th = final["Energy_thermal_kWh"].sum()
    total_el = final["Energy_electrical_kWh"].sum()
    total_energy = total_th + total_el
    
    if total_el > 0:
        ratio = total_th / total_el
        pct_el = (total_el / total_energy) * 100
        pct_th = (total_th / total_energy) * 100
        logger.info(f"✓ Allocated: Thermal={total_th:,.0f} kWh ({pct_th:.1f}%), Electrical={total_el:,.0f} kWh ({pct_el:.1f}%)")
        logger.info(f"✓ Ratio: Thermal/Electrical = {ratio:.1f}x")
    
    logger.info(f"Allocated {len(final)} records")
    return final


def add_water_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Add water-related KPIs using the formulas from product specifications."""
    df = df.copy()
    
    def calculate_water_for_product(product):
        if product in PRODUCT_SPECIFICATIONS:
            spec = PRODUCT_SPECIFICATIONS[product]
            slope = spec["slope"]
            intercept = spec["intercept"]
            water_per_mm_g = slope * SUSPENSION_KG + intercept
            pressed_thickness_mm = spec["pressed_thickness_mm"]
            water_per_plate_kg = (water_per_mm_g * pressed_thickness_mm) / 1000.0
            volume_per_plate_m3 = spec["volume_m3"]
            water_per_m3_kg = water_per_plate_kg / volume_per_plate_m3
            return water_per_m3_kg
        else:
            return np.mean(list(WATER_PER_M3_KG.values())) if WATER_PER_M3_KG else 200.0
    
    df["water_per_m3_formula"] = df["Produkt"].apply(calculate_water_for_product)
    df["Water_kg"] = df["Volume_m3"] * df["water_per_m3_formula"]
    
    df["kWh_thermal_per_m3"] = safe_divide(df["Energy_thermal_kWh"], df["Volume_m3"])
    df["kWh_per_m3"] = safe_divide(df["Energy_kWh"], df["Volume_m3"])
    df["kWh_thermal_per_kg"] = safe_divide(df["Energy_thermal_kWh"], df["Water_kg"])
    df["kWh_per_kg"] = safe_divide(df["Energy_kWh"], df["Water_kg"])
    
    for col in ["kWh_thermal_per_m3", "kWh_per_m3", "kWh_thermal_per_kg", "kWh_per_kg"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    
    return df


def calculate_water_per_plate(product: str, pressed_thickness_mm: float = None) -> float:
    if product not in PRODUCT_SPECIFICATIONS:
        return 0.0
    return PRODUCT_SPECIFICATIONS[product]["water_per_plate_kg"]


def calculate_water_per_m3_formula(product: str) -> float:
    if product not in PRODUCT_SPECIFICATIONS:
        return WATER_PER_M3_KG.get(product, 200.0)
    return PRODUCT_SPECIFICATIONS[product]["water_per_m3_kg"]


def get_product_water_curve(product: str, thickness_range: list = None) -> pd.DataFrame:
    if product not in PRODUCT_SPECIFICATIONS:
        return pd.DataFrame()
    
    spec = PRODUCT_SPECIFICATIONS[product]
    water_per_mm = spec["water_per_mm_g"]
    
    if thickness_range is None:
        center = spec["pressed_thickness_mm"]
        thickness_range = [int(center * 0.7), int(center * 1.3)]
    
    thicknesses = np.linspace(thickness_range[0], thickness_range[1], 50)
    water_values = [(water_per_mm * t) / 1000.0 for t in thicknesses]
    
    return pd.DataFrame({
        "Pressed_Thickness_mm": thicknesses,
        "Water_per_Plate_kg": water_values,
        "Product": product,
        "Water_per_mm_g": water_per_mm,
    })


def predict_production_energy(
    product_volumes_m3: dict,
    baseline_kwh_per_m3: float = None,
    baseline_kwh_per_kg: float = None,
    use_formulas: bool = True
) -> dict:
    results = {
        "products": [],
        "total_volume_m3": 0,
        "total_water_kg": 0,
        "total_energy_kwh": 0,
    }
    
    for product, volume_m3 in product_volumes_m3.items():
        if volume_m3 is None or volume_m3 <= 0:
            continue
        
        if use_formulas and product in PRODUCT_SPECIFICATIONS:
            spec = PRODUCT_SPECIFICATIONS[product]
            water_per_m3 = spec["water_per_m3_kg"]
            water_per_plate = spec["water_per_plate_kg"]
            water_per_mm = spec["water_per_mm_g"]
            num_plates = volume_m3 / spec["volume_m3"]
            formula = spec["formula"]
        else:
            water_per_m3 = WATER_PER_M3_KG.get(product, 200.0)
            water_per_plate = water_per_mm = num_plates = formula = None
        
        water_kg = volume_m3 * water_per_m3
        energy_from_water = baseline_kwh_per_kg * water_kg if baseline_kwh_per_kg else None
        
        product_result = {
            "product": product,
            "volume_m3": round(volume_m3, 3),
            "water_per_m3_kg": round(water_per_m3, 1),
            "water_kg": round(water_kg, 1),
            "num_plates": round(num_plates, 0) if num_plates else None,
            "water_per_plate_kg": round(water_per_plate, 2) if water_per_plate else None,
            "water_per_mm_g": water_per_mm,
            "formula": formula,
        }
        
        if energy_from_water:
            product_result["energy_from_water_kwh"] = round(energy_from_water, 0)
        
        results["products"].append(product_result)
        results["total_volume_m3"] += volume_m3
        results["total_water_kg"] += water_kg
        
        if energy_from_water:
            results["total_energy_kwh"] += energy_from_water
    
    results["mean_water_per_m3"] = (
        results["total_water_kg"] / results["total_volume_m3"]
        if results["total_volume_m3"] > 0 else 0
    )
    
    return results


def compute_product_wagon_stats(wagons: pd.DataFrame) -> dict:
    logger.info("Computing wagon statistics...")
    df = wagons.copy()

    if "Entnahme" in df.columns and "t0" in df.columns:
        df["residence_h"] = (df["Entnahme"] - df["t0"]).dt.total_seconds() / 3600
    else:
        df["residence_h"] = np.nan

    stats = df.groupby("Produkt", as_index=False).agg(
        avg_m3_per_wagon=("m3", "mean"),
        avg_residence_h=("residence_h", "mean"),
        wagon_count=("WG_Nr", "count") if "WG_Nr" in df.columns else ("m3", "count"),
    )
    stats["avg_residence_days"] = stats["avg_residence_h"] / 24.0

    return {
        "wagon_capacity_m3": stats.set_index("Produkt")["avg_m3_per_wagon"].to_dict(),
        "residence_h": stats.set_index("Produkt")["avg_residence_h"].to_dict(),
        "residence_days": stats.set_index("Produkt")["avg_residence_days"].to_dict(),
        "raw_stats": stats,
    }


def main():
    logger.info("Module loaded successfully.")
    logger.info(f"Configured for {len(PRODUCT_SPECIFICATIONS)} products")
    logger.info(f"Zones: {CONFIG['zones_seq']}")
    logger.info(f"Thermal zones mapped: {list(ZONE_ENERGY_MAPPING.keys())}")
    logger.info(f"Trockner column: '{TROCKNER_COLUMN}'")


if __name__ == "__main__":
    main()
