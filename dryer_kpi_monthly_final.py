# dryer_kpi_monthly_final.py
"""
Lindner Dryer KPI Calculation Module
FIXED: Correct column name handling and Trockner filtering
All original functionality preserved
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Column configurations - these are the CLEANED names (after removing newlines)
TROCKNER_COLUMN = "Trockner"
VOLUME_COLUMN = "m³"

CONFIG = {
    "energy_sheet": 0,
    "wagon_sheet": "Hordenwagenverfolgung",
    "wagon_header_row": 6,
    "gas_to_kwh": 11.5,
    "zones_seq": ["Z1", "Z2", "Z3", "Z4", "Z5"],
    "num_thermal_zones": 4,
    "trockner_column": TROCKNER_COLUMN,
    "volume_column": VOLUME_COLUMN,
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
    # ==================== L-TYPE PRODUCTS ====================
    "L20": {
        "product_type": "L",
        "final_thickness_mm": 20,
        "press_measurement_mm": 3,
        "pressed_thickness_mm": 23,
        "edge_length_mm": 602,
        "volume_m3": 0.0083,
        "suspension_kg": 330,
        "slope": -0.060,
        "intercept": 107.7,
        "formula": "-0.060x + 107.7 (extrapolated from L28)",
        "water_per_mm_g": 88,
        "water_per_plate_kg": 2.02,
        "extrapolated": True,
    },
    "L24": {
        "product_type": "L",
        "final_thickness_mm": 24,
        "press_measurement_mm": 3,
        "pressed_thickness_mm": 27,
        "edge_length_mm": 602,
        "volume_m3": 0.0098,
        "suspension_kg": 330,
        "slope": -0.060,
        "intercept": 107.7,
        "formula": "-0.060x + 107.7 (extrapolated from L28)",
        "water_per_mm_g": 88,
        "water_per_plate_kg": 2.38,
        "extrapolated": True,
    },
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
    "L37": {
        "product_type": "L",
        "final_thickness_mm": 37,
        "press_measurement_mm": 6,
        "pressed_thickness_mm": 43,
        "edge_length_mm": 602,
        "volume_m3": 0.0156,
        "suspension_kg": 330,
        "slope": -0.042,
        "intercept": 93.5,
        "formula": "-0.042x + 93.5 (interpolated L36-L38)",
        "water_per_mm_g": 80,
        "water_per_plate_kg": 3.44,
        "interpolated": True,
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
    
    # ==================== N-TYPE PRODUCTS ====================
    "N24": {
        "product_type": "N",
        "final_thickness_mm": 24,
        "press_measurement_mm": 6,
        "pressed_thickness_mm": 30,
        "edge_length_mm": 602,
        "volume_m3": 0.0109,
        "suspension_kg": 330,
        "slope": -0.103,
        "intercept": 102.5,
        "formula": "-0.103x + 102.5 (extrapolated from N40)",
        "water_per_mm_g": 69,
        "water_per_plate_kg": 2.07,
        "extrapolated": True,
    },
    "N30": {
        "product_type": "N",
        "final_thickness_mm": 30,
        "press_measurement_mm": 7,
        "pressed_thickness_mm": 37,
        "edge_length_mm": 602,
        "volume_m3": 0.0134,
        "suspension_kg": 330,
        "slope": -0.103,
        "intercept": 102.5,
        "formula": "-0.103x + 102.5 (extrapolated from N40)",
        "water_per_mm_g": 69,
        "water_per_plate_kg": 2.55,
        "extrapolated": True,
    },
    "N34": {
        "product_type": "N",
        "final_thickness_mm": 34,
        "press_measurement_mm": 8,
        "pressed_thickness_mm": 42,
        "edge_length_mm": 602,
        "volume_m3": 0.0152,
        "suspension_kg": 330,
        "slope": -0.103,
        "intercept": 102.5,
        "formula": "-0.103x + 102.5 (interpolated)",
        "water_per_mm_g": 69,
        "water_per_plate_kg": 2.90,
        "interpolated": True,
    },
    "N36": {
        "product_type": "N",
        "final_thickness_mm": 36,
        "press_measurement_mm": 9,
        "pressed_thickness_mm": 45,
        "edge_length_mm": 602,
        "volume_m3": 0.0163,
        "suspension_kg": 330,
        "slope": -0.103,
        "intercept": 102.5,
        "formula": "-0.103x + 102.5 (interpolated)",
        "water_per_mm_g": 69,
        "water_per_plate_kg": 3.11,
        "interpolated": True,
    },
    "N38": {
        "product_type": "N",
        "final_thickness_mm": 38,
        "press_measurement_mm": 9,
        "pressed_thickness_mm": 47,
        "edge_length_mm": 602,
        "volume_m3": 0.0170,
        "suspension_kg": 330,
        "slope": -0.103,
        "intercept": 102.5,
        "formula": "-0.103x + 102.5 (interpolated)",
        "water_per_mm_g": 69,
        "water_per_plate_kg": 3.24,
        "interpolated": True,
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
    "N42": {
        "product_type": "N",
        "final_thickness_mm": 42,
        "press_measurement_mm": 10,
        "pressed_thickness_mm": 52,
        "edge_length_mm": 602,
        "volume_m3": 0.0188,
        "suspension_kg": 330,
        "slope": -0.060,
        "intercept": 90.5,
        "formula": "-0.060x + 90.5 (interpolated N40-N44)",
        "water_per_mm_g": 71,
        "water_per_plate_kg": 3.69,
        "interpolated": True,
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
    
    # ==================== Y-TYPE PRODUCTS ====================
    "Y30": {
        "product_type": "Y",
        "final_thickness_mm": 30,
        "press_measurement_mm": 8,
        "pressed_thickness_mm": 38,
        "edge_length_mm": 602,
        "volume_m3": 0.0138,
        "suspension_kg": 330,
        "slope": -0.157,
        "intercept": 200.0,
        "formula": "-0.157x + 200.0 (extrapolated from Y44)",
        "water_per_mm_g": 148,
        "water_per_plate_kg": 5.62,
        "extrapolated": True,
    },
    "Y34": {
        "product_type": "Y",
        "final_thickness_mm": 34,
        "press_measurement_mm": 9,
        "pressed_thickness_mm": 43,
        "edge_length_mm": 602,
        "volume_m3": 0.0156,
        "suspension_kg": 330,
        "slope": -0.157,
        "intercept": 200.0,
        "formula": "-0.157x + 200.0 (interpolated)",
        "water_per_mm_g": 148,
        "water_per_plate_kg": 6.36,
        "interpolated": True,
    },
    "Y38": {
        "product_type": "Y",
        "final_thickness_mm": 38,
        "press_measurement_mm": 10,
        "pressed_thickness_mm": 48,
        "edge_length_mm": 602,
        "volume_m3": 0.0174,
        "suspension_kg": 330,
        "slope": -0.157,
        "intercept": 200.0,
        "formula": "-0.157x + 200.0 (interpolated)",
        "water_per_mm_g": 148,
        "water_per_plate_kg": 7.10,
        "interpolated": True,
    },
    "Y40": {
        "product_type": "Y",
        "final_thickness_mm": 40,
        "press_measurement_mm": 11,
        "pressed_thickness_mm": 51,
        "edge_length_mm": 602,
        "volume_m3": 0.0185,
        "suspension_kg": 330,
        "slope": -0.157,
        "intercept": 200.0,
        "formula": "-0.157x + 200.0 (interpolated Y38-Y44)",
        "water_per_mm_g": 148,
        "water_per_plate_kg": 7.55,
        "interpolated": True,
    },
    "Y42": {
        "product_type": "Y",
        "final_thickness_mm": 42,
        "press_measurement_mm": 11,
        "pressed_thickness_mm": 53,
        "edge_length_mm": 602,
        "volume_m3": 0.0192,
        "suspension_kg": 330,
        "slope": -0.157,
        "intercept": 200.0,
        "formula": "-0.157x + 200.0 (interpolated Y38-Y44)",
        "water_per_mm_g": 148,
        "water_per_plate_kg": 7.84,
        "interpolated": True,
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
# Calculate water_per_m3_kg for each product
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
    """Parse duration strings to timedelta."""
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

    thermal_cols = [f"E_thermal_{z_name}_kWh" for z_name in ZONE_ENERGY_MAPPING.values()]
    existing_thermal_cols = [c for c in thermal_cols if c in df.columns]
    df["E_thermal_total_kWh"] = df[existing_thermal_cols].sum(axis=1)

    if "Energieverbrauch, elektr. [kWh]" in df.columns:
        df["E_el_kWh"] = pd.to_numeric(
            df["Energieverbrauch, elektr. [kWh]"], errors='coerce'
        ).fillna(0.0)
    else:
        df["E_el_kWh"] = 0.0

    df["E_start"] = df["Zeitstempel"]
    df["E_end"] = df["Zeitstempel"] + pd.Timedelta(hours=1)

    logger.info(f"Parsed {len(df)} energy records")
    
    return df


def find_column_flexible(df: pd.DataFrame, patterns: list, description: str = "") -> Optional[str]:
    """
    Find a column matching any of the given patterns (case-insensitive, partial match).
    """
    # Try exact matches first
    for pattern in patterns:
        if pattern in df.columns:
            return pattern
    
    # Try case-insensitive and partial matches
    for col in df.columns:
        col_lower = str(col).lower()
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in col_lower or col_lower == pattern_lower:
                return col
    
    return None


def parse_wagon(df: pd.DataFrame, trockner: str = None) -> pd.DataFrame:
    """
    Parse wagon data with CORRECT Trockner filtering.
    
    FIXED VERSION:
    - Apply Trockner filter FIRST (before other filters)
    - Clean column names properly
    - Track row loss at each step
    - Product is from 'EM' (thickness) + 'Rez.' (type L/N/Y with forward-fill)
    """
    logger.info("="*70)
    logger.info("PARSING WAGON DATA - FIXED VERSION")
    logger.info(f"Trockner filter requested: {trockner or 'None (All)'}")
    logger.info("="*70)
    
    raw_row_count = len(df)
    logger.info(f"Raw input: {raw_row_count} rows, {len(df.columns)} columns")
    
    df = df.copy()
    
    # ========================================
    # STEP 1: CLEAN COLUMN NAMES
    # Remove newlines and carriage returns
    # ========================================
    original_columns = list(df.columns)
    
    cleaned_columns = []
    for col in df.columns:
        col_str = str(col)
        col_clean = col_str.replace("\n", "").replace("\r", "").strip()
        cleaned_columns.append(col_clean)
    
    df.columns = cleaned_columns
    
    logger.info("Column name cleaning:")
    changes_logged = 0
    for i, (orig, clean) in enumerate(zip(original_columns[:15], df.columns[:15])):
        if str(orig) != clean:
            logger.info(f"  [{i:2d}] {repr(str(orig))} → '{clean}'")
            changes_logged += 1
    
    if changes_logged == 0:
        logger.info("  No column names needed cleaning")
    
    logger.info(f"Columns (first 15): {list(df.columns)[:15]}")
    
    # ========================================
    # STEP 2: FIND KEY COLUMNS
    # ========================================
    
    # Find Trockner column
    trockner_col = find_column_flexible(df, ["Trockner", "Trock-ner", "Trock-", "TROCKNER"])
    if trockner_col:
        logger.info(f"✓ Trockner column: '{trockner_col}'")
    else:
        logger.warning("✗ Trockner column NOT FOUND!")
    
    # Find wagon number column (first column or WG-Nr)
    wagon_col = find_column_flexible(df, ["WG-Nr", "WG-Nr.", "WGNr", "WG Nr", "WG_Nr"])
    if not wagon_col:
        wagon_col = df.columns[0]
    logger.info(f"✓ Wagon column: '{wagon_col}'")
    
    # Find volume column (m³) - typically column AA (index 26)
    volume_col = find_column_flexible(df, ["m³", "m3", "Volumen", "Volume"])
    if not volume_col and len(df.columns) > 26:
        volume_col = df.columns[26]
    logger.info(f"✓ Volume column: '{volume_col}'")
    
    # Find EM column (thickness)
    em_col = find_column_flexible(df, ["EM", "Dicke", "Thickness"])
    if em_col:
        logger.info(f"✓ EM (thickness) column: '{em_col}'")
    else:
        logger.warning("✗ EM column NOT FOUND!")
    
    # Find Rez. column (product type)
    rez_col = find_column_flexible(df, ["Rez.", "Rez", "Rezept", "Rezeptur"])
    if rez_col:
        logger.info(f"✓ Rez. (type) column: '{rez_col}'")
    else:
        logger.warning("✗ Rez. column NOT FOUND!")
    
    # Find timestamp column
    timestamp_col = find_column_flexible(df, ["Pressdat. + Zeit", "Pressdat", "Pressdatum"])
    if timestamp_col:
        logger.info(f"✓ Timestamp column: '{timestamp_col}'")
    else:
        logger.warning("✗ Timestamp column NOT FOUND!")
    
    # ========================================
    # STEP 3: APPLY TROCKNER FILTER FIRST!
    # This is the key fix - filter by Trockner before anything else
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("STEP 3: APPLYING TROCKNER FILTER FIRST")
    logger.info("="*50)
    
    count_before_trockner = len(df)
    
    if trockner_col and trockner_col in df.columns:
        # Clean Trockner values
        df["_trockner_clean"] = df[trockner_col].astype(str).str.strip().str.upper()
        
        # Show distribution before any filtering
        logger.info("Trockner distribution (before filtering):")
        value_counts = df["_trockner_clean"].value_counts()
        for val, count in value_counts.items():
            logger.info(f"  '{val}': {count:,} rows")
        
        # Filter to valid Trockner values (A or B) only
        valid_trockner_mask = df["_trockner_clean"].isin(["A", "B"])
        invalid_count = (~valid_trockner_mask).sum()
        
        if invalid_count > 0:
            logger.info(f"Removing {invalid_count:,} rows with invalid Trockner (not A or B)")
        
        df = df[valid_trockner_mask].copy()
        count_after_valid_trockner = len(df)
        logger.info(f"After filtering to valid Trockner (A/B): {count_after_valid_trockner:,} rows")
        
        # Now apply specific Trockner selection if requested
        if trockner and trockner.upper() in ["A", "B"]:
            trockner_upper = trockner.upper().strip()
            
            # Count before selection
            count_a = (df["_trockner_clean"] == "A").sum()
            count_b = (df["_trockner_clean"] == "B").sum()
            logger.info(f"Available: Trockner A = {count_a:,}, Trockner B = {count_b:,}")
            
            # Apply selection
            before_selection = len(df)
            df = df[df["_trockner_clean"] == trockner_upper].copy()
            after_selection = len(df)
            
            logger.info(f"Selected Trockner '{trockner_upper}': {after_selection:,} rows")
            logger.info(f"Removed: {before_selection - after_selection:,} rows (other Trockner)")
        
        # Drop temp column
        df = df.drop(columns=["_trockner_clean"], errors="ignore")
    else:
        logger.warning("⚠️ Cannot filter by Trockner - column not found!")
    
    count_after_trockner = len(df)
    logger.info(f"\n>>> ROWS AFTER TROCKNER FILTER: {count_after_trockner:,} <<<")
    
    if df.empty:
        raise ValueError(f"No rows found for Trockner '{trockner}'")
    
    # ========================================
    # STEP 4: FILTER ROWS WITH VALID WAGON NUMBER
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("STEP 4: FILTERING BY WAGON NUMBER")
    logger.info("="*50)
    
    if wagon_col and wagon_col in df.columns:
        wagon_vals = df[wagon_col].astype(str).str.strip()
        valid_wagon = (wagon_vals != "") & (wagon_vals != "nan") & (wagon_vals != "NaN") & (wagon_vals != "None") & (wagon_vals.notna())
        
        count_before = len(df)
        df = df[valid_wagon].copy()
        count_after = len(df)
        
        if count_before != count_after:
            logger.info(f"Removed {count_before - count_after:,} rows with empty wagon number")
        logger.info(f"Rows with valid wagon number: {count_after:,}")
    
    # ========================================
    # STEP 5: FILTER ROWS WITH VALID VOLUME
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("STEP 5: FILTERING BY VOLUME")
    logger.info("="*50)
    
    if volume_col and volume_col in df.columns:
        df["m3"] = pd.to_numeric(df[volume_col], errors='coerce')
        
        valid_vol = df["m3"].notna() & (df["m3"] > 0)
        invalid_vol_count = (~valid_vol).sum()
        
        if invalid_vol_count > 0:
            # Show sample of invalid values
            invalid_sample = df.loc[~valid_vol, volume_col].head(10).tolist()
            logger.info(f"Found {invalid_vol_count:,} rows with invalid volume")
            logger.info(f"Sample invalid values: {invalid_sample}")
        
        count_before = len(df)
        df = df[valid_vol].copy()
        count_after = len(df)
        
        if count_before != count_after:
            logger.info(f"Removed {count_before - count_after:,} rows with invalid volume")
        
        logger.info(f"Rows with valid volume: {count_after:,}")
        logger.info(f"Volume stats: min={df['m3'].min():.4f}, max={df['m3'].max():.4f}, mean={df['m3'].mean():.4f}, sum={df['m3'].sum():.2f}")
    else:
        df["m3"] = 3.5
        logger.warning("Volume column not found - using default 3.5 m³")
    
    # ========================================
    # STEP 6: PARSE TIMESTAMPS
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("STEP 6: PARSING TIMESTAMPS")
    logger.info("="*50)
    
    if timestamp_col and timestamp_col in df.columns:
        df["t0"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        valid_ts = df["t0"].notna().sum()
        invalid_ts = df["t0"].isna().sum()
        logger.info(f"Valid timestamps: {valid_ts:,}, Invalid: {invalid_ts:,}")
        
        if invalid_ts > 0:
            invalid_sample = df.loc[df["t0"].isna(), timestamp_col].head(5).tolist()
            logger.info(f"Sample invalid timestamp values: {invalid_sample}")
    else:
        # Try alternative columns
        date_col = find_column_flexible(df, ["Datum", "Date", "Press-datum"])
        time_col = find_column_flexible(df, ["Zeit", "Time", "Uhrzeit", "Press-Zeit"])
        
        if date_col and time_col and "Entnahme" not in str(time_col):
            df["t0"] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str),
                errors="coerce"
            )
            logger.info(f"Combined timestamps from '{date_col}' + '{time_col}'")
        else:
            df["t0"] = pd.NaT
            logger.warning("Could not parse timestamps")
    
    # Filter rows without valid timestamp
    count_before = len(df)
    df = df[df["t0"].notna()].copy()
    count_after = len(df)
    if count_before != count_after:
        logger.info(f"Removed {count_before - count_after:,} rows without valid timestamp")
    logger.info(f"Rows with valid timestamp: {count_after:,}")
    
    # ========================================
    # STEP 7: PARSE PRODUCT (EM + Rez. with forward-fill)
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("STEP 7: PARSING PRODUCT (EM + Rez.)")
    logger.info("="*50)
    
    if em_col and em_col in df.columns:
        # Get thickness from EM column
        df["_thickness"] = pd.to_numeric(df[em_col], errors='coerce').astype('Int64')
        
        valid_thickness = df["_thickness"].notna().sum()
        logger.info(f"Valid thickness values: {valid_thickness:,}")
        
        logger.info(f"Thickness distribution:")
        thickness_counts = df["_thickness"].value_counts().sort_index()
        for val, count in thickness_counts.items():
            logger.info(f"  {val}: {count:,} rows")
    else:
        logger.error("Cannot parse product without EM column!")
        df["_thickness"] = pd.NA
    
    if rez_col and rez_col in df.columns:
        # Get product type from Rez. column
        df["_type_raw"] = df[rez_col].astype(str).str.strip().str.upper()
        
        # Replace empty strings and invalid values with NaN for forward-fill
        df["_type_raw"] = df["_type_raw"].replace(["", "NAN", "NONE", "NA", "<NA>", "-"], pd.NA)
        
        # Count non-empty values
        non_empty_count = df["_type_raw"].notna().sum()
        logger.info(f"Non-empty Rez. values: {non_empty_count:,}")
        if non_empty_count > 0:
            unique_types = df["_type_raw"].dropna().unique().tolist()
            logger.info(f"Unique types found: {unique_types}")
        
        # FORWARD-FILL the product type
        df["_type_filled"] = df["_type_raw"].ffill()
        
        # If still NaN at the beginning, default to 'L'
        df["_type_filled"] = df["_type_filled"].fillna("L")
        
        # Clean up - keep only valid types (L, N, Y)
        valid_types = ["L", "N", "Y"]
        df["_type_filled"] = df["_type_filled"].apply(
            lambda x: x if x in valid_types else "L"
        )
        
        logger.info(f"Product type after forward-fill:")
        type_counts = df["_type_filled"].value_counts()
        for val, count in type_counts.items():
            logger.info(f"  '{val}': {count:,} rows")
    else:
        logger.warning("No Rez. column found - defaulting all to L-type")
        df["_type_filled"] = "L"
    
    # Combine type + thickness to create product code
    def create_product_code(row):
        ptype = row.get("_type_filled", "L")
        thickness = row.get("_thickness", pd.NA)
        
        if pd.isna(thickness):
            return "Unknown"
        
        return f"{ptype}{int(thickness)}"
    
    df["Produkt"] = df.apply(create_product_code, axis=1)
    
    # Show product distribution
    logger.info(f"Combined product codes:")
    product_counts = df["Produkt"].value_counts()
    for val, count in product_counts.items():
        in_specs = "✓" if val in PRODUCT_SPECIFICATIONS else "✗"
        logger.info(f"  {in_specs} '{val}': {count:,} rows")
    
    # Clean up temporary columns
    df = df.drop(columns=["_thickness", "_type_raw", "_type_filled"], errors="ignore")
    
    # Filter to valid products
    valid_products = list(PRODUCT_SPECIFICATIONS.keys())
    count_before = len(df)
    invalid_mask = ~df["Produkt"].isin(valid_products)
    
    if invalid_mask.sum() > 0:
        invalid_prods = df.loc[invalid_mask, "Produkt"].value_counts()
        logger.warning(f"Removing {invalid_mask.sum():,} rows with invalid products:")
        for prod, count in invalid_prods.items():
            logger.warning(f"  ✗ '{prod}': {count:,} rows")
    
    df = df[df["Produkt"].isin(valid_products)].copy()
    count_after = len(df)
    
    if count_before != count_after:
        logger.info(f"After product filter: {count_after:,} rows (removed {count_before - count_after:,})")
    
    # ========================================
    # STEP 8: RENAME COLUMNS AND ADD METADATA
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("STEP 8: FINALIZING COLUMNS")
    logger.info("="*50)
    
    # Rename wagon column
    if wagon_col and wagon_col in df.columns and wagon_col != "WG_Nr":
        df = df.rename(columns={wagon_col: "WG_Nr"})
    elif "WG_Nr" not in df.columns:
        df["WG_Nr"] = df.iloc[:, 0]
    
    # Add Month/Year
    df["Month"] = df["t0"].dt.month
    df["Year"] = df["t0"].dt.year
    df["Trockner"] = trockner if trockner else "All"
    
    # ========================================
    # STEP 9: PARSE ZONE ENTRY TIMES
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("STEP 9: PARSING ZONE TIMES")
    logger.info("="*50)
    
    df["Z1_in"] = df["t0"]
    
    for z in ("Z2", "Z3", "Z4", "Z5"):
        zone_col = find_column_flexible(df, [f"In {z}", f"In{z}", f"Zone {z[-1]} In"])
        if zone_col and zone_col in df.columns:
            df[f"{z}_in"] = pd.to_datetime(df[zone_col], errors="coerce", dayfirst=True)
            valid_count = df[f"{z}_in"].notna().sum()
            logger.info(f"  {z}_in: {valid_count:,} valid")
        else:
            df[f"{z}_in"] = pd.NaT
    
    entnahme_col = find_column_flexible(df, ["Entnahme-Zeit", "EntnahmeZeit", "Entnahme Zeit", "Entnahme"])
    if entnahme_col and entnahme_col in df.columns:
        df["Entnahme"] = pd.to_datetime(df[entnahme_col], errors="coerce", dayfirst=True)
        logger.info(f"  Entnahme: {df['Entnahme'].notna().sum():,} valid")
    else:
        df["Entnahme"] = pd.NaT
    
    # ========================================
    # STEP 10: CALCULATE ZONE DURATIONS
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("STEP 10: CALCULATING ZONE DURATIONS")
    logger.info("="*50)
    
    pairs = [
        ("Z1", "Z2_in", "t0"),
        ("Z2", "Z3_in", "Z2_in"),
        ("Z3", "Z4_in", "Z3_in"),
        ("Z4", "Z5_in", "Z4_in"),
        ("Z5", "Entnahme", "Z5_in"),
    ]
    
    for z, later, earlier in pairs:
        if later in df.columns and earlier in df.columns:
            hours = (df[later] - df[earlier]).dt.total_seconds() / 3600
            df[f"{z}_dur_calc"] = hours
        else:
            df[f"{z}_dur_calc"] = np.nan
    
    for z in CONFIG["zones_seq"]:
        txt_col = find_column_flexible(df, [f"Zeit in {z}", f"Zeitin{z}", f"Zeit {z}"])
        calc = f"{z}_dur_calc"
        out = f"{z}_dur"
        
        if calc in df.columns:
            df[out] = pd.to_timedelta(df[calc], unit="h")
        else:
            df[out] = pd.NaT
        
        if txt_col and txt_col in df.columns:
            parsed = parse_duration_series(df[txt_col])
            mask = parsed.isna() | (parsed.dt.total_seconds() < 3600)
            if out in df.columns:
                df[out] = parsed.where(~mask, df[out])
        
        valid_dur = df[out].notna().sum()
        logger.info(f"  {z}_dur: {valid_dur:,} valid")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    logger.info("")
    logger.info("="*70)
    logger.info(f"FINAL RESULT - TROCKNER: {trockner or 'ALL'}")
    logger.info("="*70)
    
    total_rows = len(df)
    total_vol = df["m3"].sum() if not df.empty else 0
    avg_vol = df["m3"].mean() if total_rows > 0 else 0
    
    if total_rows > 0:
        logger.info("By Product:")
        product_summary = df.groupby("Produkt").agg({"m3": ["count", "sum", "mean"]}).round(4)
        product_summary.columns = ["Rows", "Volume_m3", "Avg_m3"]
        product_summary = product_summary.sort_values("Volume_m3", ascending=False)
        
        for prod, row in product_summary.iterrows():
            logger.info(f"  {prod:6s}: {int(row['Rows']):5d} rows | {row['Volume_m3']:8.2f} m³")
    
    logger.info("-"*70)
    logger.info(f"TOTAL: {total_rows:,} wagon rows | {total_vol:,.2f} m³ | {avg_vol:.4f} m³/row")
    logger.info("")
    logger.info("ROW TRACKING:")
    logger.info(f"  Input rows:           {raw_row_count:,}")
    logger.info(f"  After Trockner:       {count_after_trockner:,}")
    logger.info(f"  Final rows:           {total_rows:,}")
    logger.info(f"  Lost after Trockner:  {count_after_trockner - total_rows:,}")
    logger.info("="*70)
    
    # Validation for expected counts
    if trockner == "A" and total_rows < 3500:
        logger.warning(f"⚠️ Expected ~3692 rows for Trockner A, got {total_rows}")
    if trockner == "B" and total_rows < 3500:
        logger.warning(f"⚠️ Expected ~3691 rows for Trockner B, got {total_rows}")
    
    return df


def diagnose_wagon_file(df: pd.DataFrame, trockner: str = None) -> dict:
    """
    Diagnostic function to trace exactly where rows are lost.
    Call this BEFORE parse_wagon to see the raw data.
    """
    print("="*70)
    print("DIAGNOSTIC: WAGON FILE ANALYSIS")
    print("="*70)
    
    result = {
        "raw_rows": len(df),
        "steps": []
    }
    
    # Step 0: Raw data
    print(f"\n[STEP 0] Raw file: {len(df):,} rows, {len(df.columns)} columns")
    result["steps"].append(("Raw file", len(df)))
    
    # Show first 15 columns with their indices
    print("\nColumn names (first 15):")
    for i, col in enumerate(df.columns[:15]):
        print(f"  [{i:2d}] {repr(col)}")
    
    # Make a copy and clean column names
    df_clean = df.copy()
    df_clean.columns = [str(c).replace("\n", "").replace("\r", "").strip() for c in df_clean.columns]
    
    print("\nCleaned column names (first 15):")
    for i, col in enumerate(df_clean.columns[:15]):
        print(f"  [{i:2d}] '{col}'")
    
    # Find key columns
    print("\n" + "-"*50)
    print("FINDING KEY COLUMNS")
    print("-"*50)
    
    # Trockner column
    trockner_col = None
    for col in df_clean.columns:
        if "trock" in col.lower():
            trockner_col = col
            break
    
    if trockner_col:
        print(f"✓ Trockner column: '{trockner_col}'")
        trockner_vals = df_clean[trockner_col].astype(str).str.strip().str.upper()
        print(f"  Values: {trockner_vals.value_counts().to_dict()}")
        
        if trockner:
            match_count = (trockner_vals == trockner.upper()).sum()
            print(f"  Rows matching '{trockner.upper()}': {match_count:,}")
            result["trockner_match"] = match_count
    else:
        print("✗ Trockner column NOT FOUND")
    
    # EM column (thickness)
    em_col = None
    for col in df_clean.columns:
        if col.upper() == "EM":
            em_col = col
            break
    
    if em_col:
        print(f"\n✓ EM (thickness) column: '{em_col}'")
        em_vals = df_clean[em_col].astype(str).str.strip()
        print(f"  Sample values: {em_vals.head(10).tolist()}")
        print(f"  Unique values: {sorted(em_vals.unique())[:15]}")
    else:
        print("\n✗ EM column NOT FOUND")
    
    # Rez. column (product type)
    rez_col = None
    for col in df_clean.columns:
        if "rez" in col.lower():
            rez_col = col
            break
    
    if rez_col:
        print(f"\n✓ Rez. (type) column: '{rez_col}'")
        rez_vals = df_clean[rez_col].astype(str).str.strip().str.upper()
        rez_vals_clean = rez_vals.replace(["", "NAN", "NONE"], pd.NA)
        non_empty = rez_vals_clean.dropna()
        print(f"  Non-empty values: {len(non_empty):,} rows")
        print(f"  Unique non-empty: {non_empty.unique().tolist()}")
    else:
        print("\n✗ Rez. column NOT FOUND")
    
    # Volume column
    volume_col = None
    for col in df_clean.columns:
        if "m³" in col or "m3" in col.lower():
            volume_col = col
            break
    if not volume_col and len(df_clean.columns) > 26:
        volume_col = df_clean.columns[26]
    
    if volume_col:
        print(f"\n✓ Volume column: '{volume_col}'")
        vol_numeric = pd.to_numeric(df_clean[volume_col], errors='coerce')
        print(f"  Valid numeric: {vol_numeric.notna().sum():,}")
        print(f"  Positive (>0): {(vol_numeric > 0).sum():,}")
        print(f"  Zero or negative: {(vol_numeric <= 0).sum():,}")
        print(f"  NaN/invalid: {vol_numeric.isna().sum():,}")
        result["volume_positive"] = (vol_numeric > 0).sum()
    else:
        print("\n✗ Volume column NOT FOUND")
    
    # Timestamp column
    print("\n" + "-"*50)
    print("TIMESTAMP ANALYSIS")
    print("-"*50)
    
    timestamp_col = None
    for col in df_clean.columns:
        col_lower = col.lower()
        if "pressdat" in col_lower or "datum" in col_lower:
            timestamp_col = col
            break
    
    if timestamp_col:
        print(f"✓ Timestamp column: '{timestamp_col}'")
        ts = pd.to_datetime(df_clean[timestamp_col], errors='coerce')
        print(f"  Valid timestamps: {ts.notna().sum():,}")
        print(f"  Invalid (NaT): {ts.isna().sum():,}")
        result["valid_timestamps"] = ts.notna().sum()
    else:
        print("✗ Timestamp column NOT FOUND")
    
    # Simulate filtering steps
    print("\n" + "-"*50)
    print("SIMULATED FILTER PIPELINE")
    print("-"*50)
    
    df_sim = df_clean.copy()
    print(f"\n[START] {len(df_sim):,} rows")
    
    # Step 1: Trockner filter (FIRST!)
    if trockner and trockner_col:
        trockner_vals = df_sim[trockner_col].astype(str).str.strip().str.upper()
        # First filter to A/B only
        df_sim = df_sim[trockner_vals.isin(["A", "B"])]
        print(f"[AFTER Valid Trockner A/B] {len(df_sim):,} rows")
        # Then filter to specific Trockner
        trockner_vals = df_sim[trockner_col].astype(str).str.strip().str.upper()
        df_sim = df_sim[trockner_vals == trockner.upper()]
        print(f"[AFTER Trockner '{trockner}'] {len(df_sim):,} rows")
        result["steps"].append((f"After Trockner {trockner}", len(df_sim)))
    
    # Step 2: Volume filter
    if volume_col and volume_col in df_sim.columns:
        vol = pd.to_numeric(df_sim[volume_col], errors='coerce')
        before = len(df_sim)
        df_sim = df_sim[vol > 0]
        print(f"[AFTER Volume > 0] {len(df_sim):,} rows (removed {before - len(df_sim):,})")
        result["steps"].append(("After Volume > 0", len(df_sim)))
    
    # Step 3: Timestamp filter
    if timestamp_col and timestamp_col in df_sim.columns:
        ts = pd.to_datetime(df_sim[timestamp_col], errors='coerce')
        before = len(df_sim)
        df_sim = df_sim[ts.notna()]
        print(f"[AFTER Valid Timestamp] {len(df_sim):,} rows (removed {before - len(df_sim):,})")
        result["steps"].append(("After Valid Timestamp", len(df_sim)))
    
    # Step 4: Create product code
    if em_col and em_col in df_sim.columns:
        thickness = pd.to_numeric(df_sim[em_col], errors='coerce')
        
        # Get product type with forward-fill
        if rez_col and rez_col in df_sim.columns:
            ptype = df_sim[rez_col].astype(str).str.strip().str.upper()
            ptype = ptype.replace(["", "NAN", "NONE", "NA"], pd.NA)
            ptype = ptype.ffill().fillna("L")
        else:
            ptype = "L"
        
        # Create product code
        df_sim["_product"] = ptype.astype(str) + thickness.astype(str).str.replace(".0", "", regex=False)
        
        print(f"\nProduct codes created:")
        prod_counts = df_sim["_product"].value_counts()
        for p, c in prod_counts.head(15).items():
            valid = "✓" if p in PRODUCT_SPECIFICATIONS else "✗"
            print(f"  {valid} {p}: {c:,}")
        
        # Filter to valid products
        valid = df_sim["_product"].isin(PRODUCT_SPECIFICATIONS.keys())
        before = len(df_sim)
        df_sim = df_sim[valid]
        print(f"\n[AFTER Valid Products] {len(df_sim):,} rows (removed {before - len(df_sim):,})")
        result["steps"].append(("After Valid Products", len(df_sim)))
    
    print("\n" + "="*70)
    print(f"FINAL EXPECTED COUNT: {len(df_sim):,} rows")
    print("="*70)
    
    result["final_expected"] = len(df_sim)
    
    return result


def build_intervals(row: pd.Series) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Build zone intervals for a single wagon row."""
    intervals = []
    prev_end = None

    for z in CONFIG["zones_seq"]:
        z_in = row.get(f"{z}_in", pd.NaT)
        if pd.isna(z_in):
            z_in = prev_end or row.get("t0", pd.NaT)
        
        if pd.isna(z_in):
            continue

        z_dur = row.get(f"{z}_dur", pd.NaT)
        if pd.isna(z_dur):
            continue

        z_out = z_in + z_dur
        if z_out > z_in:
            intervals.append((z, z_in, z_out))
            prev_end = z_out

    return intervals


def explode_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Explode wagon data into zone intervals."""
    logger.info("Creating zone intervals...")
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
    logger.info(f"Created {len(result)} zone intervals from {len(df)} wagon rows")
    return result


def allocate_energy(e: pd.DataFrame, ivals: pd.DataFrame) -> pd.DataFrame:
    """
    Allocate energy to products based on time overlap.
    
    FIXED: 
    - Automatically detects overlapping time period between energy and wagon data
    - Only uses data from the overlapping period
    - Reports any data excluded from analysis
    """
    logger.info("="*70)
    logger.info("ALLOCATING ENERGY TO PRODUCTS")
    logger.info("="*70)
    
    if ivals.empty:
        logger.warning("No intervals to allocate energy to!")
        return pd.DataFrame()
    
    if e.empty:
        logger.warning("No energy data to allocate!")
        return pd.DataFrame()
    
    # ========================================
    # STEP 1: FIND TIME RANGES
    # ========================================
    logger.info("\n[STEP 1] Analyzing time periods...")
    
    # Energy time range
    energy_start = e["E_start"].min()
    energy_end = e["E_end"].max()
    energy_total_hours = len(e)
    energy_total_thermal = e["E_thermal_total_kWh"].sum()
    energy_total_electrical = e["E_el_kWh"].sum()
    
    logger.info(f"  ENERGY FILE:")
    logger.info(f"    Range: {energy_start.strftime('%Y-%m-%d %H:%M')} to {energy_end.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"    Total hours: {energy_total_hours:,}")
    logger.info(f"    Total thermal: {energy_total_thermal:,.0f} kWh")
    logger.info(f"    Total electrical: {energy_total_electrical:,.0f} kWh")
    
    # Wagon intervals time range
    wagon_start = ivals["P_start"].min()
    wagon_end = ivals["P_end"].max()
    wagon_total_intervals = len(ivals)
    wagon_total_volume = ivals["m3"].sum()
    
    logger.info(f"  WAGON FILE:")
    logger.info(f"    Range: {wagon_start.strftime('%Y-%m-%d %H:%M')} to {wagon_end.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"    Total intervals: {wagon_total_intervals:,}")
    logger.info(f"    Total volume: {wagon_total_volume:,.2f} m³")
    
    # ========================================
    # STEP 2: CALCULATE OVERLAP PERIOD
    # ========================================
    logger.info("\n[STEP 2] Calculating overlap period...")
    
    # Overlap is: max(start dates) to min(end dates)
    overlap_start = max(energy_start, wagon_start)
    overlap_end = min(energy_end, wagon_end)
    
    # Check if there's any overlap
    if overlap_start >= overlap_end:
        logger.error("="*50)
        logger.error("❌ CRITICAL: NO OVERLAP BETWEEN FILES!")
        logger.error("="*50)
        logger.error(f"  Energy file: {energy_start.strftime('%Y-%m-%d')} to {energy_end.strftime('%Y-%m-%d')}")
        logger.error(f"  Wagon file:  {wagon_start.strftime('%Y-%m-%d')} to {wagon_end.strftime('%Y-%m-%d')}")
        logger.error("  These files do not cover any common time period!")
        raise ValueError(
            f"No overlap between energy file ({energy_start.strftime('%Y-%m-%d')} to {energy_end.strftime('%Y-%m-%d')}) "
            f"and wagon file ({wagon_start.strftime('%Y-%m-%d')} to {wagon_end.strftime('%Y-%m-%d')})"
        )
    
    overlap_duration = overlap_end - overlap_start
    overlap_days = overlap_duration.days
    
    logger.info(f"  ✓ OVERLAP PERIOD FOUND:")
    logger.info(f"    Start: {overlap_start.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"    End:   {overlap_end.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"    Duration: {overlap_days} days")
    
    # ========================================
    # STEP 3: FILTER ENERGY DATA TO OVERLAP
    # ========================================
    logger.info("\n[STEP 3] Filtering energy data to overlap period...")
    
    # Energy BEFORE overlap (will be excluded)
    e_before_overlap = e[e["E_end"] <= overlap_start]
    e_before_count = len(e_before_overlap)
    e_before_thermal = e_before_overlap["E_thermal_total_kWh"].sum()
    e_before_electrical = e_before_overlap["E_el_kWh"].sum()
    
    # Energy AFTER overlap (will be excluded)
    e_after_overlap = e[e["E_start"] >= overlap_end]
    e_after_count = len(e_after_overlap)
    e_after_thermal = e_after_overlap["E_thermal_total_kWh"].sum()
    e_after_electrical = e_after_overlap["E_el_kWh"].sum()
    
    # Energy IN overlap (will be used)
    e_in_overlap = e[
        (e["E_start"] >= overlap_start) & 
        (e["E_end"] <= overlap_end)
    ].copy()
    e_in_count = len(e_in_overlap)
    e_in_thermal = e_in_overlap["E_thermal_total_kWh"].sum()
    e_in_electrical = e_in_overlap["E_el_kWh"].sum()
    
    if e_before_count > 0:
        logger.warning(f"  ⚠️ EXCLUDED (before overlap): {e_before_count:,} hours")
        logger.warning(f"     Energy excluded: {e_before_thermal:,.0f} kWh thermal, {e_before_electrical:,.0f} kWh electrical")
        logger.warning(f"     Period: {e_before_overlap['E_start'].min().strftime('%Y-%m-%d')} to {e_before_overlap['E_end'].max().strftime('%Y-%m-%d')}")
    
    if e_after_count > 0:
        logger.warning(f"  ⚠️ EXCLUDED (after overlap): {e_after_count:,} hours")
        logger.warning(f"     Energy excluded: {e_after_thermal:,.0f} kWh thermal, {e_after_electrical:,.0f} kWh electrical")
        logger.warning(f"     Period: {e_after_overlap['E_start'].min().strftime('%Y-%m-%d')} to {e_after_overlap['E_end'].max().strftime('%Y-%m-%d')}")
    
    logger.info(f"  ✓ INCLUDED (in overlap): {e_in_count:,} hours")
    logger.info(f"     Energy included: {e_in_thermal:,.0f} kWh thermal, {e_in_electrical:,.0f} kWh electrical")
    
    total_excluded_energy = e_before_thermal + e_before_electrical + e_after_thermal + e_after_electrical
    if total_excluded_energy > 0:
        excluded_pct = (total_excluded_energy / (energy_total_thermal + energy_total_electrical)) * 100
        logger.warning(f"  ⚠️ TOTAL ENERGY EXCLUDED: {total_excluded_energy:,.0f} kWh ({excluded_pct:.1f}%)")
    
    # Use filtered energy data
    e = e_in_overlap
    
    if e.empty:
        logger.error("❌ No energy data remaining after filtering to overlap period!")
        raise ValueError("No energy data in overlap period")
    
    # ========================================
    # STEP 4: FILTER WAGON INTERVALS TO OVERLAP
    # ========================================
    logger.info("\n[STEP 4] Filtering wagon intervals to overlap period...")
    
    # Wagon intervals BEFORE overlap (will be excluded)
    ivals_before_overlap = ivals[ivals["P_end"] <= overlap_start]
    ivals_before_count = len(ivals_before_overlap)
    ivals_before_volume = ivals_before_overlap["m3"].sum()
    
    # Wagon intervals AFTER overlap (will be excluded)
    ivals_after_overlap = ivals[ivals["P_start"] >= overlap_end]
    ivals_after_count = len(ivals_after_overlap)
    ivals_after_volume = ivals_after_overlap["m3"].sum()
    
    # Wagon intervals IN overlap (will be used)
    # Note: We include intervals that PARTIALLY overlap
    ivals_in_overlap = ivals[
        (ivals["P_end"] > overlap_start) & 
        (ivals["P_start"] < overlap_end)
    ].copy()
    ivals_in_count = len(ivals_in_overlap)
    ivals_in_volume = ivals_in_overlap["m3"].sum()
    
    if ivals_before_count > 0:
        logger.warning(f"  ⚠️ EXCLUDED (before overlap): {ivals_before_count:,} intervals")
        logger.warning(f"     Volume excluded: {ivals_before_volume:,.2f} m³")
    
    if ivals_after_count > 0:
        logger.warning(f"  ⚠️ EXCLUDED (after overlap): {ivals_after_count:,} intervals")
        logger.warning(f"     Volume excluded: {ivals_after_volume:,.2f} m³")
    
    logger.info(f"  ✓ INCLUDED (in overlap): {ivals_in_count:,} intervals")
    logger.info(f"     Volume included: {ivals_in_volume:,.2f} m³")
    
    # Use filtered wagon intervals
    ivals = ivals_in_overlap
    
    if ivals.empty:
        logger.error("❌ No wagon intervals remaining after filtering to overlap period!")
        raise ValueError("No wagon intervals in overlap period")
    
    # ========================================
    # STEP 5: SUMMARY BEFORE ALLOCATION
    # ========================================
    logger.info("\n[STEP 5] Summary of data to be used...")
    logger.info(f"  Overlap period: {overlap_start.strftime('%Y-%m-%d')} to {overlap_end.strftime('%Y-%m-%d')} ({overlap_days} days)")
    logger.info(f"  Energy: {e_in_count:,} hours | {e_in_thermal:,.0f} kWh thermal | {e_in_electrical:,.0f} kWh electrical")
    logger.info(f"  Wagons: {ivals_in_count:,} intervals | {ivals_in_volume:,.2f} m³")
    
    # ========================================
    # STEP 6: ALLOCATE ENERGY BY ZONE
    # ========================================
    logger.info("\n[STEP 6] Allocating energy by zone...")
    
    results = []
    zone_allocation_summary = {}

    for z_key, z_name in ZONE_ENERGY_MAPPING.items():
        thermal_col = f"E_thermal_{z_name}_kWh"

        if thermal_col not in e.columns:
            logger.warning(f"  {z_key}: Column '{thermal_col}' not found - SKIPPING")
            continue

        e_zone = e[e[thermal_col] > 0].copy()
        iv_zone = ivals[ivals["Zone"] == z_key].copy()

        if e_zone.empty:
            logger.info(f"  {z_key}: No energy data with {thermal_col} > 0")
            continue
            
        if iv_zone.empty:
            logger.info(f"  {z_key}: No wagon intervals in this zone")
            continue

        zone_thermal_input = e_zone[thermal_col].sum()
        logger.info(f"  {z_key}: {len(e_zone):,} hours × {len(iv_zone):,} intervals | Input: {zone_thermal_input:,.0f} kWh")

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

            # Find overlapping intervals
            merged = merged[
                (merged["P_end"] > merged["E_start"]) &
                (merged["P_start"] < merged["E_end"])
            ]

            if merged.empty:
                continue

            # Calculate overlap duration
            merged["latest_start"] = merged[["E_start", "P_start"]].max(axis=1)
            merged["earliest_end"] = merged[["E_end", "P_end"]].min(axis=1)
            
            merged["Overlap_h"] = (
                (merged["earliest_end"] - merged["latest_start"])
                .dt.total_seconds() / 3600
            ).clip(lower=0, upper=1)
            
            merged = merged[merged["Overlap_h"] > 0].copy()

            if merged.empty:
                continue

            # Calculate energy share
            merged["E_hour_key"] = merged["E_start"].dt.strftime("%Y-%m-%d %H:00")
            
            hour_total_overlap = merged.groupby("E_hour_key")["Overlap_h"].transform("sum")
            
            merged["Hour_share"] = safe_divide(merged["Overlap_h"], hour_total_overlap)
            
            merged["Energy_thermal_kWh"] = merged[thermal_col] * merged["Hour_share"]
            
            merged["Zone"] = z_key
            
            month_col = "Month_e" if "Month_e" in merged.columns else ("Month_p" if "Month_p" in merged.columns else None)
            if month_col is None:
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
            zone_thermal_allocated = zone_df["Energy_thermal_kWh"].sum()
            allocation_pct = (zone_thermal_allocated / zone_thermal_input * 100) if zone_thermal_input > 0 else 0
            logger.info(f"       → Allocated: {zone_thermal_allocated:,.0f} kWh ({allocation_pct:.1f}%)")
            
            zone_allocation_summary[z_key] = {
                "input": zone_thermal_input,
                "allocated": zone_thermal_allocated,
                "pct": allocation_pct
            }
            
            results.append(zone_df)
        else:
            logger.warning(f"       → No allocations made for {z_key}")

    if not results:
        logger.error("❌ No energy could be allocated to any zone!")
        return pd.DataFrame(columns=[
            "Month", "Produkt", "Zone", "Energy_thermal_kWh", 
            "Energy_electrical_kWh", "Energy_share_kWh", "Overlap_h", "Hour_share", "m3"
        ])

    final = pd.concat(results, ignore_index=True)
    
    # ========================================
    # STEP 7: ALLOCATE ELECTRICAL ENERGY
    # ========================================
    logger.info("\n[STEP 7] Allocating electrical energy...")
    
    hour_totals = final.groupby("E_hour_key").agg({
        "Overlap_h": "sum",
        "E_el_kWh": "first"
    }).reset_index()
    hour_totals = hour_totals.rename(columns={"Overlap_h": "total_overlap_all_zones"})
    
    final = final.merge(hour_totals[["E_hour_key", "total_overlap_all_zones"]], on="E_hour_key", how="left")
    
    final["Global_share"] = safe_divide(final["Overlap_h"], final["total_overlap_all_zones"])
    final["Energy_electrical_kWh"] = final["E_el_kWh"] * final["Global_share"]
    final["Energy_share_kWh"] = final["Energy_thermal_kWh"] + final["Energy_electrical_kWh"]
    
    # Clean up
    final = final.drop(columns=["E_hour_key", "E_el_kWh", "total_overlap_all_zones", "Global_share"], errors='ignore')
    
    # Remove invalid records
    final = final[
        (final["Energy_thermal_kWh"] >= 0) &
        (final["Energy_electrical_kWh"] >= 0) &
        (final["m3"] > 0)
    ].copy()
    
    # ========================================
    # STEP 8: FINAL VERIFICATION
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("ENERGY ALLOCATION SUMMARY")
    logger.info("="*70)
    
    total_thermal_allocated = final["Energy_thermal_kWh"].sum()
    total_electrical_allocated = final["Energy_electrical_kWh"].sum()
    total_energy_allocated = final["Energy_share_kWh"].sum()
    
    logger.info(f"\n  OVERLAP PERIOD: {overlap_start.strftime('%Y-%m-%d')} to {overlap_end.strftime('%Y-%m-%d')} ({overlap_days} days)")
    
    logger.info(f"\n  INPUT (in overlap period):")
    logger.info(f"    Thermal:    {e_in_thermal:,.0f} kWh")
    logger.info(f"    Electrical: {e_in_electrical:,.0f} kWh")
    logger.info(f"    TOTAL:      {e_in_thermal + e_in_electrical:,.0f} kWh")
    
    logger.info(f"\n  OUTPUT (allocated to wagons):")
    logger.info(f"    Thermal:    {total_thermal_allocated:,.0f} kWh")
    logger.info(f"    Electrical: {total_electrical_allocated:,.0f} kWh")
    logger.info(f"    TOTAL:      {total_energy_allocated:,.0f} kWh")
    
    # Calculate differences
    thermal_diff = e_in_thermal - total_thermal_allocated
    thermal_diff_pct = (thermal_diff / e_in_thermal * 100) if e_in_thermal > 0 else 0
    
    electrical_diff = e_in_electrical - total_electrical_allocated
    electrical_diff_pct = (electrical_diff / e_in_electrical * 100) if e_in_electrical > 0 else 0
    
    total_input = e_in_thermal + e_in_electrical
    total_diff = total_input - total_energy_allocated
    total_diff_pct = (total_diff / total_input * 100) if total_input > 0 else 0
    
    logger.info(f"\n  ENERGY BALANCE:")
    logger.info(f"    Thermal:    {thermal_diff:,.0f} kWh unallocated ({thermal_diff_pct:.1f}%)")
    logger.info(f"    Electrical: {electrical_diff:,.0f} kWh unallocated ({electrical_diff_pct:.1f}%)")
    logger.info(f"    TOTAL:      {total_diff:,.0f} kWh unallocated ({total_diff_pct:.1f}%)")
    
    if abs(total_diff_pct) < 5:
        logger.info(f"\n  ✅ ENERGY BALANCE OK - {100-abs(total_diff_pct):.1f}% of energy allocated")
    elif abs(total_diff_pct) < 15:
        logger.warning(f"\n  ⚠️ ENERGY BALANCE WARNING - {abs(total_diff_pct):.1f}% unallocated")
        logger.warning(f"     This may be due to hours when no wagons were in the dryer")
    else:
        logger.error(f"\n  ❌ ENERGY BALANCE ERROR - {abs(total_diff_pct):.1f}% unallocated")
        logger.error(f"     Please check time period alignment")
    
    # Zone summary
    logger.info(f"\n  BY ZONE:")
    for z_key, z_data in zone_allocation_summary.items():
        logger.info(f"    {z_key}: {z_data['allocated']:,.0f} / {z_data['input']:,.0f} kWh ({z_data['pct']:.1f}%)")
    
    logger.info(f"\n  Allocated {len(final):,} records")
    logger.info("="*70)
    
    # Store overlap info in the dataframe for later use
    final.attrs["overlap_start"] = overlap_start
    final.attrs["overlap_end"] = overlap_end
    final.attrs["overlap_days"] = overlap_days
    final.attrs["energy_input_thermal"] = e_in_thermal
    final.attrs["energy_input_electrical"] = e_in_electrical
    final.attrs["energy_excluded_thermal"] = e_before_thermal + e_after_thermal
    final.attrs["energy_excluded_electrical"] = e_before_electrical + e_after_electrical
    
    return final


def add_water_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Add water-related KPIs."""
    df = df.copy()
    
    def get_water_per_m3(product):
        if product in PRODUCT_SPECIFICATIONS:
            spec = PRODUCT_SPECIFICATIONS[product]
            water_per_mm_g = spec["slope"] * SUSPENSION_KG + spec["intercept"]
            water_per_plate_kg = (water_per_mm_g * spec["pressed_thickness_mm"]) / 1000.0
            return water_per_plate_kg / spec["volume_m3"]
        return np.mean(list(WATER_PER_M3_KG.values())) if WATER_PER_M3_KG else 200.0
    
    df["water_per_m3_formula"] = df["Produkt"].apply(get_water_per_m3)
    df["Water_kg"] = df["Volume_m3"] * df["water_per_m3_formula"]
    
    df["kWh_thermal_per_m3"] = safe_divide(df["Energy_thermal_kWh"], df["Volume_m3"])
    df["kWh_per_m3"] = safe_divide(df["Energy_kWh"], df["Volume_m3"])
    df["kWh_thermal_per_kg"] = safe_divide(df["Energy_thermal_kWh"], df["Water_kg"])
    df["kWh_per_kg"] = safe_divide(df["Energy_kWh"], df["Water_kg"])
    
    return df


def calculate_water_per_plate(product: str, pressed_thickness_mm: float = None) -> float:
    """Calculate water per plate for a given product."""
    if product not in PRODUCT_SPECIFICATIONS:
        return 0.0
    return PRODUCT_SPECIFICATIONS[product]["water_per_plate_kg"]


def calculate_water_per_m3_formula(product: str) -> float:
    """Calculate water per m³ for a given product using formula."""
    if product not in PRODUCT_SPECIFICATIONS:
        return WATER_PER_M3_KG.get(product, 200.0)
    return PRODUCT_SPECIFICATIONS[product]["water_per_m3_kg"]


def get_product_water_curve(product: str, thickness_range: list = None) -> pd.DataFrame:
    """Get water content curve for a product across thickness range."""
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
    """Predict energy consumption for planned production."""
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
    """Compute statistics about wagon usage per product."""
    logger.info("Computing wagon statistics...")
    df = wagons.copy()

    if "Entnahme" in df.columns and "t0" in df.columns:
        df["residence_h"] = (df["Entnahme"] - df["t0"]).dt.total_seconds() / 3600
    else:
        df["residence_h"] = np.nan

    stats = df.groupby("Produkt", as_index=False).agg(
        avg_m3_per_wagon=("m3", "mean"),
        avg_residence_h=("residence_h", "mean"),
        wagon_count=("m3", "count"),
    )
    stats["avg_residence_days"] = stats["avg_residence_h"] / 24.0

    return {
        "wagon_capacity_m3": stats.set_index("Produkt")["avg_m3_per_wagon"].to_dict(),
        "residence_h": stats.set_index("Produkt")["avg_residence_h"].to_dict(),
        "residence_days": stats.set_index("Produkt")["avg_residence_days"].to_dict(),
        "raw_stats": stats,
    }


def main():
    """Main entry point for testing."""
    logger.info("Dryer KPI Module loaded successfully.")
    logger.info(f"Products configured: {list(PRODUCT_SPECIFICATIONS.keys())}")
    logger.info(f"Suspension: {SUSPENSION_KG} kg")
    logger.info(f"Plates per wagon: {PLATES_PER_WAGON}")


if __name__ == "__main__":
    main()



