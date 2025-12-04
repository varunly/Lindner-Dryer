# dryer_kpi_monthly_final.py
"""
Lindner Dryer KPI Calculation Module
UPDATED: 
- Wagon count = rows with valid wagon number in Column A (WG-Nr)
- Volume read directly from m³ column (Column AA)
- Trockner filter applied correctly
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Column configurations
TROCKNER_COLUMN = "Trock-"
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


def find_wagon_number_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the wagon number column (Column A).
    This is the primary column for counting wagons.
    """
    logger.info("  Searching for wagon number column...")
    
    # First column is typically Column A
    first_col = df.columns[0]
    first_col_str = str(first_col).strip()
    
    logger.info(f"  → First column (A): '{first_col}'")
    
    # Common patterns for wagon number column
    wagon_patterns = ["WG-", "WG_", "WG Nr", "Wagen", "Wagon", "HW-", "Nr"]
    
    # Check first column
    for pattern in wagon_patterns:
        if pattern.lower() in first_col_str.lower():
            logger.info(f"  → Using first column as wagon number: '{first_col}'")
            return first_col
    
    # Search all columns
    for col in df.columns:
        col_str = str(col).strip()
        for pattern in wagon_patterns:
            if pattern.lower() in col_str.lower():
                logger.info(f"  → Found wagon column: '{col}'")
                return col
    
    # Default to first column
    logger.warning(f"  → No wagon pattern found, using first column: '{first_col}'")
    return first_col


def is_valid_wagon_number(value) -> bool:
    """
    Check if a value is a valid wagon number.
    
    A valid wagon number:
    - Is not empty/NaN
    - Is numeric OR starts with WG-/HW- prefix
    - Is not a header or summary row
    """
    if pd.isna(value):
        return False
    
    val_str = str(value).strip()
    
    # Empty
    if not val_str:
        return False
    
    # Exclude common non-wagon values
    exclude_values = [
        "nan", "none", "null", "-", "--", "---", "",
        "summe", "total", "gesamt", "sum",
        "wg-nr", "wg_nr", "wg nr", "wagon", "wagen", "hordenwagen",
        "produkt", "product", "datum", "date", "zeit", "time",
        "monat", "month", "jahr", "year",
    ]
    
    val_lower = val_str.lower()
    if val_lower in exclude_values:
        return False
    
    # Check for header-like text
    if any(val_lower.startswith(excl) for excl in ["wg-nr", "wg_nr", "produkt", "datum"]):
        return False
    
    # Valid patterns:
    
    # 1. Pure numeric (e.g., 1234, 5678)
    if val_str.isdigit():
        num = int(val_str)
        if 1 <= num <= 99999:  # Reasonable wagon number range
            return True
    
    # 2. Float that's actually an integer (e.g., "1234.0" from Excel)
    try:
        num_val = float(val_str)
        if num_val > 0 and num_val == int(num_val) and num_val < 100000:
            return True
    except (ValueError, TypeError):
        pass
    
    # 3. Prefixed wagon number (e.g., "WG-1234", "HW-5678")
    import re
    if re.match(r'^(WG|HW|A|B)?[-_]?\d{3,6}$', val_str, re.IGNORECASE):
        return True
    
    return False


def find_volume_column(df: pd.DataFrame) -> Optional[str]:
    """Find the volume column (m³) in the dataframe."""
    logger.info("  Searching for volume column...")
    
    volume_patterns = ["m³", "m3", "m\u00b3", "m^3", "Volumen", "Volume", "vol"]
    
    # Exact matches
    for pattern in volume_patterns:
        if pattern in df.columns:
            logger.info(f"  → Found volume column (exact): '{pattern}'")
            return pattern
    
    # Partial matches
    for col in df.columns:
        col_str = str(col).strip()
        col_lower = col_str.lower()
        
        if any([
            "m³" in col_str, "m3" in col_lower, "³" in col_str,
            "\u00b3" in col_str, col_lower.startswith("m³"), col_lower.startswith("m3"),
        ]):
            logger.info(f"  → Found volume column (pattern): '{col}'")
            return col
    
    # Check column position 26 (Column AA)
    try:
        if len(df.columns) > 26:
            col_aa = df.columns[26]
            sample = pd.to_numeric(df[col_aa], errors='coerce').dropna()
            if len(sample) > 0 and 0.5 < sample.mean() < 10:
                logger.info(f"  → Using column 26 (AA) as volume: '{col_aa}'")
                return col_aa
    except Exception as e:
        logger.warning(f"  → Error checking column 26: {e}")
    
    # Log all columns for debugging
    logger.warning("  Volume column not found! All columns:")
    for i, col in enumerate(df.columns[:30]):
        logger.info(f"    [{i:2d}] '{col}'")
    
    return None


def find_trockner_column(df: pd.DataFrame) -> Optional[str]:
    """Find the Trockner column in the dataframe."""
    if TROCKNER_COLUMN in df.columns:
        return TROCKNER_COLUMN
    
    for col in df.columns:
        col_clean = str(col).strip()
        if "Trock" in col_clean or "trock" in col_clean.lower():
            logger.info(f"  → Found Trockner column: '{col}'")
            return col
    
    return None


def parse_wagon(df: pd.DataFrame, trockner: str = None) -> pd.DataFrame:
    """
    Parse wagon data.
    
    WAGON COUNTING METHOD:
    1. Find Column A (wagon number column)
    2. Count rows where Column A has a valid wagon number
    3. If Trockner is specified, only count wagons for that Trockner
    
    Each row with a valid wagon number = 1 wagon
    Volume is read directly from the m³ column
    """
    logger.info("="*70)
    logger.info("PARSING WAGON DATA")
    logger.info("="*70)
    
    original_row_count = len(df)
    logger.info(f"Raw input: {original_row_count} rows")
    
    df = df.copy()
    
    # Clean column names
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    
    # Log columns
    logger.info(f"Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns[:30]):
        logger.info(f"  [{i:2d}] '{col}'")
    
    # ========================================
    # STEP 1: FIND WAGON NUMBER COLUMN (Column A)
    # ========================================
    wagon_col = find_wagon_number_column(df)
    
    if not wagon_col:
        raise ValueError("Cannot find wagon number column (Column A)")
    
    # Show sample values
    sample_values = df[wagon_col].head(15).tolist()
    logger.info(f"  Sample values in '{wagon_col}': {sample_values}")
    
    # ========================================
    # STEP 2: FILTER TO VALID WAGON ROWS ONLY
    # ========================================
    logger.info("Filtering valid wagon rows...")
    
    # Apply validation to each row
    valid_mask = df[wagon_col].apply(is_valid_wagon_number)
    
    valid_count = valid_mask.sum()
    invalid_count = (~valid_mask).sum()
    
    logger.info(f"  → Valid wagon rows: {valid_count}")
    logger.info(f"  → Invalid/header rows: {invalid_count}")
    
    # Show some invalid values for debugging
    if invalid_count > 0:
        invalid_samples = df.loc[~valid_mask, wagon_col].head(10).tolist()
        logger.info(f"  → Sample invalid values: {invalid_samples}")
    
    # Keep only valid wagon rows
    df = df[valid_mask].copy()
    
    logger.info(f"After wagon filter: {len(df)} valid wagons")
    
    if df.empty:
        raise ValueError("No valid wagon rows found in Column A")
    
    # Rename to standard name
    if wagon_col != "WG_Nr":
        df = df.rename(columns={wagon_col: "WG_Nr"})
    
    # ========================================
    # STEP 3: FILTER BY TROCKNER (if specified)
    # ========================================
    if trockner:
        trockner_col = find_trockner_column(df)
        
        if trockner_col:
            logger.info(f"Filtering for Trockner {trockner}...")
            
            # Show distribution before filter
            trockner_values = df[trockner_col].astype(str).str.strip().str.upper()
            value_counts = trockner_values.value_counts()
            logger.info(f"  → Trockner distribution:")
            for val, count in value_counts.items():
                logger.info(f"      {val}: {count} wagons")
            
            # Apply filter
            before_filter = len(df)
            mask = trockner_values == trockner.upper().strip()
            df = df[mask].copy()
            after_filter = len(df)
            
            logger.info(f"  → Filtered: {before_filter} → {after_filter} wagons (Trockner {trockner})")
            
            if df.empty:
                available = list(value_counts.index)
                raise ValueError(f"No wagons found for Trockner {trockner}. Available: {available}")
        else:
            logger.warning(f"  → Trockner column not found, skipping filter")
    
    # ========================================
    # STEP 4: GET VOLUME FROM m³ COLUMN
    # ========================================
    volume_col = find_volume_column(df)
    
    if volume_col:
        logger.info(f"Reading volume from column: '{volume_col}'")
        
        # Show sample raw values
        sample_vol = df[volume_col].head(10).tolist()
        logger.info(f"  → Sample raw values: {sample_vol}")
        
        # Convert to numeric
        df["m3"] = pd.to_numeric(df[volume_col], errors='coerce')
        
        # Statistics
        valid_vol = df["m3"].notna().sum()
        invalid_vol = df["m3"].isna().sum()
        
        logger.info(f"  → Valid volumes: {valid_vol}")
        logger.info(f"  → Invalid volumes: {invalid_vol}")
        
        # Fill NaN with 0 (will be filtered out)
        df["m3"] = df["m3"].fillna(0)
        
        # Filter out zero/negative volumes
        before_vol_filter = len(df)
        df = df[df["m3"] > 0].copy()
        after_vol_filter = len(df)
        
        if before_vol_filter != after_vol_filter:
            logger.info(f"  → Removed {before_vol_filter - after_vol_filter} rows with volume ≤ 0")
        
        logger.info(f"  → Final wagon count: {len(df)}")
        logger.info(f"  → Volume range: {df['m3'].min():.4f} - {df['m3'].max():.4f} m³")
        logger.info(f"  → Total volume: {df['m3'].sum():.2f} m³")
        logger.info(f"  → Avg volume/wagon: {df['m3'].mean():.4f} m³")
    else:
        logger.error("Volume column not found! Using fallback calculation...")
        df = calculate_fallback_volume(df)
    
    # ========================================
    # STEP 5: PARSE OTHER FIELDS
    # ========================================
    
    # Timestamps
    if "Pressdat. + Zeit" in df.columns:
        df["t0"] = pd.to_datetime(df["Pressdat. + Zeit"], errors="coerce")
    else:
        date = df.get("Pressen-Datum", pd.Series()).astype(str)
        time = df.get("Press-Zeit", pd.Series()).astype(str)
        df["t0"] = pd.to_datetime(date + " " + time, errors="coerce")
    
    # Product
    if "Produkt" in df.columns:
        df["Produkt"] = df["Produkt"].astype(str).str.strip()
    else:
        df["Produkt"] = "Unknown"
    
    # Zone entry times
    for z in ("Z2", "Z3", "Z4", "Z5"):
        col = f"In {z}"
        df[f"{z}_in"] = pd.to_datetime(df[col], errors="coerce", dayfirst=True) if col in df.columns else pd.NaT
    
    df["Z1_in"] = df["t0"]
    
    if "Entnahme-Zeit" in df.columns:
        df["Entnahme"] = pd.to_datetime(df["Entnahme-Zeit"], errors="coerce", dayfirst=True)
    else:
        df["Entnahme"] = pd.NaT
    
    # Calculate zone durations
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
    
    # Month/Year
    df["Month"] = df["t0"].dt.month
    df["Year"] = df["t0"].dt.year
    
    # Filter rows without valid timestamp
    before_ts = len(df)
    df = df[df["t0"].notna()].copy()
    after_ts = len(df)
    
    if before_ts != after_ts:
        logger.warning(f"Removed {before_ts - after_ts} rows without valid timestamp")
    
    # Store Trockner info
    df["Trockner"] = trockner if trockner else "ALL"
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    logger.info("")
    logger.info("="*70)
    logger.info(f"FINAL RESULT - TROCKNER {trockner or 'ALL'}")
    logger.info("="*70)
    
    # By product
    product_summary = df.groupby("Produkt").agg({
        "m3": ["count", "sum", "mean"]
    }).round(4)
    product_summary.columns = ["Wagons", "Volume_m3", "Avg_m3"]
    product_summary = product_summary.sort_values("Volume_m3", ascending=False)
    
    logger.info("By Product:")
    for prod, row in product_summary.iterrows():
        logger.info(f"  {prod:6s}: {int(row['Wagons']):5d} wagons | {row['Volume_m3']:8.2f} m³ | {row['Avg_m3']:.4f} m³/wagon")
    
    total_wagons = len(df)
    total_volume = df["m3"].sum()
    avg_volume = total_volume / total_wagons if total_wagons > 0 else 0
    
    logger.info("-"*70)
    logger.info(f"TOTAL: {total_wagons:,} wagons | {total_volume:,.2f} m³ | {avg_volume:.4f} m³/wagon")
    logger.info("="*70)
    
    return df


def calculate_fallback_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume from product specifications as fallback."""
    logger.warning("Using fallback volume calculation...")
    
    def get_volume(row):
        product = str(row.get("Produkt", "")).strip()
        if product in PRODUCT_SPECIFICATIONS:
            return PRODUCT_SPECIFICATIONS[product]["volume_m3"] * PLATES_PER_WAGON
        return 3.5  # Default average
    
    df["m3"] = df.apply(get_volume, axis=1)
    
    logger.info(f"Fallback: {df['m3'].sum():.2f} m³ total, {df['m3'].mean():.4f} m³/wagon")
    
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
    logger.info(f"Created {len(result)} zone intervals from {len(df)} wagons")
    return result


def allocate_energy(e: pd.DataFrame, ivals: pd.DataFrame) -> pd.DataFrame:
    """Allocate energy to products."""
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
            logger.info(f"Skipping {z_key}: no data")
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
        pct_th = (total_th / total_energy) * 100
        pct_el = (total_el / total_energy) * 100
        logger.info(f"✓ Allocated: Thermal={total_th:,.0f} kWh ({pct_th:.1f}%), Electrical={total_el:,.0f} kWh ({pct_el:.1f}%)")
    
    logger.info(f"Allocated {len(final)} records")
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
    logger.info("Module loaded successfully.")
    logger.info(f"Products: {len(PRODUCT_SPECIFICATIONS)}")
    logger.info(f"Zones: {CONFIG['zones_seq']}")


if __name__ == "__main__":
    main()
