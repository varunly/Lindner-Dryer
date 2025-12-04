# dryer_kpi_monthly_final.py
"""
Lindner Dryer KPI Calculation Module
FIXED: Correct wagon counting for Trockner A/B
Expected: Trockner A = 3692 rows
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


def find_column_by_pattern(df: pd.DataFrame, patterns: list, description: str) -> Optional[str]:
    """
    Find a column matching any of the given patterns.
    Returns the column name or None.
    """
    logger.info(f"  Searching for {description} column...")
    
    # Clean column names for comparison
    clean_cols = {str(c).replace("\n", " ").strip(): c for c in df.columns}
    
    # Try exact matches first
    for pattern in patterns:
        if pattern in df.columns:
            logger.info(f"  → Found exact match: '{pattern}'")
            return pattern
        # Try in cleaned names
        if pattern in clean_cols:
            logger.info(f"  → Found in cleaned names: '{clean_cols[pattern]}'")
            return clean_cols[pattern]
    
    # Try partial matches (case-insensitive)
    for col_clean, col_original in clean_cols.items():
        col_lower = col_clean.lower()
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in col_lower or col_lower.startswith(pattern_lower):
                logger.info(f"  → Found partial match: '{col_original}' (matches '{pattern}')")
                return col_original
    
    logger.warning(f"  → {description} column NOT FOUND!")
    return None


def parse_wagon(df: pd.DataFrame, trockner: str = None) -> pd.DataFrame:
    """
    Parse wagon data with CORRECT Trockner filtering.
    
    Key columns (with line breaks in original):
    - 'WG-\\nNr' → Wagon number
    - 'Trock-\\nner' → "A" or "B"
    - m³: Volume per wagon
    """
    logger.info("="*70)
    logger.info("PARSING WAGON DATA")
    logger.info(f"Trockner filter requested: {trockner or 'None (All)'}")
    logger.info("="*70)
    
    raw_row_count = len(df)
    logger.info(f"Raw input: {raw_row_count} rows")
    
    df = df.copy()
    
    # ========================================
    # STEP 1: CLEAN COLUMN NAMES
    # Remove newlines and extra spaces
    # ========================================
    original_columns = list(df.columns)
    
    # Replace newlines with empty string (not space) to join split words
    df.columns = [str(c).replace("\n", "").replace("\r", "").strip() for c in df.columns]
    
    logger.info(f"Columns after cleaning ({len(df.columns)}):")
    for i, (orig, clean) in enumerate(zip(original_columns[:30], df.columns[:30])):
        orig_repr = repr(orig)  # Show escape characters
        if str(orig) != clean:
            logger.info(f"  [{i:2d}] {orig_repr} → '{clean}'")
        else:
            logger.info(f"  [{i:2d}] '{clean}'")
    
    # ========================================
    # STEP 2: FIND KEY COLUMNS
    # ========================================
    
    # Find Trockner column - now looking for "Trockner" (after removing newline)
    trockner_col = None
    trockner_patterns = ["Trockner", "Trock-ner", "Trock- ner", "Trock-"]
    
    for col in df.columns:
        col_clean = str(col).strip()
        # Check various patterns
        for pattern in trockner_patterns:
            if col_clean == pattern or col_clean.lower() == pattern.lower():
                trockner_col = col
                logger.info(f"  → Found Trockner column: '{col}'")
                break
        if trockner_col:
            break
    
    # Fallback: partial match
    if not trockner_col:
        for col in df.columns:
            col_lower = str(col).lower()
            if "trock" in col_lower:
                trockner_col = col
                logger.info(f"  → Found Trockner column (partial match): '{col}'")
                break
    
    if not trockner_col:
        logger.warning("  → Trockner column NOT FOUND!")
        logger.warning(f"  → Available columns: {list(df.columns)[:20]}")
    
    # Find wagon number column - now looking for "WG-Nr" (after removing newline)
    wagon_col = None
    wagon_patterns = ["WG-Nr", "WG-Nr.", "WGNr", "WG Nr", "WG_Nr"]
    
    for col in df.columns:
        col_clean = str(col).strip()
        for pattern in wagon_patterns:
            if col_clean == pattern or col_clean.lower() == pattern.lower():
                wagon_col = col
                logger.info(f"  → Found wagon column: '{col}'")
                break
        if wagon_col:
            break
    
    # Fallback: first column or partial match
    if not wagon_col:
        for col in df.columns:
            if "wg" in str(col).lower():
                wagon_col = col
                logger.info(f"  → Found wagon column (partial): '{col}'")
                break
    
    if not wagon_col:
        wagon_col = df.columns[0]
        logger.info(f"  → Using first column as wagon: '{wagon_col}'")
    
    # Find volume column (m³)
    volume_col = None
    for col in df.columns:
        col_str = str(col).strip()
        if col_str == "m³" or col_str == "m3" or "³" in col_str:
            volume_col = col
            logger.info(f"  → Found volume column: '{col}'")
            break
    
    # Try position 26 (Column AA) if not found
    if not volume_col and len(df.columns) > 26:
        volume_col = df.columns[26]
        logger.info(f"  → Using column at position 26 as volume: '{volume_col}'")
    
    # ========================================
    # STEP 3: SHOW TROCKNER DISTRIBUTION (BEFORE ANY FILTER)
    # ========================================
    if trockner_col:
        # Clean the values for analysis
        df["_trockner_raw"] = df[trockner_col].astype(str).str.strip().str.upper()
        
        logger.info(f"\n{'='*50}")
        logger.info("TROCKNER DISTRIBUTION (BEFORE FILTERING)")
        logger.info(f"{'='*50}")
        
        value_counts = df["_trockner_raw"].value_counts()
        for val, count in value_counts.items():
            logger.info(f"  '{val}': {count:,} rows")
        
        # Count valid wagon rows per Trockner
        logger.info("\nBy Trockner (with valid wagon numbers):")
        for trockner_val in ["A", "B"]:
            mask_trockner = df["_trockner_raw"] == trockner_val
            count = mask_trockner.sum()
            logger.info(f"  Trockner {trockner_val}: {count:,} rows")
    
    # ========================================
    # STEP 4: FILTER BY TROCKNER (BEFORE OTHER PROCESSING)
    # ========================================
    if trockner and trockner_col:
        logger.info(f"\n{'='*50}")
        logger.info(f"APPLYING TROCKNER FILTER: '{trockner}'")
        logger.info(f"{'='*50}")
        
        count_before = len(df)
        
        # Create clean Trockner column for filtering
        df["_trockner_filter"] = df[trockner_col].astype(str).str.strip().str.upper()
        
        # Apply filter
        trockner_upper = trockner.upper().strip()
        mask = df["_trockner_filter"] == trockner_upper
        
        # Count matches
        match_count = mask.sum()
        logger.info(f"  Rows matching '{trockner_upper}': {match_count:,}")
        
        # Apply filter
        df = df[mask].copy()
        count_after = len(df)
        
        logger.info(f"  BEFORE filter: {count_before:,} rows")
        logger.info(f"  AFTER filter:  {count_after:,} rows")
        logger.info(f"  REMOVED:       {count_before - count_after:,} rows")
        
        # Clean up temp columns
        df = df.drop(columns=["_trockner_raw", "_trockner_filter"], errors="ignore")
        
        if df.empty:
            raise ValueError(f"No rows found for Trockner {trockner}")
    elif trockner and not trockner_col:
        logger.warning(f"⚠️ Cannot filter by Trockner - column not found!")
    else:
        # No filter requested, clean up temp column
        df = df.drop(columns=["_trockner_raw"], errors="ignore")
    
    # ========================================
    # STEP 5: GET VOLUME FROM m³ COLUMN
    # ========================================
    if volume_col:
        logger.info(f"\nReading volume from: '{volume_col}'")
        
        # Convert to numeric
        df["m3"] = pd.to_numeric(df[volume_col], errors='coerce')
        
        # Stats before filtering
        valid_volumes = df["m3"].notna() & (df["m3"] > 0)
        logger.info(f"  Valid volumes (>0): {valid_volumes.sum():,}")
        logger.info(f"  Invalid/zero: {(~valid_volumes).sum():,}")
        
        # DON'T filter out zero volumes yet - count them first
        zero_volume_count = (df["m3"] == 0).sum() + df["m3"].isna().sum()
        
        # Filter out invalid volumes
        count_before_vol = len(df)
        df = df[df["m3"] > 0].copy()
        count_after_vol = len(df)
        
        if count_before_vol != count_after_vol:
            logger.info(f"  Removed {count_before_vol - count_after_vol} rows with invalid volume")
        
        if not df.empty:
            logger.info(f"  Volume range: {df['m3'].min():.4f} - {df['m3'].max():.4f} m³")
            logger.info(f"  Total volume: {df['m3'].sum():,.2f} m³")
            logger.info(f"  Mean volume:  {df['m3'].mean():.4f} m³")
    else:
        logger.error("Volume column not found! Using default 3.5 m³")
        df["m3"] = 3.5
    
    # ========================================
    # STEP 6: RENAME WAGON COLUMN
    # ========================================
    if wagon_col and wagon_col != "WG_Nr" and wagon_col in df.columns:
        df = df.rename(columns={wagon_col: "WG_Nr"})
    elif "WG_Nr" not in df.columns:
        # Use first column
        df = df.rename(columns={df.columns[0]: "WG_Nr"})
    
    # ========================================
    # STEP 7: PARSE TIMESTAMPS AND OTHER COLUMNS
    # ========================================
    
    # Also clean other column names that might have newlines
    # Look for common columns with flexible matching
    
    def find_col(patterns):
        """Find column matching any pattern."""
        for col in df.columns:
            col_lower = str(col).lower()
            for p in patterns:
                if p.lower() in col_lower:
                    return col
        return None
    
    # Timestamps - look for press date/time
    press_col = find_col(["Pressdat", "Press dat", "Pressdatum"])
    
    if press_col:
        df["t0"] = pd.to_datetime(df[press_col], errors="coerce")
    else:
        # Try combining date and time columns
        date_col = find_col(["Datum"])
        time_col = find_col(["Zeit"])
        
        if date_col and time_col and "Entnahme" not in str(time_col):
            df["t0"] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str),
                errors="coerce"
            )
        else:
            df["t0"] = pd.NaT
    
    # Product
    produkt_col = find_col(["Produkt", "Product"])
    if produkt_col:
        df["Produkt"] = df[produkt_col].astype(str).str.strip()
    elif "Produkt" not in df.columns:
        df["Produkt"] = "Unknown"
    
    # Filter out rows without valid product
    valid_products = list(PRODUCT_SPECIFICATIONS.keys())
    count_before_prod = len(df)
    df = df[df["Produkt"].isin(valid_products)].copy()
    count_after_prod = len(df)
    
    if count_before_prod != count_after_prod:
        logger.info(f"Removed {count_before_prod - count_after_prod} rows with invalid product")
    
    # Zone entry times
    for z in ("Z2", "Z3", "Z4", "Z5"):
        col_pattern = f"In {z}"
        zone_col = find_col([col_pattern, f"In{z}"])
        if zone_col:
            df[f"{z}_in"] = pd.to_datetime(df[zone_col], errors="coerce", dayfirst=True)
        else:
            df[f"{z}_in"] = pd.NaT
    
    df["Z1_in"] = df["t0"]
    
    # Entnahme (removal) time
    entnahme_col = find_col(["Entnahme-Zeit", "EntnahmeZeit", "Entnahme Zeit"])
    if entnahme_col:
        df["Entnahme"] = pd.to_datetime(df[entnahme_col], errors="coerce", dayfirst=True)
    else:
        df["Entnahme"] = pd.NaT
    
    # Zone durations
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
        txt_col = find_col([f"Zeit in {z}", f"Zeitin{z}"])
        calc = f"{z}_dur_calc"
        out = f"{z}_dur"
        
        if calc in df.columns:
            df[out] = pd.to_timedelta(df[calc], unit="h")
        else:
            df[out] = pd.NaT
        
        if txt_col:
            parsed = parse_duration_series(df[txt_col])
            mask = parsed.isna() | (parsed.dt.total_seconds() < 3600)
            if out in df.columns:
                df[out] = parsed.where(~mask, df[out])
    
    # Month/Year
    df["Month"] = df["t0"].dt.month
    df["Year"] = df["t0"].dt.year
    
    # Filter rows without valid timestamp
    count_before_ts = len(df)
    df = df[df["t0"].notna()].copy()
    count_after_ts = len(df)
    
    if count_before_ts != count_after_ts:
        logger.info(f"Removed {count_before_ts - count_after_ts} rows without valid timestamp")
    
    # Store Trockner info
    df["Trockner"] = trockner if trockner else "All"
    
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
    
    # By product
    if total_rows > 0:
        product_summary = df.groupby("Produkt").agg({
            "m3": ["count", "sum", "mean"]
        }).round(4)
        product_summary.columns = ["Rows", "Volume_m3", "Avg_m3"]
        product_summary = product_summary.sort_values("Volume_m3", ascending=False)
        
        logger.info("By Product:")
        for prod, row in product_summary.iterrows():
            logger.info(f"  {prod:6s}: {int(row['Rows']):5d} rows | {row['Volume_m3']:8.2f} m³ | {row['Avg_m3']:.4f} m³/row")
    
    logger.info("-"*70)
    logger.info(f"TOTAL: {total_rows:,} wagon rows | {total_vol:,.2f} m³ | {avg_vol:.4f} m³/row")
    logger.info("="*70)
    
    return df


def build_intervals(row: pd.Series) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
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
    """Allocate energy to products."""
    logger.info("Allocating energy to products...")
    
    if ivals.empty:
        logger.warning("No intervals to allocate energy to!")
        return pd.DataFrame()
    
    results = []

    for z_key, z_name in ZONE_ENERGY_MAPPING.items():
        thermal_col = f"E_thermal_{z_name}_kWh"

        if thermal_col not in e.columns:
            continue

        e_zone = e[e[thermal_col] > 0].copy()
        iv_zone = ivals[ivals["Zone"] == z_key].copy()

        if e_zone.empty or iv_zone.empty:
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


if __name__ == "__main__":
    main()

