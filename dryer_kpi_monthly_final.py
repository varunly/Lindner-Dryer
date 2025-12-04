# dryer_kpi_monthly_final.py
"""
Lindner Dryer KPI Calculation Module
FIXED: Correct column name handling for 'WG-\\nNr' and 'Trock-\\nner'
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
    
    FIXED: Column names have newlines that need to be removed:
    - 'WG-\\nNr' → 'WG-Nr'
    - 'Trock-\\nner' → 'Trockner'
    """
    logger.info("="*70)
    logger.info("PARSING WAGON DATA")
    logger.info(f"Trockner filter requested: {trockner or 'None (All)'}")
    logger.info("="*70)
    
    raw_row_count = len(df)
    logger.info(f"Raw input: {raw_row_count} rows, {len(df.columns)} columns")
    
    df = df.copy()
    
    # ========================================
    # STEP 1: CLEAN COLUMN NAMES
    # Remove newlines and carriage returns
    # 'WG-\nNr' → 'WG-Nr'
    # 'Trock-\nner' → 'Trockner'
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
    for i, (orig, clean) in enumerate(zip(original_columns[:30], df.columns[:30])):
        if str(orig) != clean:
            logger.info(f"  [{i:2d}] {repr(str(orig))} → '{clean}'")
            changes_logged += 1
    
    if changes_logged == 0:
        logger.info("  No column names needed cleaning")
    
    # ========================================
    # STEP 2: FIND KEY COLUMNS
    # ========================================
    
    # Find Trockner column
    trockner_col = find_column_flexible(df, ["Trockner", "Trock-ner", "Trock-", "TROCKNER"])
    
    if trockner_col:
        logger.info(f"✓ Found Trockner column: '{trockner_col}'")
    else:
        logger.warning("✗ Trockner column NOT FOUND!")
    
    # Find wagon number column
    wagon_col = find_column_flexible(df, ["WG-Nr", "WG-Nr.", "WGNr", "WG Nr", "WG_Nr", "Wagen"])
    
    if not wagon_col:
        wagon_col = df.columns[0]
        logger.info(f"  Using first column as wagon number: '{wagon_col}'")
    else:
        logger.info(f"✓ Found wagon column: '{wagon_col}'")
    
    # Find volume column (m³)
    volume_col = find_column_flexible(df, ["m³", "m3", "Volumen", "Volume"])
    
    if not volume_col and len(df.columns) > 26:
        volume_col = df.columns[26]
        logger.info(f"  Using column at position 26 as volume: '{volume_col}'")
    elif volume_col:
        logger.info(f"✓ Found volume column: '{volume_col}'")
    else:
        logger.warning("✗ Volume column NOT FOUND!")
    
    # ========================================
    # STEP 3: SHOW TROCKNER DISTRIBUTION
    # ========================================
    if trockner_col:
        df["_trockner_clean"] = df[trockner_col].astype(str).str.strip().str.upper()
        
        logger.info("")
        logger.info("TROCKNER DISTRIBUTION (before filtering):")
        value_counts = df["_trockner_clean"].value_counts()
        for val, count in value_counts.head(10).items():
            logger.info(f"  '{val}': {count:,} rows")
    
    # ========================================
    # STEP 4: APPLY TROCKNER FILTER
    # ========================================
    if trockner and trockner_col:
        logger.info("")
        logger.info(f"APPLYING TROCKNER FILTER: '{trockner}'")
        
        count_before = len(df)
        trockner_upper = trockner.upper().strip()
        mask = df["_trockner_clean"] == trockner_upper
        match_count = mask.sum()
        
        logger.info(f"  Rows matching '{trockner_upper}': {match_count:,}")
        
        df = df[mask].copy()
        count_after = len(df)
        
        logger.info(f"  BEFORE: {count_before:,} rows")
        logger.info(f"  AFTER:  {count_after:,} rows")
        logger.info(f"  REMOVED: {count_before - count_after:,} rows")
        
        if df.empty:
            raise ValueError(f"No rows found for Trockner '{trockner}'")
        
        df = df.drop(columns=["_trockner_clean"], errors="ignore")
    elif trockner and not trockner_col:
        logger.warning(f"⚠️ Cannot filter by Trockner - column not found!")
    else:
        df = df.drop(columns=["_trockner_clean"], errors="ignore")
    
    # ========================================
    # STEP 5: PARSE VOLUME
    # ========================================
    if volume_col and volume_col in df.columns:
        logger.info("")
        logger.info(f"Reading volume from: '{volume_col}'")
        
        df["m3"] = pd.to_numeric(df[volume_col], errors='coerce')
        
        valid_count = df["m3"].notna().sum()
        positive_count = (df["m3"] > 0).sum()
        
        logger.info(f"  Numeric values: {valid_count:,}")
        logger.info(f"  Positive values: {positive_count:,}")
        
        count_before = len(df)
        df = df[df["m3"] > 0].copy()
        count_after = len(df)
        
        if count_before != count_after:
            logger.info(f"  Removed {count_before - count_after} rows with invalid volume")
        
        if not df.empty:
            logger.info(f"  Volume range: {df['m3'].min():.4f} - {df['m3'].max():.4f} m³")
            logger.info(f"  Total volume: {df['m3'].sum():,.2f} m³")
    else:
        logger.warning("Volume column not available - using default 3.5 m³")
        df["m3"] = 3.5
    
    # ========================================
    # STEP 6: RENAME WAGON COLUMN
    # ========================================
    if wagon_col and wagon_col in df.columns and wagon_col != "WG_Nr":
        df = df.rename(columns={wagon_col: "WG_Nr"})
    elif "WG_Nr" not in df.columns:
        df["WG_Nr"] = df.iloc[:, 0]
    
    # ========================================
    # STEP 7: PARSE TIMESTAMPS
    # ========================================
    press_col = find_column_flexible(df, ["Pressdat. + Zeit", "Pressdat", "Pressdatum"])
    
    if press_col and press_col in df.columns:
        df["t0"] = pd.to_datetime(df[press_col], errors="coerce")
        logger.info(f"✓ Parsed timestamps from '{press_col}'")
    else:
        date_col = find_column_flexible(df, ["Datum", "Date"])
        time_col = find_column_flexible(df, ["Zeit", "Time", "Uhrzeit"])
        
        if date_col and time_col and "Entnahme" not in str(time_col):
            df["t0"] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str),
                errors="coerce"
            )
            logger.info(f"✓ Combined timestamps from '{date_col}' + '{time_col}'")
        else:
            df["t0"] = pd.NaT
            logger.warning("✗ Could not parse timestamps")
    
    # ========================================
    # STEP 8: PARSE PRODUCT
    # ========================================
    produkt_col = find_column_flexible(df, ["Produkt", "Product", "Prod"])
    
    if produkt_col and produkt_col in df.columns:
        df["Produkt"] = df[produkt_col].astype(str).str.strip().str.upper()
        df["Produkt"] = df["Produkt"].str.replace(" ", "", regex=False)
    else:
        df["Produkt"] = "Unknown"
    
    # Log product distribution BEFORE filtering
    logger.info("")
    logger.info("Product distribution (before filtering):")
    prod_counts = df["Produkt"].value_counts()
    for prod, count in prod_counts.head(15).items():
        in_specs = "✓" if prod in PRODUCT_SPECIFICATIONS else "✗"
        logger.info(f"  {in_specs} '{prod}': {count:,} rows")
    
    # Filter to valid products
    valid_products = list(PRODUCT_SPECIFICATIONS.keys())
    count_before = len(df)
    invalid_mask = ~df["Produkt"].isin(valid_products)
    
    if invalid_mask.sum() > 0:
        invalid_prods = df.loc[invalid_mask, "Produkt"].value_counts()
        logger.warning(f"Removing {invalid_mask.sum()} rows with invalid products:")
        for prod, count in invalid_prods.items():
            logger.warning(f"  ✗ '{prod}': {count:,} rows")
    
    df = df[df["Produkt"].isin(valid_products)].copy()
    count_after = len(df)
    
    if count_before != count_after:
        logger.info(f"Product filter: {count_before:,} → {count_after:,} rows")
    
    # ========================================
    # STEP 9: PARSE ZONE ENTRY TIMES
    # ========================================
    for z in ("Z2", "Z3", "Z4", "Z5"):
        zone_col = find_column_flexible(df, [f"In {z}", f"In{z}", f"Zone {z[-1]} In"])
        if zone_col and zone_col in df.columns:
            df[f"{z}_in"] = pd.to_datetime(df[zone_col], errors="coerce", dayfirst=True)
        else:
            df[f"{z}_in"] = pd.NaT
    
    df["Z1_in"] = df["t0"]
    
    entnahme_col = find_column_flexible(df, ["Entnahme-Zeit", "EntnahmeZeit", "Entnahme Zeit", "Entnahme"])
    if entnahme_col and entnahme_col in df.columns:
        df["Entnahme"] = pd.to_datetime(df[entnahme_col], errors="coerce", dayfirst=True)
    else:
        df["Entnahme"] = pd.NaT
    
    # ========================================
    # STEP 10: CALCULATE ZONE DURATIONS
    # ========================================
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
    
    # ========================================
    # STEP 11: ADD MONTH/YEAR
    # ========================================
    df["Month"] = df["t0"].dt.month
    df["Year"] = df["t0"].dt.year
    
    count_before = len(df)
    df = df[df["t0"].notna()].copy()
    count_after = len(df)
    
    if count_before != count_after:
        logger.info(f"Removed {count_before - count_after} rows without valid timestamp")
    
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
    
    if total_rows > 0:
        logger.info("By Product:")
        product_summary = df.groupby("Produkt").agg({"m3": ["count", "sum", "mean"]}).round(4)
        product_summary.columns = ["Rows", "Volume_m3", "Avg_m3"]
        product_summary = product_summary.sort_values("Volume_m3", ascending=False)
        
        for prod, row in product_summary.iterrows():
            logger.info(f"  {prod:6s}: {int(row['Rows']):5d} rows | {row['Volume_m3']:8.2f} m³")
    
    logger.info("-"*70)
    logger.info(f"TOTAL: {total_rows:,} wagon rows | {total_vol:,.2f} m³ | {avg_vol:.4f} m³/row")
    logger.info("="*70)
    
    return df


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
    """Allocate energy to products based on time overlap."""
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
