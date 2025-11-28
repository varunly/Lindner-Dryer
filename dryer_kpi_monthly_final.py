"""
Lindner Dryer KPI Calculation Module (Table 1 Individual Product Formulas)

Uses individual product formulas from measurement data:
- Formula: water_per_mm_g = slope × suspension_kg + intercept
- Where suspension_kg = 330 kg (fixed for all products)
- Total water per plate = water_per_mm_g × pressed_thickness_mm / 1000
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union
from functools import lru_cache
import json
from datetime import datetime, timedelta

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
CONFIG = {
    "energy_sheet": 0,
    "wagon_sheet": "Hordenwagenverfolgung",
    "wagon_header_row": 6,
    "gas_to_kwh": 11.5,
    "zones_seq": ["Z1", "Z2", "Z3", "Z4", "Z5"],
    "cache_size": 128,  # For LRU cache
}

# ---------------------------------------------------------
# Zone energy mapping
# ---------------------------------------------------------
ZONE_ENERGY_MAPPING = {
    "Z2": "Zone 2",
    "Z3": "Zone 3",
    "Z4": "Zone 4",
    "Z5": "Zone 5",
}

# ---------------------------------------------------------
# SUSPENSION AMOUNT (fixed for all products)
# ---------------------------------------------------------
SUSPENSION_KG = 330

# ---------------------------------------------------------
# INDIVIDUAL PRODUCT SPECIFICATIONS & FORMULAS (TABLE 1)
# Each product has its own unique formula for water evaporation
# Formula: water_per_mm_g = slope × suspension_kg + intercept
# ---------------------------------------------------------
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
        "intercept": 120.5,
        "formula": "-0.103x + 120.5",
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
    },
    "Y44": {
        "product_type": "Y",
        "final_thickness_mm": 44,
        "press_measurement_mm": 12,
        "pressed_thickness_mm": 56,
        "edge_length_mm": 602,
        "volume_m3": 0.0203,
        "suspension_kg": 330,
        "slope": -0.160,
        "intercept": 200.0,
        "formula": "-0.160x + 200.0",
    },
}

# Calculate water values for each product using individual formulas
logger.info("=== Calculating Water Values (Individual Product Formulas - Table 1) ===")
for product, spec in PRODUCT_SPECIFICATIONS.items():
    # Step 1: Calculate water per mm using product-specific formula
    # water_per_mm_g = slope × suspension_kg + intercept
    water_per_mm_g = (spec["slope"] * SUSPENSION_KG) + spec["intercept"]
    spec["water_per_mm_g"] = water_per_mm_g
    
    # Step 2: Calculate total water per plate
    # water_per_plate_kg = (water_per_mm_g × pressed_thickness_mm) / 1000
    water_per_plate_g = water_per_mm_g * spec["pressed_thickness_mm"]
    spec["water_per_plate_kg"] = water_per_plate_g / 1000.0
    
    # Step 3: Calculate water per m³
    # water_per_m3_kg = water_per_plate_kg / volume_per_plate_m3
    spec["water_per_m3_kg"] = spec["water_per_plate_kg"] / spec["volume_m3"]
    
    logger.info(
        f"{product}: {spec['water_per_mm_g']:.2f} g/mm → "
        f"{spec['water_per_plate_kg']:.2f} kg/plate → "
        f"{spec['water_per_m3_kg']:.1f} kg/m³ "
        f"(formula: {spec['formula']})"
    )

# Create lookup dictionary for water per m³
WATER_PER_M3_KG = {
    product: spec["water_per_m3_kg"]
    for product, spec in PRODUCT_SPECIFICATIONS.items()
}

# Save product specifications to a file for validation
def save_product_specs():
    """Save product specifications to a JSON file for validation."""
    try:
        with open("product_specs.json", "w") as f:
            json.dump(PRODUCT_SPECIFICATIONS, f, indent=2)
        logger.info("Product specifications saved to product_specs.json")
    except Exception as e:
        logger.error(f"Failed to save product specifications: {e}")

save_product_specs()

# =====================================================================
# Data Validation Functions
# =====================================================================

def validate_energy_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate energy data for required columns and data types.
    
    Args:
        df: Energy data DataFrame
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check for required columns
    required_columns = ["Zeitstempel"]
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    # Check for zone energy columns
    zone_columns = [f"Gasmenge, {z_name} [m³]" for z_name in ZONE_ENERGY_MAPPING.values()]
    for col in zone_columns:
        if col not in df.columns:
            errors.append(f"Missing zone energy column: {col}")
    
    # Check for electrical energy column
    if "Energieverbrauch, elektr. [kWh]" not in df.columns:
        errors.append("Missing electrical energy column: Energieverbrauch, elektr. [kWh]")
    
    # Check for empty data
    if df.empty:
        errors.append("Energy data is empty")
    
    return len(errors) == 0, errors

def validate_wagon_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate wagon data for required columns and data types.
    
    Args:
        df: Wagon data DataFrame
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check for required columns
    required_columns = ["Produkt", "m³"]
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    # Check for zone entry columns
    for z in ("Z2", "Z3", "Z4", "Z5"):
        col = f"In {z}"
        if col not in df.columns:
            errors.append(f"Missing zone entry column: {col}")
    
    # Check for empty data
    if df.empty:
        errors.append("Wagon data is empty")
    
    # Check for valid product names
    valid_products = set(PRODUCT_SPECIFICATIONS.keys())
    if "Produkt" in df.columns:
        invalid_products = set(df["Produkt"].unique()) - valid_products
        if invalid_products:
            errors.append(f"Invalid product names: {invalid_products}")
    
    return len(errors) == 0, errors

def validate_product_volumes(product_volumes: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate product volumes for prediction.
    
    Args:
        product_volumes: Dictionary of product volumes
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check for valid product names
    valid_products = set(PRODUCT_SPECIFICATIONS.keys())
    invalid_products = set(product_volumes.keys()) - valid_products
    if invalid_products:
        errors.append(f"Invalid product names: {invalid_products}")
    
    # Check for valid volumes
    for product, volume in product_volumes.items():
        if volume is None or volume <= 0:
            errors.append(f"Invalid volume for product {product}: {volume}")
    
    return len(errors) == 0, errors

# =====================================================================
# Duration parsing helper
# =====================================================================
def parse_duration_series(s: pd.Series) -> pd.Series:
    """Convert 'Zeit in Zx' text to Timedelta."""
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

# =====================================================================
# ENERGY PARSING
# =====================================================================
def parse_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Parse hourly energy consumption (thermal + electrical)."""
    logger.info("Parsing energy data...")
    df = df.copy()
    
    # Validate data
    is_valid, errors = validate_energy_data(df)
    if not is_valid:
        error_msg = "Energy data validation failed:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], errors="coerce")
    df = df[df["Zeitstempel"].notna()].copy()

    df["Month"] = df["Zeitstempel"].dt.month
    df["Year"] = df["Zeitstempel"].dt.year

    # Convert gas consumption to kWh for each zone (THERMAL)
    for z_key, z_name in ZONE_ENERGY_MAPPING.items():
        gas_col = f"Gasmenge, {z_name} [m³]"
        thermal_col = f"E_thermal_{z_name}_kWh"
        
        if gas_col in df.columns:
            df[thermal_col] = pd.to_numeric(df[gas_col], errors='coerce').fillna(0) * CONFIG["gas_to_kwh"]
        else:
            df[thermal_col] = 0.0

    # Parse electrical energy (ELECTRICAL)
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

# =====================================================================
# WAGON PARSING
# =====================================================================
def parse_wagon(df: pd.DataFrame) -> pd.DataFrame:
    """Parse wagon tracking data."""
    logger.info("Parsing wagon data...")
    df = df.copy()
    
    # Validate data
    is_valid, errors = validate_wagon_data(df)
    if not is_valid:
        error_msg = "Wagon data validation failed:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

    # dryer entry time
    if "Pressdat. + Zeit" in df.columns:
        df["t0"] = pd.to_datetime(df["Pressdat. + Zeit"], errors="coerce")
    else:
        date = df.get("Pressen-Datum", pd.Series()).astype(str)
        time = df.get("Press-Zeit", pd.Series()).astype(str)
        df["t0"] = pd.to_datetime(date + " " + time, errors="coerce")

    # find wagon number column
    for col in df.columns:
        if str(col).startswith("WG-"):
            df = df.rename(columns={col: "WG_Nr"})
            break

    # clean product names
    if "Produkt" in df.columns:
        df["Produkt"] = df["Produkt"].astype(str).str.strip()
    else:
        df["Produkt"] = "Unknown"

    # volume
    if "m³" in df.columns:
        df["m3"] = pd.to_numeric(df["m³"], errors="coerce")
    else:
        thick = pd.to_numeric(df.get("Stärke", 0), errors="coerce")
        df["m3"] = 0.605 * 0.605 * (thick + 7) / 1000.0

    # zone entry timestamps
    for z in ("Z2", "Z3", "Z4", "Z5"):
        col = f"In {z}"
        df[f"{z}_in"] = (
            pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if col in df.columns else pd.NaT
        )

    # Z1
    df["Z1_in"] = df["t0"]

    # exit time
    if "Entnahme-Zeit" in df.columns:
        df["Entnahme"] = pd.to_datetime(df["Entnahme-Zeit"], errors="coerce", dayfirst=True)
    else:
        df["Entnahme"] = pd.NaT

    # estimated durations
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

    # parse & validate text durations
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
    
    logger.info(f"Parsed {len(df)} wagon records")
    return df

# =====================================================================
# BUILD ZONE INTERVALS
# =====================================================================
def build_intervals(row: pd.Series) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Create (zone, start, end) list for each wagon row."""
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
    """Explode wagons into one row per zone interval."""
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
            })
    
    result = pd.DataFrame(rows)
    logger.info(f"Created {len(result)} zone intervals")
    return result

# =====================================================================
# ENERGY ALLOCATION (OPTIMIZED)
# =====================================================================
def allocate_energy(e: pd.DataFrame, ivals: pd.DataFrame) -> pd.DataFrame:
    """Allocate energy to products based on time overlap (thermal + electrical)."""
    logger.info("Allocating energy to products...")
    results = []

    for z_key, z_name in ZONE_ENERGY_MAPPING.items():
        thermal_col = f"E_thermal_{z_name}_kWh"

        if thermal_col not in e.columns:
            logger.warning(f"Thermal energy column {thermal_col} not found")
            continue

        e_zone = e[e[thermal_col] > 0].copy()
        iv_zone = ivals[ivals["Zone"] == z_key].copy()

        if e_zone.empty or iv_zone.empty:
            logger.warning(f"No data for {z_key}")
            continue

        logger.info(f"Processing {z_key}: {len(e_zone)} energy × {len(iv_zone)} intervals")

        # Use a more efficient approach for large datasets
        if len(e_zone) * len(iv_zone) > 100000:  # Threshold for using optimized approach
            logger.info(f"Using optimized allocation for large dataset ({len(e_zone)} × {len(iv_zone)})")
            zone_res = allocate_energy_optimized(e_zone, iv_zone, z_key, thermal_col)
        else:
            zone_res = allocate_energy_standard(e_zone, iv_zone, z_key, thermal_col)
        
        if zone_res:
            results.append(zone_res)

    if results:
        final = pd.concat(results, ignore_index=True)
        
        # Calculate total energy
        final["Energy_share_kWh"] = (
            final["Energy_thermal_kWh"] + final["Energy_electrical_kWh"]
        )
        
        # Final safety check
        final = final[
            (final["Energy_thermal_kWh"] >= 0) &
            (final["Energy_electrical_kWh"] >= 0) &
            (final["Energy_share_kWh"] >= 0) &
            (final["m3"] > 0)
        ].copy()
        
        logger.info(f"Allocated {len(final)} energy records (all positive)")
        return final
    
    logger.warning("No energy could be allocated")
    return pd.DataFrame(columns=[
        "Month", "Produkt", "Zone", "Energy_thermal_kWh", 
        "Energy_electrical_kWh", "Energy_share_kWh", "Overlap_h", "m3"
    ])

def allocate_energy_standard(e_zone: pd.DataFrame, iv_zone: pd.DataFrame, 
                             z_key: str, thermal_col: str) -> pd.DataFrame:
    """Standard energy allocation for smaller datasets."""
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

        # Filter overlapping intervals
        merged = merged[
            (merged["P_end"] > merged["E_start"]) &
            (merged["P_start"] < merged["E_end"])
        ]

        if merged.empty:
            continue

        # Calculate overlap
        merged["latest_start"] = merged[["E_start","P_start"]].max(axis=1)
        merged["earliest_end"] = merged[["E_end","P_end"]].min(axis=1)
        
        # Calculate overlap in hours
        merged["Overlap_h"] = (
            (merged["earliest_end"] - merged["latest_start"])
            .dt.total_seconds() / 3600
        )
        
        # Filter out negative or zero overlaps
        merged = merged[merged["Overlap_h"] > 0].copy()

        if merged.empty:
            continue
        
        # Remove duplicates
        merged = merged.drop_duplicates(
            subset=["E_start", "E_end", "WG_Nr", "Produkt", "Zone"],
            keep="first"
        )

        # Allocate thermal energy
        merged["Energy_thermal_kWh"] = merged[thermal_col] * merged["Overlap_h"]
        
        # Allocate electrical energy proportionally
        merged["Energy_electrical_kWh"] = merged["E_el_kWh"] * merged["Overlap_h"]

        # Month detection
        if "Month_e" in merged.columns:
            month_col = "Month_e"
        elif "Month_p" in merged.columns:
            month_col = "Month_p"
        elif "Month" in merged.columns:
            month_col = "Month"
        else:
            merged["Month"] = merged["E_start"].dt.month
            month_col = "Month"

        result = merged[[
            month_col, "Produkt", "m3", "Overlap_h", 
            "Energy_thermal_kWh", "Energy_electrical_kWh"
        ]].copy()

        result = result.rename(columns={month_col: "Month"})
        result["Zone"] = z_key

        zone_res.append(result)

    if zone_res:
        return pd.concat(zone_res, ignore_index=True)
    return pd.DataFrame()

def allocate_energy_optimized(e_zone: pd.DataFrame, iv_zone: pd.DataFrame, 
                             z_key: str, thermal_col: str) -> pd.DataFrame:
    """Optimized energy allocation for large datasets."""
    # Sort by start time for more efficient processing
    e_zone = e_zone.sort_values("E_start")
    iv_zone = iv_zone.sort_values("P_start")
    
    # Initialize result list
    results = []
    
    # For each wagon interval, find overlapping energy intervals
    for _, wagon_row in iv_zone.iterrows():
        wagon_start = wagon_row["P_start"]
        wagon_end = wagon_row["P_end"]
        
        # Find energy intervals that overlap with the wagon interval
        mask = (e_zone["E_end"] > wagon_start) & (e_zone["E_start"] < wagon_end)
        overlapping_energy = e_zone[mask].copy()
        
        if overlapping_energy.empty:
            continue
        
        # Calculate overlap for each energy interval
        overlapping_energy["latest_start"] = overlapping_energy["E_start"].clip(lower=wagon_start)
        overlapping_energy["earliest_end"] = overlapping_energy["E_end"].clip(upper=wagon_end)
        overlapping_energy["Overlap_h"] = (
            (overlapping_energy["earliest_end"] - overlapping_energy["latest_start"])
            .dt.total_seconds() / 3600
        )
        
        # Filter out zero or negative overlaps
        overlapping_energy = overlapping_energy[overlapping_energy["Overlap_h"] > 0]
        
        if overlapping_energy.empty:
            continue
        
        # Calculate energy allocation
        overlapping_energy["Energy_thermal_kWh"] = overlapping_energy[thermal_col] * overlapping_energy["Overlap_h"]
        overlapping_energy["Energy_electrical_kWh"] = overlapping_energy["E_el_kWh"] * overlapping_energy["Overlap_h"]
        
        # Create result rows
        for _, energy_row in overlapping_energy.iterrows():
            results.append({
                "Month": energy_row["Month"],
                "Produkt": wagon_row["Produkt"],
                "m3": wagon_row["m3"],
                "Zone": z_key,
                "Overlap_h": energy_row["Overlap_h"],
                "Energy_thermal_kWh": energy_row["Energy_thermal_kWh"],
                "Energy_electrical_kWh": energy_row["Energy_electrical_kWh"]
            })
    
    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()

# =====================================================================
# WATER CALCULATION (STANDARDIZED)
# =====================================================================
@lru_cache(maxsize=CONFIG["cache_size"])
def calculate_water_per_mm(product: str) -> float:
    """
    Calculate water per mm using product-specific formula.
    
    Args:
        product: Product code (e.g., "L36", "N40")
    
    Returns:
        Water per mm in g
    """
    if product not in PRODUCT_SPECIFICATIONS:
        logger.warning(f"Product {product} not in specifications")
        return 0.0
    
    spec = PRODUCT_SPECIFICATIONS[product]
    water_per_mm_g = (spec["slope"] * SUSPENSION_KG) + spec["intercept"]
    return water_per_mm_g

@lru_cache(maxsize=CONFIG["cache_size"])
def calculate_water_per_plate(product: str, pressed_thickness_mm: float = None) -> float:
    """
    Calculate water evaporation per plate using individual product formula.
    
    Args:
        product: Product code (e.g., "L36", "N40")
        pressed_thickness_mm: Pressed thickness in mm (optional, uses default if None)
    
    Returns:
        Water evaporation in kg per plate
    """
    if product not in PRODUCT_SPECIFICATIONS:
        logger.warning(f"Product {product} not in specifications")
        return 0.0
    
    spec = PRODUCT_SPECIFICATIONS[product]
    
    # Use provided thickness or default
    thickness = pressed_thickness_mm if pressed_thickness_mm else spec["pressed_thickness_mm"]
    
    # Get water per mm from product-specific formula
    water_per_mm = calculate_water_per_mm(product)
    
    # Calculate total water
    total_water_g = water_per_mm * thickness
    water_kg = total_water_g / 1000.0
    
    return max(water_kg, 0.0)

@lru_cache(maxsize=CONFIG["cache_size"])
def calculate_water_per_m3(product: str) -> float:
    """
    Calculate water per m³ using individual product formula.
    
    Args:
        product: Product code
    
    Returns:
        Water density in kg/m³
    """
    if product not in PRODUCT_SPECIFICATIONS:
        return WATER_PER_M3_KG.get(product, 240.0)
    
    spec = PRODUCT_SPECIFICATIONS[product]
    return spec["water_per_m3_kg"]

def calculate_water_for_volume(product: str, volume_m3: float) -> float:
    """
    Calculate water for a given volume of product.
    
    Args:
        product: Product code
        volume_m3: Volume in m³
    
    Returns:
        Water in kg
    """
    water_per_m3 = calculate_water_per_m3(product)
    return volume_m3 * water_per_m3

def calculate_water_for_plates(product: str, num_plates: int) -> float:
    """
    Calculate water for a given number of plates.
    
    Args:
        product: Product code
        num_plates: Number of plates
    
    Returns:
        Water in kg
    """
    water_per_plate = calculate_water_per_plate(product)
    return num_plates * water_per_plate

# =====================================================================
# ADD WATER KPIs (USING STANDARDIZED WATER CALCULATION)
# =====================================================================
def add_water_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Add Water_kg and kWh_per_kg using standardized water calculation."""
    df = df.copy()

    # Calculate water for each product using standardized method
    water_kg = []
    for _, row in df.iterrows():
        product = row["Produkt"]
        volume_m3 = row["Volume_m3"]
        water = calculate_water_for_volume(product, volume_m3)
        water_kg.append(water)
    
    df["Water_kg"] = water_kg
    
    # Calculate KPIs with safe division
    df["kWh_per_m3"] = np.where(
        df["Volume_m3"] > 0,
        df["Energy_kWh"] / df["Volume_m3"],
        0
    )
    
    df["kWh_per_kg"] = np.where(
        df["Water_kg"] > 0,
        df["Energy_kWh"] / df["Water_kg"],
        0
    )
    
    # Clip any negative values
    numeric_cols = ["kWh_per_m3", "kWh_per_kg"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    
    return df

# =====================================================================
# WATER-LOSS CALCULATION FUNCTIONS (USING STANDARDIZED WATER CALCULATION)
# =====================================================================
def calculate_water_per_m3_formula(product: str) -> float:
    """
    Calculate water per m³ using individual product formula.
    
    Args:
        product: Product code
    
    Returns:
        Water density in kg/m³
    """
    return calculate_water_per_m3(product)

def get_product_water_curve(product: str, thickness_range: list = None) -> pd.DataFrame:
    """
    Generate water evaporation curve for a product across thickness range.
    
    Args:
        product: Product code
        thickness_range: List of thicknesses to evaluate [min, max]
    
    Returns:
        DataFrame with thickness and water evaporation
    """
    if product not in PRODUCT_SPECIFICATIONS:
        return pd.DataFrame()
    
    spec = PRODUCT_SPECIFICATIONS[product]
    water_per_mm = calculate_water_per_mm(product)
    
    if thickness_range is None:
        center = spec["pressed_thickness_mm"]
        thickness_range = [int(center * 0.8), int(center * 1.2)]
    
    thicknesses = np.linspace(thickness_range[0], thickness_range[1], 50)
    water_values = []
    
    for t in thicknesses:
        water_kg = (water_per_mm * t) / 1000.0
        water_values.append(water_kg)
    
    return pd.DataFrame({
        "Pressed_Thickness_mm": thicknesses,
        "Water_per_Plate_kg": water_values,
        "Product": product,
        "Formula": f"{water_per_mm:.2f} g/mm × thickness (from {spec['formula']})"
    })

def predict_production_energy(
    product_volumes_m3: dict,
    baseline_kwh_per_m3: float = None,
    baseline_kwh_per_kg: float = None,
    use_formulas: bool = True
) -> dict:
    """
    Predict energy using standardized water calculation.
    
    Args:
        product_volumes_m3: dict like {"L36": 100.5, "N40": 50.2} (m³)
        baseline_kwh_per_m3: Energy efficiency (kWh/m³)
        baseline_kwh_per_kg: Specific energy (kWh/kg water)
        use_formulas: If True, use individual formulas
    
    Returns:
        dict with detailed predictions per product
    """
    # Validate input
    is_valid, errors = validate_product_volumes(product_volumes_m3)
    if not is_valid:
        error_msg = "Product volumes validation failed:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    results = {
        "products": [],
        "total_volume_m3": 0,
        "total_water_kg": 0,
        "total_energy_kwh": 0,
    }
    
    for product, volume_m3 in product_volumes_m3.items():
        if volume_m3 is None or volume_m3 <= 0:
            continue
        
        # Calculate water load using standardized method
        if use_formulas and product in PRODUCT_SPECIFICATIONS:
            spec = PRODUCT_SPECIFICATIONS[product]
            water_per_m3 = calculate_water_per_m3(product)
            water_per_plate = calculate_water_per_plate(product)
            num_plates = volume_m3 / spec["volume_m3"]
        else:
            water_per_m3 = WATER_PER_M3_KG.get(product, 240.0)
            water_per_plate = None
            num_plates = None
        
        water_kg = calculate_water_for_volume(product, volume_m3)
        
        # Calculate energy
        energy_from_volume = baseline_kwh_per_m3 * volume_m3 if baseline_kwh_per_m3 else None
        energy_from_water = baseline_kwh_per_kg * water_kg if baseline_kwh_per_kg else None
        
        product_result = {
            "product": product,
            "volume_m3": volume_m3,
            "water_per_m3_kg": water_per_m3,
            "water_kg": water_kg,
            "num_plates": num_plates,
            "water_per_plate_kg": water_per_plate,
        }
        
        if energy_from_volume:
            product_result["energy_from_volume_kwh"] = energy_from_volume
        if energy_from_water:
            product_result["energy_from_water_kwh"] = energy_from_water
        
        results["products"].append(product_result)
        results["total_volume_m3"] += volume_m3
        results["total_water_kg"] += water_kg
        
        if energy_from_water:
            results["total_energy_kwh"] += energy_from_water
        elif energy_from_volume:
            results["total_energy_kwh"] += energy_from_volume
    
    results["mean_water_per_m3"] = (
        results["total_water_kg"] / results["total_volume_m3"]
        if results["total_volume_m3"] > 0 else 0
    )
    
    return results

# =====================================================================
# PREDICTION HELPERS
# =====================================================================
def compute_product_wagon_stats(wagons: pd.DataFrame) -> dict:
    """Compute per-product wagon statistics."""
    logger.info("Computing wagon statistics by product...")
    df = wagons.copy()

    # Compute residence time
    if "Entnahme" in df.columns and "t0" in df.columns:
        df["residence_h"] = (
            (df["Entnahme"] - df["t0"]).dt.total_seconds() / 3600
        )
    else:
        df["residence_h"] = np.nan

    # Group by Product
    stats = (
        df.groupby("Produkt", as_index=False)
        .agg(
            avg_m3_per_wagon=("m3", "mean"),
            avg_residence_h=("residence_h", "mean"),
            wagon_count=("WG_Nr", "count"),
        )
    )

    stats["avg_residence_days"] = stats["avg_residence_h"] / 24.0

    # Convert to dictionaries
    wagon_capacity = stats.set_index("Produkt")["avg_m3_per_wagon"].to_dict()
    residence_h = stats.set_index("Produkt")["avg_residence_h"].to_dict()
    residence_days = stats.set_index("Produkt")["avg_residence_days"].to_dict()

    logger.info(f"Computed stats for {len(stats)} products")
    
    return {
        "wagon_capacity_m3": wagon_capacity,
        "residence_h": residence_h,
        "residence_days": residence_days,
        "raw_stats": stats,
    }

# =====================================================================
# MAIN
# =====================================================================
def main():
    """Standalone execution for testing."""
    logger.info("Module loaded successfully. Individual product formulas (Table 1) active.")
    logger.info(f"Suspension amount: {SUSPENSION_KG} kg")
    logger.info(f"Products configured: {len(PRODUCT_SPECIFICATIONS)}")

if __name__ == "__main__":
    main()
