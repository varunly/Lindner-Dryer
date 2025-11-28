"""
Lindner Dryer KPI Calculation Module
Using MEASURED Values from Table 1 with CORRECTED Energy Allocation
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict

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
    "gas_to_kwh": 11.5,  # Calorific value of natural gas (kWh/m³)
    "zones_seq": ["Z1", "Z2", "Z3", "Z4", "Z5"],
    "num_thermal_zones": 4,  # Z2, Z3, Z4, Z5 (Z1 has no heating)
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
# SUSPENSION AMOUNT
# ---------------------------------------------------------
SUSPENSION_KG = 330

# ---------------------------------------------------------
# PRODUCT SPECIFICATIONS (MEASURED VALUES FROM TABLE 1)
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
    """
    Parse hourly energy consumption.
    
    Thermal: Gas (m³) × 11.5 kWh/m³ per zone
    Electrical: Direct from meter (total for dryer)
    """
    logger.info("Parsing energy data...")
    df = df.copy()

    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], errors="coerce")
    df = df[df["Zeitstempel"].notna()].copy()

    df["Month"] = df["Zeitstempel"].dt.month
    df["Year"] = df["Zeitstempel"].dt.year

    # Convert gas consumption to kWh for each zone
    for z_key, z_name in ZONE_ENERGY_MAPPING.items():
        gas_col = f"Gasmenge, {z_name} [m³]"
        thermal_col = f"E_thermal_{z_name}_kWh"
        
        if gas_col in df.columns:
            gas_values = pd.to_numeric(df[gas_col], errors='coerce').fillna(0)
            df[thermal_col] = gas_values * CONFIG["gas_to_kwh"]
            logger.info(f"  {z_key}: Gas column found, converted {gas_values.sum():.0f} m³ → {df[thermal_col].sum():.0f} kWh")
        else:
            df[thermal_col] = 0.0
            logger.warning(f"  {z_key}: Gas column '{gas_col}' not found")

    # Electrical energy (total for dryer)
    if "Energieverbrauch, elektr. [kWh]" in df.columns:
        df["E_el_kWh"] = pd.to_numeric(
            df["Energieverbrauch, elektr. [kWh]"], errors='coerce'
        ).fillna(0.0)
        logger.info(f"  Electrical: {df['E_el_kWh'].sum():.0f} kWh total")
    else:
        df["E_el_kWh"] = 0.0
        logger.warning("  Electrical column not found")

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

    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

    if "Pressdat. + Zeit" in df.columns:
        df["t0"] = pd.to_datetime(df["Pressdat. + Zeit"], errors="coerce")
    else:
        date = df.get("Pressen-Datum", pd.Series()).astype(str)
        time = df.get("Press-Zeit", pd.Series()).astype(str)
        df["t0"] = pd.to_datetime(date + " " + time, errors="coerce")

    for col in df.columns:
        if str(col).startswith("WG-"):
            df = df.rename(columns={col: "WG_Nr"})
            break

    if "Produkt" in df.columns:
        df["Produkt"] = df["Produkt"].astype(str).str.strip()
    else:
        df["Produkt"] = "Unknown"

    if "m³" in df.columns:
        df["m3"] = pd.to_numeric(df["m³"], errors="coerce")
    else:
        thick = pd.to_numeric(df.get("Stärke", 0), errors="coerce")
        df["m3"] = 0.605 * 0.605 * (thick + 7) / 1000.0

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
# ENERGY ALLOCATION (CORRECTED - PROPORTIONAL SHARING)
# =====================================================================
def allocate_energy(e: pd.DataFrame, ivals: pd.DataFrame) -> pd.DataFrame:
    """
    Allocate energy to products based on time overlap.
    
    IMPORTANT: Energy is SHARED PROPORTIONALLY among all products 
    present in a zone during each hour.
    
    Formula:
    - Share = (Product Overlap Hours) / (Sum of All Overlaps in that Hour)
    - Allocated Energy = Zone Energy × Share
    """
    logger.info("Allocating energy to products (proportional sharing)...")
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

        logger.info(f"Processing {z_key}: {len(e_zone)} energy hours × {len(iv_zone)} intervals")

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
            merged["latest_start"] = merged[["E_start", "P_start"]].max(axis=1)
            merged["earliest_end"] = merged[["E_end", "P_end"]].min(axis=1)
            
            # Calculate overlap in hours (0 to 1)
            merged["Overlap_h"] = (
                (merged["earliest_end"] - merged["latest_start"])
                .dt.total_seconds() / 3600
            ).clip(lower=0, upper=1)
            
            # Filter out zero overlaps
            merged = merged[merged["Overlap_h"] > 0].copy()

            if merged.empty:
                continue

            # ✅ CORRECTED: Calculate TOTAL overlap per energy hour
            merged["E_hour_id"] = merged["E_start"].astype(str)
            total_overlap = merged.groupby("E_hour_id")["Overlap_h"].transform("sum")
            
            # ✅ Calculate proportional share
            merged["Share"] = merged["Overlap_h"] / total_overlap.replace(0, 1)
            
            # ✅ Allocate thermal energy proportionally
            merged["Energy_thermal_kWh"] = merged[thermal_col] * merged["Share"]
            
            # ✅ Allocate electrical energy proportionally
            # Electrical is total for dryer, divide by number of active zones
            merged["Energy_electrical_kWh"] = (merged["E_el_kWh"] / CONFIG["num_thermal_zones"]) * merged["Share"]
            
            # Cleanup
            merged.drop("E_hour_id", axis=1, inplace=True)

            # Month detection
            if "Month_e" in merged.columns:
                month_col = "Month_e"
            elif "Month_p" in merged.columns:
                month_col = "Month_p"
            else:
                merged["Month"] = merged["E_start"].dt.month
                month_col = "Month"

            result = merged[[
                month_col, "Produkt", "m3", "Overlap_h", "Share",
                "Energy_thermal_kWh", "Energy_electrical_kWh"
            ]].copy()

            result = result.rename(columns={month_col: "Month"})
            result["Zone"] = z_key

            zone_res.append(result)

        if zone_res:
            results.append(pd.concat(zone_res, ignore_index=True))

    if results:
        final = pd.concat(results, ignore_index=True)
        
        # Calculate total energy
        final["Energy_share_kWh"] = (
            final["Energy_thermal_kWh"] + final["Energy_electrical_kWh"]
        )
        
        # Remove negative values
        final = final[
            (final["Energy_thermal_kWh"] >= 0) &
            (final["Energy_electrical_kWh"] >= 0) &
            (final["Energy_share_kWh"] >= 0) &
            (final["m3"] > 0)
        ].copy()
        
        logger.info(f"Allocated {len(final)} energy records")
        return final
    
    logger.warning("No energy could be allocated")
    return pd.DataFrame(columns=[
        "Month", "Produkt", "Zone", "Energy_thermal_kWh", 
        "Energy_electrical_kWh", "Energy_share_kWh", "Overlap_h", "Share", "m3"
    ])


# =====================================================================
# ADD WATER KPIs
# =====================================================================
def add_water_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Add Water_kg and kWh_per_kg using measured values."""
    df = df.copy()

    df["water_per_m3_bench"] = df["Produkt"].map(WATER_PER_M3_KG)
    avg_water_density = pd.Series(WATER_PER_M3_KG).mean()
    df["water_per_m3_bench"] = df["water_per_m3_bench"].fillna(avg_water_density)
    
    df["Water_kg"] = df["Volume_m3"] * df["water_per_m3_bench"]
    
    df["kWh_thermal_per_m3"] = np.where(df["Volume_m3"] > 0, df["Energy_thermal_kWh"] / df["Volume_m3"], 0)
    df["kWh_per_m3"] = np.where(df["Volume_m3"] > 0, df["Energy_kWh"] / df["Volume_m3"], 0)
    df["kWh_thermal_per_kg"] = np.where(df["Water_kg"] > 0, df["Energy_thermal_kWh"] / df["Water_kg"], 0)
    df["kWh_per_kg"] = np.where(df["Water_kg"] > 0, df["Energy_kWh"] / df["Water_kg"], 0)
    
    for col in ["kWh_thermal_per_m3", "kWh_per_m3", "kWh_thermal_per_kg", "kWh_per_kg"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    
    return df


# =====================================================================
# WATER CALCULATION FUNCTIONS
# =====================================================================
def calculate_water_per_plate(product: str, pressed_thickness_mm: float = None) -> float:
    """Get measured water evaporation per plate."""
    if product not in PRODUCT_SPECIFICATIONS:
        return 0.0
    return PRODUCT_SPECIFICATIONS[product]["water_per_plate_kg"]


def calculate_water_per_m3_formula(product: str) -> float:
    """Get measured water per m³."""
    if product not in PRODUCT_SPECIFICATIONS:
        return WATER_PER_M3_KG.get(product, 200.0)
    return PRODUCT_SPECIFICATIONS[product]["water_per_m3_kg"]


def get_product_water_curve(product: str, thickness_range: list = None) -> pd.DataFrame:
    """Generate water evaporation curve."""
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
    """Predict energy using measured values."""
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
        
        energy_from_volume = baseline_kwh_per_m3 * volume_m3 if baseline_kwh_per_m3 else None
        energy_from_water = baseline_kwh_per_kg * water_kg if baseline_kwh_per_kg else None
        
        product_result = {
            "product": product,
            "volume_m3": volume_m3,
            "water_per_m3_kg": water_per_m3,
            "water_kg": water_kg,
            "num_plates": num_plates,
            "water_per_plate_kg": water_per_plate,
            "water_per_mm_g": water_per_mm,
            "formula": formula,
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


def compute_product_wagon_stats(wagons: pd.DataFrame) -> dict:
    """Compute per-product wagon statistics."""
    logger.info("Computing wagon statistics by product...")
    df = wagons.copy()

    if "Entnahme" in df.columns and "t0" in df.columns:
        df["residence_h"] = (df["Entnahme"] - df["t0"]).dt.total_seconds() / 3600
    else:
        df["residence_h"] = np.nan

    stats = df.groupby("Produkt", as_index=False).agg(
        avg_m3_per_wagon=("m3", "mean"),
        avg_residence_h=("residence_h", "mean"),
        wagon_count=("WG_Nr", "count"),
    )

    stats["avg_residence_days"] = stats["avg_residence_h"] / 24.0

    return {
        "wagon_capacity_m3": stats.set_index("Produkt")["avg_m3_per_wagon"].to_dict(),
        "residence_h": stats.set_index("Produkt")["avg_residence_h"].to_dict(),
        "residence_days": stats.set_index("Produkt")["avg_residence_days"].to_dict(),
        "raw_stats": stats,
    }


def main():
    """Standalone execution for testing."""
    logger.info("Module loaded successfully.")
    logger.info(f"Using MEASURED values | Suspension: {SUSPENSION_KG} kg")


if __name__ == "__main__":
    main()
