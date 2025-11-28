"""
Lindner Dryer KPI Calculation Module
DEBUG VERSION - Shows exactly what's being read and calculated
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG = {
    "energy_sheet": 0,
    "wagon_sheet": "Hordenwagenverfolgung",
    "wagon_header_row": 6,
    "gas_to_kwh": 11.5,
    "zones_seq": ["Z1", "Z2", "Z3", "Z4", "Z5"],
    "num_thermal_zones": 4,
}

ZONE_ENERGY_MAPPING = {
    "Z2": "Zone 2",
    "Z3": "Zone 3",
    "Z4": "Zone 4",
    "Z5": "Zone 5",
}

SUSPENSION_KG = 330

PRODUCT_SPECIFICATIONS = {
    "L28": {"product_type": "L", "final_thickness_mm": 28, "press_measurement_mm": 4, "pressed_thickness_mm": 32, "edge_length_mm": 602, "volume_m3": 0.0116, "suspension_kg": 330, "slope": -0.060, "intercept": 107.7, "formula": "-0.060x + 107.7", "water_per_mm_g": 88, "water_per_plate_kg": 2.81},
    "L30": {"product_type": "L", "final_thickness_mm": 30, "press_measurement_mm": 4, "pressed_thickness_mm": 34, "edge_length_mm": 602, "volume_m3": 0.0123, "suspension_kg": 330, "slope": -0.042, "intercept": 102.3, "formula": "-0.042x + 102.3", "water_per_mm_g": 88, "water_per_plate_kg": 3.01},
    "L34": {"product_type": "L", "final_thickness_mm": 34, "press_measurement_mm": 5, "pressed_thickness_mm": 39, "edge_length_mm": 602, "volume_m3": 0.0141, "suspension_kg": 330, "slope": -0.056, "intercept": 111.9, "formula": "-0.056x + 111.9", "water_per_mm_g": 93, "water_per_plate_kg": 3.64},
    "L36": {"product_type": "L", "final_thickness_mm": 36, "press_measurement_mm": 6, "pressed_thickness_mm": 42, "edge_length_mm": 602, "volume_m3": 0.0152, "suspension_kg": 330, "slope": -0.025, "intercept": 76.6, "formula": "-0.025x + 76.6", "water_per_mm_g": 68, "water_per_plate_kg": 2.87},
    "L38": {"product_type": "L", "final_thickness_mm": 38, "press_measurement_mm": 7, "pressed_thickness_mm": 45, "edge_length_mm": 602, "volume_m3": 0.0163, "suspension_kg": 330, "slope": -0.058, "intercept": 110.5, "formula": "-0.058x + 110.5", "water_per_mm_g": 91, "water_per_plate_kg": 4.11},
    "L42": {"product_type": "L", "final_thickness_mm": 42, "press_measurement_mm": 8, "pressed_thickness_mm": 50, "edge_length_mm": 602, "volume_m3": 0.0181, "suspension_kg": 330, "slope": -0.007, "intercept": 74.2, "formula": "-0.007x + 74.2", "water_per_mm_g": 72, "water_per_plate_kg": 3.59},
    "L44": {"product_type": "L", "final_thickness_mm": 44, "press_measurement_mm": 9, "pressed_thickness_mm": 53, "edge_length_mm": 602, "volume_m3": 0.0192, "suspension_kg": 330, "slope": -0.011, "intercept": 77.4, "formula": "-0.011x + 77.4", "water_per_mm_g": 74, "water_per_plate_kg": 3.91},
    "N40": {"product_type": "N", "final_thickness_mm": 40, "press_measurement_mm": 10, "pressed_thickness_mm": 50, "edge_length_mm": 602, "volume_m3": 0.0181, "suspension_kg": 330, "slope": -0.103, "intercept": 102.5, "formula": "-0.103x + 102.5", "water_per_mm_g": 69, "water_per_plate_kg": 3.43},
    "N44": {"product_type": "N", "final_thickness_mm": 44, "press_measurement_mm": 11, "pressed_thickness_mm": 55, "edge_length_mm": 602, "volume_m3": 0.0199, "suspension_kg": 330, "slope": -0.017, "intercept": 78.5, "formula": "-0.017x + 78.5", "water_per_mm_g": 73, "water_per_plate_kg": 4.01},
    "Y44": {"product_type": "Y", "final_thickness_mm": 44, "press_measurement_mm": 12, "pressed_thickness_mm": 56, "edge_length_mm": 602, "volume_m3": 0.0203, "suspension_kg": 330, "slope": -0.157, "intercept": 200.0, "formula": "-0.157x + 200.0", "water_per_mm_g": 148, "water_per_plate_kg": 8.30},
}

for product, spec in PRODUCT_SPECIFICATIONS.items():
    spec["water_per_m3_kg"] = spec["water_per_plate_kg"] / spec["volume_m3"]

WATER_PER_M3_KG = {p: s["water_per_m3_kg"] for p, s in PRODUCT_SPECIFICATIONS.items()}


def parse_duration_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    s = s.str.replace(r"\bh\b", "hours", regex=True).str.replace(r"\bmin\b", "minutes", regex=True)
    s = s.replace({r"^\s*$": np.nan, r"^-$": np.nan}, regex=True)
    td = pd.to_timedelta(s, errors="coerce")
    mask = td.isna() & s.notna()
    if mask.any():
        td.loc[mask] = pd.to_datetime(s[mask], errors="coerce") - pd.Timestamp("1900-01-01")
    return td


def parse_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Parse energy with DEBUG output to show column names and values."""
    logger.info("=" * 60)
    logger.info("PARSING ENERGY DATA - DEBUG MODE")
    logger.info("=" * 60)
    
    df = df.copy()
    
    # Show all column names
    logger.info(f"Available columns in energy file:")
    for i, col in enumerate(df.columns):
        logger.info(f"  [{i}] '{col}'")
    
    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], errors="coerce")
    df = df[df["Zeitstempel"].notna()].copy()
    df["Month"] = df["Zeitstempel"].dt.month
    df["Year"] = df["Zeitstempel"].dt.year

    # Parse thermal (gas) energy per zone
    total_gas_m3 = 0
    total_thermal_kwh = 0
    
    for z_key, z_name in ZONE_ENERGY_MAPPING.items():
        gas_col = f"Gasmenge, {z_name} [m³]"
        thermal_col = f"E_thermal_{z_name}_kWh"
        
        # Try to find matching column
        matching_cols = [c for c in df.columns if z_name.lower() in c.lower() and 'gas' in c.lower()]
        
        if gas_col in df.columns:
            gas_values = pd.to_numeric(df[gas_col], errors='coerce').fillna(0)
            df[thermal_col] = gas_values * CONFIG["gas_to_kwh"]
            zone_gas = gas_values.sum()
            zone_thermal = df[thermal_col].sum()
            total_gas_m3 += zone_gas
            total_thermal_kwh += zone_thermal
            logger.info(f"  ✓ {z_key} ({gas_col}): {zone_gas:,.1f} m³ → {zone_thermal:,.0f} kWh")
        elif matching_cols:
            logger.info(f"  ! {z_key}: Expected '{gas_col}' but found similar: {matching_cols}")
            # Use the first matching column
            alt_col = matching_cols[0]
            gas_values = pd.to_numeric(df[alt_col], errors='coerce').fillna(0)
            df[thermal_col] = gas_values * CONFIG["gas_to_kwh"]
            zone_gas = gas_values.sum()
            zone_thermal = df[thermal_col].sum()
            total_gas_m3 += zone_gas
            total_thermal_kwh += zone_thermal
            logger.info(f"    Using '{alt_col}': {zone_gas:,.1f} m³ → {zone_thermal:,.0f} kWh")
        else:
            df[thermal_col] = 0.0
            logger.warning(f"  ✗ {z_key}: Column '{gas_col}' NOT FOUND")

    # Parse electrical energy
    electrical_col = "Energieverbrauch, elektr. [kWh]"
    matching_el_cols = [c for c in df.columns if 'elektr' in c.lower() or 'electrical' in c.lower()]
    
    if electrical_col in df.columns:
        df["E_el_kWh"] = pd.to_numeric(df[electrical_col], errors='coerce').fillna(0)
        total_electrical = df["E_el_kWh"].sum()
        logger.info(f"  ✓ Electrical ({electrical_col}): {total_electrical:,.0f} kWh")
    elif matching_el_cols:
        logger.info(f"  ! Expected '{electrical_col}' but found: {matching_el_cols}")
        alt_col = matching_el_cols[0]
        df["E_el_kWh"] = pd.to_numeric(df[alt_col], errors='coerce').fillna(0)
        total_electrical = df["E_el_kWh"].sum()
        logger.info(f"    Using '{alt_col}': {total_electrical:,.0f} kWh")
    else:
        df["E_el_kWh"] = 0.0
        total_electrical = 0
        logger.warning(f"  ✗ Electrical column NOT FOUND")
        logger.info(f"    Searched for columns containing 'elektr' or 'electrical'")

    # Summary
    logger.info("-" * 40)
    logger.info(f"TOTALS FROM RAW DATA:")
    logger.info(f"  Gas: {total_gas_m3:,.1f} m³")
    logger.info(f"  Thermal: {total_thermal_kwh:,.0f} kWh")
    logger.info(f"  Electrical: {total_electrical:,.0f} kWh")
    
    if total_electrical > 0 and total_thermal_kwh > 0:
        ratio = total_thermal_kwh / total_electrical
        th_pct = total_thermal_kwh / (total_thermal_kwh + total_electrical) * 100
        el_pct = total_electrical / (total_thermal_kwh + total_electrical) * 100
        logger.info(f"  Ratio: Thermal/Electrical = {ratio:.1f}x")
        logger.info(f"  Split: Thermal {th_pct:.1f}% | Electrical {el_pct:.1f}%")
        
        if ratio < 3:
            logger.warning(f"  ⚠ LOW RATIO! Expected 5-10x. Check if columns are correct.")
    
    logger.info("=" * 60)

    df["E_start"] = df["Zeitstempel"]
    df["E_end"] = df["Zeitstempel"] + pd.Timedelta(hours=1)
    
    return df


def parse_wagon(df: pd.DataFrame) -> pd.DataFrame:
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

    df["Produkt"] = df.get("Produkt", pd.Series(["Unknown"] * len(df))).astype(str).str.strip()
    
    if "m³" in df.columns:
        df["m3"] = pd.to_numeric(df["m³"], errors="coerce")
    else:
        thick = pd.to_numeric(df.get("Stärke", 0), errors="coerce")
        df["m3"] = 0.605 * 0.605 * (thick + 7) / 1000.0

    for z in ("Z2", "Z3", "Z4", "Z5"):
        col = f"In {z}"
        df[f"{z}_in"] = pd.to_datetime(df[col], errors="coerce", dayfirst=True) if col in df.columns else pd.NaT

    df["Z1_in"] = df["t0"]
    df["Entnahme"] = pd.to_datetime(df.get("Entnahme-Zeit"), errors="coerce", dayfirst=True) if "Entnahme-Zeit" in df.columns else pd.NaT

    for z, later, earlier in [("Z1", "Z2_in", "t0"), ("Z2", "Z3_in", "Z2_in"), ("Z3", "Z4_in", "Z3_in"), ("Z4", "Z5_in", "Z4_in"), ("Z5", "Entnahme", "Z5_in")]:
        df[f"{z}_dur_calc"] = (df[later] - df[earlier]).dt.total_seconds() / 3600

    for z in CONFIG["zones_seq"]:
        df[f"{z}_dur"] = pd.to_timedelta(df[f"{z}_dur_calc"], unit="h")
        if f"Zeit in {z}" in df.columns:
            parsed = parse_duration_series(df[f"Zeit in {z}"])
            mask = parsed.isna() | (parsed.dt.total_seconds() < 3600)
            df[f"{z}_dur"] = parsed.where(~mask, df[f"{z}_dur"])

    df["Month"] = df["t0"].dt.month
    df["Year"] = df["t0"].dt.year
    df = df[df["t0"].notna()].copy()
    
    logger.info(f"Parsed {len(df)} wagon records")
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
    logger.info("Exploding intervals...")
    rows = []
    for _, r in df.iterrows():
        for z, s, e in build_intervals(r):
            rows.append({"WG_Nr": r.get("WG_Nr", ""), "Produkt": r["Produkt"], "m3": r["m3"], 
                        "Zone": z, "P_start": s, "P_end": e, "Month": r["Month"], "Year": r["Year"]})
    result = pd.DataFrame(rows)
    logger.info(f"Created {len(result)} intervals")
    return result


def allocate_energy(e: pd.DataFrame, ivals: pd.DataFrame) -> pd.DataFrame:
    """Allocate energy - PRESERVING the thermal/electrical ratio from raw data."""
    logger.info("=" * 60)
    logger.info("ALLOCATING ENERGY - DEBUG MODE")
    logger.info("=" * 60)
    
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

        zone_res = []
        chunk = 1000

        for i in range(0, len(iv_zone), chunk):
            part = iv_zone.iloc[i:i+chunk]
            
            e_temp = e_zone.assign(_key=1)
            p_temp = part.assign(_key=1)
            merged = e_temp.merge(p_temp, on="_key", suffixes=("_e", "_p")).drop("_key", axis=1)

            # Filter overlaps
            merged = merged[(merged["P_end"] > merged["E_start"]) & (merged["P_start"] < merged["E_end"])]
            if merged.empty:
                continue

            # Calculate overlap
            merged["Overlap_h"] = ((merged[["E_end", "P_end"]].min(axis=1) - merged[["E_start", "P_start"]].max(axis=1)).dt.total_seconds() / 3600).clip(0, 1)
            merged = merged[merged["Overlap_h"] > 0].copy()
            if merged.empty:
                continue

            # Share calculation
            merged["Hour_key"] = merged["E_start"].astype(str)
            total_overlap = merged.groupby("Hour_key")["Overlap_h"].transform("sum")
            merged["Share"] = merged["Overlap_h"] / total_overlap.replace(0, 1)

            # ✅ THERMAL: Proportional share of zone thermal energy
            merged["Energy_thermal_kWh"] = merged[thermal_col] * merged["Share"]
            
            # ✅ ELECTRICAL: Proportional share of (hourly electrical / 4 zones)
            merged["Energy_electrical_kWh"] = (merged["E_el_kWh"] / CONFIG["num_thermal_zones"]) * merged["Share"]

            # Month
            month_col = "Month_e" if "Month_e" in merged.columns else ("Month_p" if "Month_p" in merged.columns else "Month")
            if month_col not in merged.columns:
                merged["Month"] = merged["E_start"].dt.month
                month_col = "Month"

            result = merged[[month_col, "Produkt", "m3", "Overlap_h", "Share", "Energy_thermal_kWh", "Energy_electrical_kWh"]].copy()
            result = result.rename(columns={month_col: "Month"})
            result["Zone"] = z_key
            zone_res.append(result)

        if zone_res:
            results.append(pd.concat(zone_res, ignore_index=True))

    if results:
        final = pd.concat(results, ignore_index=True)
        final["Energy_share_kWh"] = final["Energy_thermal_kWh"] + final["Energy_electrical_kWh"]
        final = final[(final["Energy_thermal_kWh"] >= 0) & (final["Energy_electrical_kWh"] >= 0) & (final["m3"] > 0)].copy()
        
        # Debug output
        total_th = final["Energy_thermal_kWh"].sum()
        total_el = final["Energy_electrical_kWh"].sum()
        total = total_th + total_el
        
        logger.info("-" * 40)
        logger.info(f"ALLOCATION RESULTS:")
        logger.info(f"  Thermal allocated: {total_th:,.0f} kWh")
        logger.info(f"  Electrical allocated: {total_el:,.0f} kWh")
        logger.info(f"  Total: {total:,.0f} kWh")
        
        if total > 0:
            logger.info(f"  Thermal %: {total_th/total*100:.1f}%")
            logger.info(f"  Electrical %: {total_el/total*100:.1f}%")
            if total_el > 0:
                logger.info(f"  Ratio: {total_th/total_el:.1f}x")
        
        logger.info("=" * 60)
        
        return final
    
    return pd.DataFrame(columns=["Month", "Produkt", "Zone", "Energy_thermal_kWh", "Energy_electrical_kWh", "Energy_share_kWh", "Overlap_h", "Share", "m3"])


def add_water_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["water_per_m3_bench"] = df["Produkt"].map(WATER_PER_M3_KG).fillna(pd.Series(WATER_PER_M3_KG).mean())
    df["Water_kg"] = df["Volume_m3"] * df["water_per_m3_bench"]
    df["kWh_thermal_per_m3"] = np.where(df["Volume_m3"] > 0, df["Energy_thermal_kWh"] / df["Volume_m3"], 0)
    df["kWh_per_m3"] = np.where(df["Volume_m3"] > 0, df["Energy_kWh"] / df["Volume_m3"], 0)
    df["kWh_thermal_per_kg"] = np.where(df["Water_kg"] > 0, df["Energy_thermal_kWh"] / df["Water_kg"], 0)
    df["kWh_per_kg"] = np.where(df["Water_kg"] > 0, df["Energy_kWh"] / df["Water_kg"], 0)
    return df


def calculate_water_per_plate(product: str, pressed_thickness_mm: float = None) -> float:
    return PRODUCT_SPECIFICATIONS.get(product, {}).get("water_per_plate_kg", 0.0)


def calculate_water_per_m3_formula(product: str) -> float:
    return PRODUCT_SPECIFICATIONS.get(product, {}).get("water_per_m3_kg", 200.0)


def get_product_water_curve(product: str, thickness_range: list = None) -> pd.DataFrame:
    if product not in PRODUCT_SPECIFICATIONS:
        return pd.DataFrame()
    spec = PRODUCT_SPECIFICATIONS[product]
    if thickness_range is None:
        center = spec["pressed_thickness_mm"]
        thickness_range = [int(center * 0.7), int(center * 1.3)]
    thicknesses = np.linspace(thickness_range[0], thickness_range[1], 50)
    return pd.DataFrame({
        "Pressed_Thickness_mm": thicknesses,
        "Water_per_Plate_kg": [(spec["water_per_mm_g"] * t) / 1000.0 for t in thicknesses],
        "Product": product, "Water_per_mm_g": spec["water_per_mm_g"]
    })


def predict_production_energy(product_volumes_m3: dict, baseline_kwh_per_m3: float = None, baseline_kwh_per_kg: float = None, use_formulas: bool = True) -> dict:
    results = {"products": [], "total_volume_m3": 0, "total_water_kg": 0, "total_energy_kwh": 0}
    for product, volume_m3 in product_volumes_m3.items():
        if volume_m3 is None or volume_m3 <= 0:
            continue
        spec = PRODUCT_SPECIFICATIONS.get(product, {})
        water_per_m3 = spec.get("water_per_m3_kg", 200.0) if use_formulas else WATER_PER_M3_KG.get(product, 200.0)
        water_kg = volume_m3 * water_per_m3
        energy = baseline_kwh_per_kg * water_kg if baseline_kwh_per_kg else 0
        results["products"].append({
            "product": product, "volume_m3": volume_m3, "water_per_m3_kg": water_per_m3,
            "water_kg": water_kg, "num_plates": volume_m3 / spec.get("volume_m3", 0.015) if spec else None,
            "water_per_plate_kg": spec.get("water_per_plate_kg"), "water_per_mm_g": spec.get("water_per_mm_g"),
            "formula": spec.get("formula"), "energy_from_water_kwh": energy
        })
        results["total_volume_m3"] += volume_m3
        results["total_water_kg"] += water_kg
        results["total_energy_kwh"] += energy
    results["mean_water_per_m3"] = results["total_water_kg"] / results["total_volume_m3"] if results["total_volume_m3"] > 0 else 0
    return results


def compute_product_wagon_stats(wagons: pd.DataFrame) -> dict:
    df = wagons.copy()
    if "Entnahme" in df.columns and "t0" in df.columns:
        df["residence_h"] = (df["Entnahme"] - df["t0"]).dt.total_seconds() / 3600
    else:
        df["residence_h"] = np.nan
    stats = df.groupby("Produkt", as_index=False).agg(avg_m3_per_wagon=("m3", "mean"), avg_residence_h=("residence_h", "mean"), wagon_count=("WG_Nr", "count"))
    return {
        "wagon_capacity_m3": stats.set_index("Produkt")["avg_m3_per_wagon"].to_dict(),
        "residence_h": stats.set_index("Produkt")["avg_residence_h"].to_dict(),
        "residence_days": (stats["avg_residence_h"] / 24).to_dict(),
        "raw_stats": stats
    }


def main():
    logger.info("Module loaded - DEBUG VERSION")

if __name__ == "__main__":
    main()
