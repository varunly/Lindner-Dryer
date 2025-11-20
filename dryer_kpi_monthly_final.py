"""
Lindner Dryer KPI Calculation Module (Final Corrected Version)

This module:
- Parses hourly energy data (kWh for each zone)
- Parses wagon tracking data (products, volumes, zone times)
- Builds zone intervals per wagon (Z1–Z5)
- Allocates zone energy to products by time-overlap
- Aggregates KPIs by month/product/zone
- Computes kWh/m³
- Uses built-in water-loss benchmarks to compute kWh/kg
- Provides a prediction helper for future product mixes
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# CONFIG (used mainly for local 'main()' testing)
# ---------------------------------------------------------
CONFIG = {
    "energy_sheet": 0,
    "wagon_sheet": "Hordenwagenverfolgung",
    "wagon_header_row": 6,
    "gas_to_kwh": 11.5,
    "zones_seq": ["Z1", "Z2", "Z3", "Z4", "Z5"],
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
# Built-in Water Benchmarks (kg per m³)
# Derived from Wasserverlust file
# ---------------------------------------------------------
WATER_PER_M3_KG = {
    "L28": 226.7,
    "L30": 231.0,
    "L32": 237.8,
    "L34": 244.7,
    "L36": 207.1,
    "L38": 235.5,
    "L40": 219.4,
    "L44": 187.2,
    "N40": 237.6,
    "N44": 237.6,
    "Y44": 439.0,
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
    """Parse hourly energy consumption."""
    logger.info("Parsing energy data...")
    df = df.copy()

    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], errors="coerce")
    df = df[df["Zeitstempel"].notna()].copy()

    df["Month"] = df["Zeitstempel"].dt.month
    df["Year"] = df["Zeitstempel"].dt.year

    for z_key, z_name in ZONE_ENERGY_MAPPING.items():
        gas_col = f"Gasmenge, {z_name} [m³]"
        out_col = f"E_{z_name}_kWh"
        if gas_col in df.columns:
            df[out_col] = df[gas_col] * CONFIG["gas_to_kwh"]
        else:
            df[out_col] = 0.0

    if "Energieverbrauch, elektr. [kWh]" in df.columns:
        df["E_el_kWh"] = df["Energieverbrauch, elektr. [kWh]"]
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
# ENERGY ALLOCATION (FIXED)
# =====================================================================
def allocate_energy(e: pd.DataFrame, ivals: pd.DataFrame) -> pd.DataFrame:
    """Allocate energy to products based on time overlap."""
    logger.info("Allocating energy to products...")
    results = []

    for z_key, z_name in ZONE_ENERGY_MAPPING.items():
        col = f"E_{z_name}_kWh"

        if col not in e.columns:
            logger.warning(f"Energy column {col} not found")
            continue

        e_zone = e[e[col] > 0].copy()
        iv_zone = ivals[ivals["Zone"] == z_key].copy()

        if e_zone.empty or iv_zone.empty:
            logger.warning(f"No data for {z_key}")
            continue

        logger.info(f"Processing {z_key}: {len(e_zone)} energy × {len(iv_zone)} intervals")

        # chunked cross join
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
            merged["Overlap_h"] = (
                (merged["earliest_end"] - merged["latest_start"])
                .dt.total_seconds() / 3600
            ).clip(lower=0)

            merged = merged[merged["Overlap_h"] > 0]

            if merged.empty:
                continue

            merged["Energy_share_kWh"] = merged[col] * merged["Overlap_h"]

            # ✅ FIX: Smart column detection for Month
            if "Month_e" in merged.columns:
                month_col = "Month_e"
            elif "Month_p" in merged.columns:
                month_col = "Month_p"
            elif "Month" in merged.columns:
                month_col = "Month"
            else:
                # Fallback: extract from timestamp
                merged["Month"] = merged["E_start"].dt.month
                month_col = "Month"

            # Select columns and rename
            result = merged[[
                month_col, "Produkt", "m3", "Overlap_h", "Energy_share_kWh"
            ]].copy()

            result = result.rename(columns={month_col: "Month"})
            result["Zone"] = z_key

            zone_res.append(result)

        if zone_res:
            results.append(pd.concat(zone_res, ignore_index=True))

    if results:
        final = pd.concat(results, ignore_index=True)
        logger.info(f"Allocated {len(final)} energy records")
        return final
    
    logger.warning("No energy could be allocated")
    return pd.DataFrame(columns=["Month","Produkt","Zone","Energy_share_kWh","Overlap_h","m3"])


# =====================================================================
# ADD WATER KPIs
# =====================================================================
def add_water_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Add Water_kg and kWh_per_kg using built-in benchmarks."""
    df = df.copy()

    df["water_per_m3_bench"] = df["Produkt"].map(WATER_PER_M3_KG)
    df["Water_kg"] = df["Volume_m3"] * df["water_per_m3_bench"]
    df["kWh_per_kg"] = df["Energy_kWh"] / df["Water_kg"].replace(0, np.nan)

    return df


# =====================================================================
# PREDICTION HELPER
# =====================================================================
def predict_mix_energy(product_mix_m3: dict,
                       baseline_kwh_per_m3=None,
                       baseline_kwh_per_kg=None) -> dict:
    """
    Estimate energy use from a planned product mix.
    product_mix_m3 = {"L36": 100, "N40": 50, ...}
    """

    total_volume = 0
    total_water = 0

    for prod, vol in product_mix_m3.items():
        if vol is None or vol <= 0:
            continue
        total_volume += vol
        if prod in WATER_PER_M3_KG:
            total_water += vol * WATER_PER_M3_KG[prod]

    mean_water = total_water / total_volume if total_volume > 0 else np.nan

    result = {
        "total_volume_m3": total_volume,
        "total_water_kg": total_water,
        "mean_water_per_m3": mean_water,
    }

    if baseline_kwh_per_m3 and total_volume > 0:
        result["energy_from_kwh_per_m3"] = baseline_kwh_per_m3 * total_volume

    if baseline_kwh_per_kg and total_water > 0:
        result["energy_from_kwh_per_kg"] = baseline_kwh_per_kg * total_water

    return result

def compute_product_wagon_stats(wagons: pd.DataFrame) -> dict:
    """
    Compute per-product wagon statistics:
      - average m3 per wagon
      - average residence time in hours/days
      - wagon count

    Requires wagons dataframe with:
      WG_Nr, Produkt, m3, t0, Entnahme
    """
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

    return {
        "wagon_capacity_m3": wagon_capacity,
        "residence_h": residence_h,
        "residence_days": residence_days,
        "raw_stats": stats,
    }

# =====================================================================
# MAIN (for standalone testing)
# =====================================================================
def main():
    """Standalone execution for testing."""
    # This is a placeholder for local testing
    # The Streamlit app uses these functions with uploaded files
    logger.info("Module loaded successfully. Use from Streamlit app.")


if __name__ == "__main__":
    main()

