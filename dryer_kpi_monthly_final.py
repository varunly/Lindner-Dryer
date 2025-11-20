"""
Lindner Dryer KPI Calculation Module

This module does the heavy lifting for the dryer KPI analysis:
- Parses hourly energy data for each dryer zone (Z2–Z5)
- Parses wagon tracking data (products, times, volumes)
- Builds time intervals per zone for each wagon
- Allocates zone energy to products based on time overlap
- Aggregates KPIs by month / product / zone

EXTENSION:
We embed water-loss knowledge (from 'Wasserverlust-Platten-W1-Endmass_2024.03.18.xlsx')
as hard-coded benchmarks:

    WATER_PER_M3_KG[product] = kg evaporated water per m³ of product

These benchmarks come from your measurement sheets:
- Grafik L (L-type products)
- Grafik N (N-type)
- Grafik Y44 (Y-type)

Using them, we can compute:
- Estimated evaporated water [kg]
- Specific energy kWh/kg (energy per kg water)
- Keep the original kWh/m³ KPIs

This means: no need to re-upload the Wasserverlust Excel each time.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Global configuration for local runs (used by main())
# The Streamlit app passes its own files and does not rely on these.
# -------------------------------------------------------------------
CONFIG = {
    "energy_file": r"E:\Lindner\Python\Energieverbrauch Trockner 1, Stundenweise - Januar - September 2025.xlsx",
    "energy_sheet": 0,
    "wagon_file": r"E:\Lindner\Python\Hordenwagenverfolgung_Stand 2025_10_12.xlsm",
    "wagon_sheet": "Hordenwagenverfolgung",
    "wagon_header_row": 6,
    "gas_to_kwh": 11.5,          # m³_gas → kWh (Brennwert)
    "takt_minutes": 65,          # not used directly in this code (we use timestamps)
    "zones_seq": ["Z1", "Z2", "Z3", "Z4", "Z5"],
    "product_filter": ["L36"],
    "month_filter": None,
    "output_file": r"E:\Lindner\Python\Dryer_KPI_Monthly_Results.xlsx",
}

# -------------------------------------------------------------------
# Which energy columns belong to which zone
# -------------------------------------------------------------------
ZONE_ENERGY_MAPPING = {
    "Z2": "Zone 2",
    "Z3": "Zone 3",
    "Z4": "Zone 4",
    "Z5": "Zone 5",
}

# -------------------------------------------------------------------
# WATER BENCHMARKS
# -------------------------------------------------------------------
"""
These values come from the measurement file:

- For each product & thickness, the sheet "Grafik L / N / Y44" gives:
    "ausgetr. Wasser pro mm gepresst [g/mm]" for a 0.605 × 0.605 m board.
- From that, we computed:
    water_per_m3 [kg/m³] = (g_per_mm / board_area)

  board_area = 0.605 m × 0.605 m = 0.366025 m²

We then:
- Averaged across all measurements for each thickness
- Interpolated for L32 and L40 (linear over thickness)
- Assumed N44 ≈ N40 (same type, no direct data)

These are "typical" water-load benchmarks per m³ product.
"""

WATER_PER_M3_KG = {
    "L28": 226.7,   # from measurements (mean)
    "L30": 231.0,
    "L32": 237.8,   # interpolated between L30 & L34
    "L34": 244.7,
    "L36": 207.1,
    "L38": 235.5,
    "L40": 219.4,   # interpolated between L38 & L44
    "L44": 187.2,
    "N40": 237.6,
    "N44": 237.6,   # assumed same as N40 (no measurement)
    "Y44": 439.0,
}


# ===================================================================
# 1. Helpers for duration parsing
# ===================================================================

def parse_duration_series(s: pd.Series) -> pd.Series:
    """
    Convert free-text "Zeit in Zx" columns to Timedelta.
    Examples of input:
        "12:34"
        "5 h 30 min"
        "0,5 h"
    We try to be robust and return NaT on failure.
    """
    s = s.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)

    # Normalize German shorthand to English words that pandas understands
    s = s.str.replace(r"\bh\b", "hours", regex=True)
    s = s.str.replace(r"\bmin\b", "minutes", regex=True)
    s = s.str.replace(r"\bst\b", "seconds", regex=True)

    # Empty / "-" → NaN
    s = s.replace({r"^\s*$": np.nan, r"^-$": np.nan}, regex=True)

    # First try direct Timedelta
    td = pd.to_timedelta(s, errors="coerce")

    # If still NaT, try parsing as datetime (Excel time format)
    mask_nat = td.isna() & s.notna()
    if mask_nat.any():
        s_datetime = pd.to_datetime(s[mask_nat], errors="coerce")
        td_from_datetime = s_datetime - pd.Timestamp("1900-01-01")
        td.loc[mask_nat] = td_from_datetime

    return td


# ===================================================================
# 2. ENERGY DATA PARSING
# ===================================================================

def parse_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse hourly energy consumption data from the raw Excel sheet.

    Expected columns include:
    - "Zeitstempel": timestamp of measurement
    - "Gasmenge, Zone 2 [m³]", "Gasmenge, Zone 3 [m³]", ...
    - "Energieverbrauch, elektr. [kWh]" (optional)

    Output:
    - one row per hour
    - added columns:
        - E_Zone 2_kWh, E_Zone 3_kWh, ...
        - E_el_kWh
        - E_start, E_end (time window of the hour)
        - Month, Year
    """
    logger.info("Parsing energy data...")
    df = df.copy()

    # Parse timestamp
    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], errors="coerce")
    df = df[df["Zeitstempel"].notna()].copy()

    df["Month"] = df["Zeitstempel"].dt.month
    df["Year"] = df["Zeitstempel"].dt.year

    # Convert gas consumption for each zone to kWh
    for zone_key, zone_name in ZONE_ENERGY_MAPPING.items():
        gas_col = f"Gasmenge, {zone_name} [m³]"
        energy_col = f"E_{zone_name}_kWh"

        if gas_col in df.columns:
            df[energy_col] = df[gas_col] * CONFIG["gas_to_kwh"]
            logger.info(f"Converted {gas_col} → {energy_col}")
        else:
            logger.warning(f"Column {gas_col} not found in energy data")

    # Electrical energy (if available)
    if "Energieverbrauch, elektr. [kWh]" in df.columns:
        df["E_el_kWh"] = df["Energieverbrauch, elektr. [kWh]"]

    # Energy time windows for overlap computation
    df["E_start"] = df["Zeitstempel"]
    df["E_end"] = df["Zeitstempel"] + pd.Timedelta(hours=1)

    logger.info("Parsed %d energy records", len(df))
    return df


# ===================================================================
# 3. WAGON DATA PARSING
# ===================================================================

def parse_wagon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse wagon tracking sheet.

    We need:
    - dryer entry time (t0)
    - product name
    - thickness / recipe if needed
    - m³ per wagon
    - timestamps for entering zones Z2–Z5 (if available)
    - text durations "Zeit in Zx" as backup

    Result:
    - one row per wagon with:
      WG_Nr, Produkt, m3, t0, Z2_in, Z3_in, Z4_in, Z5_in, Zx_dur, Month, Year
    """
    logger.info("Parsing wagon data...")
    df = df.copy()

    # Clean column names
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

    # Build starting timestamp t0 from "Pressdat. + Zeit" or date+time
    if "Pressdat. + Zeit" in df.columns:
        t0 = pd.to_datetime(df["Pressdat. + Zeit"], errors="coerce")
    else:
        press_date = df.get("Pressen-Datum", pd.Series()).astype(str)
        press_time = df.get("Press-Zeit", pd.Series()).astype(str)
        t0 = pd.to_datetime(press_date + " " + press_time, errors="coerce")

    df["t0"] = t0

    # Identify wagon number column (first column that starts with "WG-")
    for col in df.columns:
        if str(col).startswith("WG-"):
            df = df.rename(columns={col: "WG_Nr"})
            break

    # Keep only needed columns
    keep_cols = [
        "WG_Nr", "t0", "Produkt", "Rezept", "Stärke", "m³",
        "In Z2", "In Z3", "In Z4", "In Z5",
        "Zeit in Z1", "Zeit in Z2", "Zeit in Z3", "Zeit in Z4", "Zeit in Z5",
        "Entnahme-Zeit",
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()

    # Normalize product names (strip spaces)
    if "Produkt" in df.columns:
        df["Produkt"] = df["Produkt"].astype(str).strip()

    # Wagon volume m³ the same way as in your Excel / existing code
    if "m³" in df.columns:
        df["m3"] = pd.to_numeric(df["m³"], errors="coerce")
    else:
        staerke = pd.to_numeric(df.get("Stärke", 0), errors="coerce")
        # area = 0.605 x 0.605, plus 7mm safety as in your logic
        df["m3"] = 0.605 * 0.605 * (staerke + 7) / 1000.0

    # Parse zone entry timestamps
    zone_entry_cols = {f"In {z}": f"{z}_in" for z in ("Z2", "Z3", "Z4", "Z5")}
    for raw_col, new_col in zone_entry_cols.items():
        if raw_col in df.columns:
            df[new_col] = pd.to_datetime(df[raw_col], errors="coerce", dayfirst=True)
        else:
            df[new_col] = pd.NaT

    # Zone 1 entry = dryer start
    df["Z1_in"] = df["t0"]

    # Exit time (optional)
    if "Entnahme-Zeit" in df.columns:
        df["Entnahme-Zeit"] = pd.to_datetime(df["Entnahme-Zeit"], errors="coerce", dayfirst=True)
    else:
        df["Entnahme-Zeit"] = pd.NaT

    # Compute durations from timestamps (Z1→Z2, Z2→Z3, ..., Z5→exit)
    duration_pairs = [
        ("Z1", "Z2_in", "t0"),
        ("Z2", "Z3_in", "Z2_in"),
        ("Z3", "Z4_in", "Z3_in"),
        ("Z4", "Z5_in", "Z4_in"),
        ("Z5", "Entnahme-Zeit", "Z5_in"),
    ]

    for zone, later_col, earlier_col in duration_pairs:
        hours = (df[later_col] - df[earlier_col]).dt.total_seconds() / 3600
        df[f"{zone}_dur_calc"] = hours

    # Use text durations where available and plausible; otherwise use calculated
    for zone in CONFIG["zones_seq"]:
        text_col = f"Zeit in {zone}"
        calc_col = f"{zone}_dur_calc"
        dur_col = f"{zone}_dur"

        # Start with calculated duration from timestamps
        df[dur_col] = pd.to_timedelta(df[calc_col], unit="h")

        # If text info exists, parse and overwrite if it looks valid
        if text_col in df.columns:
            parsed = parse_duration_series(df[text_col])
            # Replace NaT or suspicious (<1h) parsed values by calculated ones
            mask_replace = parsed.isna() | ((parsed.dt.total_seconds() / 3600) < 1)
            df[dur_col] = parsed.where(~mask_replace, df[dur_col])

    # Month / Year for aggregation
    df["Month"] = df["t0"].dt.month
    df["Year"] = df["t0"].dt.year

    # Remove records without starting time
    df = df[df["t0"].notna()].copy()

    logger.info("Parsed %d wagon records", len(df))
    return df


# ===================================================================
# 4. BUILD INTERVALS PER ZONE
# ===================================================================

def build_intervals(row: pd.Series) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """
    For one wagon row, build (zone, start, end) intervals from entry times
    and durations. We ensure monotonic continuity:
    - If a zone entry time is missing, we fall back to previous end.
    """
    intervals = []
    prev_end = None

    for zone in CONFIG["zones_seq"]:
        zone_in = row.get(f"{zone}_in", pd.NaT)

        if pd.isna(zone_in):
            zone_in = prev_end if prev_end is not None else row["t0"]

        zone_dur = row.get(f"{zone}_dur", pd.NaT)
        if pd.isna(zone_dur):
            continue

        zone_out = zone_in + zone_dur

        if pd.notna(zone_in) and pd.notna(zone_out) and zone_out > zone_in:
            intervals.append((zone, zone_in, zone_out))
            prev_end = zone_out

    return intervals


def explode_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand wagon rows into multiple rows:
    one row per (wagon, zone interval).

    Output columns:
    - WG_Nr, Produkt, Stärke, m3, Zone, P_start, P_end, Month, Year
    """
    logger.info("Exploding wagon data into zone intervals...")
    rows = []

    for _, record in df.iterrows():
        intervals = build_intervals(record)
        for zone, start_time, end_time in intervals:
            rows.append({
                "WG_Nr": record["WG_Nr"],
                "Produkt": record["Produkt"],
                "Stärke": record.get("Stärke", np.nan),
                "m3": record["m3"],
                "Zone": zone,
                "P_start": start_time,
                "P_end": end_time,
                "Month": record["Month"],
                "Year": record.get("Year", np.nan),
            })

    result = pd.DataFrame(rows)
    logger.info("Created %d zone intervals", len(result))
    return result


# ===================================================================
# 5. ALLOCATE ENERGY TO PRODUCTS
# ===================================================================

def allocate_energy(e: pd.DataFrame, ivals: pd.DataFrame) -> pd.DataFrame:
    """
    Allocate zone energy to products based on time overlap.

    For each zone:
        - Filter energy rows with E_zone_kWh > 0.
        - For each wagon interval in that zone, compute overlap with energy hour.
        - Energy_share_kWh = E_zone_kWh * (overlap_hours)

    The result has:
        Month, Produkt, Zone, m3, Overlap_h, Energy_share_kWh
    """
    logger.info("Allocating energy to products...")
    results = []

    for zone_label, zone_name in ZONE_ENERGY_MAPPING.items():
        energy_col = f"E_{zone_name}_kWh"

        if energy_col not in e.columns:
            logger.warning("Energy column %s not found", energy_col)
            continue

        e_zone = e[e[energy_col].notna() & (e[energy_col] > 0)].copy()
        if e_zone.empty:
            logger.warning("No energy data for %s", zone_label)
            continue

        ivals_zone = ivals[ivals["Zone"] == zone_label].copy()
        if ivals_zone.empty:
            logger.warning("No intervals for %s", zone_label)
            continue

        logger.info(
            "Processing %s: %d energy rows × %d intervals",
            zone_label, len(e_zone), len(ivals_zone)
        )

        # Process in chunks so we do not explode memory
        chunk_size = 1000
        zone_results = []

        for i in range(0, len(ivals_zone), chunk_size):
            chunk = ivals_zone.iloc[i:i + chunk_size]

            # Cross join chunk with e_zone using dummy key
            e_temp = e_zone.copy()
            chunk_temp = chunk.copy()
            e_temp["_key"] = 1
            chunk_temp["_key"] = 1

            merged = e_temp.merge(chunk_temp, on="_key", suffixes=("_e", "_p"))
            merged.drop("_key", axis=1, inplace=True)

            # Filter by time overlap
            merged = merged[
                (merged["P_end"] > merged["E_start"]) &
                (merged["P_start"] < merged["E_end"])
            ]
            if merged.empty:
                continue

            # Overlap window
            merged["latest_start"] = merged[["E_start", "P_start"]].max(axis=1)
            merged["earliest_end"] = merged[["E_end", "P_end"]].min(axis=1)

            merged["overlap_h"] = (
                (merged["earliest_end"] - merged["latest_start"])
                .dt.total_seconds() / 3600.0
            ).clip(lower=0)

            merged = merged[merged["overlap_h"] > 0]
            if merged.empty:
                continue

            # Energy share proportional to overlap hours (since each row is 1h)
            merged["Energy_share_kWh"] = merged[energy_col] * merged["overlap_h"]

            # Keep only relevant columns for further aggregation
            result = merged[[
                "Month_e", "Produkt", "m3", "overlap_h", "Energy_share_kWh"
            ]].rename(columns={"Month_e": "Month", "overlap_h": "Overlap_h"})

            result["Zone"] = zone_label
            zone_results.append(result)

        if zone_results:
            results.append(pd.concat(zone_results, ignore_index=True))

    if results:
        final_result = pd.concat(results, ignore_index=True)
        logger.info("Allocated %d energy records", len(final_result))
        return final_result

    logger.warning("No energy could be allocated")
    return pd.DataFrame(
        columns=["Month", "Zone", "Produkt", "Energy_share_kWh", "Overlap_h", "m3"]
    )


# ===================================================================
# 6. WATER-BASED KPIs (kWh/kg)
# ===================================================================

def add_water_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a summary dataframe with water-related KPIs.

    Assumes df has at least:
        - 'Produkt'
        - 'Volume_m3'
        - 'Energy_kWh'

    Steps:
    1) Map each Produkt to water_per_m3_bench (kg/m³) via WATER_PER_M3_KG.
    2) Compute estimated water_kg = Volume_m3 * water_per_m3_bench.
    3) Compute kWh_per_kg = Energy_kWh / Water_kg.

    Returns a copy with new columns:
        - water_per_m3_bench
        - Water_kg
        - kWh_per_kg
    """
    df = df.copy()
    df["water_per_m3_bench"] = df["Produkt"].astype(str).map(WATER_PER_M3_KG)

    df["Water_kg"] = df["Volume_m3"] * df["water_per_m3_bench"]

    df["kWh_per_kg"] = df["Energy_kWh"] / df["Water_kg"].replace(0, np.nan)

    return df


# ===================================================================
# 7. SIMPLE PREDICTION HELPER
# ===================================================================

def predict_mix_energy(
    product_mix_m3: dict,
    baseline_kwh_per_m3: float = None,
    baseline_kwh_per_kg: float = None,
) -> dict:
    """
    Simple prediction helper to estimate:
      - total volume
      - total evaporated water (kg)
      - expected energy (kWh)

    Inputs:
    -------
    product_mix_m3:
        dict like {"L36": 100.0, "N40": 50.0}
        meaning: 100 m³ of L36, 50 m³ of N40.

    baseline_kwh_per_m3 (optional):
        If given, we assume energy ≈ baseline_kwh_per_m3 * total_volume_m3

    baseline_kwh_per_kg (optional):
        If given, we assume energy ≈ baseline_kwh_per_kg * total_water_kg

    You can pass one or both baselines:
        - If both given, we compute both energy estimates.
        - If none given, we only return volume + water.

    Returns:
    --------
    dict with keys:
        - total_volume_m3
        - total_water_kg
        - mean_water_per_m3 (kg/m³)
        - energy_from_kwh_per_m3 (if baseline_kwh_per_m3 given)
        - energy_from_kwh_per_kg (if baseline_kwh_per_kg given)
    """
    # Compute total volume and water based on benchmarks
    total_volume = 0.0
    total_water = 0.0

    for prod, vol in product_mix_m3.items():
        if vol is None:
            continue
        v = float(vol)
        if v <= 0:
            continue
        total_volume += v
        water_per_m3 = WATER_PER_M3_KG.get(prod, np.nan)
        if not np.isnan(water_per_m3):
            total_water += v * water_per_m3

    mean_water_per_m3 = total_water / total_volume if total_volume > 0 else np.nan

    result = {
        "total_volume_m3": total_volume,
        "total_water_kg": total_water,
        "mean_water_per_m3": mean_water_per_m3,
    }

    if baseline_kwh_per_m3 is not None and total_volume > 0:
        result["energy_from_kwh_per_m3"] = baseline_kwh_per_m3 * total_volume

    if baseline_kwh_per_kg is not None and total_water > 0:
        result["energy_from_kwh_per_kg"] = baseline_kwh_per_kg * total_water

    return result


# ===================================================================
# 8. MAIN (for standalone execution, e.g. debugging)
# ===================================================================

def main():
    """
    Standalone execution for testing outside Streamlit.

    - Loads energy & wagon data from CONFIG paths.
    - Parses, explodes, allocates energy.
    - Builds monthly & yearly summaries with kWh/m³ and kWh/kg.
    - Exports to an Excel file.

    The Streamlit app uses the same functions but passes its own files.
    """
    try:
        logger.info("=== Starting Dryer KPI Analysis ===")

        # --- Energy ---
        logger.info("Loading energy data from: %s", CONFIG["energy_file"])
        e_raw = pd.read_excel(CONFIG["energy_file"], sheet_name=CONFIG["energy_sheet"])
        e = parse_energy(e_raw)

        # --- Wagons ---
        logger.info("Loading wagon data from: %s", CONFIG["wagon_file"])
        w_raw = pd.read_excel(
            CONFIG["wagon_file"],
            sheet_name=CONFIG["wagon_sheet"],
            header=CONFIG["wagon_header_row"],
        )
        w = parse_wagon(w_raw)

        # Optional filters
        if CONFIG["product_filter"]:
            logger.info("Filtering products: %s", CONFIG["product_filter"])
            w = w[w["Produkt"].astype(str).isin(CONFIG["product_filter"])]

        if CONFIG["month_filter"]:
            logger.info("Filtering month: %s", CONFIG["month_filter"])
            e = e[e["Month"] == CONFIG["month_filter"]]
            w = w[w["Month"] == CONFIG["month_filter"]]

        # --- Intervals & allocation ---
        ivals = explode_intervals(w)
        alloc = allocate_energy(e, ivals)

        # --- Monthly summary ---
        logger.info("Creating monthly summary...")
        summary = alloc.groupby(["Month", "Produkt", "Zone"], as_index=False).agg(
            Energy_kWh=("Energy_share_kWh", "sum"),
            Volume_m3=("m3", "sum"),
        )
        summary["kWh_per_m3"] = (
            summary["Energy_kWh"] / summary["Volume_m3"].replace(0, np.nan)
        )

        # Add water-based KPIs (kWh/kg)
        summary = add_water_kpis(summary)

        # --- Yearly summary (aggregated over months) ---
        logger.info("Creating yearly summary...")
        yearly = summary.groupby(["Produkt", "Zone"], as_index=False).agg(
            Energy_kWh=("Energy_kWh", "sum"),
            Volume_m3=("Volume_m3", "sum"),
            Water_kg=("Water_kg", "sum"),
        )
        yearly["kWh_per_m3"] = (
            yearly["Energy_kWh"] / yearly["Volume_m3"].replace(0, np.nan)
        )
        yearly["kWh_per_kg"] = (
            yearly["Energy_kWh"] / yearly["Water_kg"].replace(0, np.nan)
        )

        # --- Export ---
        logger.info("Exporting results to: %s", CONFIG["output_file"])
        with pd.ExcelWriter(CONFIG["output_file"], engine="xlsxwriter") as writer:
            e.to_excel(writer, sheet_name="Energy_Hourly_Parsed", index=False)
            w.to_excel(writer, sheet_name="Wagons_Parsed", index=False)
            ivals.to_excel(writer, sheet_name="Intervals_By_Zone", index=False)
            alloc.to_excel(writer, sheet_name="Energy_Allocated", index=False)
            summary.to_excel(writer, sheet_name="Summary_By_Month_Zone", index=False)
            yearly.to_excel(writer, sheet_name="Yearly_Summary", index=False)

        logger.info("=== Analysis Complete ===")
        logger.info("Total Energy: %.2f kWh", yearly["Energy_kWh"].sum())
        logger.info("Total Volume: %.2f m³", yearly["Volume_m3"].sum())
        logger.info("Total Water: %.2f kg", yearly["Water_kg"].sum())
        logger.info("Average KPI (kWh/m³): %.2f", yearly["kWh_per_m3"].mean())
        logger.info("Average KPI (kWh/kg): %.2f", yearly["kWh_per_kg"].mean())

    except Exception as exc:
        logger.error("Error during analysis: %s", str(exc), exc_info=True)
        raise


if __name__ == "__main__":
    main()
