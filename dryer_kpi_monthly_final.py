"""
Lindner Dryer KPI Calculation Module (Refactored)
Calculates energy efficiency KPIs for dryer zones by allocating energy consumption to products.
Now extended with water-loss parsing to derive water-per-m³ benchmarks per product.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xlsxwriter
import logging
from typing import Dict, Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "energy_file": r"E:\Lindner\Python\Energieverbrauch Trockner 1, Stundenweise - Januar - September 2025.xlsx",
    "energy_sheet": 0,
    "wagon_file": r"E:\Lindner\Python\Hordenwagenverfolgung_Stand 2025_10_12.xlsm",
    "wagon_sheet": "Hordenwagenverfolgung",
    "wagon_header_row": 6,
    "gas_to_kwh": 11.5,
    "takt_minutes": 65,
    "zones_seq": ["Z1", "Z2", "Z3", "Z4", "Z5"],
    "product_filter": ["L36"],
    "month_filter": None,
    "output_file": r"E:\Lindner\Python\Dryer_KPI_Monthly_Results.xlsx"
}

# Zone to column mapping
ZONE_ENERGY_MAPPING = {
    "Z2": "Zone 2",
    "Z3": "Zone 3",
    "Z4": "Zone 4",
    "Z5": "Zone 5"
}


def parse_duration_series(s: pd.Series) -> pd.Series:
    """
    Convert free-text "Zeit in Zx" column (e.g. "12:34", "5 h 30 min", etc.)
    into a pandas Timedelta Series. Return NaT where parsing fails.
    """
    s = s.astype(str).str.strip()
    s = s.str.replace(',', '.', regex=False)

    # Normalize common abbreviations
    s = s.str.replace(r'\bh\b', 'hours', regex=True)
    s = s.str.replace(r'\bmin\b', 'minutes', regex=True)
    s = s.str.replace(r'\bst\b', 'seconds', regex=True)

    # Handle empty/null values
    s = s.replace({r'^\s*$': np.nan, r'^-$': np.nan}, regex=True)

    # Try pandas timedelta parsing first
    td = pd.to_timedelta(s, errors='coerce')

    # For values that failed, try parsing as datetime (Excel time format)
    mask_nat = td.isna() & s.notna()
    if mask_nat.any():
        s_datetime = pd.to_datetime(s[mask_nat], errors='coerce')
        td_from_datetime = s_datetime - pd.Timestamp('1900-01-01')
        td.loc[mask_nat] = td_from_datetime

    return td


def parse_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse hourly energy consumption data.
    """
    logger.info("Parsing energy data...")
    df = df.copy()

    # Parse timestamp
    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], errors='coerce')
    df["Month"] = df["Zeitstempel"].dt.month
    df["Year"] = df["Zeitstempel"].dt.year

    # Convert gas consumption to kWh for each zone
    for zone_key, zone_name in ZONE_ENERGY_MAPPING.items():
        gas_col = f"Gasmenge, {zone_name} [m³]"
        energy_col = f"E_{zone_name}_kWh"

        if gas_col in df.columns:
            df[energy_col] = df[gas_col] * CONFIG["gas_to_kwh"]
            logger.info(f"Converted {gas_col} to {energy_col}")
        else:
            logger.warning(f"Column {gas_col} not found in energy data")

    # Parse electrical energy
    if "Energieverbrauch, elektr. [kWh]" in df.columns:
        df["E_el_kWh"] = df["Energieverbrauch, elektr. [kWh]"]

    # Create time windows for energy allocation
    df["E_start"] = df["Zeitstempel"]
    df["E_end"] = df["Zeitstempel"] + pd.Timedelta(hours=1)

    # Remove rows with invalid timestamps
    df = df[df["Zeitstempel"].notna()].copy()

    logger.info(f"Parsed {len(df)} energy records")
    return df


def parse_wagon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse wagon tracking data with zone entry times and durations.
    """
    logger.info("Parsing wagon data...")
    df = df.copy()

    # Normalize column names
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

    # Build dryer start timestamp (t0)
    if "Pressdat. + Zeit" in df.columns:
        t0 = pd.to_datetime(df["Pressdat. + Zeit"], errors="coerce")
    else:
        press_date = df.get("Pressen-Datum", pd.Series()).astype(str)
        press_time = df.get("Press-Zeit", pd.Series()).astype(str)
        t0 = pd.to_datetime(press_date + " " + press_time, errors="coerce")

    df["t0"] = t0

    # Identify wagon number column
    for col in df.columns:
        if col.startswith("WG-"):
            df = df.rename(columns={col: "WG_Nr"})
            break

    # Keep only needed columns
    keep_cols = [
        "WG_Nr", "t0", "Produkt", "Rezept", "Stärke", "m³",
        "In Z2", "In Z3", "In Z4", "In Z5",
        "Zeit in Z1", "Zeit in Z2", "Zeit in Z3", "Zeit in Z4", "Zeit in Z5",
        "Entnahme-Zeit"
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()

    # Normalize Produkt names for reliable joins
    if "Produkt" in df.columns:
        df["Produkt"] = df["Produkt"].astype(str).str.strip()

    # Calculate volume (m³)
    if "m³" in df.columns:
        df["m3"] = pd.to_numeric(df["m³"], errors='coerce')
    else:
        staerke = pd.to_numeric(df.get("Stärke", 0), errors='coerce')
        df["m3"] = 0.605 * 0.605 * (staerke + 7) / 1000

    # Parse zone entry timestamps
    zone_entry_cols = {f"In {z}": f"{z}_in" for z in ("Z2", "Z3", "Z4", "Z5")}
    for raw_col, new_col in zone_entry_cols.items():
        if raw_col in df.columns:
            df[new_col] = pd.to_datetime(df[raw_col], errors="coerce", dayfirst=True)
        else:
            df[new_col] = pd.NaT

    # Z1 entry is dryer start
    df["Z1_in"] = df["t0"]

    # Parse exit timestamp
    if "Entnahme-Zeit" in df.columns:
        df["Entnahme-Zeit"] = pd.to_datetime(df["Entnahme-Zeit"], errors="coerce", dayfirst=True)
    else:
        df["Entnahme-Zeit"] = pd.NaT

    # Calculate zone durations from timestamps (vectorized)
    duration_pairs = [
        ("Z1", "Z2_in", "t0"),
        ("Z2", "Z3_in", "Z2_in"),
        ("Z3", "Z4_in", "Z3_in"),
        ("Z4", "Z5_in", "Z4_in"),
        ("Z5", "Entnahme-Zeit", "Z5_in")
    ]

    for zone, later_col, earlier_col in duration_pairs:
        hours = (df[later_col] - df[earlier_col]).dt.total_seconds() / 3600
        df[f"{zone}_dur_calc"] = hours

    # Use parsed text durations when available, otherwise use calculated
    for zone in CONFIG["zones_seq"]:
        text_col = f"Zeit in {zone}"
        calc_col = f"{zone}_dur_calc"
        dur_col = f"{zone}_dur"

        df[dur_col] = pd.to_timedelta(df[calc_col], unit="h")

        if text_col in df.columns:
            parsed = parse_duration_series(df[text_col])
            mask_replace = parsed.isna() | (parsed.dt.total_seconds() / 3600 < 1)
            df[dur_col] = parsed.where(~mask_replace, df[dur_col])

    # Add month/year columns for aggregation
    df["Month"] = df["t0"].dt.month
    df["Year"] = df["t0"].dt.year

    # Filter out invalid records
    df = df[df["t0"].notna()].copy()

    logger.info(f"Parsed {len(df)} wagon records")
    return df


def build_intervals(row: pd.Series) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """
    Build time intervals for each zone based on entry times and durations.
    """
    intervals = []
    prev_end = None

    for zone in CONFIG["zones_seq"]:
        zone_in = row.get(f"{zone}_in", pd.NaT)

        # Use previous end time if entry time is missing
        if pd.isna(zone_in):
            zone_in = prev_end if prev_end is not None else row["t0"]

        zone_dur = row.get(f"{zone}_dur", pd.NaT)
        zone_out = zone_in + zone_dur if pd.notna(zone_in) and pd.notna(zone_dur) else pd.NaT

        if pd.notna(zone_in) and pd.notna(zone_out) and zone_out > zone_in:
            intervals.append((zone, zone_in, zone_out))
            prev_end = zone_out

    return intervals


def explode_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode wagon data into individual zone intervals.
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
                "Year": record.get("Year", np.nan)
            })

    result = pd.DataFrame(rows)
    logger.info(f"Created {len(result)} zone intervals")
    return result


def allocate_energy(e: pd.DataFrame, ivals: pd.DataFrame) -> pd.DataFrame:
    """
    Allocate energy consumption to products based on time overlap (MEMORY OPTIMIZED).
    """
    logger.info("Allocating energy to products...")
    results = []

    for zone_label, zone_name in ZONE_ENERGY_MAPPING.items():
        energy_col = f"E_{zone_name}_kWh"

        if energy_col not in e.columns:
            logger.warning(f"Energy column {energy_col} not found")
            continue

        e_zone = e[e[energy_col].notna() & (e[energy_col] > 0)].copy()
        if e_zone.empty:
            logger.warning(f"No energy data for {zone_label}")
            continue

        ivals_zone = ivals[ivals["Zone"] == zone_label].copy()
        if ivals_zone.empty:
            logger.warning(f"No intervals for {zone_label}")
            continue

        logger.info(f"Processing {zone_label}: {len(e_zone)} energy records × {len(ivals_zone)} intervals")

        chunk_size = 1000
        zone_results = []

        for i in range(0, len(ivals_zone), chunk_size):
            chunk = ivals_zone.iloc[i:i+chunk_size]

            e_temp = e_zone.copy()
            chunk_temp = chunk.copy()

            e_temp['_key'] = 1
            chunk_temp['_key'] = 1

            merged = e_temp.merge(chunk_temp, on='_key', suffixes=('_e', '_p'))
            merged.drop('_key', axis=1, inplace=True)

            merged = merged[
                (merged['P_end'] > merged['E_start']) &
                (merged['P_start'] < merged['E_end'])
            ]

            if merged.empty:
                continue

            merged['latest_start'] = merged[['E_start', 'P_start']].max(axis=1)
            merged['earliest_end'] = merged[['E_end', 'P_end']].min(axis=1)
            merged['overlap_h'] = (
                (merged['earliest_end'] - merged['latest_start']).dt.total_seconds() / 3600
            ).clip(lower=0)

            merged = merged[merged['overlap_h'] > 0]
            if merged.empty:
                continue

            merged['Energy_share_kWh'] = merged[energy_col] * merged['overlap_h']

            result = merged[[
                'Month_e', 'Produkt', 'm3',
                'Energy_share_kWh', 'overlap_h'
            ]].rename(columns={'Month_e': 'Month', 'overlap_h': 'Overlap_h'})

            result['Zone'] = zone_label
            zone_results.append(result)

        if zone_results:
            results.append(pd.concat(zone_results, ignore_index=True))

    if results:
        final_result = pd.concat(results, ignore_index=True)
        logger.info(f"Allocated {len(final_result)} energy records")
        return final_result
    else:
        logger.warning("No energy could be allocated")
        return pd.DataFrame(columns=[
            'Month', 'Zone', 'Produkt',
            'Energy_share_kWh', 'Overlap_h', 'm3'
        ])


# -------------------------------------------------------------------
# NEW: Water-loss parsing & benchmark creation
# -------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Helper to find first column whose name contains any of the candidate substrings (case-insensitive).
    """
    cols = {str(c).strip(): str(c).strip() for c in df.columns}
    lower_map = {c.lower(): c for c in cols.keys()}

    for cand in candidates:
        cand_lower = cand.lower()
        for col_lower, orig in lower_map.items():
            if cand_lower in col_lower:
                return orig
    return None


def parse_waterloss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse water-loss Excel from 'Wasserverlust-Platten-W1-Endmass_2024.03.18.xlsx'.
    Returns a cleaned dataframe with:
      Produkt, initial_weight_kg, final_weight_kg, water_loss_kg,
      water_loss_pct, volume_m3, water_per_m3
    """
    logger.info("Parsing water-loss data...")
    df = df.copy()
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

    # Detect key columns heuristically
    prod_col = _find_column(df, ["Produkttyp", "Produkt", "Typ", "Type"])
    init_col = _find_column(df, ["Anfangsmasse", "Anfangsgewicht", "Startgewicht", "Gewicht vor", "Initial"])
    final_col = _find_column(df, ["Endmass", "Endgewicht", "Gewicht nach", "Ende", "Final"])
    vol_col = _find_column(df, ["Volumen", "m³", "m3"])
    len_col = _find_column(df, ["Länge", "Laenge", "Length"])
    wid_col = _find_column(df, ["Breite", "Width"])
    thick_col = _find_column(df, ["Dicke", "Stärke", "Staerke", "Thickness"])

    if prod_col is None or init_col is None or final_col is None:
        logger.warning("Could not detect required columns for water-loss (Produkt/initial/final). "
                       "Check column names in Wasserverlust Excel.")
    df_out = pd.DataFrame()

    if prod_col is not None:
        df_out["Produkt"] = df[prod_col].astype(str).str.strip()
    else:
        df_out["Produkt"] = "UNKNOWN"

    df_out["initial_weight_kg"] = pd.to_numeric(df[init_col], errors="coerce") if init_col else np.nan
    df_out["final_weight_kg"] = pd.to_numeric(df[final_col], errors="coerce") if final_col else np.nan

    # Volume calculation
    volume_m3 = None
    if vol_col is not None:
        volume_m3 = pd.to_numeric(df[vol_col], errors="coerce")
        logger.info(f"Using existing volume column: {vol_col}")
    else:
        # Try to compute volume from dimensions
        if all([len_col, wid_col, thick_col]):
            length = pd.to_numeric(df[len_col], errors="coerce")
            width = pd.to_numeric(df[wid_col], errors="coerce")
            thickness = pd.to_numeric(df[thick_col], errors="coerce")

            # If thickness looks like mm (>1), convert to meters
            thick_m = thickness.copy()
            thick_m.loc[thick_m > 1] = thick_m.loc[thick_m > 1] / 1000.0
            thick_m.loc[thick_m <= 1] = thick_m.loc[thick_m <= 1]  # already meters

            volume_m3 = length * width * thick_m
            logger.info("Computed volume from L × B × D")
        else:
            logger.warning("No volume or dimensions found in water-loss file.")
            volume_m3 = pd.Series(np.nan, index=df.index)

    df_out["volume_m3"] = volume_m3

    # Compute water-loss metrics
    df_out["water_loss_kg"] = df_out["initial_weight_kg"] - df_out["final_weight_kg"]
    df_out["water_loss_pct"] = df_out["water_loss_kg"] / df_out["initial_weight_kg"]

    df_out["water_per_m3"] = df_out["water_loss_kg"] / df_out["volume_m3"]

    # Drop rows with no water-loss info
    df_out = df_out.dropna(subset=["water_loss_kg", "volume_m3"], how="any").copy()

    logger.info(f"Parsed {len(df_out)} water-loss records")
    return df_out


def build_water_benchmarks(wl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-product water benchmarks from water-loss dataframe:
      Produkt, Water_per_m3_bench, Water_loss_pct_mean, Samples
    """
    if wl_df is None or wl_df.empty:
        logger.warning("Water-loss dataframe is empty; cannot build benchmarks.")
        return pd.DataFrame(columns=["Produkt", "Water_per_m3_bench", "Water_loss_pct_mean", "Samples"])

    wl_df = wl_df.copy()
    wl_df["Produkt"] = wl_df["Produkt"].astype(str).str.strip()

    grp = wl_df.groupby("Produkt", as_index=False).agg(
        Water_per_m3_bench=("water_per_m3", "mean"),
        Water_loss_pct_mean=("water_loss_pct", "mean"),
        Samples=("water_per_m3", "count")
    )

    logger.info(f"Built water benchmarks for {len(grp)} products")
    return grp


def main():
    """Main execution function"""
    try:
        logger.info("=== Starting Dryer KPI Analysis ===")

        # Load and parse energy data
        logger.info(f"Loading energy data from: {CONFIG['energy_file']}")
        e_raw = pd.read_excel(CONFIG["energy_file"], sheet_name=CONFIG["energy_sheet"])
        e = parse_energy(e_raw)

        # Load and parse wagon data
        logger.info(f"Loading wagon data from: {CONFIG['wagon_file']}")
        w_raw = pd.read_excel(
            CONFIG["wagon_file"],
            sheet_name=CONFIG["wagon_sheet"],
            header=CONFIG["wagon_header_row"]
        )
        w = parse_wagon(w_raw)

        # Apply filters
        if CONFIG["product_filter"]:
            logger.info(f"Filtering products: {CONFIG['product_filter']}")
            w = w[w["Produkt"].astype(str).isin(CONFIG["product_filter"])]

        if CONFIG["month_filter"]:
            logger.info(f"Filtering month: {CONFIG['month_filter']}")
            e = e[e["Month"] == CONFIG["month_filter"]]
            w = w[w["Month"] == CONFIG["month_filter"]]

        # Process intervals and allocate energy
        ivals = explode_intervals(w)
        alloc = allocate_energy(e, ivals)

        # Create monthly summary
        logger.info("Creating monthly summary...")
        summary = alloc.groupby(["Month", "Produkt", "Zone"], as_index=False).agg(
            Energy_kWh=("Energy_share_kWh", "sum"),
            Volume_m3=("m3", "sum")
        )
        summary["kWh_per_m3"] = summary["Energy_kWh"] / summary["Volume_m3"].replace(0, np.nan)

        # Create yearly summary
        logger.info("Creating yearly summary...")
        yearly = summary.groupby(["Produkt", "Zone"], as_index=False).agg(
            Energy_kWh=("Energy_kWh", "sum"),
            Volume_m3=("Volume_m3", "sum")
        )
        yearly["kWh_per_m3"] = yearly["Energy_kWh"] / yearly["Volume_m3"].replace(0, np.nan)

        # Export to Excel
        logger.info(f"Exporting results to: {CONFIG['output_file']}")
        with pd.ExcelWriter(CONFIG["output_file"], engine="xlsxwriter") as writer:
            e.to_excel(writer, sheet_name="Energy_Hourly_Parsed", index=False)
            w.to_excel(writer, sheet_name="Wagons_Parsed", index=False)
            ivals.to_excel(writer, sheet_name="Intervals_By_Zone", index=False)
            alloc.to_excel(writer, sheet_name="Energy_Allocated", index=False)
            summary.to_excel(writer, sheet_name="Summary_By_Month_Zone", index=False)
            yearly.to_excel(writer, sheet_name="Yearly_Summary", index=False)

            wb = writer.book
            fmt_head = wb.add_format({'bold': True, 'bg_color': '#C6E0B4', 'border': 1})
            fmt_num = wb.add_format({'num_format': '#,##0.00', 'border': 1})

            for sheet_name in ["Summary_By_Month_Zone", "Yearly_Summary"]:
                ws = writer.sheets[sheet_name]
                ws.set_row(0, 18, fmt_head)
                ws.set_column("A:F", 18, fmt_num)

        logger.info("=== Analysis Complete ===")
        logger.info(f"Total Energy: {yearly['Energy_kWh'].sum():,.2f} kWh")
        logger.info(f"Total Volume: {yearly['Volume_m3'].sum():,.2f} m³")
        logger.info(f"Average KPI: {yearly['kWh_per_m3'].mean():,.2f} kWh/m³")

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
