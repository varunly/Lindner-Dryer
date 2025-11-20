"""
Lindner Dryer KPI Calculation Module (Refactored)
Calculates energy efficiency KPIs for dryer zones by allocating energy consumption to products.
Now includes Water Loss Analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xlsxwriter
import logging
import re
from typing import Dict, Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "energy_sheet": 0,
    "wagon_sheet": "Hordenwagenverfolgung",
    "wagon_header_row": 6,
    "gas_to_kwh": 11.5,
    "zones_seq": ["Z1", "Z2", "Z3", "Z4", "Z5"],
    "std_length_m": 2.0,
    "std_width_m": 1.25
}

# Zone to column mapping
ZONE_ENERGY_MAPPING = {
    "Z2": "Zone 2",
    "Z3": "Zone 3",
    "Z4": "Zone 4",
    "Z5": "Zone 5"
}

# --- Helper Functions ---

def clean_german_float(x):
    """Converts German string floats (22,5) to Python floats (22.5)."""
    if pd.isna(x) or x == '':
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    try:
        # Remove units like 'kg', 'mm' if attached directly
        x = str(x).lower().replace('kg', '').replace('mm', '').strip()
        return float(x.replace(',', '.'))
    except ValueError:
        return np.nan

def parse_duration_series(s: pd.Series) -> pd.Series:
    """Parses free-text duration columns."""
    s = s.astype(str).str.strip()
    s = s.str.replace(',', '.', regex=False)
    s = s.str.replace(r'\bh\b', 'hours', regex=True)
    s = s.str.replace(r'\bmin\b', 'minutes', regex=True)
    s = s.replace({r'^\s*$': np.nan, r'^-$': np.nan}, regex=True)
    td = pd.to_timedelta(s, errors='coerce')
    mask_nat = td.isna() & s.notna()
    if mask_nat.any():
        s_datetime = pd.to_datetime(s[mask_nat], errors='coerce')
        td_from_datetime = s_datetime - pd.Timestamp('1900-01-01')
        td.loc[mask_nat] = td_from_datetime
    return td

# --- Water Loss Functions (New) ---

def normalize_water_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Maps messy Excel headers to standard internal variable names."""
    col_map = {}
    for col in df.columns:
        c_str = str(col).lower()
        # Weight Logic
        if 'unterlage' in c_str and 'frisch' in c_str:
            col_map[col] = 'weight_gross_wet' # Frisch + Unterlage
        elif 'unterlage' in c_str and 'frisch' not in c_str:
            col_map[col] = 'weight_support'   # Unterlage only
        elif 'trocken' in c_str and 'platte' in c_str:
            col_map[col] = 'weight_dry'
        # Dimensions / Type
        elif 'pressmaß' in c_str or 'dicke' in c_str:
            col_map[col] = 'thickness_mm'
        elif 'typ' in c_str:
            col_map[col] = 'type_letter'
        elif 'datum' in c_str:
            col_map[col] = 'date'
            
    return df.rename(columns=col_map)

def parse_waterloss(file_obj) -> pd.DataFrame:
    """
    Reads the Water-Loss Excel file and extracts relevant sample data.
    Handles multiple sheets and messy headers.
    """
    logger.info("Parsing Water Loss file...")
    try:
        xls = pd.ExcelFile(file_obj)
    except Exception as e:
        logger.error(f"Failed to open Excel file: {e}")
        return pd.DataFrame()

    all_data = []
    ignore_sheets = ['Gesamt', 'Tabelle1', 'VGL', 'alt']

    for sheet in xls.sheet_names:
        if any(x in sheet for x in ignore_sheets):
            continue
            
        # Read buffer to find header
        df_raw = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=20)
        header_idx = -1
        
        for i, row in df_raw.iterrows():
            row_str = row.astype(str).str.lower().values
            if 'typ' in row_str or 'datum' in row_str:
                header_idx = i
                break
        
        if header_idx != -1:
            df = pd.read_excel(xls, sheet_name=sheet, header=header_idx)
            df = normalize_water_columns(df)
            
            # Ensure we have minimum columns
            req_cols = ['type_letter', 'weight_gross_wet', 'weight_dry', 'thickness_mm']
            if all(col in df.columns for col in req_cols):
                all_data.append(df)

    if not all_data:
        logger.warning("No valid water loss sheets found.")
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True)

    # Clean Numeric Columns
    numeric_cols = ['weight_gross_wet', 'weight_support', 'weight_dry', 'thickness_mm']
    for col in numeric_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].apply(clean_german_float)

    # Clean Dates
    if 'date' in full_df.columns:
        full_df['date'] = pd.to_datetime(full_df['date'], errors='coerce')
        full_df = full_df.dropna(subset=['date'])

    # Create Product Key (e.g., L-30)
    full_df['thickness_clean'] = full_df['thickness_mm'].fillna(0).astype(int).astype(str)
    full_df['Produkt'] = full_df['type_letter'].fillna('?') + full_df['thickness_clean']
    
    # Cleanup Product Key (remove spaces, ensure L30 format matches wagon file)
    full_df['Produkt'] = full_df['Produkt'].str.replace('-', '').str.replace(' ', '')

    logger.info(f"Parsed {len(full_df)} water sample records.")
    return full_df

def calculate_waterloss_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes physics metrics (Water loss, %, m3) per sample."""
    if df.empty:
        return df
        
    # 1. Net Fresh Weight
    if 'weight_support' in df.columns:
        df['weight_support'] = df['weight_support'].fillna(0)
        df['weight_fresh_net'] = df['weight_gross_wet'] - df['weight_support']
    else:
        df['weight_fresh_net'] = df['weight_gross_wet'] # Assume gross is net if no support col

    # 2. Water Loss
    df['water_loss_kg'] = df['weight_fresh_net'] - df['weight_dry']
    
    # 3. Water Loss %
    df['water_loss_pct'] = (df['water_loss_kg'] / df['weight_fresh_net']) * 100
    
    # 4. Volume m3 (Dimensions in mm -> m)
    # Using standard length/width if not in file
    df['thickness_m'] = df['thickness_mm'] / 1000.0
    df['volume_m3'] = df['thickness_m'] * CONFIG['std_length_m'] * CONFIG['std_width_m']
    
    # 5. Specific Water Loss (kg/m3) - This is the key linking metric
    df['water_per_m3'] = df['water_loss_kg'] / df['volume_m3']
    
    # Filter bad data
    df = df[(df['water_loss_kg'] > 0) & (df['water_per_m3'] < 1000)]
    
    return df

def merge_energy_water(alloc_df: pd.DataFrame, water_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges Energy/Wagon data with Water Loss data.
    Logic:
    1. Calculate Average Water Loss per m3 per Product from Water Samples.
    2. Apply this factor to the Total Volume in Energy Allocation data.
    3. Calculate Totals.
    """
    if water_df.empty or alloc_df.empty:
        return pd.DataFrame()
        
    # 1. Create Reference Table (Product -> Avg Water/m3)
    water_ref = water_df.groupby('Produkt')['water_per_m3'].mean().reset_index()
    water_ref.rename(columns={'water_per_m3': 'ref_water_per_m3'}, inplace=True)
    
    # 2. Merge into Allocation Data
    # We merge on Product. If product doesn't exist in samples, we map to a global average as fallback
    global_avg_water = water_df['water_per_m3'].mean()
    
    merged = pd.merge(alloc_df, water_ref, on='Produkt', how='left')
    
    # Fill missing products with global average
    merged['ref_water_per_m3'] = merged['ref_water_per_m3'].fillna(global_avg_water)
    
    # 3. Calculate Total Water Evaporated for this wagon/interval
    # Volume (m3) * Specific Water Loss (kg/m3) = Total Water (kg)
    merged['total_water_kg'] = merged['m3'] * merged['ref_water_per_m3']
    
    return merged

def compute_kpis(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates final KPIs on the merged dataset."""
    
    # Group by Month and Product (or Month and Zone)
    # Here we calculate a Monthly Summary
    
    kpi = merged_df.groupby(['Month', 'Produkt'], as_index=False).agg({
        'Energy_share_kWh': 'sum',
        'total_water_kg': 'sum',
        'm3': 'sum',
        'Overlap_h': 'sum' # Proxy for operating time contribution
    })
    
    # KPI 1: Specific Energy per kg Water (kWh/kg)
    kpi['kwh_per_kg_water'] = kpi['Energy_share_kWh'] / kpi['total_water_kg']
    
    # KPI 2: Specific Energy per m3 Product
    kpi['kwh_per_m3'] = kpi['Energy_share_kWh'] / kpi['m3']
    
    # KPI 3: Water Density (kg/m3)
    kpi['water_density_kg_m3'] = kpi['total_water_kg'] / kpi['m3']
    
    return kpi

# --- Existing Functions (Kept as is, just ensuring imports) ---

def parse_energy(df: pd.DataFrame) -> pd.DataFrame:
    # ... (Same as before) ...
    logger.info("Parsing energy data...")
    df = df.copy()
    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], errors='coerce')
    df["Month"] = df["Zeitstempel"].dt.month
    df["Year"] = df["Zeitstempel"].dt.year
    for zone_key, zone_name in ZONE_ENERGY_MAPPING.items():
        gas_col = f"Gasmenge, {zone_name} [m³]"
        energy_col = f"E_{zone_name}_kWh"
        if gas_col in df.columns:
            df[energy_col] = df[gas_col] * CONFIG["gas_to_kwh"]
    if "Energieverbrauch, elektr. [kWh]" in df.columns:
        df["E_el_kWh"] = df["Energieverbrauch, elektr. [kWh]"]
    df["E_start"] = df["Zeitstempel"]
    df["E_end"] = df["Zeitstempel"] + pd.Timedelta(hours=1)
    df = df[df["Zeitstempel"].notna()].copy()
    return df

def parse_wagon(df: pd.DataFrame) -> pd.DataFrame:
    # ... (Same as before) ...
    logger.info("Parsing wagon data...")
    df = df.copy()
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    if "Pressdat. + Zeit" in df.columns:
        t0 = pd.to_datetime(df["Pressdat. + Zeit"], errors="coerce")
    else:
        press_date = df.get("Pressen-Datum", pd.Series()).astype(str)
        press_time = df.get("Press-Zeit", pd.Series()).astype(str)
        t0 = pd.to_datetime(press_date + " " + press_time, errors="coerce")
    df["t0"] = t0
    for col in df.columns:
        if col.startswith("WG-"):
            df = df.rename(columns={col: "WG_Nr"})
            break
    keep_cols = ["WG_Nr", "t0", "Produkt", "Rezept", "Stärke", "m³", "In Z2", "In Z3", "In Z4", "In Z5", "Zeit in Z1", "Zeit in Z2", "Zeit in Z3", "Zeit in Z4", "Zeit in Z5"]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()
    if "m³" in df.columns:
        df["m3"] = pd.to_numeric(df["m³"], errors='coerce')
    else:
        staerke = pd.to_numeric(df.get("Stärke", 0), errors='coerce')
        df["m3"] = 0.605 * 0.605 * (staerke + 7) / 1000
    zone_entry_cols = {f"In {z}": f"{z}_in" for z in ("Z2", "Z3", "Z4", "Z5")}
    for raw_col, new_col in zone_entry_cols.items():
        if raw_col in df.columns:
            df[new_col] = pd.to_datetime(df[raw_col], errors="coerce", dayfirst=True)
        else:
            df[new_col] = pd.NaT
    df["Z1_in"] = df["t0"]
    if "Entnahme-Zeit" in df.columns:
        df["Entnahme-Zeit"] = pd.to_datetime(df["Entnahme-Zeit"], errors="coerce", dayfirst=True)
    else:
        df["Entnahme-Zeit"] = pd.NaT
    duration_pairs = [("Z1", "Z2_in", "t0"), ("Z2", "Z3_in", "Z2_in"), ("Z3", "Z4_in", "Z3_in"), ("Z4", "Z5_in", "Z4_in"), ("Z5", "Entnahme-Zeit", "Z5_in")]
    for zone, later_col, earlier_col in duration_pairs:
        hours = (df[later_col] - df[earlier_col]).dt.total_seconds() / 3600
        df[f"{zone}_dur_calc"] = hours
    for zone in CONFIG["zones_seq"]:
        text_col = f"Zeit in {zone}"
        calc_col = f"{zone}_dur_calc"
        dur_col = f"{zone}_dur"
        df[dur_col] = pd.to_timedelta(df[calc_col], unit="h")
        if text_col in df.columns:
            parsed = parse_duration_series(df[text_col])
            mask_replace = parsed.isna() | (parsed.dt.total_seconds() / 3600 < 1)
            df[dur_col] = parsed.where(~mask_replace, df[dur_col])
    df["Month"] = df["t0"].dt.month
    df["Year"] = df["t0"].dt.year
    df = df[df["t0"].notna()].copy()
    return df

def explode_intervals(df: pd.DataFrame) -> pd.DataFrame:
    # ... (Same as before) ...
    logger.info("Exploding wagon data into zone intervals...")
    rows = []
    for _, record in df.iterrows():
        intervals = []
        prev_end = None
        for zone in CONFIG["zones_seq"]:
            zone_in = record.get(f"{zone}_in", pd.NaT)
            if pd.isna(zone_in):
                zone_in = prev_end if prev_end is not None else record["t0"]
            zone_dur = record.get(f"{zone}_dur", pd.NaT)
            zone_out = zone_in + zone_dur if pd.notna(zone_in) and pd.notna(zone_dur) else pd.NaT
            if pd.notna(zone_in) and pd.notna(zone_out) and zone_out > zone_in:
                intervals.append((zone, zone_in, zone_out))
                prev_end = zone_out
        for zone, start_time, end_time in intervals:
            rows.append({"WG_Nr": record["WG_Nr"], "Produkt": record["Produkt"], "m3": record["m3"], "Zone": zone, "P_start": start_time, "P_end": end_time, "Month": record["Month"], "Year": record.get("Year", np.nan)})
    return pd.DataFrame(rows)

def allocate_energy(e: pd.DataFrame, ivals: pd.DataFrame) -> pd.DataFrame:
    # ... (Same as before) ...
    logger.info("Allocating energy to products...")
    results = []
    for zone_label, zone_name in ZONE_ENERGY_MAPPING.items():
        energy_col = f"E_{zone_name}_kWh"
        if energy_col not in e.columns: continue
        e_zone = e[e[energy_col].notna() & (e[energy_col] > 0)].copy()
        ivals_zone = ivals[ivals["Zone"] == zone_label].copy()
        if e_zone.empty or ivals_zone.empty: continue
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
            merged = merged[(merged['P_end'] > merged['E_start']) & (merged['P_start'] < merged['E_end'])]
            if merged.empty: continue
            merged['latest_start'] = merged[['E_start', 'P_start']].max(axis=1)
            merged['earliest_end'] = merged[['E_end', 'P_end']].min(axis=1)
            merged['overlap_h'] = ((merged['earliest_end'] - merged['latest_start']).dt.total_seconds() / 3600).clip(lower=0)
            merged = merged[merged['overlap_h'] > 0]
            merged['Energy_share_kWh'] = merged[energy_col] * merged['overlap_h']
            result = merged[['Month_e', 'Produkt', 'm3', 'Energy_share_kWh', 'overlap_h']].rename(columns={'Month_e': 'Month', 'overlap_h': 'Overlap_h'})
            result['Zone'] = zone_label
            zone_results.append(result)
        if zone_results: results.append(pd.concat(zone_results, ignore_index=True))
    if results: return pd.concat(results, ignore_index=True)
    return pd.DataFrame(columns=['Month', 'Zone', 'Produkt', 'Energy_share_kWh', 'Overlap_h', 'm3'])
