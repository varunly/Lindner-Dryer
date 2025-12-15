# dryer_kpi_monthly_final.py
"""
Lindner Trockner KPI Berechnungsmodul
KORRIGIERT: Korrekte Behandlung von Spaltennamen und Trockner-Filter
Alle ursprünglichen Funktionen bleiben erhalten
"""

# Importiert die notwendigen Bibliotheken für Datenverarbeitung und Protokollierung
import pandas as pd  # Für die Arbeit mit DataFrames
import numpy as np  # Für numerische Operationen
import logging  # Für die Protokollierung von Ereignissen und Fehlern
from typing import List, Tuple, Dict, Optional  # Für Typ-Hinweise, um die Lesbarkeit zu verbessern

# Konfiguriert das grundlegende Logging-Setup
logging.basicConfig(
    level=logging.INFO,  # Setzt die minimale Protokollierungsstufe auf INFO
    format="%(asctime)s - %(levelname)s - %(message)s"  # Definiert das Format der Log-Nachrichten
)
logger = logging.getLogger(__name__)  # Erstellt einen Logger für dieses Modul

# Spaltenkonfigurationen - dies sind bereinigte Namen (nach dem Entfernen von Zeilenumbrüchen)
TROCKNER_COLUMN = "Trockner"  # Name der Spalte, die den Trockner (A oder B) angibt
VOLUME_COLUMN = "m³"  # Name der Spalte, die das Volumen in Kubikmetern enthält

# Zentrales Konfigurationsdictionary für die gesamte Anwendung
CONFIG = {
    "energy_sheet": 0,  # Index oder Name des Tabellenblatts in der Energie-Excel-Datei
    "wagon_sheet": "Hordenwagenverfolgung",  # Name des Tabellenblatts in der Wagen-Excel-Datei
    "wagon_header_row": 6,  # Index der Kopfzeile in der Wagen-Excel-Datei
    "gas_to_kwh": 11.5,  # Umrechnungsfaktor von Gasverbrauch (m³) zu thermischer Energie (kWh)
    "zones_seq": ["Z1", "Z2", "Z3", "Z4", "Z5"],  # Die Reihenfolge der Zonen im Trockner
    "num_thermal_zones": 4,  # Anzahl der Zonen mit thermischer Energie (Z2-Z5)
    "trockner_column": TROCKNER_COLUMN,  # Verweis auf den Namen der Trockner-Spalte
    "volume_column": VOLUME_COLUMN,  # Verweis auf den Namen der Volumen-Spalte
}

# Mapping der Zonen-Schlüssel zu ihren vollständigen Namen in den Energiestammdaten
ZONE_ENERGY_MAPPING = {
    "Z2": "Zone 2",
    "Z3": "Zone 3",
    "Z4": "Zone 4",
    "Z5": "Zone 5",
}

# Konstanten für die Produktion
SUSPENSION_KG = 330  # Menge der Suspension in Kilogramm
PLATES_PER_WAGON = 234  # Anzahl der Platten pro Wagen

# Detaillierte Produktspezifikationen für alle Produkttypen (L, N, Y)
# Jedes Produkt enthält physikalische Eigenschaften und Parameter für die Wasserberechnung
PRODUCT_SPECIFICATIONS = {
    # ==================== L-TYP PRODUKTE ====================
    "L20": {
        "product_type": "L",
        "final_thickness_mm": 20,
        "press_measurement_mm": 3,
        "pressed_thickness_mm": 23,
        "edge_length_mm": 602,
        "volume_m3": 0.0083,
        "suspension_kg": 330,
        "slope": -0.060,
        "intercept": 107.7,
        "formula": "-0.060x + 107.7 (extrapolated from L28)",
        "water_per_mm_g": 88,
        "water_per_plate_kg": 2.02,
        "extrapolated": True,
    },
    "L24": {
        "product_type": "L",
        "final_thickness_mm": 24,
        "press_measurement_mm": 3,
        "pressed_thickness_mm": 27,
        "edge_length_mm": 602,
        "volume_m3": 0.0098,
        "suspension_kg": 330,
        "slope": -0.060,
        "intercept": 107.7,
        "formula": "-0.060x + 107.7 (extrapolated from L28)",
        "water_per_mm_g": 88,
        "water_per_plate_kg": 2.38,
        "extrapolated": True,
    },
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
    "L37": {
        "product_type": "L",
        "final_thickness_mm": 37,
        "press_measurement_mm": 6,
        "pressed_thickness_mm": 43,
        "edge_length_mm": 602,
        "volume_m3": 0.0156,
        "suspension_kg": 330,
        "slope": -0.042,
        "intercept": 93.5,
        "formula": "-0.042x + 93.5 (interpolated L36-L38)",
        "water_per_mm_g": 80,
        "water_per_plate_kg": 3.44,
        "interpolated": True,
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
    
    # ==================== N-TYP PRODUKTE ====================
    "N24": {
        "product_type": "N",
        "final_thickness_mm": 24,
        "press_measurement_mm": 6,
        "pressed_thickness_mm": 30,
        "edge_length_mm": 602,
        "volume_m3": 0.0109,
        "suspension_kg": 330,
        "slope": -0.103,
        "intercept": 102.5,
        "formula": "-0.103x + 102.5 (extrapolated from N40)",
        "water_per_mm_g": 69,
        "water_per_plate_kg": 2.07,
        "extrapolated": True,
    },
    "N30": {
        "product_type": "N",
        "final_thickness_mm": 30,
        "press_measurement_mm": 7,
        "pressed_thickness_mm": 37,
        "edge_length_mm": 602,
        "volume_m3": 0.0134,
        "suspension_kg": 330,
        "slope": -0.103,
        "intercept": 102.5,
        "formula": "-0.103x + 102.5 (extrapolated from N40)",
        "water_per_mm_g": 69,
        "water_per_plate_kg": 2.55,
        "extrapolated": True,
    },
    "N34": {
        "product_type": "N",
        "final_thickness_mm": 34,
        "press_measurement_mm": 8,
        "pressed_thickness_mm": 42,
        "edge_length_mm": 602,
        "volume_m3": 0.0152,
        "suspension_kg": 330,
        "slope": -0.103,
        "intercept": 102.5,
        "formula": "-0.103x + 102.5 (interpolated)",
        "water_per_mm_g": 69,
        "water_per_plate_kg": 2.90,
        "interpolated": True,
    },
    "N36": {
        "product_type": "N",
        "final_thickness_mm": 36,
        "press_measurement_mm": 9,
        "pressed_thickness_mm": 45,
        "edge_length_mm": 602,
        "volume_m3": 0.0163,
        "suspension_kg": 330,
        "slope": -0.103,
        "intercept": 102.5,
        "formula": "-0.103x + 102.5 (interpolated)",
        "water_per_mm_g": 69,
        "water_per_plate_kg": 3.11,
        "interpolated": True,
    },
    "N38": {
        "product_type": "N",
        "final_thickness_mm": 38,
        "press_measurement_mm": 9,
        "pressed_thickness_mm": 47,
        "edge_length_mm": 602,
        "volume_m3": 0.0170,
        "suspension_kg": 330,
        "slope": -0.103,
        "intercept": 102.5,
        "formula": "-0.103x + 102.5 (interpolated)",
        "water_per_mm_g": 69,
        "water_per_plate_kg": 3.24,
        "interpolated": True,
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
    "N42": {
        "product_type": "N",
        "final_thickness_mm": 42,
        "press_measurement_mm": 10,
        "pressed_thickness_mm": 52,
        "edge_length_mm": 602,
        "volume_m3": 0.0188,
        "suspension_kg": 330,
        "slope": -0.060,
        "intercept": 90.5,
        "formula": "-0.060x + 90.5 (interpolated N40-N44)",
        "water_per_mm_g": 71,
        "water_per_plate_kg": 3.69,
        "interpolated": True,
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
    
    # ==================== Y-TYP PRODUKTE ====================
    "Y30": {
        "product_type": "Y",
        "final_thickness_mm": 30,
        "press_measurement_mm": 8,
        "pressed_thickness_mm": 38,
        "edge_length_mm": 602,
        "volume_m3": 0.0138,
        "suspension_kg": 330,
        "slope": -0.157,
        "intercept": 200.0,
        "formula": "-0.157x + 200.0 (extrapolated from Y44)",
        "water_per_mm_g": 148,
        "water_per_plate_kg": 5.62,
        "extrapolated": True,
    },
    "Y34": {
        "product_type": "Y",
        "final_thickness_mm": 34,
        "press_measurement_mm": 9,
        "pressed_thickness_mm": 43,
        "edge_length_mm": 602,
        "volume_m3": 0.0156,
        "suspension_kg": 330,
        "slope": -0.157,
        "intercept": 200.0,
        "formula": "-0.157x + 200.0 (interpolated)",
        "water_per_mm_g": 148,
        "water_per_plate_kg": 6.36,
        "interpolated": True,
    },
    "Y38": {
        "product_type": "Y",
        "final_thickness_mm": 38,
        "press_measurement_mm": 10,
        "pressed_thickness_mm": 48,
        "edge_length_mm": 602,
        "volume_m3": 0.0174,
        "suspension_kg": 330,
        "slope": -0.157,
        "intercept": 200.0,
        "formula": "-0.157x + 200.0 (interpolated)",
        "water_per_mm_g": 148,
        "water_per_plate_kg": 7.10,
        "interpolated": True,
    },
    "Y40": {
        "product_type": "Y",
        "final_thickness_mm": 40,
        "press_measurement_mm": 11,
        "pressed_thickness_mm": 51,
        "edge_length_mm": 602,
        "volume_m3": 0.0185,
        "suspension_kg": 330,
        "slope": -0.157,
        "intercept": 200.0,
        "formula": "-0.157x + 200.0 (interpolated Y38-Y44)",
        "water_per_mm_g": 148,
        "water_per_plate_kg": 7.55,
        "interpolated": True,
    },
    "Y42": {
        "product_type": "Y",
        "final_thickness_mm": 42,
        "press_measurement_mm": 11,
        "pressed_thickness_mm": 53,
        "edge_length_mm": 602,
        "volume_m3": 0.0192,
        "suspension_kg": 330,
        "slope": -0.157,
        "intercept": 200.0,
        "formula": "-0.157x + 200.0 (interpolated Y38-Y44)",
        "water_per_mm_g": 148,
        "water_per_plate_kg": 7.84,
        "interpolated": True,
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
# Berechnet den Wassergehalt pro Kubikmeter für jedes Produkt und fügt ihn den Spezifikationen hinzu
for product, spec in PRODUCT_SPECIFICATIONS.items():
    spec["water_per_m3_kg"] = spec["water_per_plate_kg"] / spec["volume_m3"]

# Erstellt ein Dictionary für schnellen Zugriff auf den Wassergehalt pro m³
WATER_PER_M3_KG = {
    product: spec["water_per_m3_kg"]
    for product, spec in PRODUCT_SPECIFICATIONS.items()
}

def add_water_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt wasserbezogene KPIs zu einem DataFrame hinzu.
    
    Parameter:
        df: DataFrame mit den Spalten 'Produkt' und 'Volume_m3'
        
    Rückgabe:
        DataFrame mit hinzugefügten Spalten:
        - Water_kg: Gesamtmenge des verdampften Wassers
        - Water_per_m3: Wassermenge pro Kubikmeter
        - kWh_per_kg: Energie pro kg Wasser (falls Energy_kWh existiert)
        - kWh_thermal_per_kg: Thermische Energie pro kg Wasser (falls vorhanden)
    """
    df = df.copy()  # Erstellt eine Kopie, um das Original zu vermeiden
    
    # Hilfsfunktion, um den Wassergehalt pro m³ für ein Produkt zu erhalten
    def get_water_per_m3(product: str) -> float:
        if product in PRODUCT_SPECIFICATIONS:
            return PRODUCT_SPECIFICATIONS[product]["water_per_m3_kg"]
        return WATER_PER_M3_KG.get(product, 180.0)  # Standard-Fallback, falls Produkt nicht gefunden
    
    # Fügt die Spalte 'Water_per_m3' hinzu
    if "Produkt" in df.columns:
        df["Water_per_m3"] = df["Produkt"].apply(get_water_per_m3)
    else:
        df["Water_per_m3"] = 180.0  # Standardwert, falls keine Produktspalte vorhanden
    
    # Berechnet die Gesamtmenge an Wasser
    if "Volume_m3" in df.columns:
        df["Water_kg"] = df["Volume_m3"] * df["Water_per_m3"]
    else:
        df["Water_kg"] = 0.0
    
    # Berechnet die Energie pro kg Wasser
    if "Energy_kWh" in df.columns and "Water_kg" in df.columns:
        df["kWh_per_kg"] = np.where(
            df["Water_kg"] > 0,
            df["Energy_kWh"] / df["Water_kg"],
            0.0
        )
    
    # Berechnet die thermische Energie pro kg Wasser
    if "Energy_thermal_kWh" in df.columns and "Water_kg" in df.columns:
        df["kWh_thermal_per_kg"] = np.where(
            df["Water_kg"] > 0,
            df["Energy_thermal_kWh"] / df["Water_kg"],
            0.0
        )
    
    return df

def safe_divide(numerator, denominator, default=0.0):
    """Sichere Division, die Null- und NaN-Werte behandelt."""
    with np.errstate(divide='ignore', invalid='ignore'):  # Ignoriert Warnungen für Division durch Null
        result = np.where(
            (denominator != 0) & (~np.isnan(denominator)) & (np.isfinite(denominator)),
            numerator / denominator,
            default
        )
    # Konvertiert NaN zu einem Standardwert
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)


def parse_duration_series(s: pd.Series) -> pd.Series:
    """Parst Dauer-Zeichenketten in timedelta-Objekte."""
    s = s.astype(str).str.strip()  # Konvertiert zu String und entfernt Leerzeichen
    s = s.str.replace(",", ".", regex=False)  # Ersetzt Komma durch Punkt für Dezimalzahlen
    s = s.str.replace(r"\bh\b", "hours", regex=True)  # Ersetzt 'h' durch 'hours'
    s = s.str.replace(r"\bmin\b", "minutes", regex=True)  # Ersetzt 'min' durch 'minutes'
    s = s.str.replace(r"\bst\b", "seconds", regex=True)  # Ersetzt 's' durch 'seconds'
    s = s.replace({r"^\s*$": np.nan, r"^-$": np.nan}, regex=True)  # Ersetzt leere oder '-' Werte durch NaN
    td = pd.to_timedelta(s, errors="coerce")  # Konvertiert zu timedelta, fehlerhaft wird zu NaN
    mask_nat = td.isna() & s.notna()  # Findet Werte, die nicht konvertiert werden konnten
    if mask_nat.any():
        dt = pd.to_datetime(s[mask_nat], errors="coerce")  # Versucht, als Datum zu parsen
        td.loc[mask_nat] = dt - pd.Timestamp("1900-01-01")  # Berechnet die Differenz zu einem Referenzdatum
    return td


def parse_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Parst stündliche Energiestammdaten."""
    logger.info("Verarbeite Energiestammdaten...")
    df = df.copy()

    # Konvertiert den Zeitstempel in ein datetime-Objekt
    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], errors="coerce")
    # Entfernt Zeilen ohne gültigen Zeitstempel
    df = df[df["Zeitstempel"].notna()].copy()

    # Extrahiert Monat und Jahr aus dem Zeitstempel
    df["Month"] = df["Zeitstempel"].dt.month
    df["Year"] = df["Zeitstempel"].dt.year

    total_thermal = 0
    # Berechnet die thermische Energie für jede Zone
    for z_key, z_name in ZONE_ENERGY_MAPPING.items():
        gas_col = f"Gasmenge, {z_name} [m³]"  # Name der Gasspalalte
        thermal_col = f"E_thermal_{z_name}_kWh"  # Name der Ergebnisspalte
        
        if gas_col in df.columns:
            # Konvertiert Gasspalalte in numerische Werte
            gas_values = pd.to_numeric(df[gas_col], errors='coerce').fillna(0)
            # Berechnet thermische Energie
            df[thermal_col] = gas_values * CONFIG["gas_to_kwh"]
            zone_total = df[thermal_col].sum()
            total_thermal += zone_total
            logger.info(f"  {z_key}: {gas_values.sum():,.0f} m³ Gas → {zone_total:,.0f} kWh thermisch")
        else:
            df[thermal_col] = 0.0

    # Summiert die thermische Energie aller Zonen
    thermal_cols = [f"E_thermal_{z_name}_kWh" for z_name in ZONE_ENERGY_MAPPING.values()]
    existing_thermal_cols = [c for c in thermal_cols if c in df.columns]
    df["E_thermal_total_kWh"] = df[existing_thermal_cols].sum(axis=1)

    # Extrahiert die elektrische Energie
    if "Energieverbrauch, elektr. [kWh]" in df.columns:
        df["E_el_kWh"] = pd.to_numeric(
            df["Energieverbrauch, elektr. [kWh]"], errors='coerce'
        ).fillna(0.0)
    else:
        df["E_el_kWh"] = 0.0

    # Definiert Start- und Endezeit für jede Energiestunde
    df["E_start"] = df["Zeitstempel"]
    df["E_end"] = df["Zeitstempel"] + pd.Timedelta(hours=1)

    logger.info(f"{len(df)} Energiestammdaten verarbeitet")
    
    return df


def find_column_flexible(df: pd.DataFrame, patterns: list, description: str = "") -> Optional[str]:
    """
    Findet eine Spalte, die einem der gegebenen Muster entspricht (Groß-/Kleinschreibung, Teilübereinstimmung).
    """
    # Versucht zuerst exakte Übereinstimmungen
    for pattern in patterns:
        if pattern in df.columns:
            return pattern
    
    # Versucht Übereinstimmungen ohne Berücksichtigung der Groß-/Kleinschreibung und Teilübereinstimmungen
    for col in df.columns:
        col_lower = str(col).lower()
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in col_lower or col_lower == pattern_lower:
                return col
    
    return None


def parse_wagon(df: pd.DataFrame, trockner: str = None) -> pd.DataFrame:
    """
    Parst Wagen-Stammdaten mit KORREKTER Trockner-Filterung.
    
    KORRIGIERTE VERSION:
    - Wendet Trockner-Filter ZUERST an (vor anderen Filtern)
    - Bereinigt Spaltennamen korrekt
    - Verfolgt Zeilenverlust bei jedem Schritt
    - Produkt setzt sich aus 'EM' (Dicke) + 'Rez.' (Typ L/N/Y mit Forward-Fill) zusammen
    """
    logger.info("="*70)
    logger.info("VERARBEITUNG DER WAGEN-STAMMDATEN - KORRIGIERTE VERSION")
    logger.info(f"Angeforderter Trockner-Filter: {trockner or 'None (Alle)'}")
    logger.info("="*70)
    
    raw_row_count = len(df)
    logger.info(f"Roh-Eingabe: {raw_row_count} Zeilen, {len(df.columns)} Spalten")
    
    df = df.copy()
    
    # ========================================
    # SCHritt 1: SPALTENNAMEN BEREINIGEN
    # Entfernt Zeilenumbrüche und Wagenrückläufe
    # ========================================
    original_columns = list(df.columns)
    
    cleaned_columns = []
    for col in df.columns:
        col_str = str(col)
        col_clean = col_str.replace("\n", "").replace("\r", "").strip()
        cleaned_columns.append(col_clean)
    
    df.columns = cleaned_columns
    
    logger.info("Bereinigung der Spaltennamen:")
    changes_logged = 0
    for i, (orig, clean) in enumerate(zip(original_columns[:15], df.columns[:15])):
        if str(orig) != clean:
            logger.info(f"  [{i:2d}] {repr(str(orig))} → '{clean}'")
            changes_logged += 1
    
    if changes_logged == 0:
        logger.info("  Es mussten keine Spaltennamen bereinigt werden")
    
    logger.info(f"Spalten (erste 15): {list(df.columns)[:15]}")
    
    # ========================================
    # SCHritt 2: WICHTIGE SPALTEN FINDEN
    # ========================================
    
    # Findet Trockner-Spalte
    trockner_col = find_column_flexible(df, ["Trockner", "Trock-ner", "Trock-", "TROCKNER"])
    if trockner_col:
        logger.info(f"✓ Trockner-Spalte: '{trockner_col}'")
    else:
        logger.warning("✗ Trockner-Spalte NICHT GEFUNDEN!")
    
    # Findet Wagennummern-Spalte (erste Spalte oder WG-Nr)
    wagon_col = find_column_flexible(df, ["WG-Nr", "WG-Nr.", "WGNr", "WG Nr", "WG_Nr"])
    if not wagon_col:
        wagon_col = df.columns[0]
    logger.info(f"✓ Wagen-Spalte: '{wagon_col}'")
    
    # Findet Volumen-Spalte (m³) - typischerweise Spalte AA (Index 26)
    volume_col = find_column_flexible(df, ["m³", "m3", "Volumen", "Volume"])
    if not volume_col and len(df.columns) > 26:
        volume_col = df.columns[26]
    logger.info(f"✓ Volumen-Spalte: '{volume_col}'")
    
    # Findet EM-Spalte (Dicke)
    em_col = find_column_flexible(df, ["EM", "Dicke", "Thickness"])
    if em_col:
        logger.info(f"✓ EM (Dicke)-Spalte: '{em_col}'")
    else:
        logger.warning("✗ EM-Spalte NICHT GEFUNDEN!")
    
    # Findet Rez.-Spalte (Produkttyp)
    rez_col = find_column_flexible(df, ["Rez.", "Rez", "Rezept", "Rezeptur"])
    if rez_col:
        logger.info(f"✓ Rez. (Typ)-Spalte: '{rez_col}'")
    else:
        logger.warning("✗ Rez.-Spalte NICHT GEFUNDEN!")
    
    # Findet Zeitstempel-Spalte
    timestamp_col = find_column_flexible(df, ["Pressdat. + Zeit", "Pressdat", "Pressdatum"])
    if timestamp_col:
        logger.info(f"✓ Zeitstempel-Spalte: '{timestamp_col}'")
    else:
        logger.warning("✗ Zeitstempel-Spalte NICHT GEFUNDEN!")
    
    # ========================================
    # SCHritt 3: TROCKNER-FILTER ZUERST ANWENDEN!
    # Dies ist der entscheidende Fix - Filtere nach Trockner, bevor irgendwas anderes passiert
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("SCHRITT 3: TROCKNER-FILTER ZUERST ANWENDEN")
    logger.info("="*50)
    
    count_before_trockner = len(df)
    
    if trockner_col and trockner_col in df.columns:
        # Bereinigt Trockner-Werte
        df["_trockner_clean"] = df[trockner_col].astype(str).str.strip().str.upper()
        
        # Zeigt die Verteilung vor dem Filtern an
        logger.info("Trockner-Verteilung (vor dem Filtern):")
        value_counts = df["_trockner_clean"].value_counts()
        for val, count in value_counts.items():
            logger.info(f"  '{val}': {count:,} Zeilen")
        
        # Filtere zu gültigen Trockner-Werten (nur A oder B)
        valid_trockner_mask = df["_trockner_clean"].isin(["A", "B"])
        invalid_count = (~valid_trockner_mask).sum()
        
        if invalid_count > 0:
            logger.info(f"Entferne {invalid_count:,} Zeilen mit ungültigem Trockner (nicht A oder B)")
        
        df = df[valid_trockner_mask].copy()
        count_after_valid_trockner = len(df)
        logger.info(f"Nach Filterung auf gültige Trockner (A/B): {count_after_valid_trockner:,} Zeilen")
        
        # Wende nun die spezifische Trockner-Auswahl an, falls gewünscht
        if trockner and trockner.upper() in ["A", "B"]:
            trockner_upper = trockner.upper().strip()
            
            # Zähle vor der Auswahl
            count_a = (df["_trockner_clean"] == "A").sum()
            count_b = (df["_trockner_clean"] == "B").sum()
            logger.info(f"Verfügbar: Trockner A = {count_a:,}, Trockner B = {count_b:,}")
            
            # Wende Auswahl an
            before_selection = len(df)
            df = df[df["_trockner_clean"] == trockner_upper].copy()
            after_selection = len(df)
            
            logger.info(f"Ausgewählter Trockner '{trockner_upper}': {after_selection:,} Zeilen")
            logger.info(f"Entfernt: {before_selection - after_selection:,} Zeilen (anderer Trockner)")
        
        # Entferne temporäre Spalte
        df = df.drop(columns=["_trockner_clean"], errors="ignore")
    else:
        logger.warning("⚠️ Kann nicht nach Trockner filtern - Spalte nicht gefunden!")
    
    count_after_trockner = len(df)
    logger.info(f"\n>>> ZEILEN NACH TROCKNER-FILTER: {count_after_trockner:,} <<<")
    
    if df.empty:
        raise ValueError(f"Keine Zeilen für Trockner '{trockner}' gefunden")
    
    # ========================================
    # SCHritt 4: FILTERE ZEILEN MIT GÜLTIGER WAGENNUMMER
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("SCHRITT 4: FILTERE NACH WAGENNUMMER")
    logger.info("="*50)
    
    if wagon_col and wagon_col in df.columns:
        wagon_vals = df[wagon_col].astype(str).str.strip()
        valid_wagon = (wagon_vals != "") & (wagon_vals != "nan") & (wagon_vals != "NaN") & (wagon_vals != "None") & (wagon_vals.notna())
        
        count_before = len(df)
        df = df[valid_wagon].copy()
        count_after = len(df)
        
        if count_before != count_after:
            logger.info(f"Entferne {count_before - count_after:,} Zeilen mit leerer Wagennummer")
        logger.info(f"Zeilen mit gültiger Wagennummer: {count_after:,}")
    
    # ========================================
    # SCHritt 5: FILTERE ZEILEN MIT GÜLTIGEM VOLUMEN
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("SCHRITT 5: FILTERE NACH VOLUMEN")
    logger.info("="*50)
    
    if volume_col and volume_col in df.columns:
        df["m3"] = pd.to_numeric(df[volume_col], errors='coerce')
        
        valid_vol = df["m3"].notna() & (df["m3"] > 0)
        invalid_vol_count = (~valid_vol).sum()
        
        if invalid_vol_count > 0:
            # Zeigt eine Stichprobe von ungültigen Werten
            invalid_sample = df.loc[~valid_vol, volume_col].head(10).tolist()
            logger.info(f"Found {invalid_vol_count:,} rows with invalid volume")
            logger.info(f"Sample invalid values: {invalid_sample}")
        
        count_before = len(df)
        df = df[valid_vol].copy()
        count_after = len(df)
        
        if count_before != count_after:
            logger.info(f"Entferne {count_before - count_after:,} Zeilen mit ungültigem Volumen")
        
        logger.info(f"Zeilen mit gültigem Volumen: {count_after:,}")
        logger.info(f"Volumen-Statistik: min={df['m3'].min():.4f}, max={df['m3'].max():.4f}, mean={df['m3'].mean():.4f}, sum={df['m3'].sum():.2f}")
    else:
        df["m3"] = 3.5
        logger.warning("Volumen-Spalte nicht gefunden - verwende Standardwert 3.5 m³")
    
    # ========================================
    # SCHritt 6: PARSE ZEITSTEMPEL
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("SCHRITT 6: VERARBEITE ZEITSTEMPEL")
    logger.info("="*50)
    
    if timestamp_col and timestamp_col in df.columns:
        df["t0"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        valid_ts = df["t0"].notna().sum()
        invalid_ts = df["t0"].isna().sum()
        logger.info(f"Gültige Zeitstempel: {valid_ts:,}, Ungültige: {invalid_ts:,}")
        
        if invalid_ts > 0:
            invalid_sample = df.loc[df["t0"].isna(), timestamp_col].head(5).tolist()
            logger.info(f"Stichprobe ungültiger Zeitstempel-Werte: {invalid_sample}")
    else:
        # Versucht alternative Spalten
        date_col = find_column_flexible(df, ["Datum", "Date", "Press-datum"])
        time_col = find_column_flexible(df, ["Zeit", "Time", "Uhrzeit", "Press-Zeit"])
        
        if date_col and time_col and "Entnahme" not in str(time_col):
            df["t0"] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str),
                errors="coerce"
            )
            logger.info(f"Kombinierte Zeitstempel aus '{date_col}' + '{time_col}'")
        else:
            df["t0"] = pd.NaT
            logger.warning("Konnte Zeitstempel nicht parsen")
    
    # Filtere Zeilen ohne gültigen Zeitstempel
    count_before = len(df)
    df = df[df["t0"].notna()].copy()
    count_after = len(df)
    if count_before != count_after:
        logger.info(f"Entferne {count_before - count_after:,} Zeilen ohne gültigen Zeitstempel")
    logger.info(f"Zeilen mit gültigem Zeitstempel: {count_after:,}")
    
    # ========================================
    # SCHritt 7: PARSE PRODUKT (EM + Rez. mit Forward-Fill)
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("SCHRITT 7: PARSE PRODUKT (EM + Rez.)")
    logger.info("="*50)
    
    if em_col and em_col in df.columns:
        # Erhalte die Dicke aus der EM-Spalte
        df["_thickness"] = pd.to_numeric(df[em_col], errors='coerce').astype('Int64')
        
        valid_thickness = df["_thickness"].notna().sum()
        logger.info(f"Gültige Dicken-Werte: {valid_thickness:,}")
        
        logger.info(f"Dicken-Verteilung:")
        thickness_counts = df["_thickness"].value_counts().sort_index()
        for val, count in thickness_counts.items():
            logger.info(f"  {val}: {count:,} Zeilen")
    else:
        logger.error("Kann Produkt nicht ohne EM-Spalte parsen!")
        df["_thickness"] = pd.NA
    
    if rez_col and rez_col in df.columns:
        # Erhalte den Produkttyp aus der Rez.-Spalte
        df["_type_raw"] = df[rez_col].astype(str).str.strip().str.upper()
        
        # Ersetze leere Zeichenketten und ungültige Werte durch NaN für Forward-Fill
        df["_type_raw"] = df["_type_raw"].replace(["", "NAN", "NONE", "NA", "<NA>", "-"], pd.NA)
        
        # Zähle nicht-leere Werte
        non_empty_count = df["_type_raw"].notna().sum()
        logger.info(f"Nicht-leere Rez.-Werte: {non_empty_count:,}")
        if non_empty_count > 0:
            unique_types = df["_type_raw"].dropna().unique().tolist()
            logger.info(f"Gefundene einzigartige Typen: {unique_types}")
        
        # FORWARD-FILL des Produkttyps
        df["_type_filled"] = df["_type_raw"].ffill()
        
        # Falls am Anfang noch NaN, setze Standardwert 'L'
        df["_type_filled"] = df["_type_filled"].fillna("L")
        
        # Bereinige - behalte nur gültige Typen (L, N, Y)
        valid_types = ["L", "N", "Y"]
        df["_type_filled"] = df["_type_filled"].apply(
            lambda x: x if x in valid_types else "L"
        )
        
        logger.info(f"Produkttyp nach Forward-Fill:")
        type_counts = df["_type_filled"].value_counts()
        for val, count in type_counts.items():
            logger.info(f"  '{val}': {count:,} Zeilen")
    else:
        logger.warning("Keine Rez.-Spalte gefunden - setze alle auf L-Typ")
        df["_type_filled"] = "L"
    
    # Kombiniere Typ + Dicke, um Produktcode zu erstellen
    def create_product_code(row):
        ptype = row.get("_type_filled", "L")
        thickness = row.get("_thickness", pd.NA)
        
        if pd.isna(thickness):
            return "Unknown"
        
        return f"{ptype}{int(thickness)}"
    
    df["Produkt"] = df.apply(create_product_code, axis=1)
    
    # Zeige Produktverteilung an
    logger.info(f"Kombinierte Produktcodes:")
    product_counts = df["Produkt"].value_counts()
    for val, count in product_counts.items():
        in_specs = "✓" if val in PRODUCT_SPECIFICATIONS else "✗"
        logger.info(f"  {in_specs} '{val}': {count:,} Zeilen")
    
    # Bereinige temporäre Spalten
    df = df.drop(columns=["_thickness", "_type_raw", "_type_filled"], errors="ignore")
    
    # Filtere zu gültigen Produkten
    valid_products = list(PRODUCT_SPECIFICATIONS.keys())
    count_before = len(df)
    invalid_mask = ~df["Produkt"].isin(valid_products)
    
    if invalid_mask.sum() > 0:
        invalid_prods = df.loc[invalid_mask, "Produkt"].value_counts()
        logger.warning(f"Entferne {invalid_mask.sum():,} Zeilen mit ungültigen Produkten:")
        for prod, count in invalid_prods.items():
            logger.warning(f"  ✗ '{prod}': {count:,} Zeilen")
    
    df = df[df["Produkt"].isin(valid_products)].copy()
    count_after = len(df)
    
    if count_before != count_after:
        logger.info(f"Nach Produktfilter: {count_after:,} Zeilen (entfernt {count_before - count_after:,})")
    
    # ========================================
    # SCHritt 8: SPALTEN UMBENENNEN UND METADATEN HINZUFÜGEN
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("SCHRITT 8: SPALTEN FINALISIEREN")
    logger.info("="*50)
    
    # Benennt Wagenspalte um
    if wagon_col and wagon_col in df.columns and wagon_col != "WG_Nr":
        df = df.rename(columns={wagon_col: "WG_Nr"})
    elif "WG_Nr" not in df.columns:
        df["WG_Nr"] = df.iloc[:, 0]
    
    # Füge Monat/Jahr hinzu
    df["Month"] = df["t0"].dt.month
    df["Year"] = df["t0"].dt.year
    df["Trockner"] = trockner if trockner else "All"
    
    # ========================================
    # SCHritt 9: PARSE ZONEN-EINTRITTSZEITEN
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("SCHRITT 9: VERARBEITE ZONEN-ZEITEN")
    logger.info("="*50)
    
    df["Z1_in"] = df["t0"]
    
    for z in ("Z2", "Z3", "Z4", "Z5"):
        zone_col = find_column_flexible(df, [f"In {z}", f"In{z}", f"Zone {z[-1]} In"])
        if zone_col and zone_col in df.columns:
            df[f"{z}_in"] = pd.to_datetime(df[zone_col], errors="coerce", dayfirst=True)
            valid_count = df[f"{z}_in"].notna().sum()
            logger.info(f"  {z}_in: {valid_count:,} gültig")
        else:
            df[f"{z}_in"] = pd.NaT
    
    entnahme_col = find_column_flexible(df, ["Entnahme-Zeit", "EntnahmeZeit", "Entnahme Zeit", "Entnahme"])
    if entnahme_col and entnahme_col in df.columns:
        df["Entnahme"] = pd.to_datetime(df[entnahme_col], errors="coerce", dayfirst=True)
        logger.info(f"  Entnahme: {df['Entnahme'].notna().sum():,} gültig")
    else:
        df["Entnahme"] = pd.NaT
    
    # ========================================
    # SCHritt 10: BERECHNE ZONEN-DAUERN
    # ========================================
    logger.info("")
    logger.info("="*50)
    logger.info("SCHRITT 10: BERECHNE ZONEN-DAUERN")
    logger.info("="*50)
    
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
        
        valid_dur = df[out].notna().sum()
        logger.info(f"  {z}_dur: {valid_dur:,} gültig")
    
    # ========================================
    # FINALE ZUSAMMENFASSUNG
    # ========================================
    logger.info("")
    logger.info("="*70)
    logger.info(f"FINALE ERGEBNISSE - TROCKNER: {trockner or 'ALLE'}")
    logger.info("="*70)
    
    total_rows = len(df)
    total_vol = df["m3"].sum() if not df.empty else 0
    avg_vol = df["m3"].mean() if total_rows > 0 else 0
    
    if total_rows > 0:
        logger.info("Nach Produkt:")
        product_summary = df.groupby("Produkt").agg({"m3": ["count", "sum", "mean"]}).round(4)
        product_summary.columns = ["Rows", "Volume_m3", "Avg_m3"]
        product_summary = product_summary.sort_values("Volume_m3", ascending=False)
        
        for prod, row in product_summary.iterrows():
            logger.info(f"  {prod:6s}: {int(row['Rows']):5d} Zeilen | {row['Volume_m3']:8.2f} m³")
    
    logger.info("-"*70)
    logger.info(f"GESAMT: {total_rows:,} Wagen-Zeilen | {total_vol:,.2f} m³ | {avg_vol:.4f} m³/Zeile")
    logger.info("")
    logger.info("ZEILEN-VERFOLG:")
    logger.info(f"  Eingabezeilen:           {raw_row_count:,}")
    logger.info(f"  Nach Trockner:       {count_after_trockner:,}")
    logger.info(f"  Finale Zeilen:           {total_rows:,}")
    logger.info(f"  Verloren nach Trockner:  {count_after_trockner - total_rows:,}")
    logger.info("="*70)
    
    # Validierung für erwartete Anzahl
    if trockner == "A" and total_rows < 3500:
        logger.warning(f"⚠️ Erwartet ~3692 Zeilen für Trockner A, erhalten {total_rows}")
    if trockner == "B" and total_rows < 3500:
        logger.warning(f"⚠️ Erwartet ~3691 Zeilen für Trockner B, erhalten {total_rows}")
    
    # =============================================
    # NEU: Füge Zonendauer in Stunden für die Anzeige hinzu
    # =============================================
    for z in CONFIG["zones_seq"]:
        dur_col = f"{z}_dur"
        hours_col = f"{z}_dur_hours"
        if dur_col in df.columns:
            df[hours_col] = df[dur_col].dt.total_seconds() / 3600
        else:
            df[hours_col] = np.nan
    
    # Berechne Gesamt-Verweildauer
    if "Entnahme" in df.columns and "t0" in df.columns:
        df["Total_residence_hours"] = (df["Entnahme"] - df["t0"]).dt.total_seconds() / 3600
    else:
        df["Total_residence_hours"] = np.nan
    
    # Füge Woche/Jahr basierend auf Z2-Eintritt (für Energieberechnungen) hinzu
    if "Z2_in" in df.columns and df["Z2_in"].notna().any():
        df["Z2_entry_time"] = df["Z2_in"]
        df["Week_Z2"] = df["Z2_in"].dt.isocalendar().week
        df["Year_Z2"] = df["Z2_in"].dt.year
    else:
        df["Z2_entry_time"] = df["t0"]
        df["Week_Z2"] = df["t0"].dt.isocalendar().week
        df["Year_Z2"] = df["t0"].dt.year
    
    logger.info(f"Finales Ergebnis: {len(df)} Zeilen, {df['m3'].sum():.2f} m³")
    
    return df
    

def compute_zone_duration_stats(wagons: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Statistiken für die Zeit, die in jeder Zone verbracht wird.
    
    Gibt einen DataFrame mit durchschnittlicher, minimaler, maximaler Dauer pro Zone und Produkt zurück.
    """
    logger.info("Berechne Zonen-Verweildauer-Statistiken...")
    
    zone_cols = ["Z1_dur_hours", "Z2_dur_hours", "Z3_dur_hours", "Z4_dur_hours", "Z5_dur_hours"]
    available_cols = [c for c in zone_cols if c in wagons.columns]
    
    if not available_cols:
        logger.warning("Keine Zonen-Dauer-Spalten gefunden!")
        return pd.DataFrame()
    
    # Gesamtdaten pro Zone
    zone_stats = []
    
    for col in available_cols:
        zone_name = col.replace("_dur_hours", "")
        valid_data = wagons[col].dropna()
        
        if len(valid_data) > 0:
            zone_stats.append({
                "Zone": zone_name,
                "Avg_Hours": valid_data.mean(),
                "Min_Hours": valid_data.min(),
                "Max_Hours": valid_data.max(),
                "Std_Hours": valid_data.std(),
                "Count": len(valid_data),
            })
    
    overall_stats = pd.DataFrame(zone_stats)
    
    # Daten pro Produkt
    product_zone_stats = []
    
    for product in wagons["Produkt"].unique():
        prod_data = wagons[wagons["Produkt"] == product]
        
        for col in available_cols:
            zone_name = col.replace("_dur_hours", "")
            valid_data = prod_data[col].dropna()
            
            if len(valid_data) > 0:
                product_zone_stats.append({
                    "Produkt": product,
                    "Zone": zone_name,
                    "Avg_Hours": valid_data.mean(),
                    "Min_Hours": valid_data.min(),
                    "Max_Hours": valid_data.max(),
                    "Count": len(valid_data),
                })
    
    product_stats = pd.DataFrame(product_zone_stats)
    
    return overall_stats, product_stats
def diagnose_wagon_file(df: pd.DataFrame, trockner: str = None) -> dict:
    """
    Diagnosefunktion, um genau nachzuvollziehen, wo Zeilen verloren gehen.
    Rufe diese Funktion VOR parse_wagon auf, um die Rohdaten zu sehen.
    """
    print("="*70)
    print("DIAGNOSE: WAGEN-DATEI-ANALYSE")
    print("="*70)
    
    result = {
        "raw_rows": len(df),
        "steps": []
    }
    
    # Schritt 0: Rohdaten
    print(f"\n[SCHRITT 0] Rohdatei: {len(df):,} Zeilen, {len(df.columns)} Spalten")
    result["steps"].append(("Rohdatei", len(df)))
    
    # Zeige die ersten 15 Spalten mit ihren Indizes
    print("\nSpaltennamen (erste 15):")
    for i, col in enumerate(df.columns[:15]):
        print(f"  [{i:2d}] {repr(col)}")
    
    # Erstelle eine Kopie und bereinige Spaltennamen
    df_clean = df.copy()
    df_clean.columns = [str(c).replace("\n", "").replace("\r", "").strip() for c in df_clean.columns]
    
    print("\nBereinigte Spaltennamen (erste 15):")
    for i, col in enumerate(df_clean.columns[:15]):
        print(f"  [{i:2d}] '{col}'")
    
    # Finde wichtige Spalten
    print("\n" + "-"*50)
    print("FINDE WICHTIGE SPALTEN")
    print("-"*50)
    
    # Trockner-Spalte
    trockner_col = None
    for col in df_clean.columns:
        if "trock" in col.lower():
            trockner_col = col
            break
    
    if trockner_col:
        print(f"✓ Trockner-Spalte: '{trockner_col}'")
        trockner_vals = df_clean[trockner_col].astype(str).str.strip().str.upper()
        print(f"  Werte: {trockner_vals.value_counts().to_dict()}")
        
        if trockner:
            match_count = (trockner_vals == trockner.upper()).sum()
            print(f"  Zeilen, die auf '{trockner.upper()}' passen: {match_count:,}")
            result["trockner_match"] = match_count
    else:
        print("✗ Trockner-Spalte NICHT GEFUNDEN")
    
    # EM-Spalte (Dicke)
    em_col = None
    for col in df_clean.columns:
        if col.upper() == "EM":
            em_col = col
            break
    
    if em_col:
        print(f"\n✓ EM (Dicke)-Spalte: '{em_col}'")
        em_vals = df_clean[em_col].astype(str).str.strip()
        print(f"  Stichprobenwerte: {em_vals.head(10).tolist()}")
        print(f"  Einzigartige Werte: {sorted(em_vals.unique())[:15]}")
    else:
        print("\n✗ EM-Spalte NICHT GEFUNDEN")
    
    # Rez.-Spalte (Produkttyp)
    rez_col = None
    for col in df_clean.columns:
        if "rez" in col.lower():
            rez_col = col
            break
    
    if rez_col:
        print(f"\n✓ Rez. (Typ)-Spalte: '{rez_col}'")
        rez_vals = df_clean[rez_col].astype(str).str.strip().str.upper()
        rez_vals_clean = rez_vals.replace(["", "NAN", "NONE"], pd.NA)
        non_empty = rez_vals_clean.dropna()
        print(f"  Nicht-leere Werte: {len(non_empty):,} Zeilen")
        print(f"  Einzigartige nicht-leere: {non_empty.unique().tolist()}")
    else:
        print("\n✗ Rez.-Spalte NICHT GEFUNDEN")
    
    # Volumen-Spalte
    volume_col = None
    for col in df_clean.columns:
        if "m³" in col or "m3" in col.lower():
            volume_col = col
            break
    if not volume_col and len(df_clean.columns) > 26:
        volume_col = df_clean.columns[26]
    
    if volume_col:
        print(f"\n✓ Volumen-Spalte: '{volume_col}'")
        vol_numeric = pd.to_numeric(df_clean[volume_col], errors='coerce')
        print(f"  Gültig numerisch: {vol_numeric.notna().sum():,}")
        print(f"  Positiv (>0): {(vol_numeric > 0).sum():,}")
        print(f"  Null oder negativ: {(vol_numeric <= 0).sum():,}")
        print(f"  NaN/ungültig: {vol_numeric.isna().sum():,}")
        result["volume_positive"] = (vol_numeric > 0).sum()
    else:
        print("\n✗ Volumen-Spalte NICHT GEFUNDEN")
    
    # Zeitstempel-Spalte
    print("\n" + "-"*50)
    print("ZEITSTEMPEL-ANALYSE")
    print("-"*50)
    
    timestamp_col = None
    for col in df_clean.columns:
        col_lower = col.lower()
        if "pressdat" in col_lower or "datum" in col_lower:
            timestamp_col = col
            break
    
    if timestamp_col:
        print(f"✓ Zeitstempel-Spalte: '{timestamp_col}'")
        ts = pd.to_datetime(df_clean[timestamp_col], errors='coerce')
        print(f"  Gültige Zeitstempel: {ts.notna().sum():,}")
        print(f"  Ungültig (NaT): {ts.isna().sum():,}")
        result["valid_timestamps"] = ts.notna().sum()
    else:
        print("✗ Zeitstempel-Spalte NICHT GEFUNDEN")
    
    # Simuliere Filterschritte
    print("\n" + "-"*50)
    print("SIMULIERTER FILTER-PROZESS")
    print("-"*50)
    
    df_sim = df_clean.copy()
    print(f"\n[START] {len(df_sim):,} Zeilen")
    
    # Schritt 1: Trockner-Filter (ZUERST!)
    if trockner and trockner_col:
        trockner_vals = df_sim[trockner_col].astype(str).str.strip().str.upper()
        # Filtere zuerst zu A/B nur
        df_sim = df_sim[trockner_vals.isin(["A", "B"])]
        print(f"[NACH GÜLTIGEM TROCKNER A/B] {len(df_sim):,} Zeilen")
        # Filtere dann zu spezifischem Trockner
        trockner_vals = df_sim[trockner_col].astype(str).str.strip().str.upper()
        df_sim = df_sim[trockner_vals == trockner.upper()]
        print(f"[NACH TROCKNER '{trockner}'] {len(df_sim):,} Zeilen")
        result["steps"].append((f"Nach Trockner {trockner}", len(df_sim)))
    
    # Schritt 2: Volumen-Filter
    if volume_col and volume_col in df_sim.columns:
        vol = pd.to_numeric(df_sim[volume_col], errors='coerce')
        before = len(df_sim)
        df_sim = df_sim[vol > 0]
        print(f"[NACH VOLUMEN > 0] {len(df_sim):,} Zeilen (entfernt {before - len(df_sim):,})")
        result["steps"].append(("Nach Volumen > 0", len(df_sim)))
    
    # Schritt 3: Zeitstempel-Filter
    if timestamp_col and timestamp_col in df_sim.columns:
        ts = pd.to_datetime(df_sim[timestamp_col], errors='coerce')
        before = len(df_sim)
        df_sim = df_sim[ts.notna()]
        print(f"[NACH GÜLTIGEM ZEITSTEMPEL] {len(df_sim):,} Zeilen (entfernt {before - len(df_sim):,})")
        result["steps"].append(("Nach gültigem Zeitstempel", len(df_sim)))
    
    # Schritt 4: Erstelle Produktcode
    if em_col and em_col in df_sim.columns:
        thickness = pd.to_numeric(df_sim[em_col], errors='coerce')
        
        # Erhalte Produkttyp mit Forward-Fill
        if rez_col and rez_col in df_sim.columns:
            ptype = df_sim[rez_col].astype(str).str.strip().str.upper()
            ptype = ptype.replace(["", "NAN", "NONE", "NA", "<NA>", "-"], pd.NA)
            ptype = ptype.ffill().fillna("L")
        else:
            ptype = "L"
        
        # Erstelle Produktcode
        df_sim["_product"] = ptype.astype(str) + thickness.astype(str).str.replace(".0", "", regex=False)
        
        print(f"\nProduktcodes erstellt:")
        prod_counts = df_sim["_product"].value_counts()
        for p, c in prod_counts.head(15).items():
            valid = "✓" if p in PRODUCT_SPECIFICATIONS else "✗"
            print(f"  {valid} {p}: {c:,}")
        
        # Filtere zu gültigen Produkten
        valid = df_sim["_product"].isin(PRODUCT_SPECIFICATIONS.keys())
        before = len(df_sim)
        df_sim = df_sim[valid]
        print(f"\n[NACH GÜLTIGEN PRODUKTEN] {len(df_sim):,} Zeilen (entfernt {before - len(df_sim):,})")
        result["steps"].append(("Nach gültigen Produkten", len(df_sim)))
    
    print("\n" + "="*70)
    print(f"FINALE ERWARTETE ANZAHL: {len(df_sim):,} Zeilen")
    print("="*70)
    
    result["final_expected"] = len(df_sim)
    
    return result


def build_intervals(row: pd.Series) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Erstellt Zonen-Intervalle für eine einzelne Wagenzeile."""
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
    """Explodiert Wagen-Daten in Zonen-Intervalle."""
    logger.info("Erstelle Zonen-Intervalle...")
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
    logger.info(f"{len(result)} Zonen-Intervalle aus {len(df)} Wagen-Zeilen erstellt")
    return result


def allocate_energy(e: pd.DataFrame, ivals: pd.DataFrame, full_electrical_kwh: float = None) -> pd.DataFrame:
    """
    VERSION 6.1 - GARANTIERT KORREKTE ELEKTRISCHE ENERGIE
    """
    logger.info("="*70)
    logger.info("ENERGIEZUWEISUNG VERSION 6.1")
    logger.info("="*70)
    
    if ivals.empty or e.empty:
        logger.warning("Keine Daten zum Zuweisen vorhanden!")
        return pd.DataFrame()
    
    # Thermisch aus gefilterten Daten
    INPUT_THERMAL = e["E_thermal_total_kWh"].sum()
    
    # Elektrisch: Verwende full_electrical_kwh, falls bereitgestellt, andernfalls gefilterte Daten
    if full_electrical_kwh is not None and full_electrical_kwh > 0:
        INPUT_ELECTRICAL = float(full_electrical_kwh)
        logger.info(f"✅ Verwende VOLLE elektrische Energie: {INPUT_ELECTRICAL:,.0f} kWh")
    else:
        INPUT_ELECTRICAL = e["E_el_kWh"].sum()
        logger.warning(f"⚠️ Verwende gefilterte elektrische Energie: {INPUT_ELECTRICAL:,.0f} kWh")
    
    INPUT_TOTAL = INPUT_THERMAL + INPUT_ELECTRICAL
    
    logger.info(f"INPUT: Thermisch={INPUT_THERMAL:,.0f}, Elektrisch={INPUT_ELECTRICAL:,.0f}, Gesamt={INPUT_TOTAL:,.0f}")
    
    # WEISE THERMISCHE ENERGIE PRO ZONE ZU
    thermal_results = []
    
    for z_key, z_name in ZONE_ENERGY_MAPPING.items():
        thermal_col = f"E_thermal_{z_name}_kWh"
        
        if thermal_col not in e.columns:
            continue
        
        e_zone = e[e[thermal_col] > 0].copy()
        iv_zone = ivals[ivals["Zone"] == z_key].copy()
        
        if e_zone.empty or iv_zone.empty:
            continue
        
        zone_input = e_zone[thermal_col].sum()
        zone_records = []
        
        # Verarbeite in Blöcken von 1000, um Speicherprobleme zu vermeiden
        for i in range(0, len(iv_zone), 1000):
            chunk = iv_zone.iloc[i:i+1000]
            
            e_temp = e_zone.assign(_key=1)
            p_temp = chunk.assign(_key=1)
            merged = e_temp.merge(p_temp, on="_key", suffixes=("_e", "_p")).drop("_key", axis=1)
            
            merged = merged[
                (merged["P_end"] > merged["E_start"]) & 
                (merged["P_start"] < merged["E_end"])
            ]
            
            if merged.empty:
                continue
            
            merged["overlap_start"] = merged[["E_start", "P_start"]].max(axis=1)
            merged["overlap_end"] = merged[["E_end", "P_end"]].min(axis=1)
            merged["Overlap_h"] = (
                (merged["overlap_end"] - merged["overlap_start"])
                .dt.total_seconds() / 3600
            ).clip(0, 1)
            
            merged = merged[merged["Overlap_h"] > 0]
            if merged.empty:
                continue
            
            merged["E_hour_key"] = merged["E_start"].dt.strftime("%Y-%m-%d %H:00")
            hour_totals = merged.groupby("E_hour_key")["Overlap_h"].transform("sum")
            merged["share"] = (merged["Overlap_h"] / hour_totals).clip(0, 1)
            merged["Energy_thermal_kWh"] = merged[thermal_col] * merged["share"]
            merged["Zone"] = z_key
            
            if "Month_e" in merged.columns:
                merged["Month"] = merged["Month_e"]
            else:
                merged["Month"] = merged["E_start"].dt.month
            
            zone_records.append(merged[["Month", "Produkt", "m3", "Overlap_h", "Energy_thermal_kWh", "Zone"]])
        
        if zone_records:
            zone_df = pd.concat(zone_records, ignore_index=True)
            thermal_results.append(zone_df)
    
    if not thermal_results:
        logger.error("Keine thermische Energie zugewiesen!")
        return pd.DataFrame()
    
    final = pd.concat(thermal_results, ignore_index=True)
    thermal_allocated = final["Energy_thermal_kWh"].sum()
    
    # WEISE ELEKTRISCHE ENERGIE PROPORTIONAL ZUM THERMISCHEN ANTEIL ZU
    if thermal_allocated > 0:
        final["thermal_share"] = final["Energy_thermal_kWh"] / thermal_allocated
        final["Energy_electrical_kWh"] = final["thermal_share"] * INPUT_ELECTRICAL
        final = final.drop(columns=["thermal_share"])
    else:
        final["Energy_electrical_kWh"] = 0.0
    
    # Gesamt
    final["Energy_share_kWh"] = final["Energy_thermal_kWh"] + final["Energy_electrical_kWh"]
    final = final.rename(columns={"Overlap_h": "Hour_share"})
    final = final[(final["Energy_thermal_kWh"] >= 0) & (final["m3"] > 0)]
    
    # LOG FINALE
    output_thermal = final["Energy_thermal_kWh"].sum()
    output_electrical = final["Energy_electrical_kWh"].sum()
    output_total = final["Energy_share_kWh"].sum()
    
    logger.info("="*70)
    logger.info("ERGEBNIS DER ZUWEISUNG:")
    logger.info(f"  Thermisch:    {output_thermal:>12,.0f} kWh")
    logger.info(f"  Elektrisch: {output_electrical:>12,.0f} kWh  ← SOLLTE {INPUT_ELECTRICAL:,.0f} SEIN")
    logger.info(f"  Gesamt:      {output_total:>12,.0f} kWh")
    logger.info("="*70)
    
    return final


def calculate_water_per_plate(product: str, pressed_thickness_mm: float = None) -> float:
    """Berechnet die Wassermenge pro Platte für ein gegebenes Produkt."""
    if product not in PRODUCT_SPECIFICATIONS:
        return 0.0
    return PRODUCT_SPECIFICATIONS[product]["water_per_plate_kg"]


def calculate_water_per_m3_formula(product: str) -> float:
    """Berechnet die Wassermenge pro m³ für ein gegebenes Produkt unter Verwendung der Formel."""
    if product not in PRODUCT_SPECIFICATIONS:
        return WATER_PER_M3_KG.get(product, 200.0)
    return PRODUCT_SPECIFICATIONS[product]["water_per_m3_kg"]


def get_product_water_curve(product: str, thickness_range: list = None) -> pd.DataFrame:
    """Ruft die Wassergehaltskurve für ein Produkt über einen Dickenbereich ab."""
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
    """Sagt den Energieverbrauch für eine geplante Produktion voraus."""
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
    """Berechnet Statistiken über die Wagen-Nutzung pro Produkt."""
    logger.info("Berechne Wagen-Statistiken...")
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
    """Haupteinstiegspunkt für Tests."""
    logger.info("Trockner KPI Modul erfolgreich geladen.")
    logger.info(f"Produkte konfiguriert: {list(PRODUCT_SPECIFICATIONS.keys())}")
    logger.info(f"Suspension: {SUSPENSION_KG} kg")
    logger.info(f"Platten pro Wagen: {PLATES_PER_WAGON}")


if __name__ == "__main__":
    main()
