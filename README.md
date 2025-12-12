# Lindner Dryer KPI Dashboard — Handover Manual

Author: Varun Solanki  
Project: Dryer KPI Monitoring & Energy Allocation Framework  
Date: 2025

---

## 1. Overview
This project provides a complete energy KPI monitoring and analysis system for the Lindner ceramic dryers (Trockner A & B). It consists of:

- A Streamlit dashboard (`dryer_kpi_app.py`) for interactive KPI exploration
- A backend calculation engine (`dryer_kpi_monthly_final.py`) for parsing, cleaning, computing and allocating energy

Key KPIs:
- kWh/m³
- kWh/kg evaporated water
- Energy distribution by product, zone and month
- Total thermal + electrical consumption
- Zone residence times
- Product-level water & energy profiles

---

## 2. Folder Structure

Lindner_Dryer_KPI_Handover/
│
├── README.md
├── Dryer_KPI_User_Manual.pdf
│
├── sample_files/
│     ├── energy_sample.xlsx
│     └── wagon_sample.xlsx
│
├── src/
│     ├── dryer_kpi_app.py
│     └── dryer_kpi_monthly_final.py
│
├── requirements.txt
└── RUN_APP.bat

---

## 3. Installation

1. Install Python 3.9 or newer
2. Create virtual environment:
   python -m venv venv
   venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt

---

## 4. Running the Dashboard

Option A — double-click:
    RUN_APP.bat

Option B — manual:
    streamlit run src/dryer_kpi_app.py

The dashboard opens at:
http://localhost:8501

---

## 5. Required Input Files

### Energy File (.xlsx)
Columns:
- Zeitstempel
- Gasmenge, Zone 2–5 [m³]
- Energieverbrauch, elektr. [kWh]

### Wagon File (.xlsx/.xlsm)
Columns (system detects flexible variants):
- Trockner (A/B)
- WG-Nr
- m³ / m3 / Volume
- Pressdat. + Zeit
- EM (thickness)
- Rez. (product type L/N/Y)
- In Z2, In Z3, … (if available)
- Zeit in Z2, … (text durations)

---

## 6. Analysis Workflow

1. Energy parsing → converts gas to thermal kWh & defines hourly windows
2. Wagon parsing → cleans columns, detects product codes, computes zone durations
3. Overlap calculation → determines the time window usable for both datasets
4. Interval explosion → splits each wagon stay into Z1–Z5 intervals
5. Energy Allocation:
   - thermal energy allocated by overlap share
   - electrical energy allocated proportional to thermal
6. KPI computation:
   - kWh/m³, kWh/kg
   - Water evaporation
   - Product and zone summaries
7. Export → download consolidated Excel report

---

## 7. Troubleshooting

| Problem | Likely cause | Fix |
|--------|--------------|-----|
| No wagons appear | Trockner column variant not detected | Check "Raw Wagon File Analysis" in UI |
| Zero allocated energy | No overlap between timestamps | Verify Date/Time in both files |
| kWh/kg unrealistic | Volume missing or invalid | Check volume column detection |
| Product = Unknown | EM or Rez. columns not found | Ensure thickness & recipe columns exist |

---

## 8. Developer Notes

Backend functions:
- parse_energy
- parse_wagon
- explode_intervals
- allocate_energy
- add_water_kpis
- compute_zone_duration_stats
- predict_production_energy

Configurable parameters:
- PRODUCT_SPECIFICATIONS
- gas_to_kwh
- zones_seq

---

## 9. Contact

For internal questions, please refer to the Engineering / Digitalization team.

