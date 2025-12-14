

---

# Lindner Dryer KPI Dashboard

Energy Allocation & KPI Monitoring Framework

---

## Overview

This repository contains a complete KPI monitoring and energy allocation system for Lindner ceramic dryers (Trockner A and B).

The system combines **hourly energy measurements** with **wagon tracking data** to calculate transparent, auditable KPIs such as:

* kWh per m³ of product
* kWh per kg of evaporated water
* Energy distribution by product, zone, month, and week
* Thermal vs. electrical energy split
* Zone residence times
* Production-based energy forecasts

The solution is designed for **industrial robustness**, **traceability**, and **future extensibility**.

---

## Project Structure

```
.
├── dryer_kpi_app.py
├── dryer_kpi_monthly_final.py
├── README.md
```

### dryer_kpi_monthly_final.py

Backend module containing all business logic and calculations.

Responsibilities:

* Parsing and cleaning energy data
* Parsing and validating wagon tracking data
* Dryer (Trockner) filtering
* Product derivation (EM + Rez.)
* Zone duration calculation
* Time-overlap based energy allocation
* KPI computation
* Forecasting utilities

This module is independent of the user interface and can be tested standalone.

---

### dryer_kpi_app.py

Streamlit-based frontend application.

Responsibilities:

* File upload (energy and wagon Excel files)
* Filter selection (dryer, product, month, date range)
* Visualization of KPIs and trends
* Debug and validation views
* Excel export of results

The app contains **no calculation logic** and relies entirely on backend functions.

---

## Data Inputs

### Energy File (Excel)

Required content:

* Hourly timestamps
* Gas consumption per zone (Z2–Z5)
* Electrical energy consumption

Key assumptions:

* Gas is converted to thermal energy using a fixed factor (default: 11.5 kWh/m³)
* Electrical energy is treated as a global value and allocated proportionally

---

### Wagon Tracking File (Excel)

Required content:

* Wagon number
* Press timestamp
* Zone entry times (Z2–Z5)
* Product thickness (EM)
* Product type (Rez.)
* Volume per wagon (m³)
* Dryer identifier (A or B)

The parser is robust against:

* Column name variations
* Empty cells
* Forward-filled product type rows

---

## Core Concepts

### Dryer Separation (Trockner A / B)

Dryer filtering is applied **before any other processing step**.
This guarantees that energy and wagon data from different dryers are never mixed.

---

### Overlapping Energy Allocation

Energy is measured hourly, while wagon residence times are continuous.

The system:

* Builds zone-level time intervals per wagon
* Computes exact time overlaps with energy hours
* Allocates energy proportionally based on overlap duration

This guarantees:

* Energy conservation
* No double counting
* Physically correct attribution

---

### Electrical Energy Handling

Electrical energy is:

* Not zone-specific
* Optionally taken from the full, unfiltered dataset
* Distributed proportionally to thermal energy shares

This avoids distortion when applying time or product filters.

---

## KPIs Provided

* Total thermal energy (kWh)
* Total electrical energy (kWh)
* Total energy (kWh)
* kWh per m³
* kWh per kg of evaporated water
* Thermal and electrical energy share
* Zone-specific energy consumption
* Zone residence times
* Product-level performance
* Weekly energy forecasts

---

## Running the Application

### Requirements

* Python 3.9+
* pandas
* numpy
* streamlit
* plotly
* openpyxl
* xlsxwriter

Install dependencies:

```bash
pip install -r requirements.txt
```

### Start the App

```bash
streamlit run dryer_kpi_app.py
```

---

## Extending or Modifying the System

Guidelines:

* All calculation logic must remain in `dryer_kpi_monthly_final.py`
* UI changes only belong in `dryer_kpi_app.py`
* Product definitions are centralized in `PRODUCT_SPECIFICATIONS`
* Volume must always be stored in column `m3`
* Energy totals must always remain conserved

A detailed **Developer Guide** is included in the handover documentation.

---

## Validation & Debugging

The application includes:

* Row count tracking after each filter step
* Volume and energy consistency checks
* Before/after comparisons for filters
* Detailed debug views (expandable in UI)

These should remain enabled to ensure traceability.

---

## Intended Audience

* Production and process engineers
* Energy management teams
* Data analysts
* Software developers
* Future maintainers and working students

---

## License / Usage

Internal industrial analytics tool.
Reuse and modification permitted within the organization.

---

## Contact / Handover

This repository is delivered as part of a structured handover package including:

* Functional documentation
* Technical documentation
* Developer guide
* Validation methodology

All logic is fully documented to ensure continuity after handover.

---

If you want, I can also:

* Create a **short GitHub description**
* Add **badges** (Python, Streamlit)
* Write a **CONTRIBUTING.md**
* Prepare a **requirements.txt**

Just tell me.
