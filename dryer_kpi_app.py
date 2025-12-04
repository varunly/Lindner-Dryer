import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os

# ---------------------------------------------------------
# Import KPI engine functions from backend module
# ---------------------------------------------------------
try:
    from dryer_kpi_monthly_final import (
        parse_energy,
        parse_wagon,
        explode_intervals,
        allocate_energy,
        add_water_kpis,
        compute_product_wagon_stats,
        predict_production_energy,
        calculate_water_per_m3_formula,
        get_product_water_curve,
        WATER_PER_M3_KG,
        PRODUCT_SPECIFICATIONS,
        SUSPENSION_KG,
        CONFIG,
        safe_divide,
        PLATES_PER_WAGON,
    )
except ImportError as e:
    st.error(f"❌ Unable to import dryer_kpi_monthly_final module: {e}")
    st.stop()

# ---------------------------------------------------------
# Streamlit page configuration & CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Lindner Dryer KPI Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        color: #003366;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        color: #003366;
        font-size: 22px;
        font-weight: 600;
        margin-top: 40px;
        margin-bottom: 20px;
        border-bottom: 2px solid #003366;
        padding-bottom: 6px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        color: white;
    }
    .metric-card h3 { margin: 0; font-size: 16px; opacity: 0.9; }
    .metric-card h2 { margin: 10px 0 0 0; font-size: 32px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🏭 Lindner – Dryer KPI Monitoring Dashboard</div>',
    unsafe_allow_html=True,
)

st.info(
    f"📊 **Varun Solanki** | "
    f"Suspension: {SUSPENSION_KG} kg | "
    f"Lindner Dryer KPI Calculation"
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("📁 Data Upload")

    energy_file = st.file_uploader(
        "📊 Energy File (.xlsx)",
        type=["xlsx"],
        key="energy_uploader"
    )
    wagon_file = st.file_uploader(
        "🚛 Hordenwagen File (.xlsm, .xlsx)",
        type=["xlsm", "xlsx"],
        key="wagon_uploader"
    )

    st.markdown("---")
    st.subheader("⚙️ Filters")

    products = st.multiselect(
        "🧱 Product(s):",
        ["L28", "L30", "L32", "L34", "L36", "L38", "L40", "L42", "L44", "N40", "N44", "Y44"],
        default=["L28", "L30", "L32", "L34", "L36", "L38", "L40", "L42", "L44", "N40", "N44", "Y44"],
    )

    month = st.number_input(
        "📅 Month (0 = all):",
        min_value=0,
        max_value=12,
        value=0,
    )

    st.markdown("---")
    run_button = st.button("▶️ Run Analysis")


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def create_kpi_card(title: str, value, unit: str) -> str:
    if value is None:
        text, unit_str = "–", ""
    else:
        try:
            if np.isnan(value):
                text, unit_str = "–", ""
            else:
                text, unit_str = f"{value:,.2f}", f" {unit}"
        except (TypeError, ValueError):
            text, unit_str = str(value), f" {unit}"
    return f'<div class="metric-card"><h3>{title}</h3><h2>{text}{unit_str}</h2></div>'


def create_excel_download(results: dict) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for key, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                sheet_name = key.replace("_", " ").title()[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output


def save_uploaded_file(uploaded_file, suffix: str) -> str:
    """Safely save uploaded file to temporary location."""
    try:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        if len(file_bytes) == 0:
            raise ValueError(f"Uploaded file '{uploaded_file.name}' is empty")
        
        if not file_bytes[:4] == b'PK\x03\x04':
            raise ValueError(
                f"File '{uploaded_file.name}' does not appear to be a valid Excel file."
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        
        if os.path.getsize(tmp_path) != len(file_bytes):
            raise ValueError("File was not written correctly to temporary storage")
        
        return tmp_path
        
    except Exception as e:
        raise ValueError(f"Error saving uploaded file: {e}")


def validate_excel_file(file_path: str, file_description: str) -> bool:
    """Validate that an Excel file can be opened."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        wb.close()
        return True
    except Exception as e:
        raise ValueError(f"Cannot read {file_description}: {e}")


def run_analysis(energy_path: str, wagon_path: str, products_filter, month_filter) -> dict:
    progress = st.progress(0)
    status = st.empty()

    try:
        status.text("🔄 Parsing energy data...")
        progress.progress(15)
        
        try:
            e_raw = pd.read_excel(energy_path, sheet_name=CONFIG["energy_sheet"])
        except Exception as e:
            raise ValueError(f"Cannot read energy file: {e}")
        
        e = parse_energy(e_raw)
        if e.empty:
            raise ValueError("Parsed energy data is empty.")

        status.text("🔄 Parsing wagon tracking data...")
        progress.progress(35)
        
        try:
            w_raw = pd.read_excel(
                wagon_path,
                sheet_name=CONFIG["wagon_sheet"],
                header=CONFIG["wagon_header_row"],
            )
        except Exception as e:
            raise ValueError(f"Cannot read wagon file: {e}")
        
        w = parse_wagon(w_raw)
        if w.empty:
            raise ValueError("Parsed wagon data is empty.")

        status.text("🔄 Applying filters...")
        progress.progress(50)
        if products_filter:
            w = w[w["Produkt"].astype(str).isin(products_filter)]
            if w.empty:
                raise ValueError(f"No wagon records found for selected products: {products_filter}")

        total_volume_raw = w["m3"].sum()
        total_wagons = len(w)

        status.text(f"📊 Total volume: {total_volume_raw:,.0f} m³ from {total_wagons:,} wagons")

        if month_filter:
            e = e[e["Month"] == month_filter]
            w = w[w["Month"] == month_filter]
            if e.empty or w.empty:
                raise ValueError(f"No data found for month = {month_filter}.")

        status.text("🔄 Building zone intervals...")
        progress.progress(65)
        ivals = explode_intervals(w)
        if ivals.empty:
            raise ValueError("Zone intervals could not be created.")

        status.text("🔄 Allocating energy to products...")
        progress.progress(80)
        alloc = allocate_energy(e, ivals)
        if alloc.empty:
            raise ValueError("Energy allocation result is empty.")

        status.text("🔄 Aggregating KPIs...")
        progress.progress(90)

        summary = alloc.groupby(["Month", "Produkt", "Zone"], as_index=False).agg(
            Energy_thermal_kWh=("Energy_thermal_kWh", "sum"),
            Energy_electrical_kWh=("Energy_electrical_kWh", "sum"),
            Energy_kWh=("Energy_share_kWh", "sum"),
            Volume_m3=("m3", "sum"),
        )

        summary["kWh_thermal_per_m3"] = np.where(
            summary["Volume_m3"] > 0,
            summary["Energy_thermal_kWh"] / summary["Volume_m3"],
            0
        )
        summary["kWh_per_m3"] = np.where(
            summary["Volume_m3"] > 0,
            summary["Energy_kWh"] / summary["Volume_m3"],
            0
        )
        summary = add_water_kpis(summary)
        summary = summary.fillna(0)

        yearly = summary.groupby(["Produkt", "Zone"], as_index=False).agg(
            Energy_thermal_kWh=("Energy_thermal_kWh", "sum"),
            Energy_electrical_kWh=("Energy_electrical_kWh", "sum"),
            Energy_kWh=("Energy_kWh", "sum"),
            Volume_m3=("Volume_m3", "sum"),
            Water_kg=("Water_kg", "sum"),
        )

        yearly["kWh_thermal_per_m3"] = np.where(
            yearly["Volume_m3"] > 0,
            yearly["Energy_thermal_kWh"] / yearly["Volume_m3"],
            0
        )
        yearly["kWh_per_m3"] = np.where(
            yearly["Volume_m3"] > 0,
            yearly["Energy_kWh"] / yearly["Volume_m3"],
            0
        )
        yearly["kWh_thermal_per_kg"] = np.where(
            yearly["Water_kg"] > 0,
            yearly["Energy_thermal_kWh"] / yearly["Water_kg"],
            0
        )
        yearly["kWh_per_kg"] = np.where(
            yearly["Water_kg"] > 0,
            yearly["Energy_kWh"] / yearly["Water_kg"],
            0
        )
        yearly = yearly.fillna(0)

        product_totals = summary.groupby(["Month", "Produkt"], as_index=False).agg(
            Energy_thermal_kWh=("Energy_thermal_kWh", "sum"),
            Energy_electrical_kWh=("Energy_electrical_kWh", "sum"),
            Energy_kWh=("Energy_kWh", "sum"),
            Volume_m3=("Volume_m3", "sum"),
            Water_kg=("Water_kg", "sum"),
        )

        product_totals["kWh_thermal_per_m3"] = np.where(
            product_totals["Volume_m3"] > 0,
            product_totals["Energy_thermal_kWh"] / product_totals["Volume_m3"],
            0
        )
        product_totals["kWh_per_m3"] = np.where(
            product_totals["Volume_m3"] > 0,
            product_totals["Energy_kWh"] / product_totals["Volume_m3"],
            0
        )
        product_totals["kWh_thermal_per_kg"] = np.where(
            product_totals["Water_kg"] > 0,
            product_totals["Energy_thermal_kWh"] / product_totals["Water_kg"],
            0
        )
        product_totals["kWh_per_kg"] = np.where(
            product_totals["Water_kg"] > 0,
            product_totals["Energy_kWh"] / product_totals["Water_kg"],
            0
        )
        product_totals = product_totals.fillna(0)

        progress.progress(100)
        status.text("✅ Analysis complete!")

        return {
            "energy": e,
            "wagons": w,
            "intervals": ivals,
            "allocation": alloc,
            "summary": summary,
            "yearly": yearly,
            "product_totals": product_totals,
        }

    finally:
        import time
        time.sleep(0.4)
        progress.empty()
        status.empty()


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if energy_file and wagon_file:
    current_files = (energy_file.name, wagon_file.name)
    if "last_files" not in st.session_state:
        st.session_state.last_files = current_files
    elif st.session_state.last_files != current_files:
        st.session_state.results = None
        st.session_state.analysis_complete = False
        st.session_state.last_files = current_files

# ---------------------------------------------------------
# Run Button
# ---------------------------------------------------------
if run_button:
    if not energy_file or not wagon_file:
        st.error("⚠️ Please upload both Energy and Hordenwagen files.")
    else:
        tmp_e = tmp_w = None
        try:
            st.info(f"📄 Energy file: {energy_file.name} ({energy_file.size:,} bytes)")
            st.info(f"📄 Wagon file: {wagon_file.name} ({wagon_file.size:,} bytes)")
            
            with st.spinner("📁 Saving uploaded files..."):
                tmp_e = save_uploaded_file(energy_file, ".xlsx")
                wagon_suffix = ".xlsm" if wagon_file.name.endswith(".xlsm") else ".xlsx"
                tmp_w = save_uploaded_file(wagon_file, wagon_suffix)
            
            with st.spinner("🔍 Validating Excel files..."):
                validate_excel_file(tmp_e, "Energy file")
                validate_excel_file(tmp_w, "Wagon file")
            
            results = run_analysis(
                tmp_e,
                tmp_w,
                products if products else None,
                month if month != 0 else None
            )
            st.session_state.results = results
            st.session_state.analysis_complete = True

        except ValueError as ve:
            st.error(f"⚠️ Validation Error: {ve}")
            st.warning(
                "💡 **Troubleshooting Tips:**\n"
                "1. Try re-downloading the original Excel file\n"
                "2. Open and re-save the file in Excel\n"
                "3. Ensure the file is not password protected\n"
                "4. Check if the file opens correctly on your computer"
            )
            st.session_state.results = None
            st.session_state.analysis_complete = False
            
        except Exception as err:
            st.error(f"❌ Error during analysis: {err}")
            with st.expander("🔍 View Error Details"):
                st.exception(err)
            st.session_state.results = None
            st.session_state.analysis_complete = False

        finally:
            for p in (tmp_e, tmp_w):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------
if st.session_state.analysis_complete and st.session_state.results:
    try:
        results = st.session_state.results
        summary = results["summary"]
        yearly = results["yearly"]
        product_totals = results.get("product_totals")

        if summary.empty:
            st.warning("⚠️ No data available after filtering.")
        else:
            # ===== ENERGY TOTALS =====
            total_thermal = float(yearly["Energy_thermal_kWh"].sum())
            total_electrical = float(yearly["Energy_electrical_kWh"].sum())
            total_energy = float(yearly["Energy_kWh"].sum())

            # ===== CORRECT VOLUME: FROM WAGONS =====
            total_volume = float(results["wagons"]["m3"].sum())

            # ===== WATER CALCULATION - FIXED TO AVOID ZONE MULTIPLICATION =====
            product_volume_unique = results["wagons"].groupby("Produkt")["m3"].sum().reset_index()
            
            water_calc_details = []
            total_water = 0.0
            
            for _, row in product_volume_unique.iterrows():
                prod = row["Produkt"]
                vol = row["m3"]
                
                if prod in PRODUCT_SPECIFICATIONS:
                    spec = PRODUCT_SPECIFICATIONS[prod]
                    water_per_mm_g = spec["slope"] * SUSPENSION_KG + spec["intercept"]
                    water_per_plate_kg = (water_per_mm_g * spec["pressed_thickness_mm"]) / 1000.0
                    water_per_m3 = water_per_plate_kg / spec["volume_m3"]
                    water_kg = vol * water_per_m3
                else:
                    water_per_m3 = WATER_PER_M3_KG.get(prod, 180.0)
                    water_kg = vol * water_per_m3
                    water_per_mm_g = 0
                    water_per_plate_kg = 0
                
                total_water += water_kg
                water_calc_details.append({
                    "Product": prod,
                    "Volume (m³)": round(vol, 2),
                    "Water/m³ (kg)": round(water_per_m3, 1),
                    "Total Water (kg)": round(water_kg, 0),
                    "Water/mm (g)": round(water_per_mm_g, 1),
                    "Water/Plate (kg)": round(water_per_plate_kg, 3),
                })

            # ===== KPI CALCULATIONS =====
            avg_kwh_per_m3 = safe_divide(total_energy, total_volume)
            avg_kwh_thermal_per_m3 = safe_divide(total_thermal, total_volume)
            avg_kwh_per_kg = safe_divide(total_energy, total_water)
            avg_kwh_thermal_per_kg = safe_divide(total_thermal, total_water)

            thermal_pct = (total_thermal / total_energy * 100) if total_energy > 0 else 0
            electrical_pct = (total_electrical / total_energy * 100) if total_energy > 0 else 0

            total_wagons = len(results["wagons"])
            avg_water_per_m3 = safe_divide(total_water, total_volume)

            # ============================================================
            #                     SUMMARY KPIs SECTION
            # ============================================================
            st.markdown('<div class="section-header">📈 Summary KPIs</div>', unsafe_allow_html=True)

            # --- ENERGY CARDS ---
            st.subheader("⚡ Energy Consumption")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(create_kpi_card("Thermal Energy", total_thermal, "kWh"), unsafe_allow_html=True)
            with c2:
                st.markdown(create_kpi_card("Electrical Energy", total_electrical, "kWh"), unsafe_allow_html=True)
            with c3:
                st.markdown(create_kpi_card("Total Energy", total_energy, "kWh"), unsafe_allow_html=True)

            # --- PRODUCTION CARDS ---
            st.subheader("🏭 Production")
            c4, c5, c6 = st.columns(3)

            with c4:
                st.markdown(create_kpi_card("Total Volume", total_volume, "m³"), unsafe_allow_html=True)
            with c5:
                st.markdown(create_kpi_card("Water Evaporated", total_water, "kg"), unsafe_allow_html=True)
            with c6:
                st.markdown(create_kpi_card("Water/m³", avg_water_per_m3, "kg/m³"), unsafe_allow_html=True)

            # --- EFFICIENCY CARDS ---
            st.subheader("📊 Energy Efficiency")
            c7, c8, c9, c10 = st.columns(4)

            with c7:
                st.markdown(create_kpi_card("kWh/kg water", avg_kwh_per_kg, "kWh/kg"), unsafe_allow_html=True)
            with c8:
                st.markdown(create_kpi_card("Thermal kWh/kg", avg_kwh_thermal_per_kg, "kWh/kg"), unsafe_allow_html=True)
            with c9:
                st.markdown(create_kpi_card("kWh/m³", avg_kwh_per_m3, "kWh/m³"), unsafe_allow_html=True)
            with c10:
                st.markdown(create_kpi_card("Thermal kWh/m³", avg_kwh_thermal_per_m3, "kWh/m³"), unsafe_allow_html=True)

            # ===== INFO BOX =====
            st.info(
                f"⚡ **Energy Mix:** Thermal = **{thermal_pct:.1f}%** ({total_thermal:,.0f} kWh) | "
                f"Electrical = **{electrical_pct:.1f}%** ({total_electrical:,.0f} kWh) | "
                f"🚛 **Production:** {total_wagons:,} wagons | {total_volume:,.0f} m³ | "
                f"💧 **Water:** {total_water:,.0f} kg ({total_water/1000:,.1f} tons) evaporated"
            )
    # Add this section right after the INFO BOX and before the kWh/kg expander
# Around line 550 in your code

# ===== INFO BOX =====
            st.info(
                f"⚡ **Energiemix:** Thermisch = **{thermal_pct:.1f}%** ({total_thermal:,.0f} kWh) | "
                f"Elektrisch = **{electrical_pct:.1f}%** ({total_electrical:,.0f} kWh) | "
                f"🚛 **Produktion:** {total_wagons:,} Wagen | {total_volume:,.0f} m³ | "
                f"💧 **Wasser:** {total_water:,.0f} kg ({total_water/1000:,.1f} Tonnen) verdampft"
            )
            
            # ============================================================
            #    ENERGY CALCULATION EXPLANATION
            # ============================================================
            with st.expander("⚡ Energieverbrauch Berechnung - Detaillierte Erklärung"):
                st.markdown("### 📊 Wie wird der Energieverbrauch berechnet?")
                
                st.markdown("""
                **Überblick:**  
                Der Gesamtenergieverbrauch wird aus der stündlichen Energie-Datei gelesen, 
                den Produkten und Zonen zugeordnet und dann aggregiert.
                """)
                
                # ===== STEP 1: INPUT ENERGY =====
                st.markdown("---")
                st.markdown("### 📥 Schritt 1: Energie-Eingangsdaten")
                
                # Calculate input energy totals
                if "energy" in results and not results["energy"].empty:
                    energy_df = results["energy"]
                    
                    # Input totals from original energy file
                    input_thermal_total = energy_df["E_thermal_total_kWh"].sum()
                    input_electrical_total = energy_df["E_el_kWh"].sum()
                    input_total_energy = input_thermal_total + input_electrical_total
                    
                    total_hours = len(energy_df)
                    avg_thermal_per_hour = input_thermal_total / total_hours if total_hours > 0 else 0
                    avg_electrical_per_hour = input_electrical_total / total_hours if total_hours > 0 else 0
                    
                    col_e1, col_e2, col_e3 = st.columns(3)
                    
                    with col_e1:
                        st.metric("Eingangsdaten: Thermische Energie", f"{input_thermal_total:,.0f} kWh")
                        st.caption(f"Durchschnitt: {avg_thermal_per_hour:.1f} kWh/Stunde")
                    
                    with col_e2:
                        st.metric("Eingangsdaten: Elektrische Energie", f"{input_electrical_total:,.0f} kWh")
                        st.caption(f"Durchschnitt: {avg_electrical_per_hour:.1f} kWh/Stunde")
                    
                    with col_e3:
                        st.metric("Eingangsdaten: Gesamtenergie", f"{input_total_energy:,.0f} kWh")
                        st.caption(f"Über {total_hours:,} Stunden")
                    
                    st.markdown("**Quelle:** Energie-Excel-Datei (stündliche Messungen)")
                    
                    st.code(f"""
            Thermische Energie = Gas-Verbrauch (m³) × 11,5 kWh/m³
            
            Zonen: Z2, Z3, Z4, Z5
            Thermische Gesamtenergie = Z2 + Z3 + Z4 + Z5
            
            Beispiel für eine Stunde:
            - Gas Z2: 15,2 m³ × 11,5 = 174,8 kWh
            - Gas Z3: 20,3 m³ × 11,5 = 233,4 kWh
            - Gas Z4: 17,3 m³ × 11,5 = 198,9 kWh
            - Gas Z5: 13,6 m³ × 11,5 = 156,4 kWh
            - Thermisch gesamt: 763,5 kWh
            - Elektrisch: 45,0 kWh (direkt gemessen)
            - Stunde gesamt: 808,5 kWh
                    """, language="text")
                    
                    # Show sample of energy data
                    st.markdown("**Beispieldaten aus Energie-Datei:**")
                    sample_energy = energy_df.head(10).copy()
                    
                    # Select relevant columns for display
                    display_cols = ["E_start", "E_thermal_total_kWh", "E_el_kWh", "Month"]
                    if "E_thermal_Z2_kWh" in sample_energy.columns:
                        display_cols = ["E_start", "E_thermal_Z2_kWh", "E_thermal_Z3_kWh", 
                                      "E_thermal_Z4_kWh", "E_thermal_Z5_kWh", "E_thermal_total_kWh", 
                                      "E_el_kWh", "Month"]
                    
                    available_cols = [col for col in display_cols if col in sample_energy.columns]
                    st.dataframe(sample_energy[available_cols], use_container_width=True, hide_index=True)
                
                # ===== STEP 2: ALLOCATION =====
                st.markdown("---")
                st.markdown("### 🔄 Schritt 2: Energiezuordnung zu Produkten & Zonen")
                
                st.markdown("""
                **Allokationsmethode:**  
                Für jede Stunde wird die Energie auf alle Wagen verteilt, die zu diesem Zeitpunkt im Trockner waren.
                """)
                
                if "allocation" in results and not results["allocation"].empty:
                    allocation_df = results["allocation"]
                    
                    # Calculate allocation statistics
                    allocated_thermal = allocation_df["Energy_thermal_kWh"].sum()
                    allocated_electrical = allocation_df["Energy_electrical_kWh"].sum()
                    allocated_total = allocation_df["Energy_share_kWh"].sum()
                    
                    num_allocation_rows = len(allocation_df)
                    unique_wagons_allocated = allocation_df["WG_Nr"].nunique() if "WG_Nr" in allocation_df.columns else "N/A"
                    
                    st.code(f"""
            Beispiel: Stunde 01.01.2024 10:00-11:00
            
            Energie verfügbar:
            - Thermisch: 850 kWh
            - Elektrisch: 45 kWh
            - Gesamt: 895 kWh
            
            Wagen im Trockner während dieser Stunde:
            - Wagen 1234 (L36) in Z2: 08:00-10:30 → überlappt 30 min (10:00-10:30)
            - Wagen 1235 (L38) in Z3: 09:00-12:00 → überlappt 60 min (10:00-11:00)
            - Wagen 1236 (L36) in Z4: 10:30-14:00 → überlappt 30 min (10:30-11:00)
            - Wagen 1237 (L42) in Z5: 09:30-11:30 → überlappt 60 min (10:00-11:00)
            
            Gesamte Überlappungszeit: 30 + 60 + 30 + 60 = 180 Minuten
            
            Anteil berechnen:
            - Wagen 1234: 30/180 = 16,67%  → 850 × 0,1667 = 141,7 kWh thermisch
            - Wagen 1235: 60/180 = 33,33%  → 850 × 0,3333 = 283,3 kWh thermisch
            - Wagen 1236: 30/180 = 16,67%  → 850 × 0,1667 = 141,7 kWh thermisch
            - Wagen 1237: 60/180 = 33,33%  → 850 × 0,3333 = 283,3 kWh thermisch
            
            Summe: 141,7 + 283,3 + 141,7 + 283,3 = 850,0 kWh ✅
            
            Dieser Prozess wird für jede Stunde wiederholt.
                    """, language="text")
                    
                    col_a1, col_a2 = st.columns(2)
                    
                    with col_a1:
                        st.metric("Zugeordnete Zeilen", f"{num_allocation_rows:,}")
                        st.caption("Anzahl Wagen×Zone×Stunde Zuordnungen")
                    
                    with col_a2:
                        st.metric("Zugeordnete Wagen", f"{unique_wagons_allocated}")
                        st.caption("Eindeutige Wagen mit Energiezuordnung")
                    
                    # Show sample allocation
                    st.markdown("**Beispiel Zuordnungsdaten:**")
                    sample_alloc = allocation_df.head(10).copy()
                    
                    alloc_display_cols = ["Produkt", "Zone", "Energy_thermal_kWh", "Energy_electrical_kWh", 
                                          "Energy_share_kWh", "m3"]
                    if "E_start" in sample_alloc.columns:
                        alloc_display_cols.insert(0, "E_start")
                    
                    available_alloc_cols = [col for col in alloc_display_cols if col in sample_alloc.columns]
                    st.dataframe(sample_alloc[available_alloc_cols].round(2), use_container_width=True, hide_index=True)
                
                # ===== STEP 3: AGGREGATION =====
                st.markdown("---")
                st.markdown("### 📊 Schritt 3: Aggregation nach Produkt & Zone")
                
                st.markdown("""
                **Aggregationsstufen:**  
                1. Monatlich nach Produkt & Zone (`summary`)  
                2. Jährlich nach Produkt & Zone (`yearly`)  
                3. Gesamtsumme über alle Produkte & Zonen (Summary KPIs)
                """)
                
                # Show aggregation by product and zone
                if "yearly" in results and not results["yearly"].empty:
                    yearly_display = yearly.groupby("Produkt", as_index=False).agg({
                        "Energy_thermal_kWh": "sum",
                        "Energy_electrical_kWh": "sum",
                        "Energy_kWh": "sum",
                    })
                    
                    yearly_display = yearly_display.rename(columns={
                        "Produkt": "Produkt",
                        "Energy_thermal_kWh": "Thermisch (kWh)",
                        "Energy_electrical_kWh": "Elektrisch (kWh)",
                        "Energy_kWh": "Gesamt (kWh)"
                    })
                    
                    yearly_display = yearly_display.sort_values("Gesamt (kWh)", ascending=False)
                    
                    st.markdown("**Energieverbrauch nach Produkt (alle Zonen summiert):**")
                    st.dataframe(yearly_display.round(0), use_container_width=True, hide_index=True)
                    
                    # Zone breakdown for one product example
                    if len(yearly["Produkt"].unique()) > 0:
                        example_product = yearly["Produkt"].unique()[0]
                        zone_breakdown = yearly[yearly["Produkt"] == example_product].copy()
                        zone_breakdown = zone_breakdown[["Zone", "Energy_thermal_kWh", "Energy_electrical_kWh", "Energy_kWh"]]
                        zone_breakdown = zone_breakdown.rename(columns={
                            "Zone": "Zone",
                            "Energy_thermal_kWh": "Thermisch (kWh)",
                            "Energy_electrical_kWh": "Elektrisch (kWh)",
                            "Energy_kWh": "Gesamt (kWh)"
                        })
                        
                        st.markdown(f"**Beispiel: Zonenverteilung für Produkt {example_product}:**")
                        st.dataframe(zone_breakdown.round(0), use_container_width=True, hide_index=True)
                        
                        total_product = zone_breakdown["Gesamt (kWh)"].sum()
                        st.caption(f"Summe über alle Zonen: {total_product:,.0f} kWh")
                
                st.code(f"""
            Aggregation:
            
            1. summary (Monat + Produkt + Zone):
               - Gruppiert: ["Month", "Produkt", "Zone"]
               - Summe: Energy_thermal_kWh, Energy_electrical_kWh, Energy_kWh
            
            2. yearly (Produkt + Zone):
               - Gruppiert: ["Produkt", "Zone"]
               - Summe über alle Monate
            
            3. Summary KPIs (Gesamt):
               - Summe: yearly["Energy_thermal_kWh"].sum()
               - Ergebnis: {total_thermal:,.0f} kWh thermisch
                         {total_electrical:,.0f} kWh elektrisch
                         {total_energy:,.0f} kWh gesamt
                """, language="text")
                
                # ===== STEP 4: VALIDATION =====
                #
                
                # ===== STEP 5: SUMMARY CALCULATION =====
                st.markdown("---")
                st.markdown("### 📈 Schritt 5: Summary KPI Berechnung")
                
                st.markdown("""
                Die in den **Summary KPI-Karten** angezeigten Werte stammen aus:
                """)
                
                st.code(f"""
            Berechnung der Summary KPIs:
            
            # Schritt 1: Summe aus yearly DataFrame
            total_thermal = yearly["Energy_thermal_kWh"].sum()
                          = {total_thermal:,.0f} kWh
            
            total_electrical = yearly["Energy_electrical_kWh"].sum()
                             = {total_electrical:,.0f} kWh
            
            total_energy = yearly["Energy_kWh"].sum()
                         = {total_energy:,.0f} kWh
            
            # Schritt 2: Prozentuale Anteile
            thermal_pct = (total_thermal / total_energy) × 100
                        = ({total_thermal:,.0f} / {total_energy:,.0f}) × 100
                        = {thermal_pct:.1f}%
            
            electrical_pct = (total_electrical / total_energy) × 100
                           = ({total_electrical:,.0f} / {total_energy:,.0f}) × 100
                           = {electrical_pct:.1f}%
            
            # Schritt 3: Anzeige in KPI-Karten
            ✅ Diese Werte werden in den violetten Karten oben angezeigt
                """, language="text")
                
                # Create summary comparison table
                summary_comparison = pd.DataFrame({
                    "Metrik": ["Thermische Energie", "Elektrische Energie", "Gesamtenergie"],
                    "Wert (kWh)": [total_thermal, total_electrical, total_energy],
                    "Anteil (%)": [thermal_pct, electrical_pct, 100.0],
                    "Quelle": [
                        "yearly → Energy_thermal_kWh → sum()",
                        "yearly → Energy_electrical_kWh → sum()",
                        "yearly → Energy_kWh → sum()"
                    ]
                })
                
                st.markdown("**Summary KPI Quelltabelle:**")
                st.dataframe(summary_comparison.round(0), use_container_width=True, hide_index=True)
                
                st.success("✅ Alle Summary KPIs für Energieverbrauch sind vollständig dokumentiert und validiert.")
            
            # Rest of the code continues...
            # (kWh/kg expander, volume breakdown, etc.)        
            # ============================================================
            #    DETAILED kWh/kg CALCULATION TRANSPARENCY
            # ============================================================
            with st.expander("🔍 kWh/kg Water Calculation - Detailed Breakdown"):
                st.markdown("### 📊 How kWh/kg is Calculated")
                
                st.markdown("""
                **Formula:** `kWh/kg = Total Energy (kWh) ÷ Total Water Evaporated (kg)`
                
                **Important Notes:**
                - Water is calculated **per product** based on the formula-derived water content
                - Volume comes from wagon tracking data (unique per wagon, NOT multiplied by zones)
                - Energy is the total allocated energy from all zones
                """)
                
                st.markdown("---")
                st.markdown("### 📦 Step 1: Volume from Wagons")
                st.code(f"""
Total Wagons: {total_wagons:,}
Total Volume: {total_volume:,.2f} m³
Average Volume/Wagon: {total_volume/total_wagons:.3f} m³
                """, language="text")
                
                st.markdown("### 💧 Step 2: Water Calculation per Product")
                water_df = pd.DataFrame(water_calc_details)
                st.dataframe(water_df, use_container_width=True, hide_index=True)
                
                st.code(f"""
Total Water = Σ(Volume × Water/m³) for each product
            = {total_water:,.0f} kg
            = {total_water/1000:,.1f} tons
                """, language="text")
                
                st.markdown("### ⚡ Step 3: Energy Totals")
                st.code(f"""
Thermal Energy:    {total_thermal:,.0f} kWh
Electrical Energy: {total_electrical:,.0f} kWh
Total Energy:      {total_energy:,.0f} kWh
                """, language="text")
                
                st.markdown("### 📈 Step 4: Final KPI Calculation")
                st.code(f"""
kWh/kg (Total) = Total Energy / Total Water
               = {total_energy:,.0f} / {total_water:,.0f}
               = {avg_kwh_per_kg:.4f} kWh/kg

kWh/kg (Thermal) = Thermal Energy / Total Water
                 = {total_thermal:,.0f} / {total_water:,.0f}
                 = {avg_kwh_thermal_per_kg:.4f} kWh/kg
                """, language="text")
                
                st.markdown("### 🔗 Cross-Check: kWh/m³ vs kWh/kg")
                calculated_kwh_m3 = avg_kwh_per_kg * avg_water_per_m3
                st.code(f"""
kWh/m³ should ≈ kWh/kg × Water/m³
Calculated: {avg_kwh_per_kg:.4f} × {avg_water_per_m3:.1f} = {calculated_kwh_m3:.1f} kWh/m³
Actual:     {avg_kwh_per_m3:.1f} kWh/m³
Difference: {abs(calculated_kwh_m3 - avg_kwh_per_m3):.1f} kWh/m³
                """, language="text")
                
            
            # ===== VOLUME BREAKDOWN & VALIDATION =====
            with st.expander("📊 Volume Breakdown & Validation"):
                st.subheader("Volume Statistics")

                col_v1, col_v2, col_v3, col_v4 = st.columns(4)

                with col_v1:
                    st.metric("Total Wagons", f"{total_wagons:,}")
                with col_v2:
                    st.metric("Total Volume", f"{total_volume:,.0f} m³")
                with col_v3:
                    avg_volume_wagon = results["wagons"]["m3"].mean()
                    st.metric("Avg Volume/Wagon", f"{avg_volume_wagon:.2f} m³")
                with col_v4:
                    in_range = 12000 <= total_volume <= 15000
                    st.metric("Status", "✅ In Range" if in_range else "⚠️ Check Range")

                st.subheader("Volume by Product")
                vol_by_product = results["wagons"].groupby("Produkt").agg({
                    "m3": ["sum", "mean", "count"]
                }).round(2)
                vol_by_product.columns = ["Total (m³)", "Avg/Wagon (m³)", "Wagon Count"]
                vol_by_product = vol_by_product.sort_values("Total (m³)", ascending=False)
                vol_by_product["% of Total"] = (
                    vol_by_product["Total (m³)"] / total_volume * 100
                ).round(1)
                st.dataframe(vol_by_product, use_container_width=True)

                st.subheader("Volume by Month")
                vol_by_month = results["wagons"].groupby("Month").agg({
                    "m3": "sum",
                    "WG_Nr": "count"
                }).round(0)
                vol_by_month.columns = ["Volume (m³)", "Wagons"]
                vol_by_month["Avg m³/Wagon"] = (
                    vol_by_month["Volume (m³)"] / vol_by_month["Wagons"]
                ).round(2)
                st.dataframe(vol_by_month, use_container_width=True)

            # ===== 2. ZONE COMPARISON =====
            st.markdown('<div class="section-header">📉 Zone Comparison</div>', unsafe_allow_html=True)

            zone_totals = yearly.groupby("Zone", as_index=False).agg({
                "Energy_thermal_kWh": "sum",
                "Energy_electrical_kWh": "sum",
                "Energy_kWh": "sum",
                "Volume_m3": "sum",
                "Water_kg": "sum",
            })
            zone_totals["kWh_per_m3"] = np.where(
                zone_totals["Volume_m3"] > 0,
                zone_totals["Energy_kWh"] / zone_totals["Volume_m3"],
                0
            )

            col_z1, col_z2 = st.columns(2)

            with col_z1:
                fig_zone_energy = go.Figure()
                fig_zone_energy.add_trace(go.Bar(
                    name='Thermal (Gas)',
                    x=zone_totals['Zone'],
                    y=zone_totals['Energy_thermal_kWh'],
                    marker_color='#FF6B6B',
                    text=[f"{v:,.0f}" for v in zone_totals['Energy_thermal_kWh']],
                    textposition='inside'
                ))
                fig_zone_energy.add_trace(go.Bar(
                    name='Electrical',
                    x=zone_totals['Zone'],
                    y=zone_totals['Energy_electrical_kWh'],
                    marker_color='#4ECDC4',
                    text=[f"{v:,.0f}" for v in zone_totals['Energy_electrical_kWh']],
                    textposition='inside'
                ))
                fig_zone_energy.update_layout(
                    title="Energy Consumption by Zone (kWh)",
                    yaxis_title="Energy (kWh)",
                    barmode='stack',
                    height=400,
                    plot_bgcolor="white"
                )
                st.plotly_chart(fig_zone_energy, use_container_width=True)

            with col_z2:
                fig_pie = px.pie(
                    zone_totals,
                    values="Energy_kWh",
                    names="Zone",
                    title="Energy Distribution by Zone (%)",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("Zone Energy Summary")
            zone_display = zone_totals.copy()
            zone_display["Thermal %"] = (
                zone_display["Energy_thermal_kWh"] / zone_display["Energy_kWh"] * 100
            ).round(1)
            zone_display["Electrical %"] = (
                zone_display["Energy_electrical_kWh"] / zone_display["Energy_kWh"] * 100
            ).round(1)
            zone_display = zone_display.rename(columns={
                "Energy_thermal_kWh": "Thermal (kWh)",
                "Energy_electrical_kWh": "Electrical (kWh)",
                "Energy_kWh": "Total (kWh)",
                "Volume_m3": "Volume (m³)",
                "Water_kg": "Water (kg)",
                "kWh_per_m3": "kWh/m³"
            })
            st.dataframe(zone_display, use_container_width=True, hide_index=True)

            # ===== 5. MONTHLY & WEEKLY TRENDS WITH PRODUCT FILTER =====
            st.markdown(
                '<div class="section-header">📊 Monthly & Weekly KPI Trends</div>',
                unsafe_allow_html=True
            )

            # Get available products for filter
            available_products = sorted(summary["Produkt"].unique().tolist())
            
            # Product filter for trends
            st.subheader("🎯 Filter by Product")
            col_filter1, col_filter2 = st.columns([3, 1])
            
            with col_filter1:
                selected_products_trends = st.multiselect(
                    "Select products to display:",
                    options=available_products,
                    default=available_products,
                    key="trends_product_filter"
                )
            
            with col_filter2:
                select_all = st.button("Select All", key="select_all_trends")
                clear_all = st.button("Clear All", key="clear_all_trends")
            
            if select_all:
                selected_products_trends = available_products
            if clear_all:
                selected_products_trends = []

            if not selected_products_trends:
                st.warning("⚠️ Please select at least one product to display trends.")
            else:
                # Filter data based on selected products
                summary_filtered = summary[summary["Produkt"].isin(selected_products_trends)]
                
                # Show filter info
                st.info(f"📊 Showing data for **{len(selected_products_trends)}** product(s): {', '.join(selected_products_trends)}")

                # By Product (filtered)
                monthly_product = summary_filtered.groupby(["Month", "Produkt"], as_index=False).agg({
                    "Energy_thermal_kWh": "sum",
                    "Energy_electrical_kWh": "sum",
                    "Energy_kWh": "sum",
                    "Volume_m3": "sum",
                    "Water_kg": "sum",
                })
                monthly_product["kWh_per_m3"] = safe_divide(
                    monthly_product["Energy_kWh"], monthly_product["Volume_m3"]
                )
                monthly_product["kWh_per_kg"] = safe_divide(
                    monthly_product["Energy_kWh"], monthly_product["Water_kg"]
                )
                monthly_product["kWh_thermal_per_m3"] = safe_divide(
                    monthly_product["Energy_thermal_kWh"], monthly_product["Volume_m3"]
                )

                # By Zone (filtered)
                monthly_zone = summary_filtered.groupby(["Month", "Zone"], as_index=False).agg({
                    "Energy_thermal_kWh": "sum",
                    "Energy_electrical_kWh": "sum",
                    "Energy_kWh": "sum",
                    "Volume_m3": "sum",
                    "Water_kg": "sum",
                })
                monthly_zone["kWh_per_m3"] = safe_divide(
                    monthly_zone["Energy_kWh"], monthly_zone["Volume_m3"]
                )
                monthly_zone["kWh_per_kg"] = safe_divide(
                    monthly_zone["Energy_kWh"], monthly_zone["Water_kg"]
                )

                # Overall (filtered)
                monthly_overall = summary_filtered.groupby(["Month"], as_index=False).agg({
                    "Energy_thermal_kWh": "sum",
                    "Energy_electrical_kWh": "sum",
                    "Energy_kWh": "sum",
                    "Volume_m3": "sum",
                    "Water_kg": "sum",
                })
                monthly_overall["kWh_per_m3"] = safe_divide(
                    monthly_overall["Energy_kWh"], monthly_overall["Volume_m3"]
                )
                monthly_overall["kWh_per_kg"] = safe_divide(
                    monthly_overall["Energy_kWh"], monthly_overall["Water_kg"]
                )

                # ===== WEEKLY DATA PREPARATION (FILTERED) =====
                weekly_energy = None
                if "energy" in results and not results["energy"].empty:
                    energy_df = results["energy"].copy()
                    energy_df["Week"] = energy_df["E_start"].dt.isocalendar().week
                    energy_df["Year"] = energy_df["E_start"].dt.year

                    agg_dict = {}
                    if "E_thermal_total_kWh" in energy_df.columns:
                        agg_dict["E_thermal_total_kWh"] = "sum"
                    else:
                        thermal_cols = [
                            col for col in energy_df.columns
                            if col.startswith("E_thermal_") and col.endswith("_kWh")
                        ]
                        for col in thermal_cols:
                            agg_dict[col] = "sum"

                    if "E_el_kWh" in energy_df.columns:
                        agg_dict["E_el_kWh"] = "sum"

                    if agg_dict:
                        weekly_energy = energy_df.groupby(["Year", "Week"], as_index=False).agg(agg_dict)

                        if "E_thermal_total_kWh" not in weekly_energy.columns:
                            thermal_cols = [
                                col for col in weekly_energy.columns
                                if col.startswith("E_thermal_") and col.endswith("_kWh")
                            ]
                            if thermal_cols:
                                weekly_energy["E_thermal_total_kWh"] = weekly_energy[thermal_cols].sum(axis=1)
                            else:
                                weekly_energy["E_thermal_total_kWh"] = 0

                        weekly_energy["Total_kWh"] = (
                            weekly_energy.get("E_thermal_total_kWh", 0) +
                            weekly_energy.get("E_el_kWh", 0)
                        )
                        weekly_energy["Week_Label"] = (
                            weekly_energy["Year"].astype(str) + "-W" +
                            weekly_energy["Week"].astype(str).str.zfill(2)
                        )
                        
                        # Clip negative values
                        if "E_thermal_total_kWh" in weekly_energy.columns:
                            weekly_energy["E_thermal_total_kWh"] = weekly_energy["E_thermal_total_kWh"].clip(lower=0)
                        if "E_el_kWh" in weekly_energy.columns:
                            weekly_energy["E_el_kWh"] = weekly_energy["E_el_kWh"].clip(lower=0)
                        weekly_energy["Total_kWh"] = weekly_energy["Total_kWh"].clip(lower=0)

                # SECTION 1: OVERALL TRENDS (FILTERED) - INCLUDING WEEKLY
                st.subheader("📈 Overall Performance Trends (Filtered)")
                
                # Summary metrics for filtered data
                filtered_thermal = monthly_overall["Energy_thermal_kWh"].sum()
                filtered_electrical = monthly_overall["Energy_electrical_kWh"].sum()
                filtered_energy = monthly_overall["Energy_kWh"].sum()
                filtered_volume = monthly_overall["Volume_m3"].sum()
                filtered_water = monthly_overall["Water_kg"].sum()
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Total Energy", f"{filtered_energy:,.0f} kWh")
                with col_m2:
                    st.metric("Total Volume", f"{filtered_volume:,.0f} m³")
                with col_m3:
                    st.metric("Total Water", f"{filtered_water:,.0f} kg")
                with col_m4:
                    filtered_kwh_kg = safe_divide(filtered_energy, filtered_water)
                    st.metric("kWh/kg", f"{filtered_kwh_kg:.3f}")

                # Monthly Charts
                col_o1, col_o2 = st.columns(2)

                with col_o1:
                    fig_monthly_energy = go.Figure()
                    fig_monthly_energy.add_trace(go.Bar(
                        name='Thermal',
                        x=monthly_overall["Month"],
                        y=monthly_overall["Energy_thermal_kWh"],
                        marker_color='#FF6B6B',
                        text=[f"{v:,.0f}" for v in monthly_overall["Energy_thermal_kWh"]],
                        textposition='inside'
                    ))
                    fig_monthly_energy.add_trace(go.Bar(
                        name='Electrical',
                        x=monthly_overall["Month"],
                        y=monthly_overall["Energy_electrical_kWh"],
                        marker_color='#4ECDC4',
                        text=[f"{v:,.0f}" for v in monthly_overall["Energy_electrical_kWh"]],
                        textposition='inside'
                    ))
                    fig_monthly_energy.update_layout(
                        title="Monthly Energy Consumption (kWh)",
                        xaxis_title="Month",
                        yaxis_title="Energy (kWh)",
                        barmode='stack',
                        height=350,
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_monthly_energy, use_container_width=True)

                with col_o2:
                    fig_monthly_volume = px.bar(
                        monthly_overall,
                        x="Month",
                        y="Volume_m3",
                        title="Monthly Production Volume (m³)",
                        labels={"Volume_m3": "Volume (m³)", "Month": "Month"},
                        text_auto='.0f'
                    )
                    fig_monthly_volume.update_traces(marker_color='#26de81', textposition='outside')
                    fig_monthly_volume.update_layout(height=350, plot_bgcolor="white", showlegend=False)
                    st.plotly_chart(fig_monthly_volume, use_container_width=True)

                # ===== WEEKLY TRENDS (INSIDE FILTERED SECTION) =====
                st.subheader("📅 Weekly Energy Trends")
                
                if weekly_energy is not None and not weekly_energy.empty:
                    col_w1, col_w2 = st.columns(2)

                    with col_w1:
                        fig_weekly_total = go.Figure()

                        if "E_thermal_total_kWh" in weekly_energy.columns:
                            thermal_data = weekly_energy["E_thermal_total_kWh"].fillna(0).clip(lower=0)
                        else:
                            thermal_data = pd.Series([0] * len(weekly_energy), index=weekly_energy.index)

                        if "E_el_kWh" in weekly_energy.columns:
                            electrical_data = weekly_energy["E_el_kWh"].fillna(0).clip(lower=0)
                        else:
                            electrical_data = pd.Series([0] * len(weekly_energy), index=weekly_energy.index)

                        if thermal_data.sum() > 0:
                            fig_weekly_total.add_trace(go.Bar(
                                name='Thermal',
                                x=weekly_energy["Week_Label"],
                                y=thermal_data,
                                marker_color='#FF6B6B',
                                text=[f"{v:,.0f}" if v > 0 else "" for v in thermal_data],
                                textposition='inside'
                            ))

                        if electrical_data.sum() > 0:
                            fig_weekly_total.add_trace(go.Bar(
                                name='Electrical',
                                x=weekly_energy["Week_Label"],
                                y=electrical_data,
                                marker_color='#4ECDC4',
                                text=[f"{v:,.0f}" if v > 0 else "" for v in electrical_data],
                                textposition='inside'
                            ))

                        fig_weekly_total.update_layout(
                            title="Weekly Energy Consumption (kWh)",
                            xaxis_title="Week",
                            yaxis_title="Energy (kWh)",
                            barmode='stack',
                            height=350,
                            plot_bgcolor="white",
                            xaxis_tickangle=-45,
                            yaxis=dict(rangemode='tozero')
                        )
                        st.plotly_chart(fig_weekly_total, use_container_width=True)

                    with col_w2:
                        total_data = weekly_energy["Total_kWh"].fillna(0).clip(lower=0)

                        fig_weekly_trend = px.line(
                            weekly_energy,
                            x="Week_Label",
                            y=total_data,
                            markers=True,
                            title="Weekly Total Energy Trend (kWh)",
                            labels={"y": "Total Energy (kWh)", "Week_Label": "Week"}
                        )
                        fig_weekly_trend.update_traces(
                            line_color='#667eea',
                            line_width=3,
                            marker=dict(size=8),
                            text=[f"{v:,.0f}" for v in total_data],
                            textposition="top center"
                        )
                        fig_weekly_trend.update_layout(
                            height=350,
                            plot_bgcolor="white",
                            showlegend=False,
                            xaxis_tickangle=-45,
                            yaxis=dict(rangemode='tozero')
                        )
                        st.plotly_chart(fig_weekly_trend, use_container_width=True)

                    # Weekly statistics
                    valid_total = weekly_energy["Total_kWh"].clip(lower=0)
                    avg_weekly = valid_total.mean()
                    max_weekly = valid_total.max()
                    min_weekly = valid_total.min()

                    if not weekly_energy.empty and max_weekly > 0:
                        max_week_idx = valid_total.idxmax()
                        if pd.notna(max_week_idx):
                            max_week = weekly_energy.loc[max_week_idx, "Week_Label"]
                            st.info(
                                f"📊 **Weekly Statistics:** Average = **{avg_weekly:,.0f} kWh/week** | "
                                f"Peak week: **{max_week}** ({max_weekly:,.0f} kWh) | "
                                f"Range: {min_weekly:,.0f} - {max_weekly:,.0f} kWh"
                            )
                else:
                    st.info("📅 Weekly energy data not available for the selected period")

                # Monthly KPI Charts
                col_o3, col_o4 = st.columns(2)
                
                with col_o3:
                    fig_monthly_kwh_m3 = px.line(
                        monthly_overall,
                        x="Month",
                        y="kWh_per_m3",
                        markers=True,
                        title="Monthly kWh/m³ Trend",
                        labels={"kWh_per_m3": "kWh/m³", "Month": "Month"}
                    )
                    fig_monthly_kwh_m3.update_traces(line_color='#667eea', line_width=3)
                    fig_monthly_kwh_m3.update_layout(height=300, plot_bgcolor="white")
                    st.plotly_chart(fig_monthly_kwh_m3, use_container_width=True)

                with col_o4:
                    fig_monthly_kwh_kg = px.line(
                        monthly_overall,
                        x="Month",
                        y="kWh_per_kg",
                        markers=True,
                        title="Monthly kWh/kg Trend",
                        labels={"kWh_per_kg": "kWh/kg", "Month": "Month"}
                    )
                    fig_monthly_kwh_kg.update_traces(line_color='#f093fb', line_width=3)
                    fig_monthly_kwh_kg.update_layout(height=300, plot_bgcolor="white")
                    st.plotly_chart(fig_monthly_kwh_kg, use_container_width=True)

                # ===== TRENDS BY PRODUCT (EXPANDER) =====
                with st.expander("🧱 Trends by Product - Detailed Charts"):
                    st.markdown("### Energy Efficiency by Product")
                    
                    col_p1, col_p2 = st.columns(2)

                    with col_p1:
                        fig_prod_efficiency = px.line(
                            monthly_product,
                            x="Month",
                            y="kWh_per_m3",
                            color="Produkt",
                            markers=True,
                            title="Energy Efficiency by Product (kWh/m³)",
                            labels={"kWh_per_m3": "kWh/m³", "Month": "Month"}
                        )
                        fig_prod_efficiency.update_layout(height=350, plot_bgcolor="white")
                        st.plotly_chart(fig_prod_efficiency, use_container_width=True)

                    with col_p2:
                        fig_prod_specific = px.line(
                            monthly_product,
                            x="Month",
                            y="kWh_per_kg",
                            color="Produkt",
                            markers=True,
                            title="Specific Energy by Product (kWh/kg water)",
                            labels={"kWh_per_kg": "kWh/kg", "Month": "Month"}
                        )
                        fig_prod_specific.update_layout(height=350, plot_bgcolor="white")
                        st.plotly_chart(fig_prod_specific, use_container_width=True)

                    st.markdown("### Thermal Efficiency and Volume by Product")
                    
                    col_p3, col_p4 = st.columns(2)

                    with col_p3:
                        fig_prod_thermal = px.line(
                            monthly_product,
                            x="Month",
                            y="kWh_thermal_per_m3",
                            color="Produkt",
                            markers=True,
                            title="Thermal Efficiency by Product (kWh/m³)",
                            labels={"kWh_thermal_per_m3": "Thermal kWh/m³", "Month": "Month"}
                        )
                        fig_prod_thermal.update_layout(height=350, plot_bgcolor="white")
                        st.plotly_chart(fig_prod_thermal, use_container_width=True)

                    with col_p4:
                        fig_prod_volume = px.line(
                            monthly_product,
                            x="Month",
                            y="Volume_m3",
                            color="Produkt",
                            markers=True,
                            title="Production Volume by Product (m³)",
                            labels={"Volume_m3": "Volume (m³)", "Month": "Month"}
                        )
                        fig_prod_volume.update_layout(height=350, plot_bgcolor="white")
                        st.plotly_chart(fig_prod_volume, use_container_width=True)

                    st.markdown("### Energy and Water by Product")
                    
                    col_p5, col_p6 = st.columns(2)
                    
                    with col_p5:
                        fig_prod_energy = px.line(
                            monthly_product,
                            x="Month",
                            y="Energy_kWh",
                            color="Produkt",
                            markers=True,
                            title="Total Energy by Product (kWh)",
                            labels={"Energy_kWh": "Energy (kWh)", "Month": "Month"}
                        )
                        fig_prod_energy.update_layout(height=350, plot_bgcolor="white")
                        st.plotly_chart(fig_prod_energy, use_container_width=True)

                    with col_p6:
                        fig_prod_water = px.line(
                            monthly_product,
                            x="Month",
                            y="Water_kg",
                            color="Produkt",
                            markers=True,
                            title="Water Evaporated by Product (kg)",
                            labels={"Water_kg": "Water (kg)", "Month": "Month"}
                        )
                        fig_prod_water.update_layout(height=350, plot_bgcolor="white")
                        st.plotly_chart(fig_prod_water, use_container_width=True)

                    # Product comparison table
                    st.markdown("### 📊 Product Comparison Summary")
                    product_comparison = monthly_product.groupby("Produkt", as_index=False).agg({
                        "Energy_thermal_kWh": "sum",
                        "Energy_electrical_kWh": "sum",
                        "Energy_kWh": "sum",
                        "Volume_m3": "sum",
                        "Water_kg": "sum",
                    })
                    product_comparison["kWh/m³"] = safe_divide(
                        product_comparison["Energy_kWh"], product_comparison["Volume_m3"]
                    )
                    product_comparison["kWh/kg"] = safe_divide(
                        product_comparison["Energy_kWh"], product_comparison["Water_kg"]
                    )
                    product_comparison["Water/m³"] = safe_divide(
                        product_comparison["Water_kg"], product_comparison["Volume_m3"]
                    )
                    product_comparison = product_comparison.rename(columns={
                        "Produkt": "Product",
                        "Energy_thermal_kWh": "Thermal (kWh)",
                        "Energy_electrical_kWh": "Electrical (kWh)",
                        "Energy_kWh": "Total Energy (kWh)",
                        "Volume_m3": "Volume (m³)",
                        "Water_kg": "Water (kg)"
                    })
                    st.dataframe(product_comparison.round(2), use_container_width=True, hide_index=True)

                # ===== TRENDS BY ZONE (EXPANDER) =====
                with st.expander("🏭 Trends by Zone - Detailed Charts"):
                    st.markdown("### Zone Performance Over Time")
                    
                    col_z1, col_z2 = st.columns(2)

                    with col_z1:
                        fig_zone_efficiency = px.line(
                            monthly_zone,
                            x="Month",
                            y="kWh_per_m3",
                            color="Zone",
                            markers=True,
                            title="Energy Efficiency by Zone (kWh/m³)",
                            labels={"kWh_per_m3": "kWh/m³", "Month": "Month"}
                        )
                        fig_zone_efficiency.update_layout(height=350, plot_bgcolor="white")
                        st.plotly_chart(fig_zone_efficiency, use_container_width=True)

                    with col_z2:
                        fig_zone_specific = px.line(
                            monthly_zone,
                            x="Month",
                            y="kWh_per_kg",
                            color="Zone",
                            markers=True,
                            title="Specific Energy by Zone (kWh/kg water)",
                            labels={"kWh_per_kg": "kWh/kg", "Month": "Month"}
                        )
                        fig_zone_specific.update_layout(height=350, plot_bgcolor="white")
                        st.plotly_chart(fig_zone_specific, use_container_width=True)

                    col_z3, col_z4 = st.columns(2)

                    with col_z3:
                        fig_zone_thermal = px.line(
                            monthly_zone,
                            x="Month",
                            y="Energy_thermal_kWh",
                            color="Zone",
                            markers=True,
                            title="Thermal Energy by Zone (kWh)",
                            labels={"Energy_thermal_kWh": "Thermal (kWh)", "Month": "Month"}
                        )
                        fig_zone_thermal.update_layout(height=350, plot_bgcolor="white")
                        st.plotly_chart(fig_zone_thermal, use_container_width=True)

                    with col_z4:
                        fig_zone_volume = px.line(
                            monthly_zone,
                            x="Month",
                            y="Volume_m3",
                            color="Zone",
                            markers=True,
                            title="Volume by Zone (m³)",
                            labels={"Volume_m3": "Volume (m³)", "Month": "Month"}
                        )
                        fig_zone_volume.update_layout(height=350, plot_bgcolor="white")
                        st.plotly_chart(fig_zone_volume, use_container_width=True)

            # ===== SINGLE PRODUCT DEEP DIVE =====
            st.markdown(
                '<div class="section-header">🔬 Single Product Deep Dive</div>',
                unsafe_allow_html=True
            )
            
            selected_single_product = st.selectbox(
                "Select a product for detailed analysis:",
                options=available_products,
                key="single_product_select"
            )
            
            if selected_single_product:
                single_product_data = summary[summary["Produkt"] == selected_single_product]
                
                if single_product_data.empty:
                    st.warning(f"No data available for {selected_single_product}")
                else:
                    single_monthly = single_product_data.groupby("Month", as_index=False).agg({
                        "Energy_thermal_kWh": "sum",
                        "Energy_electrical_kWh": "sum",
                        "Energy_kWh": "sum",
                        "Volume_m3": "sum",
                        "Water_kg": "sum",
                    })
                    single_monthly["kWh_per_m3"] = safe_divide(
                        single_monthly["Energy_kWh"], single_monthly["Volume_m3"]
                    )
                    single_monthly["kWh_per_kg"] = safe_divide(
                        single_monthly["Energy_kWh"], single_monthly["Water_kg"]
                    )
                    
                    total_energy_prod = single_monthly["Energy_kWh"].sum()
                    total_volume_prod = single_monthly["Volume_m3"].sum()
                    total_water_prod = single_monthly["Water_kg"].sum()
                    avg_kwh_m3_prod = safe_divide(total_energy_prod, total_volume_prod)
                    avg_kwh_kg_prod = safe_divide(total_energy_prod, total_water_prod)
                    
                    st.subheader(f"📊 {selected_single_product} - Summary")
                    
                    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
                    with col_s1:
                        st.metric("Total Energy", f"{total_energy_prod:,.0f} kWh")
                    with col_s2:
                        st.metric("Total Volume", f"{total_volume_prod:,.0f} m³")
                    with col_s3:
                        st.metric("Total Water", f"{total_water_prod:,.0f} kg")
                    with col_s4:
                        st.metric("kWh/m³", f"{avg_kwh_m3_prod:.1f}")
                    with col_s5:
                        st.metric("kWh/kg", f"{avg_kwh_kg_prod:.3f}")
                    
                    col_sp1, col_sp2 = st.columns(2)
                    
                    with col_sp1:
                        fig_single_energy = go.Figure()
                        fig_single_energy.add_trace(go.Bar(
                            name='Thermal',
                            x=single_monthly["Month"],
                            y=single_monthly["Energy_thermal_kWh"],
                            marker_color='#FF6B6B'
                        ))
                        fig_single_energy.add_trace(go.Bar(
                            name='Electrical',
                            x=single_monthly["Month"],
                            y=single_monthly["Energy_electrical_kWh"],
                            marker_color='#4ECDC4'
                        ))
                        fig_single_energy.update_layout(
                            title=f"{selected_single_product} - Monthly Energy (kWh)",
                            barmode='stack',
                            height=300,
                            plot_bgcolor="white"
                        )
                        st.plotly_chart(fig_single_energy, use_container_width=True)
                    
                    with col_sp2:
                        fig_single_kpi = go.Figure()
                        fig_single_kpi.add_trace(go.Scatter(
                            x=single_monthly["Month"],
                            y=single_monthly["kWh_per_m3"],
                            mode='lines+markers',
                            name='kWh/m³',
                            line=dict(color='#667eea', width=3)
                        ))
                        fig_single_kpi.add_trace(go.Scatter(
                            x=single_monthly["Month"],
                            y=single_monthly["kWh_per_kg"] * 100,
                            mode='lines+markers',
                            name='kWh/kg (×100)',
                            line=dict(color='#f093fb', width=3),
                            yaxis='y2'
                        ))
                        fig_single_kpi.update_layout(
                            title=f"{selected_single_product} - Monthly KPIs",
                            yaxis=dict(title="kWh/m³"),
                            yaxis2=dict(title="kWh/kg (×100)", overlaying='y', side='right'),
                            height=300,
                            plot_bgcolor="white"
                        )
                        st.plotly_chart(fig_single_kpi, use_container_width=True)
                    
                    st.subheader(f"📋 {selected_single_product} - Monthly Details")
                    display_single = single_monthly.copy()
                    display_single = display_single.rename(columns={
                        "Energy_thermal_kWh": "Thermal (kWh)",
                        "Energy_electrical_kWh": "Electrical (kWh)",
                        "Energy_kWh": "Total (kWh)",
                        "Volume_m3": "Volume (m³)",
                        "Water_kg": "Water (kg)",
                        "kWh_per_m3": "kWh/m³",
                        "kWh_per_kg": "kWh/kg"
                    })
                    st.dataframe(display_single.round(2), use_container_width=True, hide_index=True)
                    
                    if selected_single_product in PRODUCT_SPECIFICATIONS:
                        spec = PRODUCT_SPECIFICATIONS[selected_single_product]
                        st.subheader(f"📐 {selected_single_product} - Specifications")
                        
                        col_spec1, col_spec2, col_spec3 = st.columns(3)
                        
                        with col_spec1:
                            st.markdown("**Physical Properties**")
                            st.write(f"- Edge Length: {spec['edge_length_mm']} mm")
                            st.write(f"- Final Thickness: {spec['final_thickness_mm']} mm")
                            st.write(f"- Pressed Thickness: {spec['pressed_thickness_mm']} mm")
                            st.write(f"- Volume/Plate: {spec['volume_m3']:.6f} m³")
                        
                        with col_spec2:
                            st.markdown("**Water Content**")
                            water_per_mm = spec["slope"] * SUSPENSION_KG + spec["intercept"]
                            water_per_plate = (water_per_mm * spec["pressed_thickness_mm"]) / 1000
                            water_per_m3_spec = water_per_plate / spec["volume_m3"]
                            st.write(f"- Formula: {spec['formula']}")
                            st.write(f"- Water/mm: {water_per_mm:.1f} g")
                            st.write(f"- Water/Plate: {water_per_plate:.3f} kg")
                            st.write(f"- Water/m³: {water_per_m3_spec:.1f} kg/m³")
                        
                        with col_spec3:
                            st.markdown("**Formula Parameters**")
                            st.write(f"- Slope: {spec['slope']}")
                            st.write(f"- Intercept: {spec['intercept']}")
                            st.write(f"- Product Type: {spec['product_type']}")

            # ===== 3. PRODUCT PERFORMANCE =====
            if product_totals is not None and not product_totals.empty:
                st.markdown(
                    '<div class="section-header">📊 Product Performance</div>',
                    unsafe_allow_html=True
                )

                prod_agg = product_totals.groupby("Produkt", as_index=False).agg({
                    "Energy_thermal_kWh": "sum",
                    "Energy_electrical_kWh": "sum",
                    "Energy_kWh": "sum",
                    "Volume_m3": "sum",
                    "Water_kg": "sum",
                })

                prod_agg["kWh_per_m3"] = np.where(
                    prod_agg["Volume_m3"] > 0,
                    prod_agg["Energy_kWh"] / prod_agg["Volume_m3"],
                    0
                )
                prod_agg["kWh_per_kg"] = np.where(
                    prod_agg["Water_kg"] > 0,
                    prod_agg["Energy_kWh"] / prod_agg["Water_kg"],
                    0
                )
                prod_agg["Thermal_pct"] = np.where(
                    prod_agg["Energy_kWh"] > 0,
                    (prod_agg["Energy_thermal_kWh"] / prod_agg["Energy_kWh"] * 100),
                    0
                )
                prod_agg = prod_agg.fillna(0)

                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    fig_prod_energy = go.Figure()
                    fig_prod_energy.add_trace(go.Bar(
                        name='Thermal (Gas)',
                        x=prod_agg['Produkt'],
                        y=prod_agg['Energy_thermal_kWh'],
                        marker_color='#FF6B6B',
                        text=[f"{v:,.0f}" for v in prod_agg['Energy_thermal_kWh']],
                        textposition='auto'
                    ))
                    fig_prod_energy.add_trace(go.Bar(
                        name='Electrical',
                        x=prod_agg['Produkt'],
                        y=prod_agg['Energy_electrical_kWh'],
                        marker_color='#4ECDC4',
                        text=[f"{v:,.0f}" for v in prod_agg['Energy_electrical_kWh']],
                        textposition='auto'
                    ))
                    fig_prod_energy.update_layout(
                        title="Total Energy Consumption by Product (kWh)",
                        xaxis_title="Product",
                        yaxis_title="Energy (kWh)",
                        barmode='stack',
                        height=400,
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_prod_energy, use_container_width=True)

                with col_p2:
                    prod_sorted = prod_agg.sort_values("Energy_thermal_kWh", ascending=True)
                    fig_thermal = go.Figure()
                    fig_thermal.add_trace(go.Bar(
                        x=prod_sorted['Energy_thermal_kWh'],
                        y=prod_sorted['Produkt'],
                        orientation='h',
                        marker=dict(
                            color=prod_sorted['Energy_thermal_kWh'],
                            colorscale='Reds',
                            showscale=True,
                            colorbar=dict(title="kWh")
                        ),
                        text=[f"{v:,.0f} kWh" for v in prod_sorted['Energy_thermal_kWh']],
                        textposition='outside'
                    ))
                    fig_thermal.update_layout(
                        title="Thermal Energy Consumption by Product (kWh)",
                        xaxis_title="Thermal Energy (kWh)",
                        yaxis_title="Product",
                        height=400,
                        plot_bgcolor="white",
                        showlegend=False
                    )
                    st.plotly_chart(fig_thermal, use_container_width=True)

                st.subheader("Product Energy Summary")
                prod_display = prod_agg.copy()
                prod_display = prod_display.rename(columns={
                    "Produkt": "Product",
                    "Energy_thermal_kWh": "Thermal (kWh)",
                    "Energy_electrical_kWh": "Electrical (kWh)",
                    "Energy_kWh": "Total (kWh)",
                    "Thermal_pct": "Thermal %",
                    "Volume_m3": "Volume (m³)",
                    "Water_kg": "Water (kg)",
                    "kWh_per_m3": "kWh/m³",
                    "kWh_per_kg": "kWh/kg"
                })
                display_cols = [
                    "Product", "Thermal (kWh)", "Electrical (kWh)", "Total (kWh)",
                    "Thermal %", "Volume (m³)", "Water (kg)", "kWh/m³", "kWh/kg"
                ]
                display_cols = [c for c in display_cols if c in prod_display.columns]
                st.dataframe(prod_display[display_cols].round(2), use_container_width=True, hide_index=True)

            # ===== 4. PRODUCT SPECIFICATIONS =====
            st.markdown(
                '<div class="section-header">📐 Product Specifications</div>',
                unsafe_allow_html=True
            )

            st.write(f"**Formula:** Water/mm (g) = Slope × Suspension ({SUSPENSION_KG} kg) + Intercept")

            specs_data = []
            for prod, spec in PRODUCT_SPECIFICATIONS.items():
                slope = spec["slope"]
                intercept = spec["intercept"]
                water_per_mm_g = slope * SUSPENSION_KG + intercept
                pressed_thickness_mm = spec["pressed_thickness_mm"]
                water_per_plate_kg = (water_per_mm_g * pressed_thickness_mm) / 1000.0
                water_per_m3_kg = water_per_plate_kg / spec["volume_m3"]
                water_per_wagon_kg = water_per_plate_kg * PLATES_PER_WAGON

                is_interpolated = spec.get("interpolated", False)
                formula_display = spec["formula"]
                if is_interpolated:
                    formula_display += " ⚠️"

                specs_data.append({
                    "Product": prod,
                    "Type": spec["product_type"],
                    "Formula": formula_display,
                    "Water/mm (g)": round(water_per_mm_g, 1),
                    "Pressed (mm)": pressed_thickness_mm,
                    "Water/Plate (kg)": round(water_per_plate_kg, 3),
                    "Water/m³ (kg)": round(water_per_m3_kg, 1),
                })

            specs_df = pd.DataFrame(specs_data)
            st.dataframe(specs_df, use_container_width=True, hide_index=True)
            st.info("⚠️ Products marked with ⚠️ are interpolated values")

            # ===== 6. DATA TABLES =====
            with st.expander("📋 View Detailed Data Tables"):
                tab1, tab2, tab3 = st.tabs(["Monthly Summary", "Yearly Summary", "Product Totals"])
                with tab1:
                    st.dataframe(summary, use_container_width=True)
                with tab2:
                    st.dataframe(yearly, use_container_width=True)
                with tab3:
                    if product_totals is not None:
                        st.dataframe(product_totals, use_container_width=True)

            # ===== 7. WEEKLY PREDICTION =====
            st.markdown(
                '<div class="section-header">🔮 Weekly Energy Prediction</div>',
                unsafe_allow_html=True
            )

            wagon_stats = compute_product_wagon_stats(results["wagons"])
            wagon_capacity = wagon_stats.get("wagon_capacity_m3", {})

            baseline_kwh_m3 = float(yearly["kWh_per_m3"].mean()) if len(yearly) > 0 else 0.0
            baseline_kwh_kg = float(yearly["kWh_per_kg"].mean()) if len(yearly) > 0 else 0.0

            st.info(
                f"📈 **Historical Baseline KPIs:** "
                f"**{baseline_kwh_kg:.3f} kWh/kg** | **{baseline_kwh_m3:.1f} kWh/m³**"
            )

            use_custom_kpis = st.checkbox("🔧 Use custom KPIs", value=False)

            if use_custom_kpis:
                col_kpi1, col_kpi2 = st.columns(2)
                with col_kpi1:
                    prediction_kwh_m3 = st.number_input("Target kWh/m³", min_value=0.0, value=baseline_kwh_m3)
                with col_kpi2:
                    prediction_kwh_kg = st.number_input("Target kWh/kg", min_value=0.0, value=baseline_kwh_kg)
            else:
                prediction_kwh_m3 = baseline_kwh_m3
                prediction_kwh_kg = baseline_kwh_kg

            with st.form("weekly_prediction_form"):
                st.write("### 📅 Planned Production per Week")
                planned_wagons = {}

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**L-Type (Light)**")
                    for p in ["L28", "L30", "L34", "L36"]:
                        cap = wagon_capacity.get(p, 0)
                        cap_text = f" ({cap:.2f} m³/w)" if cap > 0 else ""
                        planned_wagons[p] = st.number_input(
                            f"{p}{cap_text}", min_value=0, value=0, step=10, key=f"w_{p}"
                        )

                with col2:
                    st.write("**L-Type (Heavy)**")
                    for p in ["L38", "L42", "L44"]:
                        cap = wagon_capacity.get(p, 0)
                        cap_text = f" ({cap:.2f} m³/w)" if cap > 0 else ""
                        planned_wagons[p] = st.number_input(
                            f"{p}{cap_text}", min_value=0, value=0, step=10, key=f"w_{p}"
                        )

                with col3:
                    st.write("**N & Y-Type**")
                    for p in ["N40", "N44", "Y44"]:
                        cap = wagon_capacity.get(p, 0)
                        cap_text = f" ({cap:.2f} m³/w)" if cap > 0 else ""
                        planned_wagons[p] = st.number_input(
                            f"{p}{cap_text}", min_value=0, value=0, step=10, key=f"w_{p}"
                        )

                submitted = st.form_submit_button("🔮 Calculate Prediction", type="primary")

            if submitted:
                product_volumes = {}
                total_wagons_pred = 0

                for prod, wagons in planned_wagons.items():
                    if wagons > 0:
                        capacity = wagon_capacity.get(prod, 1.5) or 1.5
                        product_volumes[prod] = wagons * capacity
                        total_wagons_pred += wagons

                if product_volumes:
                    pred = predict_production_energy(
                        product_volumes_m3=product_volumes,
                        baseline_kwh_per_m3=prediction_kwh_m3,
                        baseline_kwh_per_kg=prediction_kwh_kg,
                        use_formulas=True
                    )

                    st.markdown("---")
                    st.subheader("📊 Weekly Forecast")

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Total Wagons", f"{total_wagons_pred:,}")
                    with c2:
                        st.metric("Total Volume", f"{pred['total_volume_m3']:,.1f} m³")
                    with c3:
                        st.metric("Water to Evaporate", f"{pred['total_water_kg']:,.0f} kg")
                    with c4:
                        if pred.get("total_energy_kwh", 0) > 0:
                            energy = pred["total_energy_kwh"]
                            st.metric("Energy Required", f"{energy:,.0f} kWh")

                    if pred.get("products"):
                        st.write("### 📦 Product Breakdown")
                        breakdown = pd.DataFrame(pred["products"])
                        display_cols = {
                            "product": "Product",
                            "volume_m3": "Volume (m³)",
                            "water_per_plate_kg": "Water/Plate (kg)",
                            "water_kg": "Total Water (kg)",
                        }
                        if "energy_from_water_kwh" in breakdown.columns:
                            display_cols["energy_from_water_kwh"] = "Energy (kWh)"
                        breakdown = breakdown.rename(columns=display_cols)
                        cols = [c for c in display_cols.values() if c in breakdown.columns]
                        st.dataframe(breakdown[cols], use_container_width=True, hide_index=True)

                    st.success("✅ Prediction complete!")
                else:
                    st.warning("⚠️ Enter wagon counts for at least one product.")

            # ===== 8. EXPORT =====
            st.markdown(
                '<div class="section-header">📥 Export Results</div>',
                unsafe_allow_html=True
            )

            excel_data = create_excel_download(results)
            st.download_button(
                "📥 Download Complete Excel Report",
                excel_data,
                "Dryer_KPI_Analysis.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success("✅ Analysis complete!")

    except Exception as e:
        st.error(f"❌ Display error: {e}")
        with st.expander("Details"):
            st.exception(e)


