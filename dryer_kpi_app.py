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
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #17a2b8;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
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
    f"Plates/Wagon: {PLATES_PER_WAGON} | "
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

    # Trockner selection
    trockner_option = st.selectbox(
        "🏭 Trockner (Dryer):",
        options=["All", "A", "B"],
        index=0,
        help="Select Trockner A, B, or All"
    )

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
    run_button = st.button("▶️ Run Analysis", type="primary")


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


def run_analysis(energy_path: str, wagon_path: str, products_filter, month_filter, trockner_filter) -> dict:
    """
    Run the complete KPI analysis.
    
    Wagon counting method:
    - Each row in Column A (WG-Nr) with a valid wagon number = 1 wagon
    - If Trockner filter is applied, only count wagons for that Trockner
    - Volume is read directly from the m³ column
    """
    progress = st.progress(0)
    status = st.empty()

    try:
        # ===== PARSE ENERGY =====
        status.text("🔄 Parsing energy data...")
        progress.progress(15)
        
        try:
            e_raw = pd.read_excel(energy_path, sheet_name=CONFIG["energy_sheet"])
        except Exception as e:
            raise ValueError(f"Cannot read energy file: {e}")
        
        e = parse_energy(e_raw)
        if e.empty:
            raise ValueError("Parsed energy data is empty.")

        # ===== PARSE WAGON DATA =====
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
        
        # Apply Trockner filter during parsing
        trockner_to_use = trockner_filter if trockner_filter != "All" else None
        
        w = parse_wagon(w_raw, trockner=trockner_to_use)
        
        if w.empty:
            raise ValueError("Parsed wagon data is empty.")

        # Store filter info
        applied_trockner = trockner_to_use or "All"
        
        # ===== APPLY PRODUCT FILTER =====
        status.text("🔄 Applying product filters...")
        progress.progress(50)
        
        wagon_count_before_product_filter = len(w)
        
        if products_filter:
            w = w[w["Produkt"].astype(str).isin(products_filter)]
            if w.empty:
                raise ValueError(f"No wagon records found for selected products: {products_filter}")
        
        wagon_count_after_product_filter = len(w)
        
        # Key metrics from wagon data
        total_wagons = len(w)  # This is the correct wagon count!
        total_volume = w["m3"].sum()
        
        status.text(f"📊 Found {total_wagons:,} wagons with {total_volume:,.2f} m³ total volume")

        # ===== APPLY MONTH FILTER =====
        if month_filter:
            e = e[e["Month"] == month_filter]
            w = w[w["Month"] == month_filter]
            if e.empty or w.empty:
                raise ValueError(f"No data found for month = {month_filter}.")

        # ===== BUILD INTERVALS =====
        status.text("🔄 Building zone intervals...")
        progress.progress(65)
        ivals = explode_intervals(w)
        if ivals.empty:
            raise ValueError("Zone intervals could not be created.")

        # ===== ALLOCATE ENERGY =====
        status.text("🔄 Allocating energy to products...")
        progress.progress(80)
        alloc = allocate_energy(e, ivals)
        if alloc.empty:
            raise ValueError("Energy allocation result is empty.")

        # ===== AGGREGATE KPIs =====
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
            "applied_trockner": applied_trockner,
            "wagon_count_before_product_filter": wagon_count_before_product_filter,
            "wagon_count_after_product_filter": wagon_count_after_product_filter,
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
    current_files = (energy_file.name, wagon_file.name, trockner_option)
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
            st.info(f"🏭 Trockner filter: {trockner_option}")
            
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
                month if month != 0 else None,
                trockner_option
            )
            st.session_state.results = results
            st.session_state.analysis_complete = True

        except ValueError as ve:
            st.error(f"⚠️ Validation Error: {ve}")
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
        wagons_df = results["wagons"]
        
        # Get Trockner info
        applied_trockner = results.get("applied_trockner", "All")

        if summary.empty:
            st.warning("⚠️ No data available after filtering.")
        else:
            # ============================================================
            #   CORRECT WAGON AND VOLUME CALCULATION
            # ============================================================
            
            # WAGON COUNT: Number of rows in wagons DataFrame
            # This is already filtered by Trockner and Product in run_analysis()
            total_wagons = len(wagons_df)
            
            # VOLUME: Sum of m³ column from wagons DataFrame
            total_volume = wagons_df["m3"].sum()
            
            # Average volume per wagon
            avg_volume_per_wagon = total_volume / total_wagons if total_wagons > 0 else 0

            # ===== ENERGY TOTALS =====
            total_thermal = float(yearly["Energy_thermal_kWh"].sum())
            total_electrical = float(yearly["Energy_electrical_kWh"].sum())
            total_energy = float(yearly["Energy_kWh"].sum())

            # ===== WATER CALCULATION =====
            product_volume_unique = wagons_df.groupby("Produkt")["m3"].sum().reset_index()
            
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
                    "Wagons": len(wagons_df[wagons_df["Produkt"] == prod]),
                    "Volume (m³)": round(vol, 2),
                    "Water/m³ (kg)": round(water_per_m3, 1),
                    "Total Water (kg)": round(water_kg, 0),
                })

            # ===== KPI CALCULATIONS =====
            avg_kwh_per_m3 = safe_divide(total_energy, total_volume)
            avg_kwh_thermal_per_m3 = safe_divide(total_thermal, total_volume)
            avg_kwh_per_kg = safe_divide(total_energy, total_water)
            avg_kwh_thermal_per_kg = safe_divide(total_thermal, total_water)

            thermal_pct = (total_thermal / total_energy * 100) if total_energy > 0 else 0
            electrical_pct = (total_electrical / total_energy * 100) if total_energy > 0 else 0

            avg_water_per_m3 = safe_divide(total_water, total_volume)

            # ============================================================
            #                 TROCKNER INFO BANNER
            # ============================================================
            if applied_trockner != "All":
                st.success(f"🏭 **Showing data for Trockner {applied_trockner} only**")
            else:
                st.info("🏭 **Showing data for all Trockner (A + B)**")

            # ============================================================
            #                     SUMMARY KPIs SECTION
            # ============================================================
            st.markdown('<div class="section-header">📈 Summary KPIs</div>', unsafe_allow_html=True)

            # --- PRODUCTION CARDS (with wagon count) ---
            st.subheader("🏭 Production")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(create_kpi_card("Total Wagons", total_wagons, ""), unsafe_allow_html=True)
            with c2:
                st.markdown(create_kpi_card("Total Volume", total_volume, "m³"), unsafe_allow_html=True)
            with c3:
                st.markdown(create_kpi_card("Avg Volume/Wagon", avg_volume_per_wagon, "m³"), unsafe_allow_html=True)
            with c4:
                st.markdown(create_kpi_card("Water Evaporated", total_water, "kg"), unsafe_allow_html=True)

            # --- ENERGY CARDS ---
            st.subheader("⚡ Energy Consumption")
            c5, c6, c7 = st.columns(3)

            with c5:
                st.markdown(create_kpi_card("Thermal Energy", total_thermal, "kWh"), unsafe_allow_html=True)
            with c6:
                st.markdown(create_kpi_card("Electrical Energy", total_electrical, "kWh"), unsafe_allow_html=True)
            with c7:
                st.markdown(create_kpi_card("Total Energy", total_energy, "kWh"), unsafe_allow_html=True)

            # --- EFFICIENCY CARDS ---
            st.subheader("📊 Energy Efficiency")
            c8, c9, c10, c11 = st.columns(4)

            with c8:
                st.markdown(create_kpi_card("kWh/kg water", avg_kwh_per_kg, "kWh/kg"), unsafe_allow_html=True)
            with c9:
                st.markdown(create_kpi_card("Thermal kWh/kg", avg_kwh_thermal_per_kg, "kWh/kg"), unsafe_allow_html=True)
            with c10:
                st.markdown(create_kpi_card("kWh/m³", avg_kwh_per_m3, "kWh/m³"), unsafe_allow_html=True)
            with c11:
                st.markdown(create_kpi_card("Thermal kWh/m³", avg_kwh_thermal_per_m3, "kWh/m³"), unsafe_allow_html=True)

            # ===== INFO BOX =====
            st.info(
                f"🏭 **Trockner {applied_trockner}** | "
                f"🚛 **{total_wagons:,} wagons** | "
                f"📦 **{total_volume:,.0f} m³** | "
                f"💧 **{total_water:,.0f} kg** water | "
                f"⚡ **{total_energy:,.0f} kWh** (Thermal: {thermal_pct:.1f}%)"
            )

            # ============================================================
            #    VOLUME BREAKDOWN & VALIDATION (CORRECTED)
            # ============================================================
            with st.expander("📊 Volume Breakdown & Wagon Count Validation", expanded=True):
                st.markdown("### 🚛 Wagon Count Methodology")
                
                st.markdown(f"""
                **How wagons are counted:**
                1. **Column A (WG-Nr)** in the Excel file contains wagon numbers
                2. Each row with a valid wagon number = **1 wagon**
                3. Invalid rows (headers, summaries, empty) are excluded
                4. Trockner filter: **{applied_trockner}**
                
                **Current counts:**
                - Wagons before product filter: **{results.get('wagon_count_before_product_filter', 'N/A'):,}**
                - Wagons after product filter: **{results.get('wagon_count_after_product_filter', 'N/A'):,}**
                - Final wagon count: **{total_wagons:,}**
                """)
                
                st.markdown("---")
                st.markdown("### 📊 Summary Statistics")
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                
                with col_s1:
                    st.metric("Total Wagons", f"{total_wagons:,}")
                with col_s2:
                    st.metric("Total Volume", f"{total_volume:,.2f} m³")
                with col_s3:
                    st.metric("Avg Volume/Wagon", f"{avg_volume_per_wagon:.4f} m³")
                with col_s4:
                    unique_products = wagons_df["Produkt"].nunique()
                    st.metric("Unique Products", f"{unique_products}")
                
                st.markdown("---")
                st.markdown("### 📦 Breakdown by Product")
                
                # Create product breakdown table
                product_breakdown = wagons_df.groupby("Produkt").agg({
                    "m3": ["count", "sum", "mean", "min", "max"]
                }).round(4)
                product_breakdown.columns = ["Wagon Count", "Total Volume (m³)", "Avg Volume (m³)", "Min Volume (m³)", "Max Volume (m³)"]
                product_breakdown = product_breakdown.reset_index()
                product_breakdown = product_breakdown.sort_values("Total Volume (m³)", ascending=False)
                
                # Add percentage columns
                product_breakdown["% of Wagons"] = (product_breakdown["Wagon Count"] / total_wagons * 100).round(1)
                product_breakdown["% of Volume"] = (product_breakdown["Total Volume (m³)"] / total_volume * 100).round(1)
                
                # Display table
                st.dataframe(
                    product_breakdown,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Produkt": "Product",
                        "Wagon Count": st.column_config.NumberColumn("Wagons", format="%d"),
                        "Total Volume (m³)": st.column_config.NumberColumn("Total Volume (m³)", format="%.2f"),
                        "Avg Volume (m³)": st.column_config.NumberColumn("Avg Vol/Wagon (m³)", format="%.4f"),
                        "Min Volume (m³)": st.column_config.NumberColumn("Min (m³)", format="%.4f"),
                        "Max Volume (m³)": st.column_config.NumberColumn("Max (m³)", format="%.4f"),
                        "% of Wagons": st.column_config.NumberColumn("% Wagons", format="%.1f%%"),
                        "% of Volume": st.column_config.NumberColumn("% Volume", format="%.1f%%"),
                    }
                )
                
                # Totals row
                st.markdown(f"""
                **Totals:** {total_wagons:,} wagons | {total_volume:,.2f} m³ | {avg_volume_per_wagon:.4f} m³/wagon
                """)
                
                st.markdown("---")
                st.markdown("### 📅 Breakdown by Month")
                
                monthly_breakdown = wagons_df.groupby("Month").agg({
                    "m3": ["count", "sum", "mean"]
                }).round(4)
                monthly_breakdown.columns = ["Wagon Count", "Total Volume (m³)", "Avg Volume (m³)"]
                monthly_breakdown = monthly_breakdown.reset_index()
                
                st.dataframe(
                    monthly_breakdown,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Month": st.column_config.NumberColumn("Month", format="%d"),
                        "Wagon Count": st.column_config.NumberColumn("Wagons", format="%d"),
                        "Total Volume (m³)": st.column_config.NumberColumn("Volume (m³)", format="%.2f"),
                        "Avg Volume (m³)": st.column_config.NumberColumn("Avg Vol/Wagon (m³)", format="%.4f"),
                    }
                )
                
                st.markdown("---")
                st.markdown("### 🔍 Sample Wagon Data (First 20 rows)")
                
                sample_cols = ["WG_Nr", "Produkt", "m3", "Month", "t0", "Trockner"]
                available_cols = [c for c in sample_cols if c in wagons_df.columns]
                
                sample_df = wagons_df[available_cols].head(20).copy()
                if "t0" in sample_df.columns:
                    sample_df["t0"] = sample_df["t0"].dt.strftime("%Y-%m-%d %H:%M")
                
                st.dataframe(sample_df, use_container_width=True, hide_index=True)
                
                # Validation check
                st.markdown("---")
                st.markdown("### ✅ Validation")
                
                # Check if volume sums match
                volume_from_breakdown = product_breakdown["Total Volume (m³)"].sum()
                wagon_count_from_breakdown = product_breakdown["Wagon Count"].sum()
                
                col_v1, col_v2 = st.columns(2)
                
                with col_v1:
                    if abs(volume_from_breakdown - total_volume) < 0.01:
                        st.success(f"✅ Volume check passed: {total_volume:,.2f} m³")
                    else:
                        st.error(f"❌ Volume mismatch: {total_volume:,.2f} vs {volume_from_breakdown:,.2f}")
                
                with col_v2:
                    if wagon_count_from_breakdown == total_wagons:
                        st.success(f"✅ Wagon count check passed: {total_wagons:,} wagons")
                    else:
                        st.error(f"❌ Wagon count mismatch: {total_wagons:,} vs {wagon_count_from_breakdown:,}")

            # ============================================================
            #    Continue with the rest of the dashboard...
            # ============================================================
            
            # Zone Comparison
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

            # ===== EXPORT =====
            st.markdown(
                '<div class="section-header">📥 Export Results</div>',
                unsafe_allow_html=True
            )

            excel_data = create_excel_download(results)
            st.download_button(
                "📥 Download Complete Excel Report",
                excel_data,
                f"Dryer_KPI_Analysis_Trockner_{applied_trockner}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success("✅ Analysis complete!")

    except Exception as e:
        st.error(f"❌ Display error: {e}")
        with st.expander("Details"):
            st.exception(e)
