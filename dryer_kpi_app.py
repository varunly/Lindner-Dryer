# dryer_kpi_app.py
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
        TROCKNER_COLUMN,
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
    .trockner-a {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    }
    .trockner-b {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%) !important;
    }
    .trockner-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 18px;
        margin: 10px 0;
    }
    .trockner-badge-a {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .trockner-badge-b {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
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

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("📁 Data Upload")

    energy_file = st.file_uploader(
        "📊 Energy File (.xlsx)",
        type=["xlsx"],
        key="energy_uploader",
        help="Upload the energy consumption file"
    )
    wagon_file = st.file_uploader(
        "🚛 Hordenwagen File (.xlsm, .xlsx)",
        type=["xlsm", "xlsx"],
        key="wagon_uploader",
        help="Upload the wagon tracking file"
    )

    st.markdown("---")
    
    # ===== TROCKNER SELECTION (MANDATORY) =====
    st.subheader("🏭 Trockner Selection")
    
    trockner_selection = st.radio(
        "Select Dryer:",
        options=["Trockner A", "Trockner B"],
        index=0,
        help=f"Wagons are filtered by column '{TROCKNER_COLUMN}' = 'A' or 'B'"
    )
    
    # Extract just A or B
    selected_trockner = "A" if trockner_selection == "Trockner A" else "B"
    
    # Show badge
    badge_class = "trockner-badge-a" if selected_trockner == "A" else "trockner-badge-b"
    st.markdown(
        f'<span class="trockner-badge {badge_class}">🏭 Trockner {selected_trockner}</span>',
        unsafe_allow_html=True
    )
    
    st.caption(f"Will filter wagons where '{TROCKNER_COLUMN}' = '{selected_trockner}'")

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
    run_button = st.button("▶️ Run Analysis", type="primary")

# Show current configuration
st.info(
    f"📊 **Configuration:** Suspension = {SUSPENSION_KG} kg | "
    f"Trockner Column = '{TROCKNER_COLUMN}' | "
    f"**Selected: Trockner {selected_trockner}**"
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def create_kpi_card(title: str, value, unit: str, trockner: str = None) -> str:
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
    
    extra_class = ""
    if trockner == "A":
        extra_class = " trockner-a"
    elif trockner == "B":
        extra_class = " trockner-b"
    
    return f'<div class="metric-card{extra_class}"><h3>{title}</h3><h2>{text}{unit_str}</h2></div>'


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
    try:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        wb.close()
        return True
    except Exception as e:
        raise ValueError(f"Cannot read {file_description}: {e}")


def run_analysis(energy_path: str, wagon_path: str, products_filter, month_filter, 
                 trockner: str) -> dict:
    """Run analysis with Trockner filtering."""
    progress = st.progress(0)
    status = st.empty()

    try:
        status.text(f"🔄 Parsing energy data...")
        progress.progress(15)
        
        try:
            e_raw = pd.read_excel(energy_path, sheet_name=CONFIG["energy_sheet"])
        except Exception as e:
            raise ValueError(f"Cannot read energy file: {e}")
        
        e = parse_energy(e_raw)
        if e.empty:
            raise ValueError("Parsed energy data is empty.")

        status.text(f"🔄 Parsing wagon data for Trockner {trockner}...")
        progress.progress(35)
        
        try:
            w_raw = pd.read_excel(
                wagon_path,
                sheet_name=CONFIG["wagon_sheet"],
                header=CONFIG["wagon_header_row"],
            )
        except Exception as e:
            raise ValueError(f"Cannot read wagon file: {e}")
        
        # Parse wagons WITH Trockner filter
        w = parse_wagon(w_raw, trockner=trockner)
        if w.empty:
            raise ValueError(f"No wagon records found for Trockner {trockner}.")

        status.text("🔄 Applying product filters...")
        progress.progress(50)
        if products_filter:
            w = w[w["Produkt"].astype(str).isin(products_filter)]
            if w.empty:
                raise ValueError(f"No wagon records found for selected products: {products_filter}")

        total_volume_raw = w["m3"].sum()
        total_wagons = len(w)

        status.text(f"📊 Trockner {trockner}: {total_volume_raw:,.0f} m³ from {total_wagons:,} wagons")

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
        status.text(f"✅ Analysis complete for Trockner {trockner}!")

        return {
            "energy": e,
            "wagons": w,
            "intervals": ivals,
            "allocation": alloc,
            "summary": summary,
            "yearly": yearly,
            "product_totals": product_totals,
            "trockner": trockner,
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

# Reset if files or trockner changed
if energy_file and wagon_file:
    current_config = (energy_file.name, wagon_file.name, selected_trockner)
    if "last_config" not in st.session_state:
        st.session_state.last_config = current_config
    elif st.session_state.last_config != current_config:
        st.session_state.results = None
        st.session_state.analysis_complete = False
        st.session_state.last_config = current_config

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
            
            # Show Trockner selection prominently
            trockner_color = "#11998e" if selected_trockner == "A" else "#eb3349"
            st.markdown(
                f'<div style="background: {trockner_color}; color: white; padding: 10px 20px; '
                f'border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; '
                f'margin: 10px 0;">🏭 Analyzing: Trockner {selected_trockner}</div>',
                unsafe_allow_html=True
            )
            
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
                trockner=selected_trockner
            )
            st.session_state.results = results
            st.session_state.analysis_complete = True

        except ValueError as ve:
            st.error(f"⚠️ Validation Error: {ve}")
            st.warning(
                f"💡 **Troubleshooting Tips:**\n"
                f"1. Check that the wagon file has column '{TROCKNER_COLUMN}'\n"
                f"2. Verify the column contains 'A' or 'B' values\n"
                f"3. Make sure there are wagons for Trockner {selected_trockner}\n"
                f"4. Check the energy file corresponds to Trockner {selected_trockner}"
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
        active_trockner = results.get("trockner", "Unknown")

        if summary.empty:
            st.warning("⚠️ No data available after filtering.")
        else:
            # ===== TROCKNER HEADER =====
            trockner_color = "#11998e" if active_trockner == "A" else "#eb3349"
            st.markdown(
                f'<div style="background: {trockner_color}; color: white; padding: 15px 25px; '
                f'border-radius: 15px; text-align: center; font-size: 28px; font-weight: bold; '
                f'margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">'
                f'🏭 Trockner {active_trockner} - Analysis Results</div>',
                unsafe_allow_html=True
            )

            # ===== ENERGY TOTALS =====
            total_thermal = float(yearly["Energy_thermal_kWh"].sum())
            total_electrical = float(yearly["Energy_electrical_kWh"].sum())
            total_energy = float(yearly["Energy_kWh"].sum())

            # ===== VOLUME FROM WAGONS =====
            total_volume = float(results["wagons"]["m3"].sum())

            # ===== WATER CALCULATION =====
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
            st.markdown(
                f'<div class="section-header">📈 Summary KPIs - Trockner {active_trockner}</div>',
                unsafe_allow_html=True
            )

            # --- ENERGY CARDS ---
            st.subheader("⚡ Energy Consumption")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(create_kpi_card("Thermal Energy", total_thermal, "kWh", active_trockner), unsafe_allow_html=True)
            with c2:
                st.markdown(create_kpi_card("Electrical Energy", total_electrical, "kWh", active_trockner), unsafe_allow_html=True)
            with c3:
                st.markdown(create_kpi_card("Total Energy", total_energy, "kWh", active_trockner), unsafe_allow_html=True)

            # --- PRODUCTION CARDS ---
            st.subheader("🏭 Production")
            c4, c5, c6 = st.columns(3)

            with c4:
                st.markdown(create_kpi_card("Total Volume", total_volume, "m³", active_trockner), unsafe_allow_html=True)
            with c5:
                st.markdown(create_kpi_card("Water Evaporated", total_water, "kg", active_trockner), unsafe_allow_html=True)
            with c6:
                st.markdown(create_kpi_card("Water/m³", avg_water_per_m3, "kg/m³", active_trockner), unsafe_allow_html=True)

            # --- EFFICIENCY CARDS ---
            st.subheader("📊 Energy Efficiency")
            c7, c8, c9, c10 = st.columns(4)

            with c7:
                st.markdown(create_kpi_card("kWh/kg water", avg_kwh_per_kg, "kWh/kg", active_trockner), unsafe_allow_html=True)
            with c8:
                st.markdown(create_kpi_card("Thermal kWh/kg", avg_kwh_thermal_per_kg, "kWh/kg", active_trockner), unsafe_allow_html=True)
            with c9:
                st.markdown(create_kpi_card("kWh/m³", avg_kwh_per_m3, "kWh/m³", active_trockner), unsafe_allow_html=True)
            with c10:
                st.markdown(create_kpi_card("Thermal kWh/m³", avg_kwh_thermal_per_m3, "kWh/m³", active_trockner), unsafe_allow_html=True)

            # ===== INFO BOX =====
            st.info(
                f"🏭 **Trockner {active_trockner}** | "
                f"⚡ **Energy Mix:** Thermal = **{thermal_pct:.1f}%** ({total_thermal:,.0f} kWh) | "
                f"Electrical = **{electrical_pct:.1f}%** ({total_electrical:,.0f} kWh) | "
                f"🚛 **Production:** {total_wagons:,} wagons | {total_volume:,.0f} m³ | "
                f"💧 **Water:** {total_water:,.0f} kg ({total_water/1000:,.1f} tons)"
            )

            # ===== REST OF YOUR DISPLAY CODE =====
            # (Zone comparison, charts, data tables, etc.)
            # Copy the rest from your original file...

            # ===== 2. ZONE COMPARISON =====
            st.markdown(
                f'<div class="section-header">📉 Zone Comparison - Trockner {active_trockner}</div>',
                unsafe_allow_html=True
            )

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
                    title=f"Energy by Zone - Trockner {active_trockner} (kWh)",
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
                    title=f"Energy Distribution - Trockner {active_trockner} (%)",
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

            # ===== DATA TABLES =====
            with st.expander("📋 View Detailed Data Tables"):
                tab1, tab2, tab3, tab4 = st.tabs(["Wagons", "Monthly Summary", "Yearly Summary", "Product Totals"])
                with tab1:
                    st.write(f"**Wagons for Trockner {active_trockner}:** {len(results['wagons'])} records")
                    st.dataframe(results["wagons"].head(100), use_container_width=True)
                with tab2:
                    st.dataframe(summary, use_container_width=True)
                with tab3:
                    st.dataframe(yearly, use_container_width=True)
                with tab4:
                    if product_totals is not None:
                        st.dataframe(product_totals, use_container_width=True)

            # ===== EXPORT =====
            st.markdown(
                f'<div class="section-header">📥 Export Results - Trockner {active_trockner}</div>',
                unsafe_allow_html=True
            )

            excel_data = create_excel_download(results)
            st.download_button(
                f"📥 Download Trockner {active_trockner} Report (Excel)",
                excel_data,
                f"Dryer_KPI_Trockner_{active_trockner}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success(f"✅ Analysis complete for Trockner {active_trockner}!")

    except Exception as e:
        st.error(f"❌ Display error: {e}")
        with st.expander("Details"):
            st.exception(e)
