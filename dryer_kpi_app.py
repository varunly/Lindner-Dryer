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
    f"📊 **Using MEASURED Values from Table 1** | "
    f"Suspension: {SUSPENSION_KG} kg | "
    f"Each product has measured water evaporation values"
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("📁 Data Upload")

    energy_file = st.file_uploader(
        "📊 Energy File (.xlsx)",
        type=["xlsx"],
    )
    wagon_file = st.file_uploader(
        "🚛 Hordenwagen File (.xlsm, .xlsx)",
        type=["xlsm", "xlsx"],
    )

    st.markdown("---")
    st.subheader("⚙️ Filters")

    products = st.multiselect(
        "🧱 Product(s):",
        ["L28", "L30", "L32", "L34", "L36", "L38","L40", "L42", "L44", "N40", "N44", "Y44"],
        default=["L28", "L30", "L32", "L34", "L36", "L38","L40", "L42", "L44", "N40", "N44", "Y44"],
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


def run_analysis(energy_path: str, wagon_path: str, products_filter, month_filter) -> dict:
    progress = st.progress(0)
    status = st.empty()

    try:
        status.text("🔄 Parsing energy data...")
        progress.progress(15)
        e_raw = pd.read_excel(energy_path, sheet_name=CONFIG["energy_sheet"])
        e = parse_energy(e_raw)
        if e.empty:
            raise ValueError("Parsed energy data is empty.")

        status.text("🔄 Parsing wagon tracking data...")
        progress.progress(35)
        w_raw = pd.read_excel(
            wagon_path,
            sheet_name=CONFIG["wagon_sheet"],
            header=CONFIG["wagon_header_row"],
        )
        w = parse_wagon(w_raw)
        if w.empty:
            raise ValueError("Parsed wagon data is empty.")

        status.text("🔄 Applying filters...")
        progress.progress(50)
        if products_filter:
            w = w[w["Produkt"].astype(str).isin(products_filter)]
            if w.empty:
                raise ValueError(f"No wagon records found for selected products: {products_filter}")

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
        
        summary["kWh_thermal_per_m3"] = np.where(summary["Volume_m3"] > 0, summary["Energy_thermal_kWh"] / summary["Volume_m3"], 0)
        summary["kWh_per_m3"] = np.where(summary["Volume_m3"] > 0, summary["Energy_kWh"] / summary["Volume_m3"], 0)
        summary = add_water_kpis(summary)
        summary = summary.fillna(0)

        yearly = summary.groupby(["Produkt", "Zone"], as_index=False).agg(
            Energy_thermal_kWh=("Energy_thermal_kWh", "sum"),
            Energy_electrical_kWh=("Energy_electrical_kWh", "sum"),
            Energy_kWh=("Energy_kWh", "sum"),
            Volume_m3=("Volume_m3", "sum"),
            Water_kg=("Water_kg", "sum"),
        )
        
        yearly["kWh_thermal_per_m3"] = np.where(yearly["Volume_m3"] > 0, yearly["Energy_thermal_kWh"] / yearly["Volume_m3"], 0)
        yearly["kWh_per_m3"] = np.where(yearly["Volume_m3"] > 0, yearly["Energy_kWh"] / yearly["Volume_m3"], 0)
        yearly["kWh_thermal_per_kg"] = np.where(yearly["Water_kg"] > 0, yearly["Energy_thermal_kWh"] / yearly["Water_kg"], 0)
        yearly["kWh_per_kg"] = np.where(yearly["Water_kg"] > 0, yearly["Energy_kWh"] / yearly["Water_kg"], 0)
        yearly = yearly.fillna(0)
        
        product_totals = summary.groupby(["Month", "Produkt"], as_index=False).agg(
            Energy_thermal_kWh=("Energy_thermal_kWh", "sum"),
            Energy_electrical_kWh=("Energy_electrical_kWh", "sum"),
            Energy_kWh=("Energy_kWh", "sum"),
            Volume_m3=("Volume_m3", "sum"),
            Water_kg=("Water_kg", "sum"),
        )
        
        product_totals["kWh_thermal_per_m3"] = np.where(product_totals["Volume_m3"] > 0, product_totals["Energy_thermal_kWh"] / product_totals["Volume_m3"], 0)
        product_totals["kWh_per_m3"] = np.where(product_totals["Volume_m3"] > 0, product_totals["Energy_kWh"] / product_totals["Volume_m3"], 0)
        product_totals["kWh_thermal_per_kg"] = np.where(product_totals["Water_kg"] > 0, product_totals["Energy_thermal_kWh"] / product_totals["Water_kg"], 0)
        product_totals["kWh_per_kg"] = np.where(product_totals["Water_kg"] > 0, product_totals["Energy_kWh"] / product_totals["Water_kg"], 0)
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as f_e:
                f_e.write(energy_file.read())
                tmp_e = f_e.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsm") as f_w:
                f_w.write(wagon_file.read())
                tmp_w = f_w.name

            results = run_analysis(tmp_e, tmp_w, products if products else None, month if month != 0 else None)
            st.session_state.results = results
            st.session_state.analysis_complete = True

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
                    except:
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
            # ===== 1. SUMMARY KPIs =====
            st.markdown('<div class="section-header">📈 Summary KPIs</div>', unsafe_allow_html=True)

            total_thermal = float(yearly["Energy_thermal_kWh"].sum())
            total_electrical = float(yearly["Energy_electrical_kWh"].sum())
            total_energy = float(yearly["Energy_kWh"].sum())
            total_volume = float(yearly["Volume_m3"].sum())
            total_water = float(yearly["Water_kg"].sum())
            avg_kwh_per_m3 = float(yearly["kWh_per_m3"].mean()) if len(yearly) > 0 else 0
            avg_kwh_per_kg = float(yearly["kWh_per_kg"].mean()) if len(yearly) > 0 else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(create_kpi_card("Thermal Energy", total_thermal, "kWh"), unsafe_allow_html=True)
            with c2:
                st.markdown(create_kpi_card("Total Energy", total_energy, "kWh"), unsafe_allow_html=True)
            with c3:
                st.markdown(create_kpi_card("Total Volume", total_volume, "m³"), unsafe_allow_html=True)
            with c4:
                st.markdown(create_kpi_card("Avg kWh/m³", avg_kwh_per_m3, "kWh/m³"), unsafe_allow_html=True)
            with c5:
                st.markdown(create_kpi_card("Avg kWh/kg", avg_kwh_per_kg, "kWh/kg"), unsafe_allow_html=True)
            
            electrical_pct = (total_electrical / total_energy * 100) if total_energy > 0 else 0
            st.info(f"⚡ Electrical energy represents **{electrical_pct:.1f}%** of total energy consumption | 💧 Total water evaporated: **{total_water:,.0f} kg**")

            # ===== 2. ZONE COMPARISON (MOVED UP) =====
            st.markdown('<div class="section-header">📉 Zone Comparison</div>', unsafe_allow_html=True)

            # Aggregate by zone only
            zone_totals = yearly.groupby("Zone", as_index=False).agg({
                "Energy_thermal_kWh": "sum",
                "Energy_electrical_kWh": "sum",
                "Energy_kWh": "sum",
                "Volume_m3": "sum",
                "Water_kg": "sum",
            })
            zone_totals["kWh_per_m3"] = np.where(zone_totals["Volume_m3"] > 0, zone_totals["Energy_kWh"] / zone_totals["Volume_m3"], 0)

            col_z1, col_z2 = st.columns(2)
            
            with col_z1:
                # Stacked bar: Thermal + Electrical by Zone
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
                # Pie chart: Energy distribution
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

            # Zone summary table
            st.subheader("Zone Energy Summary")
            zone_display = zone_totals.copy()
            zone_display["Thermal %"] = (zone_display["Energy_thermal_kWh"] / zone_display["Energy_kWh"] * 100).round(1)
            zone_display["Electrical %"] = (zone_display["Energy_electrical_kWh"] / zone_display["Energy_kWh"] * 100).round(1)
            zone_display = zone_display.rename(columns={
                "Energy_thermal_kWh": "Thermal (kWh)",
                "Energy_electrical_kWh": "Electrical (kWh)",
                "Energy_kWh": "Total (kWh)",
                "Volume_m3": "Volume (m³)",
                "Water_kg": "Water (kg)",
                "kWh_per_m3": "kWh/m³"
            })
            st.dataframe(zone_display, use_container_width=True, hide_index=True)

            # ===== 3. PRODUCT PERFORMANCE (MOVED DOWN) =====
            if product_totals is not None and not product_totals.empty:
                st.markdown('<div class="section-header">📊 Product Performance</div>', unsafe_allow_html=True)
                
                prod_agg = product_totals.groupby("Produkt", as_index=False).agg({
                    "Energy_thermal_kWh": "sum",
                    "Energy_electrical_kWh": "sum",
                    "Energy_kWh": "sum",
                    "Volume_m3": "sum",
                    "Water_kg": "sum",
                })
                
                prod_agg["kWh_per_m3"] = np.where(prod_agg["Volume_m3"] > 0, prod_agg["Energy_kWh"] / prod_agg["Volume_m3"], 0)
                prod_agg["kWh_per_kg"] = np.where(prod_agg["Water_kg"] > 0, prod_agg["Energy_kWh"] / prod_agg["Water_kg"], 0)
                prod_agg = prod_agg.fillna(0)
                
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    # Total Energy by Product (Thermal + Electrical stacked)
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
                        yaxis_title="Energy (kWh)",
                        barmode='stack',
                        height=400,
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_prod_energy, use_container_width=True)
                
                with col_p2:
                    # Energy Efficiency by Product (kWh/m³)
                    fig_efficiency = px.bar(
                        prod_agg,
                        x="Produkt",
                        y="kWh_per_m3",
                        color="kWh_per_m3",
                        color_continuous_scale="RdYlGn_r",
                        title="Energy Efficiency by Product (kWh/m³)",
                        text_auto=".1f"
                    )
                    fig_efficiency.update_layout(height=400, plot_bgcolor="white")
                    fig_efficiency.update_coloraxes(colorbar_title="kWh/m³")
                    st.plotly_chart(fig_efficiency, use_container_width=True)
                
                # Product summary table
                st.subheader("Product Energy Summary")
                prod_display = prod_agg.rename(columns={
                    "Produkt": "Product",
                    "Energy_thermal_kWh": "Thermal (kWh)",
                    "Energy_electrical_kWh": "Electrical (kWh)",
                    "Energy_kWh": "Total (kWh)",
                    "Volume_m3": "Volume (m³)",
                    "Water_kg": "Water (kg)",
                    "kWh_per_m3": "kWh/m³",
                    "kWh_per_kg": "kWh/kg"
                })
                st.dataframe(prod_display, use_container_width=True, hide_index=True)

            # ===== 4. PRODUCT SPECIFICATIONS (MEASURED VALUES) =====
            st.markdown('<div class="section-header">📐 Product Specifications (MEASURED Values)</div>', unsafe_allow_html=True)
            
            st.write(f"**Formula:** Water/mm (g) = Slope × {SUSPENSION_KG}kg + Intercept | All values are **MEASURED** from production data")
            
            specs_data = []
            for prod, spec in PRODUCT_SPECIFICATIONS.items():
                specs_data.append({
                    "Product": prod,
                    "Type": spec["product_type"],
                    "Formula": spec["formula"],
                    "Thickness (mm)": spec["pressed_thickness_mm"],
                    "Water/mm (g)": spec["water_per_mm_g"],
                    "Water/Plate (kg)": spec["water_per_plate_kg"],
                    "Water/m³ (kg)": round(spec["water_per_m3_kg"], 1),
                })
            
            specs_df = pd.DataFrame(specs_data)
            st.dataframe(specs_df, use_container_width=True, hide_index=True)
            
            # Water curves
            st.subheader("Water Evaporation Curves by Product")
            
            products_to_plot = st.multiselect(
                "Select products to compare:",
                list(PRODUCT_SPECIFICATIONS.keys()),
                default=["L36", "L38", "N40"],
                key="curve_products"
            )
            
            if products_to_plot:
                all_curves = []
                for prod in products_to_plot:
                    curve_df = get_product_water_curve(prod)
                    if curve_df is not None and not curve_df.empty:
                        all_curves.append(curve_df)
                
                if all_curves:
                    combined_curves = pd.concat(all_curves, ignore_index=True)
                    
                    fig_curves = px.line(
                        combined_curves,
                        x="Pressed_Thickness_mm",
                        y="Water_per_Plate_kg",
                        color="Product",
                        markers=True,
                        title="Water Evaporation vs. Pressed Thickness",
                        labels={"Pressed_Thickness_mm": "Pressed Thickness (mm)", "Water_per_Plate_kg": "Water per Plate (kg)"}
                    )
                    
                    for prod in products_to_plot:
                        if prod in PRODUCT_SPECIFICATIONS:
                            spec = PRODUCT_SPECIFICATIONS[prod]
                            fig_curves.add_scatter(
                                x=[spec["pressed_thickness_mm"]],
                                y=[spec["water_per_plate_kg"]],
                                mode='markers',
                                marker=dict(size=14, symbol='star'),
                                name=f"{prod} (measured)"
                            )
                    
                    fig_curves.update_layout(height=450, plot_bgcolor="white")
                    st.plotly_chart(fig_curves, use_container_width=True)
                    st.info("⭐ **Star markers** = MEASURED values from production data")

            # ===== 5. MONTHLY TRENDS =====
            st.markdown('<div class="section-header">📊 Monthly KPI Trends</div>', unsafe_allow_html=True)

            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                fig_kwh_m3 = px.line(
                    summary,
                    x="Month",
                    y="kWh_per_m3",
                    color="Zone",
                    markers=True,
                    title="Energy Efficiency Trend (kWh/m³)"
                )
                fig_kwh_m3.update_layout(height=400, plot_bgcolor="white")
                st.plotly_chart(fig_kwh_m3, use_container_width=True)

            with col_m2:
                if "kWh_per_kg" in summary.columns:
                    fig_kwh_kg = px.line(
                        summary,
                        x="Month",
                        y="kWh_per_kg",
                        color="Zone",
                        markers=True,
                        title="Specific Energy Trend (kWh/kg water)"
                    )
                    fig_kwh_kg.update_layout(height=400, plot_bgcolor="white")
                    st.plotly_chart(fig_kwh_kg, use_container_width=True)

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
            st.markdown('<div class="section-header">🔮 Weekly Energy Prediction</div>', unsafe_allow_html=True)

            wagon_stats = compute_product_wagon_stats(results["wagons"])
            wagon_capacity = wagon_stats.get("wagon_capacity_m3", {})

            # Automatic baseline KPIs
            baseline_kwh_m3 = float(yearly["kWh_per_m3"].mean()) if len(yearly) > 0 else 0.0
            baseline_kwh_kg = float(yearly["kWh_per_kg"].mean()) if len(yearly) > 0 else 0.0

            st.info(f"📈 **Historical Baseline KPIs** (automatic): **{baseline_kwh_kg:.3f} kWh/kg** | **{baseline_kwh_m3:.1f} kWh/m³**")

            use_custom_kpis = st.checkbox("🔧 Use custom KPIs (scenario testing)", value=False)

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
                        planned_wagons[p] = st.number_input(f"{p}{cap_text}", min_value=0, value=0, step=10, key=f"w_{p}")

                with col2:
                    st.write("**L-Type (Heavy)**")
                    for p in ["L38", "L42", "L44"]:
                        cap = wagon_capacity.get(p, 0)
                        cap_text = f" ({cap:.2f} m³/w)" if cap > 0 else ""
                        planned_wagons[p] = st.number_input(f"{p}{cap_text}", min_value=0, value=0, step=10, key=f"w_{p}")

                with col3:
                    st.write("**N & Y-Type**")
                    for p in ["N40", "N44", "Y44"]:
                        cap = wagon_capacity.get(p, 0)
                        cap_text = f" ({cap:.2f} m³/w)" if cap > 0 else ""
                        planned_wagons[p] = st.number_input(f"{p}{cap_text}", min_value=0, value=0, step=10, key=f"w_{p}")

                submitted = st.form_submit_button("🔮 Calculate Prediction", type="primary")

            if submitted:
                product_volumes = {}
                total_wagons = 0
                
                for prod, wagons in planned_wagons.items():
                    if wagons > 0:
                        capacity = wagon_capacity.get(prod, 1.5) or 1.5
                        product_volumes[prod] = wagons * capacity
                        total_wagons += wagons
                
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
                        st.metric("Total Wagons", f"{total_wagons:,}")
                    with c2:
                        st.metric("Total Volume", f"{pred['total_volume_m3']:,.1f} m³")
                    with c3:
                        st.metric("Water to Evaporate", f"{pred['total_water_kg']:,.0f} kg")
                    with c4:
                        if pred.get("total_energy_kwh", 0) > 0:
                            energy = pred["total_energy_kwh"]
                            st.metric("Energy Required", f"{energy:,.0f} kWh")
                            st.caption(f"📅 {energy/7:,.0f} kWh/day | ⏱️ {energy/168:,.0f} kWh/hour")
                    
                    if pred.get("products"):
                        st.write("### 📦 Product Breakdown")
                        breakdown = pd.DataFrame(pred["products"])
                        
                        display_cols = {
                            "product": "Product",
                            "formula": "Formula",
                            "volume_m3": "Volume (m³)",
                            "num_plates": "Plates",
                            "water_per_mm_g": "Water/mm (g)",
                            "water_per_plate_kg": "Water/Plate (kg)",
                            "water_kg": "Total Water (kg)",
                        }
                        if "energy_from_water_kwh" in breakdown.columns:
                            display_cols["energy_from_water_kwh"] = "Energy (kWh)"
                        
                        breakdown = breakdown.rename(columns=display_cols)
                        cols = [c for c in display_cols.values() if c in breakdown.columns]
                        st.dataframe(breakdown[cols], use_container_width=True, hide_index=True)
                    
                    with st.expander("🔍 How is this calculated?"):
                        st.markdown(f"""
**Calculation Method:**

| Step | Calculation | Example |
|------|-------------|---------|
| 1. Volume | Wagons × Capacity | 100 wagons × 1.5 m³ = 150 m³ |
| 2. Water | Volume × Water/m³ | 150 m³ × 188.8 kg/m³ = 28,320 kg |
| 3. Energy | Water × kWh/kg | 28,320 kg × {prediction_kwh_kg:.3f} = {28320 * prediction_kwh_kg:,.0f} kWh |

**Baseline KPI:** {prediction_kwh_kg:.3f} kWh/kg {'(custom)' if use_custom_kpis else '(from historical data)'}
                        """)
                    
                    st.success("✅ Prediction complete!")
                else:
                    st.warning("⚠️ Enter wagon counts for at least one product.")

            # ===== 8. EXPORT =====
            st.markdown('<div class="section-header">📥 Export Results</div>', unsafe_allow_html=True)
            
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

