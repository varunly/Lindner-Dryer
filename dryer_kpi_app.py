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
        predict_weekly_energy_from_wagons,
        compute_product_wagon_stats,
        predict_mix_energy,
        predict_production_energy,
        calculate_water_per_m3_formula,
        get_product_water_curve,
        WATER_PER_M3_KG,
        PRODUCT_SPECIFICATIONS,
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
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
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
    .metric-card h3 {
        margin: 0;
        font-size: 16px;
        opacity: 0.9;
    }
    .metric-card h2 {
        margin: 10px 0 0 0;
        font-size: 32px;
        font-weight: 700;
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
    "📊 Upload your **Energy** and **Hordenwagen** files. "
    "Water-loss formulas are embedded from actual measurements."
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("📁 Data Upload")

    energy_file = st.file_uploader(
        "📊 Energy File (.xlsx)",
        type=["xlsx"],
        help="Upload the hourly energy consumption Excel file",
    )
    wagon_file = st.file_uploader(
        "🚛 Hordenwagen File (.xlsm, .xlsx)",
        type=["xlsm", "xlsx"],
        help="Upload the wagon tracking Excel file",
    )

    st.markdown("---")
    st.subheader("⚙️ Filters")

    products = st.multiselect(
        "🧱 Product(s):",
        ["L28", "L30", "L34", "L36", "L38", "L42", "L44", "N40", "N44", "Y44"],
        default=["L36", "L38", "N40"],
        help="Select products to include in the analysis",
    )

    month = st.number_input(
        "📅 Month (0 = all):",
        min_value=0,
        max_value=12,
        value=0,
        help="Filter by month (1–12) or 0 for all months",
    )

    st.markdown("---")
    run_button = st.button("▶️ Run Analysis")


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def safe_format(df: pd.DataFrame, format_dict: dict) -> pd.io.formats.style.Styler:
    """Safely format dataframe, only formatting columns that exist."""
    # Filter format_dict to only include columns that exist
    existing_cols = set(df.columns)
    safe_dict = {k: v for k, v in format_dict.items() if k in existing_cols}
    
    # Fill NaN with 0 for numeric columns before formatting
    df_display = df.copy()
    for col in safe_dict.keys():
        if col in df_display.columns:
            if df_display[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                df_display[col] = df_display[col].fillna(0)
    
    return df_display.style.format(safe_dict)


def create_kpi_card(title: str, value, unit: str) -> str:
    """Return HTML for a KPI card."""
    if value is None or (isinstance(value, float) and (pd.isna(value) or np.isnan(value))):
        text = "–"
        unit_str = ""
    else:
        try:
            text = f"{value:,.2f}"
            unit_str = f" {unit}"
        except:
            text = str(value)
            unit_str = f" {unit}"
    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <h2>{text}{unit_str}</h2>
    </div>
    """


def create_excel_download(results: dict) -> BytesIO:
    """Create an Excel file in memory with all result tables."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for key, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Clean sheet name
                sheet_name = key.replace("_", " ").title()[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output


def run_analysis(energy_path: str, wagon_path: str, products_filter, month_filter) -> dict:
    """Orchestrate the full KPI calculation"""
    progress = st.progress(0)
    status = st.empty()

    try:
        # 1) Energy
        status.text("🔄 Parsing energy data...")
        progress.progress(15)
        e_raw = pd.read_excel(energy_path, sheet_name=CONFIG["energy_sheet"])
        e = parse_energy(e_raw)
        if e.empty:
            raise ValueError("Parsed energy data is empty.")

        # 2) Wagons
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

        # 3) Filters
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
                raise ValueError(f"No energy/wagon data found for month = {month_filter}.")

        # 4) Intervals
        status.text("🔄 Building zone intervals...")
        progress.progress(65)
        ivals = explode_intervals(w)
        if ivals.empty:
            raise ValueError("Zone intervals could not be created (empty result).")

        # 5) Allocation
        status.text("🔄 Allocating energy to products...")
        progress.progress(80)
        alloc = allocate_energy(e, ivals)
        if alloc.empty:
            raise ValueError("Energy allocation result is empty.")

        # 6) Summaries
        status.text("🔄 Aggregating KPIs...")
        progress.progress(90)

        summary = alloc.groupby(["Month", "Produkt", "Zone"], as_index=False).agg(
            Energy_thermal_kWh=("Energy_thermal_kWh", "sum"),
            Energy_electrical_kWh=("Energy_electrical_kWh", "sum"),
            Energy_kWh=("Energy_share_kWh", "sum"),
            Volume_m3=("m3", "sum"),
        )
        
        # Safe division
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

        # Add water KPIs
        summary = add_water_kpis(summary)
        
        # Fill NaN values
        summary = summary.fillna(0)

        # Yearly by zone
        yearly = summary.groupby(["Produkt", "Zone"], as_index=False).agg(
            Energy_thermal_kWh=("Energy_thermal_kWh", "sum"),
            Energy_electrical_kWh=("Energy_electrical_kWh", "sum"),
            Energy_kWh=("Energy_kWh", "sum"),
            Volume_m3=("Volume_m3", "sum"),
            Water_kg=("Water_kg", "sum"),
        )
        
        # Safe division for yearly
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
        
        # Product totals
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

# Reset when files change
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
        st.error("⚠️ Please upload both Energy and Hordenwagen files before running.")
    else:
        tmp_e = None
        tmp_w = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as f_e:
                f_e.write(energy_file.read())
                tmp_e = f_e.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsm") as f_w:
                f_w.write(wagon_file.read())
                tmp_w = f_w.name

            results = run_analysis(
                tmp_e,
                tmp_w,
                products if products else None,
                month if month != 0 else None,
            )
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
            # ===== KPI CARDS =====
            st.markdown('<div class="section-header">📈 Summary KPIs</div>', unsafe_allow_html=True)

            total_thermal = float(yearly["Energy_thermal_kWh"].sum())
            total_electrical = float(yearly["Energy_electrical_kWh"].sum())
            total_energy = float(yearly["Energy_kWh"].sum())
            total_volume = float(yearly["Volume_m3"].sum())
            total_water = float(yearly["Water_kg"].sum())

            avg_kwh_thermal_kg = float(yearly["kWh_thermal_per_kg"].mean()) if len(yearly) > 0 else 0
            avg_kwh_kg = float(yearly["kWh_per_kg"].mean()) if len(yearly) > 0 else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(create_kpi_card("Thermal Energy", total_thermal, "kWh"), unsafe_allow_html=True)
            with c2:
                st.markdown(create_kpi_card("Total Energy", total_energy, "kWh"), unsafe_allow_html=True)
            with c3:
                st.markdown(create_kpi_card("Total Volume", total_volume, "m³"), unsafe_allow_html=True)
            with c4:
                st.markdown(create_kpi_card("Thermal kWh/kg", avg_kwh_thermal_kg, "kWh/kg"), unsafe_allow_html=True)
            with c5:
                st.markdown(create_kpi_card("Total kWh/kg", avg_kwh_kg, "kWh/kg"), unsafe_allow_html=True)
            
            electrical_pct = (total_electrical / total_energy * 100) if total_energy > 0 else 0
            st.info(f"⚡ Electrical energy represents **{electrical_pct:.1f}%** of total energy consumption")

            # ===== PRODUCT TOTALS =====
            if product_totals is not None and not product_totals.empty:
                st.markdown('<div class="section-header">📊 Product Performance (All Zones Combined)</div>', unsafe_allow_html=True)
                
                prod_agg = product_totals.groupby("Produkt", as_index=False).agg({
                    "Energy_thermal_kWh": "sum",
                    "Energy_electrical_kWh": "sum",
                    "Energy_kWh": "sum",
                    "Volume_m3": "sum",
                    "Water_kg": "sum",
                })
                
                prod_agg["kWh_thermal_per_m3"] = np.where(
                    prod_agg["Volume_m3"] > 0,
                    prod_agg["Energy_thermal_kWh"] / prod_agg["Volume_m3"],
                    0
                )
                prod_agg["kWh_per_m3"] = np.where(
                    prod_agg["Volume_m3"] > 0,
                    prod_agg["Energy_kWh"] / prod_agg["Volume_m3"],
                    0
                )
                prod_agg["kWh_thermal_per_kg"] = np.where(
                    prod_agg["Water_kg"] > 0,
                    prod_agg["Energy_thermal_kWh"] / prod_agg["Water_kg"],
                    0
                )
                prod_agg["kWh_per_kg"] = np.where(
                    prod_agg["Water_kg"] > 0,
                    prod_agg["Energy_kWh"] / prod_agg["Water_kg"],
                    0
                )
                prod_agg["Electrical_pct"] = np.where(
                    prod_agg["Energy_kWh"] > 0,
                    (prod_agg["Energy_electrical_kWh"] / prod_agg["Energy_kWh"]) * 100,
                    0
                )
                
                prod_agg = prod_agg.fillna(0)
                
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    fig_thermal = go.Figure()
                    fig_thermal.add_trace(go.Bar(
                        name='Thermal (Gas)',
                        x=prod_agg['Produkt'],
                        y=prod_agg['Energy_thermal_kWh'],
                        marker_color='#FF6B6B'
                    ))
                    fig_thermal.add_trace(go.Bar(
                        name='Electrical',
                        x=prod_agg['Produkt'],
                        y=prod_agg['Energy_electrical_kWh'],
                        marker_color='#4ECDC4'
                    ))
                    fig_thermal.update_layout(
                        title="Total Energy by Product (Thermal + Electrical)",
                        yaxis_title="Energy (kWh)",
                        barmode='stack',
                        height=400,
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_thermal, use_container_width=True)
                
                with col_p2:
                    fig_prod_thermal = px.bar(
                        prod_agg,
                        x="Produkt",
                        y="kWh_thermal_per_kg",
                        color="kWh_thermal_per_kg",
                        color_continuous_scale="Reds",
                        title="Thermal Specific Energy by Product (kWh thermal/kg water)",
                        text_auto=".2f"
                    )
                    fig_prod_thermal.update_layout(height=400, plot_bgcolor="white")
                    st.plotly_chart(fig_prod_thermal, use_container_width=True)
                
                st.subheader("Product Energy Breakdown")
                # Display without complex styling to avoid errors
                st.dataframe(prod_agg, use_container_width=True)

            # ===== PRODUCT SPECIFICATIONS =====
            st.markdown('<div class="section-header">📐 Product Specifications & Water-Loss Formulas</div>', unsafe_allow_html=True)
            
            st.write(
                "Each product has a **linear formula** for water evaporation based on pressed thickness: "
                "**Water (g) = Slope × Thickness (mm) + Intercept**"
            )
            
            specs_data = []
            for prod, spec in PRODUCT_SPECIFICATIONS.items():
                specs_data.append({
                    "Product": prod,
                    "Final Thickness (mm)": spec["final_thickness_mm"],
                    "Pressed Thickness (mm)": spec["pressed_thickness_mm"],
                    "Volume per Plate (m³)": spec["volume_m3"],
                    "Formula": spec["formula"],
                    "Water per Plate (kg)": spec["water_per_plate_kg"],
                    "Water per m³ (kg/m³)": spec["water_per_m3_kg"],
                })
            
            specs_df = pd.DataFrame(specs_data)
            
            with st.expander("📊 View Complete Product Specifications"):
                st.dataframe(specs_df, use_container_width=True)
            
            # Water curves
            st.subheader("Water Evaporation Curves by Product")
            
            products_to_plot = st.multiselect(
                "Select products to compare:",
                list(PRODUCT_SPECIFICATIONS.keys()),
                default=["L36", "N40", "Y44"],
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
                        labels={
                            "Pressed_Thickness_mm": "Pressed Thickness (mm)",
                            "Water_per_Plate_kg": "Water per Plate (kg)"
                        }
                    )
                    
                    # Add measured points
                    for prod in products_to_plot:
                        if prod in PRODUCT_SPECIFICATIONS:
                            spec = PRODUCT_SPECIFICATIONS[prod]
                            fig_curves.add_scatter(
                                x=[spec["pressed_thickness_mm"]],
                                y=[spec["water_per_plate_kg"]],
                                mode='markers',
                                marker=dict(size=12, symbol='star'),
                                name=f"{prod} (measured)",
                                showlegend=True
                            )
                    
                    fig_curves.update_layout(height=500, plot_bgcolor="white")
                    st.plotly_chart(fig_curves, use_container_width=True)
                    
                    st.info("⭐ **Star markers** show measured values. Lines show predicted water evaporation.")

            # ===== Monthly Trends =====
            st.markdown('<div class="section-header">📊 Monthly KPI Trends</div>', unsafe_allow_html=True)

            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                fig_kwh_m3 = px.line(
                    summary,
                    x="Month",
                    y="kWh_per_m3",
                    color="Zone",
                    markers=True,
                    hover_data=["Produkt", "Energy_kWh", "Volume_m3"],
                    title="Total Energy Efficiency (kWh/m³)",
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
                        hover_data=["Produkt", "Energy_kWh", "Water_kg"],
                        title="Specific Energy (kWh/kg H₂O)",
                    )
                    fig_kwh_kg.update_layout(height=400, plot_bgcolor="white")
                    st.plotly_chart(fig_kwh_kg, use_container_width=True)

            # ===== Zone Comparison =====
            st.markdown('<div class="section-header">📉 Zone Comparison</div>', unsafe_allow_html=True)

            col_z1, col_z2 = st.columns(2)
            with col_z1:
                fig_zone = px.bar(
                    yearly,
                    x="Zone",
                    y="kWh_per_m3",
                    color="Produkt",
                    text_auto=".2f",
                    title="Yearly KPI by Zone (kWh/m³)",
                )
                fig_zone.update_layout(height=400, plot_bgcolor="white")
                st.plotly_chart(fig_zone, use_container_width=True)

            with col_z2:
                fig_pie = px.pie(
                    yearly,
                    values="Energy_kWh",
                    names="Zone",
                    title="Energy Distribution by Zone",
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

            # ===== Data Tables =====
            with st.expander("📋 View Detailed Data Tables"):
                tab1, tab2, tab3 = st.tabs(["Monthly Summary", "Yearly Summary", "Product Totals"])
                
                with tab1:
                    st.dataframe(summary, use_container_width=True)
                
                with tab2:
                    st.dataframe(yearly, use_container_width=True)
                
                with tab3:
                    if product_totals is not None:
                        st.dataframe(product_totals, use_container_width=True)

            # ===== Weekly Prediction =====
            st.markdown('<div class="section-header">🔮 Weekly Energy Prediction</div>', unsafe_allow_html=True)

            wagon_stats = compute_product_wagon_stats(results["wagons"])
            wagon_capacity = wagon_stats.get("wagon_capacity_m3", {})
            residence_days = wagon_stats.get("residence_days", {})

            with st.form("weekly_prediction_form"):
                st.write("### Planned Wagons per Week")
                planned_wagons = {}

                prod_left = ["L28", "L30", "L34", "L36"]
                prod_mid = ["L38", "L42", "L44"]
                prod_right = ["N40", "N44", "Y44"]

                col1, col2, col3 = st.columns(3)

                with col1:
                    for p in prod_left:
                        planned_wagons[p] = st.number_input(
                            f"{p} wagons/week", 
                            min_value=0, 
                            value=0, 
                            step=10,
                            key=f"weekly_{p}"
                        )

                with col2:
                    for p in prod_mid:
                        planned_wagons[p] = st.number_input(
                            f"{p} wagons/week", 
                            min_value=0, 
                            value=0, 
                            step=10,
                            key=f"weekly_{p}"
                        )

                with col3:
                    for p in prod_right:
                        planned_wagons[p] = st.number_input(
                            f"{p} wagons/week", 
                            min_value=0, 
                            value=0, 
                            step=10,
                            key=f"weekly_{p}"
                        )

                st.write("### Baseline KPIs")
                avg_kwh_m3 = float(yearly["kWh_per_m3"].mean()) if len(yearly) > 0 else 0.0
                avg_kwh_kg = float(yearly["kWh_per_kg"].mean()) if len(yearly) > 0 else 0.0
                
                base_kwh_m3 = st.number_input(
                    "Baseline kWh/m³",
                    min_value=0.0,
                    value=avg_kwh_m3,
                )
                base_kwh_kg = st.number_input(
                    "Baseline kWh/kg",
                    min_value=0.0,
                    value=avg_kwh_kg,
                )

                submitted_weekly = st.form_submit_button("Calculate Prediction")

            if submitted_weekly:
                product_volumes = {}
                for prod, wagons in planned_wagons.items():
                    if wagons > 0:
                        capacity = wagon_capacity.get(prod, 1.5)
                        product_volumes[prod] = wagons * capacity
                
                if product_volumes:
                    detailed_pred = predict_production_energy(
                        product_volumes_m3=product_volumes,
                        baseline_kwh_per_m3=base_kwh_m3,
                        baseline_kwh_per_kg=base_kwh_kg,
                        use_formulas=True
                    )
                    
                    st.subheader("📊 Weekly Prediction Results")
                    
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.metric("Total Volume (m³/week)", f"{detailed_pred['total_volume_m3']:,.2f}")
                    with r2:
                        st.metric("Total Water (kg/week)", f"{detailed_pred['total_water_kg']:,.2f}")
                    with r3:
                        if detailed_pred.get("total_energy_kwh", 0) > 0:
                            st.metric("Total Energy (kWh/week)", f"{detailed_pred['total_energy_kwh']:,.0f}")
                    
                    if detailed_pred.get("products"):
                        product_breakdown = pd.DataFrame(detailed_pred["products"])
                        st.dataframe(product_breakdown, use_container_width=True)
                    
                    st.success("✅ Weekly energy prediction completed.")
                else:
                    st.warning("Please enter wagon counts for at least one product.")

            # ===== Export =====
            st.markdown('<div class="section-header">📥 Export Results</div>', unsafe_allow_html=True)

            excel_data = create_excel_download(results)
            st.download_button(
                label="📥 Download Complete Excel Report",
                data=excel_data,
                file_name="Dryer_KPI_Analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.success("✅ Analysis complete! Explore the visualizations above or download the full report.")

    except Exception as display_error:
        st.error(f"❌ Error displaying results: {display_error}")
        with st.expander("🔍 View Error Details"):
            st.exception(display_error)
        
        # Still show raw data
        st.subheader("📋 Raw Data (Fallback View)")
        if "summary" in results:
            st.write("Summary Data:")
            st.dataframe(results["summary"])
        if "yearly" in results:
            st.write("Yearly Data:")
            st.dataframe(results["yearly"])
