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
        # ✅ ADD VOLUME VALIDATION HERE
        total_volume_raw = w["m3"].sum()
        total_wagons = len(w)
        avg_volume_per_wagon = w["m3"].mean()
        
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
                                    # ===== 1. SUMMARY KPIs (FIXED) =====
            st.markdown('<div class="section-header">📈 Summary KPIs</div>', unsafe_allow_html=True)

            # ✅ FIX: Use yearly for energy, but wagons for volume to avoid double-counting
            total_thermal = float(yearly["Energy_thermal_kWh"].sum())
            total_electrical = float(yearly["Energy_electrical_kWh"].sum())
            total_energy = float(yearly["Energy_kWh"].sum())
            
            # ✅ CORRECT: Get volume from wagon data (not yearly which has zone duplication)
            total_volume = float(results["wagons"]["m3"].sum())
            
            # Water calculation - use the summary data aggregated by product only
            product_summary_for_water = summary.groupby("Produkt", as_index=False).agg({
                "Volume_m3": "sum",
                "Water_kg": "sum",
            })
            total_water = float(product_summary_for_water["Water_kg"].sum())
            
            # Calculate KPIs
            avg_kwh_per_m3 = safe_divide(total_energy, total_volume)
            avg_kwh_thermal_per_m3 = safe_divide(total_thermal, total_volume)
            avg_kwh_per_kg = safe_divide(total_energy, total_water)
            avg_kwh_thermal_per_kg = safe_divide(total_thermal, total_water)

            # Row 1: Energy Metrics
            st.subheader("⚡ Energy Consumption")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(create_kpi_card("Thermal Energy", total_thermal, "kWh"), unsafe_allow_html=True)
            with c2:
                st.markdown(create_kpi_card("Electrical Energy", total_electrical, "kWh"), unsafe_allow_html=True)
            with c3:
                st.markdown(create_kpi_card("Total Energy", total_energy, "kWh"), unsafe_allow_html=True)
            
            # Row 2: Production & Water Metrics
            st.subheader("🏭 Production")
            c4, c5, c6 = st.columns(3)
            with c4:
                st.markdown(create_kpi_card("Total Volume of products", total_volume, "m³"), unsafe_allow_html=True)
            with c5:
                st.markdown(create_kpi_card("Total Water Evaporated", total_water, "kg"), unsafe_allow_html=True)
            with c6:
                water_per_m3 = safe_divide(total_water, total_volume)
                st.markdown(create_kpi_card("Water/m³", water_per_m3, "kg/m³"), unsafe_allow_html=True)
            
                        
            # Summary info box
            thermal_pct = (total_thermal / total_energy * 100) if total_energy > 0 else 0
            electrical_pct = (total_electrical / total_energy * 100) if total_energy > 0 else 0
            total_wagons = len(results["wagons"])
            
            st.info(
                f"⚡ **Energy Mix:** Thermal = **{thermal_pct:.1f}%** ({total_thermal:,.0f} kWh) | "
                f"Electrical = **{electrical_pct:.1f}%** ({total_electrical:,.0f} kWh) | "
                f"🚛 **Production:** {total_wagons:,} wagons | {total_volume:,.0f} m³ | "
                f"💧 **Water:** {total_water:,.0f} kg ({total_water/1000:,.1f} tons) evaporated"
            
            )
            
            # Row 4: Energy Intensity (per volume)
            st.subheader("📦 Energy per Volume")
            c9, c10 = st.columns(2)
            with c9:
                st.markdown(create_kpi_card("Total kWh/m³", avg_kwh_per_m3, "kWh/m³"), unsafe_allow_html=True)
            with c10:
                st.markdown(create_kpi_card("Thermal kWh/m³", avg_kwh_thermal_per_m3, "kWh/m³"), unsafe_allow_html=True)
            # After Summary KPIs section, add this:
            
            # ===== VOLUME BREAKDOWN & VALIDATION =====
            with st.expander("📊 Volume Breakdown & Validation"):
                st.subheader("Volume Statistics")
                
                # Overall statistics
                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                
                total_wagons_filtered = len(results["wagons"])
                total_volume_filtered = results["wagons"]["m3"].sum()
                avg_volume_wagon = results["wagons"]["m3"].mean()
                
                with col_v1:
                    st.metric("Total Wagons", f"{total_wagons_filtered:,}")
                with col_v2:
                    st.metric("Total Volume", f"{total_volume_filtered:,.0f} m³")
                with col_v3:
                    st.metric("Avg Volume/Wagon", f"{avg_volume_wagon:.2f} m³")
                with col_v4:
                    expected_range = "12,000-15,000 m³"
                    in_range = 12000 <= total_volume_filtered <= 15000
                    st.metric("Status", "✅ In Range" if in_range else "⚠️ Out of Range")
                
                # Volume by product
                st.subheader("Volume by Product")
                vol_by_product = results["wagons"].groupby("Produkt").agg({
                    "m3": ["sum", "mean", "count"]
                }).round(2)
                vol_by_product.columns = ["Total (m³)", "Avg/Wagon (m³)", "Wagon Count"]
                vol_by_product = vol_by_product.sort_values("Total (m³)", ascending=False)
                
                # Add percentage
                vol_by_product["% of Total"] = (vol_by_product["Total (m³)"] / total_volume_filtered * 100).round(1)
                
                st.dataframe(vol_by_product, use_container_width=True)
                
                # Volume by month
                st.subheader("Volume by Month")
                vol_by_month = results["wagons"].groupby("Month").agg({
                    "m3": "sum",
                    "WG_Nr": "count"
                }).round(0)
                vol_by_month.columns = ["Volume (m³)", "Wagons"]
                vol_by_month["Avg m³/Wagon"] = (vol_by_month["Volume (m³)"] / vol_by_month["Wagons"]).round(2)
                
                st.dataframe(vol_by_month, use_container_width=True)
                
                # Theoretical vs Actual comparison
                st.subheader("Theoretical vs Actual Volume per Wagon")
                
                comparison_data = []
                for prod in results["wagons"]["Produkt"].unique():
                    if prod in PRODUCT_SPECIFICATIONS:
                        spec = PRODUCT_SPECIFICATIONS[prod]
                        
                        # Theoretical volume
                        edge_m = spec["edge_length_mm"] / 1000.0
                        thick_m = spec["final_thickness_mm"] / 1000.0
                        vol_theoretical = 234 * edge_m * edge_m * thick_m
                        
                        # Actual volume
                        actual_vols = results["wagons"][results["wagons"]["Produkt"] == prod]["m3"]
                        vol_actual = actual_vols.mean()
                        
                        difference = vol_actual - vol_theoretical
                        diff_pct = (difference / vol_theoretical * 100) if vol_theoretical > 0 else 0
                        
                        comparison_data.append({
                            "Product": prod,
                            "Theoretical (m³)": round(vol_theoretical, 3),
                            "Actual (m³)": round(vol_actual, 3),
                            "Difference (m³)": round(difference, 3),
                            "Difference (%)": round(diff_pct, 1),
                            "Wagons": len(actual_vols)
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Flag large discrepancies
                large_diff = comparison_df[abs(comparison_df["Difference (%)"]) > 10]
                if not large_diff.empty:
                    st.warning(
                        f"⚠️ **Large discrepancies detected** (>10%) for products: {', '.join(large_diff['Product'].tolist())}"
                    )
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

                        # ===== 5. MONTHLY & WEEKLY TRENDS (FIXED) =====
            st.markdown('<div class="section-header">📊 Monthly & Weekly KPI Trends</div>', unsafe_allow_html=True)
            
            # Prepare aggregated data for different views
            # 1. By Product (across all zones)
            monthly_product = summary.groupby(["Month", "Produkt"], as_index=False).agg({
                "Energy_thermal_kWh": "sum",
                "Energy_electrical_kWh": "sum",
                "Energy_kWh": "sum",
                "Volume_m3": "sum",
                "Water_kg": "sum",
            })
            monthly_product["kWh_per_m3"] = safe_divide(monthly_product["Energy_kWh"], monthly_product["Volume_m3"])
            monthly_product["kWh_per_kg"] = safe_divide(monthly_product["Energy_kWh"], monthly_product["Water_kg"])
            monthly_product["kWh_thermal_per_m3"] = safe_divide(monthly_product["Energy_thermal_kWh"], monthly_product["Volume_m3"])
            
            # 2. By Zone (across all products)
            monthly_zone = summary.groupby(["Month", "Zone"], as_index=False).agg({
                "Energy_thermal_kWh": "sum",
                "Energy_electrical_kWh": "sum",
                "Energy_kWh": "sum",
                "Volume_m3": "sum",
                "Water_kg": "sum",
            })
            monthly_zone["kWh_per_m3"] = safe_divide(monthly_zone["Energy_kWh"], monthly_zone["Volume_m3"])
            monthly_zone["kWh_per_kg"] = safe_divide(monthly_zone["Energy_kWh"], monthly_zone["Water_kg"])
            monthly_zone["kWh_thermal_per_m3"] = safe_divide(monthly_zone["Energy_thermal_kWh"], monthly_zone["Volume_m3"])
            
            # 3. Overall (total across everything)
            monthly_overall = summary.groupby(["Month"], as_index=False).agg({
                "Energy_thermal_kWh": "sum",
                "Energy_electrical_kWh": "sum",
                "Energy_kWh": "sum",
                "Volume_m3": "sum",
                "Water_kg": "sum",
            })
            monthly_overall["kWh_per_m3"] = safe_divide(monthly_overall["Energy_kWh"], monthly_overall["Volume_m3"])
            monthly_overall["kWh_per_kg"] = safe_divide(monthly_overall["Energy_kWh"], monthly_overall["Water_kg"])
            monthly_overall["kWh_thermal_per_m3"] = safe_divide(monthly_overall["Energy_thermal_kWh"], monthly_overall["Volume_m3"])
            monthly_overall["Thermal_pct"] = safe_divide(monthly_overall["Energy_thermal_kWh"], monthly_overall["Energy_kWh"]) * 100
            
            # 4. Weekly data (from energy data if available) - FIXED
            if "energy" in results and not results["energy"].empty:
                energy_df = results["energy"].copy()
                energy_df["Week"] = energy_df["E_start"].dt.isocalendar().week
                energy_df["Year"] = energy_df["E_start"].dt.year
                
                # Check which columns exist and aggregate accordingly
                agg_dict = {}
                
                # Check for thermal total column
                if "E_thermal_total_kWh" in energy_df.columns:
                    agg_dict["E_thermal_total_kWh"] = "sum"
                else:
                    # If total doesn't exist, sum individual zone thermal columns
                    thermal_cols = [col for col in energy_df.columns if col.startswith("E_thermal_") and col.endswith("_kWh")]
                    for col in thermal_cols:
                        agg_dict[col] = "sum"
                
                # Electrical column
                if "E_el_kWh" in energy_df.columns:
                    agg_dict["E_el_kWh"] = "sum"
                
                if agg_dict:
                    weekly_energy = energy_df.groupby(["Year", "Week"], as_index=False).agg(agg_dict)
                    
                    # Calculate total thermal if we have individual zone columns
                    if "E_thermal_total_kWh" not in weekly_energy.columns:
                        thermal_cols = [col for col in weekly_energy.columns if col.startswith("E_thermal_") and col.endswith("_kWh")]
                        if thermal_cols:
                            weekly_energy["E_thermal_total_kWh"] = weekly_energy[thermal_cols].sum(axis=1)
                        else:
                            weekly_energy["E_thermal_total_kWh"] = 0
                    
                    # Calculate total energy
                    weekly_energy["Total_kWh"] = weekly_energy.get("E_thermal_total_kWh", 0) + weekly_energy.get("E_el_kWh", 0)
                    weekly_energy["Week_Label"] = weekly_energy["Year"].astype(str) + "-W" + weekly_energy["Week"].astype(str).str.zfill(2)
                else:
                    weekly_energy = None
            else:
                weekly_energy = None
            
            # ========== SECTION 1: OVERALL TRENDS ==========
            st.subheader("📈 Overall Performance Trends")
            
            col_o3, col_o4 = st.columns(2)
            
            with col_o3:
                # Monthly energy consumption (stacked)
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
            
            with col_o4:
                # Monthly production volume
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
            
                       # ========== SECTION 2: WEEKLY ENERGY CONSUMPTION (FIXED) ==========
            if weekly_energy is not None and not weekly_energy.empty:
                st.subheader("📅 Weekly Energy Consumption")
                
                # Ensure no negative values - clip to 0
                if "E_thermal_total_kWh" in weekly_energy.columns:
                    weekly_energy["E_thermal_total_kWh"] = weekly_energy["E_thermal_total_kWh"].clip(lower=0)
                if "E_el_kWh" in weekly_energy.columns:
                    weekly_energy["E_el_kWh"] = weekly_energy["E_el_kWh"].clip(lower=0)
                weekly_energy["Total_kWh"] = weekly_energy["Total_kWh"].clip(lower=0)
                
                col_w1, col_w2 = st.columns(2)
                
                with col_w1:
                    # Weekly total energy consumption
                    fig_weekly_total = go.Figure()
                    
                    # Properly handle column access
                    if "E_thermal_total_kWh" in weekly_energy.columns:
                        thermal_data = weekly_energy["E_thermal_total_kWh"].fillna(0).clip(lower=0)
                    else:
                        thermal_data = pd.Series([0] * len(weekly_energy), index=weekly_energy.index)
                    
                    if "E_el_kWh" in weekly_energy.columns:
                        electrical_data = weekly_energy["E_el_kWh"].fillna(0).clip(lower=0)
                    else:
                        electrical_data = pd.Series([0] * len(weekly_energy), index=weekly_energy.index)
                    
                    # Add traces only if there's actual data
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
                        yaxis=dict(rangemode='tozero')  # Force y-axis to start at 0
                    )
                    st.plotly_chart(fig_weekly_total, use_container_width=True)
                
                with col_w2:
                    # Weekly trend line
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
                        yaxis=dict(rangemode='tozero')  # Force y-axis to start at 0
                    )
                    st.plotly_chart(fig_weekly_trend, use_container_width=True)
                
                # Weekly statistics (with validation)
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
                        st.info("📊 **Weekly Statistics:** Data available but no valid maximum found")
                
                # Debug information (optional - remove in production)
                with st.expander("🔍 Debug: Check Weekly Data"):
                    st.write("Columns available:", list(weekly_energy.columns))
                    st.write("Data preview:")
                    st.dataframe(weekly_energy.head())
                    st.write(f"Thermal sum: {thermal_data.sum():,.0f}")
                    st.write(f"Electrical sum: {electrical_data.sum():,.0f}")
                    st.write(f"Total sum: {valid_total.sum():,.0f}")
            else:
                st.info("📅 Weekly energy data not available for the selected period")
            
            # ========== SECTION 3: BY PRODUCT TRENDS ==========
            st.subheader("🧱 Trends by Product")
            
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
            
            # ========== SECTION 4: BY ZONE TRENDS ==========
            st.subheader("🏭 Trends by Zone")
            
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
                    labels={"Energy_thermal_kWh": "Thermal Energy (kWh)", "Month": "Month"}
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
                    title="Production Volume by Zone (m³)",
                    labels={"Volume_m3": "Volume (m³)", "Month": "Month"}
                )
                fig_zone_volume.update_layout(height=350, plot_bgcolor="white")
                st.plotly_chart(fig_zone_volume, use_container_width=True)


            # ===== 3. PRODUCT PERFORMANCE (SIMPLIFIED) =====
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
                prod_agg["kWh_thermal_per_m3"] = np.where(prod_agg["Volume_m3"] > 0, prod_agg["Energy_thermal_kWh"] / prod_agg["Volume_m3"], 0)
                prod_agg["Thermal_pct"] = np.where(prod_agg["Energy_kWh"] > 0, (prod_agg["Energy_thermal_kWh"] / prod_agg["Energy_kWh"] * 100), 0)
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
                        xaxis_title="Product",
                        yaxis_title="Energy (kWh)",
                        barmode='stack',
                        height=400,
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_prod_energy, use_container_width=True)
                
                with col_p2:
                    # Thermal Energy Consumed by Each Product (Horizontal Bar Chart)
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
                
                # Thermal energy distribution pie chart (centered)
                col_pie1, col_pie2, col_pie3 = st.columns([1, 2, 1])
                
                with col_pie2:
                    fig_thermal_pie = px.pie(
                        prod_agg,
                        values="Energy_thermal_kWh",
                        names="Produkt",
                        title="Thermal Energy Distribution by Product (%)",
                        color_discrete_sequence=px.colors.sequential.Reds_r
                    )
                    fig_thermal_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_thermal_pie.update_layout(height=400)
                    st.plotly_chart(fig_thermal_pie, use_container_width=True)
                
                # Product summary table
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
                    "kWh_per_kg": "kWh/kg",
                    "kWh_thermal_per_m3": "Thermal kWh/m³"
                })
                
                # Select and order columns for display
                display_cols = [
                    "Product", "Thermal (kWh)", "Electrical (kWh)", "Total (kWh)", 
                    "Thermal %", "Volume (m³)", "Water (kg)", "Thermal kWh/m³", "kWh/m³", "kWh/kg"
                ]
                display_cols = [c for c in display_cols if c in prod_display.columns]
                
                st.dataframe(prod_display[display_cols], use_container_width=True, hide_index=True)
                
                # Summary metrics for thermal energy
                total_thermal_prod = prod_agg["Energy_thermal_kWh"].sum()
                avg_thermal_per_product = prod_agg["Energy_thermal_kWh"].mean()
                max_thermal_product = prod_agg.loc[prod_agg["Energy_thermal_kWh"].idxmax(), "Produkt"]
                max_thermal_value = prod_agg["Energy_thermal_kWh"].max()
                
                st.info(
                    f"🔥 **Thermal Energy Summary:** Total = **{total_thermal_prod:,.0f} kWh** | "
                    f"Average per product = **{avg_thermal_per_product:,.0f} kWh** | "
                    f"Highest consumer: **{max_thermal_product}** ({max_thermal_value:,.0f} kWh)"
                )
                

            # ===== 4. PRODUCT SPECIFICATIONS (MEASURED VALUES) =====
                        # ===== PRODUCT SPECIFICATIONS (FORMULA-BASED) =====
            st.markdown('<div class="section-header">📐 Product Specifications (Formula-Based Calculations)</div>', unsafe_allow_html=True)
            
            st.write(f"**Formula:** Water/mm (g) = Slope × Suspension ({SUSPENSION_KG} kg) + Intercept")
            st.write(f"**Then:** Water/Plate (kg) = [Water/mm (g) × Pressed Thickness (mm)] / 1000")
            
            specs_data = []
            for prod, spec in PRODUCT_SPECIFICATIONS.items():
                # Calculate using formula
                slope = spec["slope"]
                intercept = spec["intercept"]
                water_per_mm_g = slope * SUSPENSION_KG + intercept
                pressed_thickness_mm = spec["pressed_thickness_mm"]
                water_per_plate_kg = (water_per_mm_g * pressed_thickness_mm) / 1000.0
                water_per_m3_kg = water_per_plate_kg / spec["volume_m3"]
                water_per_wagon_kg = water_per_plate_kg * PLATES_PER_WAGON
                
                # Check if interpolated
                is_interpolated = spec.get("interpolated", False)
                formula_display = spec["formula"]
                if is_interpolated:
                    formula_display += " ⚠️"
                
                specs_data.append({
                    "Product": prod,
                    "Type": spec["product_type"],
                    "Formula": formula_display,
                    "Suspension (kg)": SUSPENSION_KG,
                    "Water/mm (g)": round(water_per_mm_g, 1),
                    "Pressed Thickness (mm)": pressed_thickness_mm,
                    "Water/Plate (kg)": round(water_per_plate_kg, 3),
                    "Water/Wagon (kg)": round(water_per_wagon_kg, 1),
                    "Water/m³ (kg/m³)": round(water_per_m3_kg, 1),
                })
            
            specs_df = pd.DataFrame(specs_data)
            st.dataframe(specs_df, use_container_width=True, hide_index=True)
            
            st.info("⚠️ Products marked with ⚠️ are interpolated values")
            
            # Comparison: Formula vs Measured
            st.subheader("Formula Calculation vs Measured Values")
            
            comparison_data = []
            for prod, spec in PRODUCT_SPECIFICATIONS.items():
                # Formula calculation
                slope = spec["slope"]
                intercept = spec["intercept"]
                water_per_mm_g_calc = slope * SUSPENSION_KG + intercept
                water_per_plate_kg_calc = (water_per_mm_g_calc * spec["pressed_thickness_mm"]) / 1000.0
                
                # Measured value
                water_per_plate_kg_measured = spec["water_per_plate_kg"]
                
                # Difference
                difference = water_per_plate_kg_calc - water_per_plate_kg_measured
                diff_pct = (difference / water_per_plate_kg_measured * 100) if water_per_plate_kg_measured > 0 else 0
                
                comparison_data.append({
                    "Product": prod,
                    "Measured (kg)": round(water_per_plate_kg_measured, 3),
                    "Calculated (kg)": round(water_per_plate_kg_calc, 3),
                    "Difference (kg)": round(difference, 3),
                    "Difference (%)": round(diff_pct, 1)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Check for large discrepancies
            large_diff = comparison_df[abs(comparison_df["Difference (%)"]) > 5]
            if not large_diff.empty:
                st.warning(
                    f"⚠️ **Discrepancies >5% found** for: {', '.join(large_diff['Product'].tolist())}. "
                    f"This might indicate measurement errors or formula issues."
                )
            else:
                st.success("✅ Formula calculations match measured values within 5%")

           
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






















