import streamlit as st
import pandas as pd
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
from io import BytesIO

# Import the KPI calculation module
try:
    from dryer_kpi_monthly_final import (
        parse_energy, parse_wagon, explode_intervals, 
        allocate_energy, parse_waterloss, calculate_waterloss_metrics,
        merge_energy_water, compute_kpis, CONFIG
    )
except ImportError:
    st.error("❌ Unable to import dryer_kpi_monthly_final module")
    st.stop()

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Lindner Dryer KPI Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom CSS ------------------
st.markdown("""
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
    .stDownloadButton button {
        background-color: #003366;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown('<div class="main-title">🏭 Lindner – Dryer KPI Monitoring Dashboard</div>', 
            unsafe_allow_html=True)

st.info("📊 Upload Energy, Hordenwagen, and Water Loss files to analyze efficiency.")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.image("https://www.karrieretag.org/wp-content/uploads/2023/10/lindner-logo-1.png", 
             use_column_width=True)
    st.markdown("---")
    
    st.subheader("📁 Data Upload")
    energy_file = st.file_uploader("📊 Energy File (.xlsx)", type=["xlsx"])
    wagon_file = st.file_uploader("🚛 Hordenwagen File (.xlsm, .xlsx)", type=["xlsm", "xlsx"])
    water_file = st.file_uploader("💧 Water Loss File (.xlsx)", type=["xlsx"])
    
    st.markdown("---")
    st.subheader("⚙️ Filters")
    
    products = st.multiselect(
        "🧱 Product(s):",
        ["L30", "L32", "L34", "L36", "L38", "L40", "N40", "N44"],
        default=["L30", "L32", "L34", "L36", "L38", "L40", "N40", "N44"]
    )
    
    month = st.number_input("📅 Month (0 = all):", min_value=0, max_value=12, value=0)
    
    st.markdown("---")
    run_button = st.button("▶️ Run Analysis", use_container_width=True)

# ------------------ Helper Functions ------------------
def create_kpi_card(title, value, unit):
    return f'''
    <div class="metric-card">
        <h3>{title}</h3>
        <h2>{value:,.2f} {unit}</h2>
    </div>
    '''

def create_excel_download(results):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        results['summary'].to_excel(writer, sheet_name="Monthly_Summary", index=False)
        results['yearly'].to_excel(writer, sheet_name="Yearly_Summary", index=False)
        results['kpi_merged'].to_excel(writer, sheet_name="Detailed_KPIs", index=False)
        if not results['water_metrics'].empty:
            results['water_metrics'].to_excel(writer, sheet_name="Water_Samples_Raw", index=False)
        results['allocation'].to_excel(writer, sheet_name="Energy_Allocation", index=False)
    output.seek(0)
    return output

def run_analysis(energy_path, wagon_path, water_path, products_filter, month_filter):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Energy
        status_text.text("🔄 Parsing energy data...")
        progress_bar.progress(10)
        e = parse_energy(pd.read_excel(energy_path, sheet_name=CONFIG["energy_sheet"]))
        
        # 2. Wagons
        status_text.text("🔄 Parsing wagon data...")
        progress_bar.progress(30)
        w = parse_wagon(pd.read_excel(wagon_path, sheet_name=CONFIG["wagon_sheet"], header=CONFIG["wagon_header_row"]))
        
        # 3. Water Loss (Optional but recommended)
        water_metrics = pd.DataFrame()
        if water_path:
            status_text.text("🔄 Parsing water loss data...")
            progress_bar.progress(40)
            w_loss_raw = parse_waterloss(water_path)
            water_metrics = calculate_waterloss_metrics(w_loss_raw)

        # 4. Filtering
        if products_filter:
            w = w[w["Produkt"].astype(str).isin(products_filter)]
        if month_filter:
            e = e[e["Month"] == month_filter]
            w = w[w["Month"] == month_filter]

        # 5. Allocation
        status_text.text("🔄 Allocating energy...")
        progress_bar.progress(60)
        ivals = explode_intervals(w)
        alloc = allocate_energy(e, ivals)
        
        # 6. Merging Water Data
        status_text.text("🔄 Merging physics data...")
        progress_bar.progress(80)
        
        # If we have water data, we merge it. If not, we create a dummy merge for code stability
        if not water_metrics.empty:
            merged_data = merge_energy_water(alloc, water_metrics)
            kpi_df = compute_kpis(merged_data)
        else:
            # Fallback if no water file
            st.warning("⚠️ No Water File uploaded. Water-based KPIs will be 0.")
            alloc['total_water_kg'] = 0
            merged_data = alloc
            kpi_df = compute_kpis(merged_data)

        # 7. Summaries
        summary = alloc.groupby(["Month", "Produkt", "Zone"], as_index=False).agg(
            Energy_kWh=("Energy_share_kWh", "sum"),
            Volume_m3=("m3", "sum")
        )
        summary["kWh_per_m3"] = summary["Energy_kWh"] / summary["Volume_m3"].replace(0, pd.NA)
        
        yearly = summary.groupby(["Produkt", "Zone"], as_index=False).agg(
            Energy_kWh=("Energy_kWh", "sum"),
            Volume_m3=("Volume_m3", "sum")
        )
        yearly["kWh_per_m3"] = yearly["Energy_kWh"] / yearly["Volume_m3"].replace(0, pd.NA)

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        return {
            'summary': summary,
            'yearly': yearly,
            'water_metrics': water_metrics,
            'kpi_merged': kpi_df,
            'allocation': alloc
        }
        
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ------------------ Main Processing ------------------

if 'results' not in st.session_state:
    st.session_state.results = None

if run_button:
    if not energy_file or not wagon_file:
        st.error("⚠️ Energy and Wagon files are mandatory!")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as te, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".xlsm") as tw:
            te.write(energy_file.read())
            tw.write(wagon_file.read())
            
            # Handle Water File (passed as object or temp file)
            # Since parse_waterloss takes a file object or path, we can pass the BytesIO directly if not None
            
            results = run_analysis(
                te.name, tw.name, water_file, 
                products if products else None, 
                month if month != 0 else None
            )
            st.session_state.results = results

# Display Logic
if st.session_state.results:
    res = st.session_state.results
    kpi = res['kpi_merged']
    water = res['water_metrics']
    
    # --- KPI Cards ---
    st.markdown('<div class="section-header">📈 Efficiency KPIs</div>', unsafe_allow_html=True)
    
    # Aggregates
    total_e = kpi['Energy_share_kWh'].sum()
    total_w = kpi['total_water_kg'].sum()
    total_v = kpi['m3'].sum()
    
    avg_kwh_m3 = total_e / total_v if total_v else 0
    avg_kwh_kg = total_e / total_w if total_w else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(create_kpi_card("Total Energy", total_e, "kWh"), unsafe_allow_html=True)
    c2.markdown(create_kpi_card("Total Water Evap.", total_w/1000, "Tons"), unsafe_allow_html=True)
    c3.markdown(create_kpi_card("Spec. Energy (Vol)", avg_kwh_m3, "kWh/m³"), unsafe_allow_html=True)
    c4.markdown(create_kpi_card("Spec. Energy (H2O)", avg_kwh_kg, "kWh/kg"), unsafe_allow_html=True)
    
    # --- Plots ---
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💦 Water Removal Density (kg/m³) by Product")
        if not water.empty:
            fig_w = px.box(water, x="Produkt", y="water_per_m3", color="Produkt",
                          title="Measured Water Loss per m³ (Sample Data)")
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.info("Upload Water Loss file to see density analytics.")
            
    with col2:
        st.markdown("### ⚡ Efficiency: kWh per kg Water")
        if not kpi.empty:
            # Aggregate by product for bar chart
            kpi_prod = kpi.groupby('Produkt')[['Energy_share_kWh', 'total_water_kg']].sum().reset_index()
            kpi_prod['kwh_kg'] = kpi_prod['Energy_share_kWh'] / kpi_prod['total_water_kg']
            
            fig_e = px.bar(kpi_prod, x="Produkt", y="kwh_kg", color="kwh_kg",
                           title="Energy required to evaporate 1kg Water",
                           color_continuous_scale="RdYlGn_r") # Red is high (bad), Green is low (good)
            fig_e.add_hline(y=1.0, line_dash="dot", annotation_text="Target (1.0)")
            st.plotly_chart(fig_e, use_container_width=True)

    # --- Data Tables ---
    with st.expander("📋 View KPI Details"):
        st.dataframe(kpi.style.format("{:.2f}"))
        
    # --- Export ---
    st.download_button(
        "📥 Download Full Excel Report",
        data=create_excel_download(res),
        file_name="Dryer_KPI_Analysis_Water_Integrated.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
