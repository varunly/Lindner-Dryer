import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os

try:
    from dryer_kpi_monthly_final import (
        parse_energy, parse_wagon, explode_intervals, allocate_energy,
        add_water_kpis, compute_product_wagon_stats, predict_production_energy,
        get_product_water_curve, WATER_PER_M3_KG, PRODUCT_SPECIFICATIONS,
        SUSPENSION_KG, CONFIG,
    )
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

st.set_page_config(page_title="Lindner Dryer KPI", page_icon="🏭", layout="wide")

st.markdown("""
<style>
.main-title { font-size: 36px; color: #003366; font-weight: 700; text-align: center; margin-bottom: 20px; }
.section-header { color: #003366; font-size: 22px; font-weight: 600; margin-top: 40px; margin-bottom: 20px; border-bottom: 2px solid #003366; padding-bottom: 6px; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); color: white; }
.metric-card h3 { margin: 0; font-size: 16px; opacity: 0.9; }
.metric-card h2 { margin: 10px 0 0 0; font-size: 28px; font-weight: 700; }
.debug-box { background: #fffde7; border: 2px solid #ffc107; padding: 15px; border-radius: 8px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏭 Lindner – Dryer KPI Dashboard</div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("📁 Upload")
    energy_file = st.file_uploader("📊 Energy (.xlsx)", type=["xlsx"])
    wagon_file = st.file_uploader("🚛 Wagon (.xlsm/.xlsx)", type=["xlsm", "xlsx"])
    st.markdown("---")
    st.subheader("⚙️ Filters")
    products = st.multiselect("Products", list(PRODUCT_SPECIFICATIONS.keys()), default=["L36", "L38", "N40"])
    month = st.number_input("Month (0=all)", 0, 12, 0)
    st.markdown("---")
    show_debug = st.checkbox("🐛 Show Debug Info", value=True)
    run_button = st.button("▶️ Run Analysis")


def create_kpi_card(title, value, unit):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f'<div class="metric-card"><h3>{title}</h3><h2>–</h2></div>'
    return f'<div class="metric-card"><h3>{title}</h3><h2>{value:,.1f} {unit}</h2></div>'


def run_analysis(energy_path, wagon_path, products_filter, month_filter):
    progress = st.progress(0)
    status = st.empty()
    debug_info = {}

    try:
        status.text("🔄 Reading energy file...")
        progress.progress(10)
        e_raw = pd.read_excel(energy_path, sheet_name=CONFIG["energy_sheet"])
        debug_info["energy_columns"] = list(e_raw.columns)
        debug_info["energy_rows"] = len(e_raw)
        
        # Show first few values of key columns
        for col in e_raw.columns:
            if 'gas' in col.lower() or 'elektr' in col.lower():
                debug_info[f"sample_{col}"] = e_raw[col].head(5).tolist()
        
        status.text("🔄 Parsing energy...")
        progress.progress(20)
        e = parse_energy(e_raw)
        
        # Capture totals from raw energy
        debug_info["raw_thermal_total"] = sum(
            e[f"E_thermal_{z}_kWh"].sum() 
            for z in ["Zone 2", "Zone 3", "Zone 4", "Zone 5"] 
            if f"E_thermal_{z}_kWh" in e.columns
        )
        debug_info["raw_electrical_total"] = e["E_el_kWh"].sum() if "E_el_kWh" in e.columns else 0

        status.text("🔄 Reading wagon file...")
        progress.progress(30)
        w_raw = pd.read_excel(wagon_path, sheet_name=CONFIG["wagon_sheet"], header=CONFIG["wagon_header_row"])
        
        status.text("🔄 Parsing wagons...")
        progress.progress(40)
        w = parse_wagon(w_raw)

        status.text("🔄 Filtering...")
        progress.progress(50)
        if products_filter:
            w = w[w["Produkt"].isin(products_filter)]
        if month_filter:
            e = e[e["Month"] == month_filter]
            w = w[w["Month"] == month_filter]

        status.text("🔄 Building intervals...")
        progress.progress(60)
        ivals = explode_intervals(w)

        status.text("🔄 Allocating energy...")
        progress.progress(80)
        alloc = allocate_energy(e, ivals)
        
        # Capture allocation results
        debug_info["alloc_thermal_total"] = alloc["Energy_thermal_kWh"].sum() if "Energy_thermal_kWh" in alloc.columns else 0
        debug_info["alloc_electrical_total"] = alloc["Energy_electrical_kWh"].sum() if "Energy_electrical_kWh" in alloc.columns else 0
        debug_info["alloc_rows"] = len(alloc)

        status.text("🔄 Aggregating...")
        progress.progress(90)

        summary = alloc.groupby(["Month", "Produkt", "Zone"], as_index=False).agg(
            Energy_thermal_kWh=("Energy_thermal_kWh", "sum"),
            Energy_electrical_kWh=("Energy_electrical_kWh", "sum"),
            Energy_kWh=("Energy_share_kWh", "sum"),
            Volume_m3=("m3", "sum"),
        )
        summary["kWh_per_m3"] = np.where(summary["Volume_m3"] > 0, summary["Energy_kWh"] / summary["Volume_m3"], 0)
        summary = add_water_kpis(summary).fillna(0)

        yearly = summary.groupby(["Produkt", "Zone"], as_index=False).agg({
            "Energy_thermal_kWh": "sum", "Energy_electrical_kWh": "sum",
            "Energy_kWh": "sum", "Volume_m3": "sum", "Water_kg": "sum",
        })
        yearly["kWh_per_m3"] = np.where(yearly["Volume_m3"] > 0, yearly["Energy_kWh"] / yearly["Volume_m3"], 0)
        yearly["kWh_per_kg"] = np.where(yearly["Water_kg"] > 0, yearly["Energy_kWh"] / yearly["Water_kg"], 0)

        product_totals = summary.groupby(["Month", "Produkt"], as_index=False).agg({
            "Energy_thermal_kWh": "sum", "Energy_electrical_kWh": "sum",
            "Energy_kWh": "sum", "Volume_m3": "sum", "Water_kg": "sum",
        })

        progress.progress(100)
        status.text("✅ Done!")

        return {
            "energy": e, "wagons": w, "intervals": ivals, "allocation": alloc,
            "summary": summary, "yearly": yearly, "product_totals": product_totals,
            "debug": debug_info
        }
    finally:
        import time
        time.sleep(0.3)
        progress.empty()
        status.empty()


if "results" not in st.session_state:
    st.session_state.results = None

if run_button:
    if not energy_file or not wagon_file:
        st.error("⚠️ Upload both files first.")
    else:
        tmp_e = tmp_w = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as f:
                f.write(energy_file.read())
                tmp_e = f.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsm") as f:
                f.write(wagon_file.read())
                tmp_w = f.name
            st.session_state.results = run_analysis(tmp_e, tmp_w, products or None, month or None)
        except Exception as err:
            st.error(f"❌ Error: {err}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state.results = None
        finally:
            for p in (tmp_e, tmp_w):
                if p and os.path.exists(p):
                    os.unlink(p)


if st.session_state.results:
    r = st.session_state.results
    yearly = r["yearly"]
    summary = r["summary"]
    product_totals = r["product_totals"]
    debug = r.get("debug", {})

    # ===== DEBUG INFO =====
    if show_debug:
        st.markdown('<div class="section-header">🐛 Debug Information</div>', unsafe_allow_html=True)
        
        with st.expander("📊 Raw Data Analysis", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Energy File Columns:**")
                for i, col in enumerate(debug.get("energy_columns", [])):
                    if 'gas' in col.lower():
                        st.write(f"  🔥 `{col}`")
                    elif 'elektr' in col.lower():
                        st.write(f"  ⚡ `{col}`")
                    else:
                        st.write(f"  📄 `{col}`")
            
            with col2:
                st.write("**Raw Energy Totals (before allocation):**")
                raw_th = debug.get("raw_thermal_total", 0)
                raw_el = debug.get("raw_electrical_total", 0)
                raw_total = raw_th + raw_el
                
                st.metric("Raw Thermal", f"{raw_th:,.0f} kWh")
                st.metric("Raw Electrical", f"{raw_el:,.0f} kWh")
                
                if raw_total > 0:
                    st.write(f"**Thermal %:** {raw_th/raw_total*100:.1f}%")
                    st.write(f"**Electrical %:** {raw_el/raw_total*100:.1f}%")
                    if raw_el > 0:
                        st.write(f"**Ratio:** {raw_th/raw_el:.1f}x")
        
        with st.expander("📈 Allocation Results"):
            alloc_th = debug.get("alloc_thermal_total", 0)
            alloc_el = debug.get("alloc_electrical_total", 0)
            alloc_total = alloc_th + alloc_el
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Allocated Thermal", f"{alloc_th:,.0f} kWh")
            col2.metric("Allocated Electrical", f"{alloc_el:,.0f} kWh")
            col3.metric("Total Allocated", f"{alloc_total:,.0f} kWh")
            
            if alloc_total > 0:
                st.write(f"**After allocation - Thermal:** {alloc_th/alloc_total*100:.1f}% | **Electrical:** {alloc_el/alloc_total*100:.1f}%")

    # ===== 1. SUMMARY KPIs =====
    st.markdown('<div class="section-header">📈 Summary KPIs</div>', unsafe_allow_html=True)
    
    total_thermal = yearly["Energy_thermal_kWh"].sum()
    total_electrical = yearly["Energy_electrical_kWh"].sum()
    total_energy = yearly["Energy_kWh"].sum()
    total_volume = yearly["Volume_m3"].sum()
    total_water = yearly["Water_kg"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(create_kpi_card("Thermal", total_thermal, "kWh"), unsafe_allow_html=True)
    c2.markdown(create_kpi_card("Electrical", total_electrical, "kWh"), unsafe_allow_html=True)
    c3.markdown(create_kpi_card("Total", total_energy, "kWh"), unsafe_allow_html=True)
    c4.markdown(create_kpi_card("Volume", total_volume, "m³"), unsafe_allow_html=True)
    c5.markdown(create_kpi_card("Water", total_water, "kg"), unsafe_allow_html=True)

    if total_energy > 0:
        th_pct = total_thermal / total_energy * 100
        el_pct = total_electrical / total_energy * 100
        ratio = total_thermal / total_electrical if total_electrical > 0 else 0
        
        if th_pct > 80:
            st.success(f"✅ Energy Split: 🔥 Thermal **{th_pct:.1f}%** | ⚡ Electrical **{el_pct:.1f}%** | Ratio: **{ratio:.1f}x**")
        else:
            st.warning(f"⚠️ Energy Split: 🔥 Thermal **{th_pct:.1f}%** | ⚡ Electrical **{el_pct:.1f}%** | Ratio: **{ratio:.1f}x** (Expected: Thermal > 85%)")

    # ===== 2. ZONE COMPARISON =====
    st.markdown('<div class="section-header">📉 Zone Comparison</div>', unsafe_allow_html=True)

    zone_totals = yearly.groupby("Zone", as_index=False).agg({
        "Energy_thermal_kWh": "sum", "Energy_electrical_kWh": "sum", "Energy_kWh": "sum", "Volume_m3": "sum",
    })

    z1, z2 = st.columns(2)
    with z1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Thermal', x=zone_totals['Zone'], y=zone_totals['Energy_thermal_kWh'], marker_color='#FF6B6B'))
        fig.add_trace(go.Bar(name='Electrical', x=zone_totals['Zone'], y=zone_totals['Energy_electrical_kWh'], marker_color='#4ECDC4'))
        fig.update_layout(title="Energy by Zone", barmode='stack', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with z2:
        fig = px.pie(zone_totals, values="Energy_kWh", names="Zone", title="Zone Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # ===== 3. PRODUCT PERFORMANCE =====
    st.markdown('<div class="section-header">📊 Product Performance</div>', unsafe_allow_html=True)

    prod_agg = product_totals.groupby("Produkt", as_index=False).agg({
        "Energy_thermal_kWh": "sum", "Energy_electrical_kWh": "sum", "Energy_kWh": "sum", "Volume_m3": "sum", "Water_kg": "sum",
    })

    p1, p2 = st.columns(2)
    with p1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Thermal', x=prod_agg['Produkt'], y=prod_agg['Energy_thermal_kWh'], marker_color='#FF6B6B'))
        fig.add_trace(go.Bar(name='Electrical', x=prod_agg['Produkt'], y=prod_agg['Energy_electrical_kWh'], marker_color='#4ECDC4'))
        fig.update_layout(title="Total Energy by Product", barmode='stack', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        fig = px.bar(prod_agg, x="Produkt", y="Energy_thermal_kWh", color="Energy_thermal_kWh",
                     color_continuous_scale="Oranges", title="Thermal Energy by Product", text_auto=",.0f")
        fig.update_layout(height=400, yaxis_title="Thermal (kWh)")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(prod_agg, use_container_width=True, hide_index=True)

    # ===== 4. EXPORT =====
    st.markdown('<div class="section-header">📥 Export</div>', unsafe_allow_html=True)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in r.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=name[:31], index=False)
    output.seek(0)
    st.download_button("📥 Download Excel", output, "KPI_Report.xlsx")
    
    st.success("✅ Analysis complete!")
