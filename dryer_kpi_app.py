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
        allocate_energy, parse_waterloss, build_water_benchmarks,
        CONFIG
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

st.info("📊 Upload your Energy, Hordenwagen and Water-Loss files to analyze dryer efficiency across zones, products, and evaporated water.")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.image("https://www.karrieretag.org/wp-content/uploads/2023/10/lindner-logo-1.png", 
             use_column_width=True)
    st.markdown("---")
    
    st.subheader("📁 Data Upload")
    energy_file = st.file_uploader(
        "📊 Energy File (.xlsx)", 
        type=["xlsx"],
        help="Upload the hourly energy consumption Excel file"
    )
    wagon_file = st.file_uploader(
        "🚛 Hordenwagen File (.xlsm, .xlsx)", 
        type=["xlsm", "xlsx"],
        help="Upload the wagon tracking Excel file"
    )
    water_file = st.file_uploader(
        "💧 Water-Loss File (.xlsx)",
        type=["xlsx"],
        help="Upload the Wasserverlust-Platten Excel file (optional but recommended)"
    )
    
    st.markdown("---")
    st.subheader("⚙️ Filters")
    
    products = st.multiselect(
        "🧱 Product(s):",
        ["L30", "L32", "L34", "L36", "L38", "L40", "N40", "N44"],
        default=["L30", "L32", "L34", "L36", "L38", "L40", "N40", "N44"],
        help="Select one or more products to analyze"
    )
    
    month = st.number_input(
        "📅 Month (0 = all):",
        min_value=0,
        max_value=12,
        value=0,
        help="Filter by specific month (1-12) or 0 for all months"
    )
    
    st.markdown("---")
    run_button = st.button("▶️ Run Analysis", use_container_width=True)

# ------------------ Helper Functions ------------------
def create_kpi_card(title, value, unit):
    """Create a styled KPI metric card"""
    if value is None or pd.isna(value):
        display = "–"
        unit_str = ""
    else:
        display = f"{value:,.2f}"
        unit_str = f" {unit}"
    return f'''
    <div class="metric-card">
        <h3>{title}</h3>
        <h2>{display}{unit_str}</h2>
    </div>
    '''

def create_excel_download(results):
    """Create Excel file in memory for download"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        results['energy'].to_excel(writer, sheet_name="Energy_Data", index=False)
        results['wagons'].to_excel(writer, sheet_name="Wagon_Data", index=False)
        results['intervals'].to_excel(writer, sheet_name="Zone_Intervals", index=False)
        results['allocation'].to_excel(writer, sheet_name="Energy_Allocation", index=False)
        results['summary'].to_excel(writer, sheet_name="Monthly_Summary", index=False)
        results['yearly'].to_excel(writer, sheet_name="Yearly_Summary", index=False)
        
        if 'waterloss' in results and results['waterloss'] is not None:
            results['waterloss'].to_excel(writer, sheet_name="Waterloss_Data", index=False)
        if 'water_benchmarks' in results and results['water_benchmarks'] is not None:
            results['water_benchmarks'].to_excel(writer, sheet_name="Water_Benchmarks", index=False)
    
    output.seek(0)
    return output

def run_analysis(energy_path, wagon_path, water_path, products_filter, month_filter):
    """Run the KPI analysis with progress tracking, including optional water-loss integration"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Parse energy data
        status_text.text("🔄 Parsing energy data...")
        progress_bar.progress(15)
        e_raw = pd.read_excel(energy_path, sheet_name=CONFIG["energy_sheet"])
        e = parse_energy(e_raw)
        
        if e.empty:
            raise ValueError("Energy data is empty after parsing")
        
        # Step 2: Parse wagon data
        status_text.text("🔄 Parsing wagon tracking data...")
        progress_bar.progress(30)
        w_raw = pd.read_excel(
            wagon_path, 
            sheet_name=CONFIG["wagon_sheet"], 
            header=CONFIG["wagon_header_row"]
        )
        w = parse_wagon(w_raw)
        
        if w.empty:
            raise ValueError("Wagon data is empty after parsing")
        
        # Step 3: Apply filters
        status_text.text("🔄 Applying filters...")
        progress_bar.progress(45)
        
        if products_filter:
            w = w[w["Produkt"].astype(str).isin(products_filter)]
            if w.empty:
                raise ValueError(f"No wagons found for products: {products_filter}")
        
        if month_filter:
            e = e[e["Month"] == month_filter]
            w = w[w["Month"] == month_filter]
            if e.empty or w.empty:
                raise ValueError(f"No data found for month: {month_filter}")
        
        # Step 4: Process intervals
        status_text.text("🔄 Processing zone intervals...")
        progress_bar.progress(60)
        ivals = explode_intervals(w)
        
        if ivals.empty:
            raise ValueError("No valid zone intervals could be created")
        
        # Step 5: Allocate energy
        status_text.text("🔄 Allocating energy to products...")
        progress_bar.progress(75)
        alloc = allocate_energy(e, ivals)
        
        if alloc.empty:
            raise ValueError("Energy allocation produced no results")
        
        # Step 6: Create summaries (base)
        status_text.text("🔄 Generating summaries...")
        progress_bar.progress(85)
        
        summary = alloc.groupby(["Month", "Produkt", "Zone"], as_index=False).agg(
            Energy_kWh=("Energy_share_kWh", "sum"),
            Volume_m3=("m3", "sum")
        )
        summary["kWh_per_m3"] = (
            summary["Energy_kWh"] / summary["Volume_m3"].replace(0, pd.NA)
        )
        
        yearly = summary.groupby(["Produkt", "Zone"], as_index=False).agg(
            Energy_kWh=("Energy_kWh", "sum"),
            Volume_m3=("Volume_m3", "sum")
        )
        yearly["kWh_per_m3"] = (
            yearly["Energy_kWh"] / yearly["Volume_m3"].replace(0, pd.NA)
        )
        
        waterloss_df = None
        water_benchmarks = None
        
        # Step 7: If water-loss file provided, integrate water KPIs
        if water_path is not None:
            status_text.text("💧 Parsing water-loss data and building benchmarks...")
            progress_bar.progress(92)
            wloss_raw = pd.read_excel(water_path, sheet_name=0)
            waterloss_df = parse_waterloss(wloss_raw)
            
            if not waterloss_df.empty:
                water_benchmarks = build_water_benchmarks(waterloss_df)
                
                # Merge water benchmarks (per product) into summary/yearly
                if not water_benchmarks.empty:
                    summary = summary.merge(
                        water_benchmarks,
                        on="Produkt",
                        how="left"
                    )
                    yearly = yearly.merge(
                        water_benchmarks,
                        on="Produkt",
                        how="left"
                    )
                    
                    # Estimate total water evaporated from volume * water_per_m3 benchmark
                    summary["Water_kg"] = summary["Volume_m3"] * summary["Water_per_m3_bench"]
                    yearly["Water_kg"] = yearly["Volume_m3"] * yearly["Water_per_m3_bench"]
                    
                    # Specific energy per kg of evaporated water
                    summary["kWh_per_kg"] = (
                        summary["Energy_kWh"] / summary["Water_kg"].replace(0, pd.NA)
                    )
                    yearly["kWh_per_kg"] = (
                        yearly["Energy_kWh"] / yearly["Water_kg"].replace(0, pd.NA)
                    )
            else:
                st.warning("⚠️ Water-loss file could be read, but no valid rows were parsed.")
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        return {
            'summary': summary,
            'yearly': yearly,
            'energy': e,
            'wagons': w,
            'intervals': ivals,
            'allocation': alloc,
            'waterloss': waterloss_df,
            'water_benchmarks': water_benchmarks
        }
        
    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        raise e
    finally:
        import time
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

# ------------------ Main Processing ------------------

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Clear results if files are changed
if energy_file and wagon_file:
    current_files = (
        energy_file.name,
        wagon_file.name,
        water_file.name if water_file else None
    )
    if 'last_files' not in st.session_state:
        st.session_state.last_files = current_files
    elif st.session_state.last_files != current_files:
        st.session_state.results = None
        st.session_state.analysis_complete = False
        st.session_state.last_files = current_files

if run_button:
    if not energy_file or not wagon_file:
        st.error("⚠️ Please upload at least energy and wagon files before running analysis.")
    else:
        tmp_e_path = None
        tmp_w_path = None
        tmp_wl_path = None
        
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_e:
                tmp_e.write(energy_file.read())
                tmp_e_path = tmp_e.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsm") as tmp_w:
                tmp_w.write(wagon_file.read())
                tmp_w_path = tmp_w.name
            
            if water_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_wl:
                    tmp_wl.write(water_file.read())
                    tmp_wl_path = tmp_wl.name
            
            # Run analysis
            results = run_analysis(
                tmp_e_path,
                tmp_w_path,
                tmp_wl_path,
                products if products else None,
                month if month != 0 else None
            )
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.analysis_complete = True
            
        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            with st.expander("🔍 View Error Details"):
                st.exception(e)
            st.session_state.results = None
            st.session_state.analysis_complete = False
        
        finally:
            # Clean up temporary files
            if tmp_e_path and os.path.exists(tmp_e_path):
                try:
                    os.unlink(tmp_e_path)
                except:
                    pass
            
            if tmp_w_path and os.path.exists(tmp_w_path):
                try:
                    os.unlink(tmp_w_path)
                except:
                    pass
            
            if tmp_wl_path and os.path.exists(tmp_wl_path):
                try:
                    os.unlink(tmp_wl_path)
                except:
                    pass

# Display results if available
if st.session_state.analysis_complete and st.session_state.results:
    results = st.session_state.results
    summary = results['summary']
    yearly = results['yearly']
    water_benchmarks = results.get('water_benchmarks')
    
    if summary.empty:
        st.warning("⚠️ No data found matching the selected filters.")
    else:
        # --------------- KPI Cards ---------------
        st.markdown('<div class="section-header">📈 Summary KPIs</div>', 
                   unsafe_allow_html=True)
        
        total_energy = yearly["Energy_kWh"].sum()
        avg_kpi_m3 = yearly["kWh_per_m3"].mean()
        total_volume = yearly["Volume_m3"].sum()
        
        avg_kpi_kg = None
        total_water = None
        if "kWh_per_kg" in yearly.columns and "Water_kg" in yearly.columns:
            avg_kpi_kg = yearly["kWh_per_kg"].mean()
            total_water = yearly["Water_kg"].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                create_kpi_card("Total Energy", total_energy, "kWh"),
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                create_kpi_card("Avg. Efficiency", avg_kpi_m3, "kWh/m³"),
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                create_kpi_card("Total Volume", total_volume, "m³"),
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                create_kpi_card("Avg. Spec. Energy", avg_kpi_kg, "kWh/kg"),
                unsafe_allow_html=True
            )
        
        # --------------- Monthly Trend ---------------
        st.markdown('<div class="section-header">📊 Monthly KPI Trend</div>', 
                   unsafe_allow_html=True)
        
        fig1 = px.line(
            summary,
            x="Month",
            y="kWh_per_m3",
            color="Zone",
            markers=True,
            hover_data=["Produkt", "Energy_kWh", "Volume_m3"],
            title="Energy Efficiency by Month and Zone (kWh/m³)"
        )
        fig1.update_layout(
            height=500,
            xaxis_title="Month",
            yaxis_title="kWh/m³",
            plot_bgcolor="white",
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # If water benchmarks exist, add kWh/kg trend plot
        if "kWh_per_kg" in summary.columns:
            fig1b = px.line(
                summary.dropna(subset=["kWh_per_kg"]),
                x="Month",
                y="kWh_per_kg",
                color="Zone",
                markers=True,
                hover_data=["Produkt", "Energy_kWh", "Water_kg"],
                title="Specific Energy by Month and Zone (kWh/kg H₂O)"
            )
            fig1b.update_layout(
                height=500,
                xaxis_title="Month",
                yaxis_title="kWh/kg",
                plot_bgcolor="white",
                hovermode='x unified'
            )
            st.plotly_chart(fig1b, use_container_width=True)
        
        # --------------- Zone Comparison ---------------
        st.markdown('<div class="section-header">📉 Zone Comparison</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig2 = px.bar(
                yearly,
                x="Zone",
                y="kWh_per_m3",
                color="Produkt",
                text_auto=".2f",
                title="Yearly KPI by Zone (kWh/m³)"
            )
            fig2.update_layout(height=400, plot_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            fig3 = px.pie(
                yearly,
                values="Energy_kWh",
                names="Zone",
                title="Energy Distribution by Zone"
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        # --------------- Water Benchmark View ---------------
        if water_benchmarks is not None and not water_benchmarks.empty:
            st.markdown('<div class="section-header">💧 Water Benchmarks per Product</div>', 
                        unsafe_allow_html=True)
            fig_w = px.bar(
                water_benchmarks,
                x="Produkt",
                y="Water_per_m3_bench",
                text_auto=".1f",
                title="Mean Water per m³ by Product (kg/m³)"
            )
            fig_w.update_layout(height=400, plot_bgcolor="white")
            st.plotly_chart(fig_w, use_container_width=True)
        
        # --------------- Data Tables ---------------
        with st.expander("📋 View Detailed Data Tables"):
            tab1, tab2, tab3 = st.tabs(["Monthly Summary", "Yearly Summary", "Water Benchmarks"])
            
            with tab1:
                fmt_cols = {
                    "Energy_kWh": "{:.2f}",
                    "Volume_m3": "{:.2f}",
                    "kWh_per_m3": "{:.2f}"
                }
                if "Water_kg" in summary.columns:
                    fmt_cols["Water_kg"] = "{:.2f}"
                if "kWh_per_kg" in summary.columns:
                    fmt_cols["kWh_per_kg"] = "{:.2f}"
                
                st.dataframe(
                    summary.style.format(fmt_cols),
                    use_container_width=True
                )
            
            with tab2:
                fmt_cols_y = {
                    "Energy_kWh": "{:.2f}",
                    "Volume_m3": "{:.2f}",
                    "kWh_per_m3": "{:.2f}"
                }
                if "Water_kg" in yearly.columns:
                    fmt_cols_y["Water_kg"] = "{:.2f}"
                if "kWh_per_kg" in yearly.columns:
                    fmt_cols_y["kWh_per_kg"] = "{:.2f}"
                
                st.dataframe(
                    yearly.style.format(fmt_cols_y),
                    use_container_width=True
                )
            
            with tab3:
                if water_benchmarks is not None and not water_benchmarks.empty:
                    st.dataframe(
                        water_benchmarks.style.format({
                            "Water_per_m3_bench": "{:.2f}",
                            "Water_loss_pct_mean": "{:.2%}",
                            "Samples": "{:.0f}"
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No water benchmark data available. Upload a water-loss file to enable this view.")
        
        # --------------- Download Section ---------------
        st.markdown('<div class="section-header">📥 Export Results</div>', 
                   unsafe_allow_html=True)
        
        excel_data = create_excel_download(results)
        
        st.download_button(
            label="📥 Download Complete Excel Report",
            data=excel_data,
            file_name="Dryer_KPI_Analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        st.success("✅ Analysis complete! Explore the visualizations above or download the full report.")
