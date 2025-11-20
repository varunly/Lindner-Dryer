import streamlit as st
import pandas as pd
import tempfile
import plotly.express as px
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
        WATER_PER_M3_KG,
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
    .stDownloadButton button {
        background-color: #003366;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
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
    "Water-loss behaviour is already embedded as product benchmarks, "
    "so kWh/kg H₂O is computed automatically – no extra Wasserverlust file needed."
)

# ---------------------------------------------------------
# Sidebar – file upload & filters
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
        ["L28", "L30", "L32", "L34", "L36", "L38", "L40", "L44", "N40", "N44", "Y44"],
        default=["L30", "L32", "L34", "L36", "L38", "L40", "N40"],
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
    run_button = st.button("▶️ Run Analysis", use_container_width=True)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def create_kpi_card(title: str, value, unit: str) -> str:
    """Return HTML for a KPI card."""
    if value is None or pd.isna(value):
        text = "–"
        unit_str = ""
    else:
        text = f"{value:,.2f}"
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
        results["energy"].to_excel(writer, sheet_name="Energy_Data", index=False)
        results["wagons"].to_excel(writer, sheet_name="Wagon_Data", index=False)
        results["intervals"].to_excel(writer, sheet_name="Zone_Intervals", index=False)
        results["allocation"].to_excel(
            writer, sheet_name="Energy_Allocation", index=False
        )
        results["summary"].to_excel(writer, sheet_name="Monthly_Summary", index=False)
        results["yearly"].to_excel(writer, sheet_name="Yearly_Summary", index=False)

    output.seek(0)
    return output


def run_analysis(
    energy_path: str, wagon_path: str, products_filter, month_filter
) -> dict:
    """
    Orchestrate the full KPI calculation:
    1. Parse energy & wagon files
    2. Filter by product & month
    3. Build zone intervals & allocate energy
    4. Aggregate KPIs (kWh/m³)
    5. Add water-based KPIs (kWh/kg) using built-in benchmarks
    """
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
                raise ValueError(
                    f"No wagon records found for selected products: {products_filter}"
                )

        if month_filter:
            e = e[e["Month"] == month_filter]
            w = w[w["Month"] == month_filter]
            if e.empty or w.empty:
                raise ValueError(
                    f"No energy/wagon data found for month = {month_filter}."
                )

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
            Energy_kWh=("Energy_share_kWh", "sum"),
            Volume_m3=("m3", "sum"),
        )
        summary["kWh_per_m3"] = (
            summary["Energy_kWh"] / summary["Volume_m3"].replace(0, pd.NA)
        )

        # Add water-related KPIs
        summary = add_water_kpis(summary)

        yearly = summary.groupby(["Produkt", "Zone"], as_index=False).agg(
            Energy_kWh=("Energy_kWh", "sum"),
            Volume_m3=("Volume_m3", "sum"),
            Water_kg=("Water_kg", "sum"),
        )
        yearly["kWh_per_m3"] = (
            yearly["Energy_kWh"] / yearly["Volume_m3"].replace(0, pd.NA)
        )
        yearly["kWh_per_kg"] = (
            yearly["Energy_kWh"] / yearly["Water_kg"].replace(0, pd.NA)
        )

        progress.progress(100)
        status.text("✅ Analysis complete!")

        return {
            "energy": e,
            "wagons": w,
            "intervals": ivals,
            "allocation": alloc,
            "summary": summary,
            "yearly": yearly,
        }

    finally:
        import time

        time.sleep(0.4)
        progress.empty()
        status.empty()


# ---------------------------------------------------------
# Session state initialisation
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
# Run analysis when button clicked
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
# Show results
# ---------------------------------------------------------
if st.session_state.analysis_complete and st.session_state.results:
    results = st.session_state.results
    summary = results["summary"]
    yearly = results["yearly"]

    if summary.empty:
        st.warning("⚠️ No data available after filtering.")
    else:
        # ---------------- KPI CARDS ----------------
        st.markdown(
            '<div class="section-header">📈 Summary KPIs</div>',
            unsafe_allow_html=True,
        )

        total_energy = yearly["Energy_kWh"].sum()
        total_volume = yearly["Volume_m3"].sum()
        total_water = yearly["Water_kg"].sum()

        avg_kwh_m3 = yearly["kWh_per_m3"].mean()
        avg_kwh_kg = yearly["kWh_per_kg"].mean()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                create_kpi_card("Total Energy", total_energy, "kWh"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                create_kpi_card("Total Volume", total_volume, "m³"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                create_kpi_card("Total Water", total_water, "kg"),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                create_kpi_card("Avg. Spec. Energy", avg_kwh_kg, "kWh/kg"),
                unsafe_allow_html=True,
            )

        # ---------------- Monthly KPI Trends ----------------
        st.markdown(
            '<div class="section-header">📊 Monthly KPI Trends</div>',
            unsafe_allow_html=True,
        )

        fig_kwh_m3 = px.line(
            summary,
            x="Month",
            y="kWh_per_m3",
            color="Zone",
            markers=True,
            hover_data=["Produkt", "Energy_kWh", "Volume_m3"],
            title="Energy Efficiency by Month & Zone (kWh/m³)",
        )
        fig_kwh_m3.update_layout(
            height=450, xaxis_title="Month", yaxis_title="kWh/m³", plot_bgcolor="white"
        )
        st.plotly_chart(fig_kwh_m3, use_container_width=True)

        if "kWh_per_kg" in summary.columns:
            fig_kwh_kg = px.line(
                summary,
                x="Month",
                y="kWh_per_kg",
                color="Zone",
                markers=True,
                hover_data=["Produkt", "Energy_kWh", "Water_kg"],
                title="Specific Energy by Month & Zone (kWh/kg H₂O)",
            )
            fig_kwh_kg.update_layout(
                height=450,
                xaxis_title="Month",
                yaxis_title="kWh/kg",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_kwh_kg, use_container_width=True)

        # ---------------- Zone Comparison ----------------
        st.markdown(
            '<div class="section-header">📉 Zone Comparison</div>',
            unsafe_allow_html=True,
        )

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

        # ---------------- Water Benchmarks ----------------
        st.markdown(
            '<div class="section-header">💧 Water Benchmarks per Product</div>',
            unsafe_allow_html=True,
        )

        bench_df = (
            pd.DataFrame(
                [
                    {"Produkt": prod, "Water_per_m3_bench": val}
                    for prod, val in WATER_PER_M3_KG.items()
                ]
            )
            .sort_values("Produkt")
            .reset_index(drop=True)
        )

        fig_bench = px.bar(
            bench_df,
            x="Produkt",
            y="Water_per_m3_bench",
            text_auto=".1f",
            title="Embedded Water Benchmarks (kg water / m³ product)",
        )
        fig_bench.update_layout(height=400, plot_bgcolor="white")
        st.plotly_chart(fig_bench, use_container_width=True)

        # ---------------- Detailed Tables ----------------
        with st.expander("📋 View Detailed Data Tables"):
            tab1, tab2 = st.tabs(["Monthly Summary", "Yearly Summary"])

            with tab1:
                fmt = {
                    "Energy_kWh": "{:.2f}",
                    "Volume_m3": "{:.2f}",
                    "kWh_per_m3": "{:.2f}",
                    "Water_kg": "{:.2f}",
                    "kWh_per_kg": "{:.2f}",
                }
                st.dataframe(summary.style.format(fmt), use_container_width=True)

            with tab2:
                fmt_y = {
                    "Energy_kWh": "{:.2f}",
                    "Volume_m3": "{:.2f}",
                    "kWh_per_m3": "{:.2f}",
                    "Water_kg": "{:.2f}",
                    "kWh_per_kg": "{:.2f}",
                }
                st.dataframe(yearly.style.format(fmt_y), use_container_width=True)

        # ---------------- Weekly Prediction Helper ----------------
        st.markdown(
            '<div class="section-header">🔮 Weekly Energy Prediction (Wagons/Week)</div>',
            unsafe_allow_html=True,
        )

        st.write(
            "Enter your weekly production plan in **wagons per product per week**. "
            "The system will automatically convert wagons → m³ → water load → energy."
        )

        # Compute wagon capacities & residence times
        wagon_stats = compute_product_wagon_stats(results["wagons"])
        wagon_capacity = wagon_stats["wagon_capacity_m3"]
        residence_days = wagon_stats["residence_days"]

        # UI form
        with st.form("weekly_prediction_form"):

            st.write("### Planned Wagons per Week")
            planned_wagons = {}

            # Split products into 3 columns
            prod_left  = ["L28", "L30", "L32", "L34"]
            prod_mid   = ["L36", "L38", "L40", "L44"]
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
            base_kwh_m3 = st.number_input(
                "Baseline kWh/m³ (default: measured avg)",
                min_value=0.0,
                value=float(avg_kwh_m3) if not pd.isna(avg_kwh_m3) else 0.0,
            )
            base_kwh_kg = st.number_input(
                "Baseline kWh/kg (default: measured avg)",
                min_value=0.0,
                value=float(avg_kwh_kg) if not pd.isna(avg_kwh_kg) else 0.0,
            )

            submitted_weekly = st.form_submit_button("Predict Weekly Energy")

        # Process prediction (OUTSIDE the form)
        if submitted_weekly:
            
            pred_week = predict_weekly_energy_from_wagons(
                product_wagons_per_week=planned_wagons,
                wagons_df=results["wagons"],
                baseline_kwh_per_m3=base_kwh_m3,
                baseline_kwh_per_kg=base_kwh_kg,
            )
        
            st.subheader("📊 Weekly Prediction Results")
        
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Total Wagons/Week", f"{pred_week['total_wagons_week']:,}")
            with r2:
                st.metric("Total Volume (m³/week)", f"{pred_week['total_volume_m3_week']:,.2f}")
            with r3:
                st.metric("Total Water (kg/week)", f"{pred_week['total_water_kg_week']:,.2f}")
        
            # Energy predictions
            if "energy_week_from_kwh_per_m3" in pred_week:
                st.write(f"**Predicted Energy (kWh/week) using kWh/m³:** "
                         f"{pred_week['energy_week_from_kwh_per_m3']:,.0f} kWh")
        
                st.write(f"**Average Energy per Day (kWh/day):** "
                         f"{pred_week['avg_energy_per_day_from_kwh_per_m3']:,.0f} kWh")
        
            if "energy_week_from_kwh_per_kg" in pred_week:
                st.write(f"**Predicted Energy (kWh/week) using kWh/kg:** "
                         f"{pred_week['energy_week_from_kwh_per_kg']:,.0f} kWh")
        
                st.write(f"**Average Energy per Day (kWh/day):** "
                         f"{pred_week['avg_energy_per_day_from_kwh_per_kg']:,.0f} kWh")
        
            # Residence time
            st.subheader("⏱️ Average Residence Time (from your real dryer data)")
            res_df = pd.DataFrame([
                {"Produkt": p, "Residence Time (days)": residence_days.get(p, float("nan"))}
                for p in planned_wagons.keys()
            ])
            st.dataframe(res_df, use_container_width=True)
        
            # WIP water load
            st.subheader("💧 Work-in-Progress (Water Inventory Inside Dryer)")
            st.metric("Estimated WIP Water (kg)", f"{pred_week['wip_water_kg_estimate']:,.0f}")
        
            st.success("✅ Weekly energy prediction completed.")

        # ---------------- Export ----------------
        st.markdown(
            '<div class="section-header">📥 Export Results</div>',
            unsafe_allow_html=True,
        )

        excel_data = create_excel_download(results)
        st.download_button(
            label="📥 Download Complete Excel Report",
            data=excel_data,
            file_name="Dryer_KPI_Analysis.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

        st.success("✅ Analysis complete! You can explore the visualizations above or download the full report.")
