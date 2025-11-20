import streamlit as st
import pandas as pd
import tempfile
import plotly.express as px
from io import BytesIO
import os

# Import the KPI calculation module
try:
    from dryer_kpi_monthly_final import (
        parse_energy,
        parse_wagon,
        explode_intervals,
        allocate_energy,
        add_water_kpis,
        predict_mix_energy,
        WATER_PER_M3_KG,
        CONFIG,
    )
except ImportError:
    st.error("❌ Unable to import dryer_kpi_monthly_final module")
    st.stop()

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Lindner Dryer KPI Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------ Custom CSS ------------------
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

# ------------------ Header ------------------
st.markdown(
    '<div class="main-title">🏭 Lindner – Dryer KPI Monitoring Dashboard</div>',
    unsafe_allow_html=True,
)

st.info(
    "📊 Upload your **Energy** and **Hordenwagen** files. "
    "Water-loss curves are already embedded from the Wasserverlust file, "
    "so kWh/kg and water KPIs are available without extra uploads."
)

# ------------------ Sidebar ------------------
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
        ["L30", "L32", "L34", "L36", "L38", "L40", "N40", "N44", "Y44"],
        default=["L30", "L32", "L34", "L36", "L38", "L40", "N40"],
        help="Select one or more products to analyze",
    )

    month = st.number_input(
        "📅 Month (0 = all):",
        min_value=0,
        max_value=12,
        value=0,
        help="Filter by specific month (1–12) or 0 for all months",
    )

    st.markdown("---")
    run_button = st.button("▶️ Run Analysis", use_container_width=True)


# ------------------ Helper Functions ------------------
def create_kpi_card(title, value, unit):
    """Render a nice KPI card."""
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


def create_excel_download(results):
    """Create Excel file in memory for download."""
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


def run_analysis(energy_path, wagon_path, products_filter, month_filter):
    """
    Orchestrates the whole analysis:

    1. Parse energy & wagon data
    2. Filter by product and month
    3. Build zone intervals and allocate energy
    4. Aggregate monthly and yearly KPIs
    5. Add water-based KPIs (kWh/kg)
    """
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # --- Parse energy ---
        status_text.text("🔄 Parsing energy data...")
        progress_bar.progress(15)
        e_raw = pd.read_excel(energy_path, sheet_name=CONFIG["energy_sheet"])
        e = parse_energy(e_raw)
        if e.empty:
            raise ValueError("Energy data is empty after parsing")

        # --- Parse wagons ---
        status_text.text("🔄 Parsing wagon tracking data...")
        progress_bar.progress(35)
        w_raw = pd.read_excel(
            wagon_path,
            sheet_name=CONFIG["wagon_sheet"],
            header=CONFIG["wagon_header_row"],
        )
        w = parse_wagon(w_raw)
        if w.empty:
            raise ValueError("Wagon data is empty after parsing")

        # --- Filters ---
        status_text.text("🔄 Applying filters...")
        progress_bar.progress(50)

        if products_filter:
            w = w[w["Produkt"].astype(str).isin(products_filter)]
            if w.empty:
                raise ValueError(f"No wagons found for products: {products_filter}")

        if month_filter:
            e = e[e["Month"] == month_filter]
            w = w[w["Month"] == month_filter]
            if e.empty or w.empty:
                raise ValueError(f"No data found for month: {month_filter}")

        # --- Intervals ---
        status_text.text("🔄 Processing zone intervals...")
        progress_bar.progress(65)
        ivals = explode_intervals(w)
        if ivals.empty:
            raise ValueError("No valid zone intervals could be created")

        # --- Allocation ---
        status_text.text("🔄 Allocating energy to products...")
        progress_bar.progress(80)
        alloc = allocate_energy(e, ivals)
        if alloc.empty:
            raise ValueError("Energy allocation produced no results")

        # --- Summaries ---
        status_text.text("🔄 Generating summaries...")
        progress_bar.progress(90)

        summary = alloc.groupby(["Month", "Produkt", "Zone"], as_index=False).agg(
            Energy_kWh=("Energy_share_kWh", "sum"),
            Volume_m3=("m3", "sum"),
        )
        summary["kWh_per_m3"] = (
            summary["Energy_kWh"] / summary["Volume_m3"].replace(0, pd.NA)
        )

        # Add water-based KPIs using built-in benchmarks
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

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

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
        progress_bar.empty()
        status_text.empty()


# ------------------ Session State ------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# Clear results when files change
if energy_file and wagon_file:
    current_files = (energy_file.name, wagon_file.name)
    if "last_files" not in st.session_state:
        st.session_state.last_files = current_files
    elif st.session_state.last_files != current_files:
        st.session_state.results = None
        st.session_state.analysis_complete = False
        st.session_state.last_files = current_files

# ------------------ Run Button ------------------
if run_button:
    if not energy_file or not wagon_file:
        st.error("⚠️ Please upload both Energy and Hordenwagen files.")
    else:
        tmp_e_path = None
        tmp_w_path = None

        try:
            # Save uploads as temp files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_e:
                tmp_e.write(energy_file.read())
                tmp_e_path = tmp_e.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsm") as tmp_w:
                tmp_w.write(wagon_file.read())
                tmp_w_path = tmp_w.name

            # Run analysis
            results = run_analysis(
                tmp_e_path,
                tmp_w_path,
                products if products else None,
                month if month != 0 else None,
            )

            st.session_state.results = results
            st.session_state.analysis_complete = True

        except Exception as exc:
            st.error(f"❌ Error during analysis: {exc}")
            with st.expander("🔍 View Error Details"):
                st.exception(exc)
            st.session_state.results = None
            st.session_state.analysis_complete = False

        finally:
            # Cleanup
            for p in (tmp_e_path, tmp_w_path):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except:
                        pass

# ------------------ Display Results ------------------
if st.session_state.analysis_complete and st.session_state.results:
    results = st.session_state.results
    summary = results["summary"]
    yearly = results["yearly"]

    if summary.empty:
        st.warning("⚠️ No data found with the selected filters.")
    else:
        # ===== KPI CARDS =====
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

        # ===== Monthly KPI Trends =====
        st.markdown(
            '<div class="section-header">📊 Monthly KPI Trends</div>',
            unsafe_allow_html=True,
        )

        # kWh/m³ vs month
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

        # kWh/kg vs month
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

        # ===== Zone Comparison =====
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

        # ===== Water Benchmarks per Product =====
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

        # ===== Detailed Tables =====
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

        # ===== Prediction Helper =====
        st.markdown(
            '<div class="section-header">🔮 Simple Prediction Helper</div>',
            unsafe_allow_html=True,
        )

        st.write(
            "Use the embedded water benchmarks and your measured KPIs "
            "to estimate water load and energy for a planned product mix."
        )

        with st.form("prediction_form"):
            st.write("### Define planned daily volume per product (m³/day)")
            pred_volumes = {}
            col_pred1, col_pred2, col_pred3 = st.columns(3)
            product_list = ["L30", "L32", "L34", "L36", "L38", "L40", "N40", "N44", "Y44"]

            # Split into columns just for nicer layout
            col_lists = [
                product_list[0:3],
                product_list[3:6],
                product_list[6:9],
            ]
            for col, plist in zip([col_pred1, col_pred2, col_pred3], col_lists):
                with col:
                    for p in plist:
                        pred_volumes[p] = st.number_input(
                            f"{p} [m³/day]",
                            min_value=0.0,
                            value=0.0,
                            step=10.0,
                        )

            st.write("### Baseline KPIs for prediction")
            # From the measured data: overall average
            overall_kwh_m3 = avg_kwh_m3
            overall_kwh_kg = avg_kwh_kg

            baseline_kwh_m3 = st.number_input(
                "Baseline kWh/m³ (leave as default to use measured avg)",
                min_value=0.0,
                value=float(overall_kwh_m3) if not pd.isna(overall_kwh_m3) else 0.0,
            )
            baseline_kwh_kg = st.number_input(
                "Baseline kWh/kg (leave as default to use measured avg)",
                min_value=0.0,
                value=float(overall_kwh_kg) if not pd.isna(overall_kwh_kg) else 0.0,
            )

            submitted = st.form_submit_button("Calculate Prediction")

        if submitted:
            # Use the helper from the KPI module
            pred_result = predict_mix_energy(
                product_mix_m3=pred_volumes,
                baseline_kwh_per_m3=baseline_kwh_m3 or None,
                baseline_kwh_per_kg=baseline_kwh_kg or None,
            )

            st.write("#### Prediction Results")
            colr1, colr2, colr3 = st.columns(3)
            with colr1:
                st.metric(
                    "Total Volume (m³/day)",
                    f"{pred_result['total_volume_m3']:,.2f}",
                )
            with colr2:
                st.metric(
                    "Total Water (kg/day)",
                    f"{pred_result['total_water_kg']:,.2f}",
                )
            with colr3:
                st.metric(
                    "Mean Water per m³ (kg/m³)",
                    f"{pred_result['mean_water_per_m3']:,.2f}",
                )

            # Show energy predictions if available
            if "energy_from_kwh_per_m3" in pred_result:
                st.write(
                    f"**Predicted Energy (kWh/day) "
                    f"based on kWh/m³:** "
                    f"{pred_result['energy_from_kwh_per_m3']:,.0f} kWh"
                )
            if "energy_from_kwh_per_kg" in pred_result:
                st.write(
                    f"**Predicted Energy (kWh/day) "
                    f"based on kWh/kg:** "
                    f"{pred_result['energy_from_kwh_per_kg']:,.0f} kWh"
                )

        # ===== Export =====
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

        st.success("✅ Analysis complete! Explore the charts or download the report.")
