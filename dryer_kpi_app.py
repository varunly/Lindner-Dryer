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
        compute_zone_duration_stats,
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
# German Number Formatting Functions
# ---------------------------------------------------------
def format_german(value, decimals=2):
    """
    Format number in German style:
    - Dot (.) as thousands separator
    - Comma (,) as decimal separator
    
    Examples:
        1234.56 -> "1.234,56"
        1234567.89 -> "1.234.567,89"
    """
    if value is None:
        return "–"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "–"
        # Format with specified decimals using US style first
        formatted = f"{value:,.{decimals}f}"
        # Swap commas and dots: comma->X, dot->comma, X->dot
        formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
        return formatted
    except (TypeError, ValueError):
        return str(value)


def format_german_int(value):
    """Format integer in German style (dot as thousands separator)"""
    if value is None:
        return "–"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "–"
        formatted = f"{int(value):,}"
        formatted = formatted.replace(",", ".")
        return formatted
    except (TypeError, ValueError):
        return str(value)


def format_german_pct(value, decimals=1):
    """Format percentage in German style"""
    if value is None:
        return "–"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "–"
        formatted = f"{value:.{decimals}f}"
        formatted = formatted.replace(".", ",")
        return f"{formatted}%"
    except (TypeError, ValueError):
        return str(value)


def format_df_german(df, int_cols=None, float_cols=None, pct_cols=None, decimals=2):
    """
    Format DataFrame columns to German number format.
    Returns a copy with formatted string columns.
    """
    df_display = df.copy()
    
    if int_cols:
        for col in int_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(format_german_int)
    
    if float_cols:
        for col in float_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: format_german(x, decimals))
    
    if pct_cols:
        for col in pct_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(format_german_pct)
    
    return df_display


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
    .trockner-select {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #17a2b8;
        margin-bottom: 20px;
    }
    .debug-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .filter-summary {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
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
    f"Suspension: {format_german_int(SUSPENSION_KG)} kg | "
    f"Plates/Wagon: {format_german_int(PLATES_PER_WAGON)} | "
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
    
    # ===== TROCKNER SELECTION (HIGHLIGHTED WITH RADIO BUTTONS) =====
    st.markdown('<div class="trockner-select">', unsafe_allow_html=True)
    st.markdown("### 🏭 Select Trockner (Dryer)")
    trockner_option = st.radio(
        "Choose dryer:",
        options=["All", "A", "B"],
        index=0,
        horizontal=True,
        help="Select which Trockner to analyze: A, B, or All (both)"
    )
    if trockner_option == "All":
        st.info("📊 Analyzing data from **both Trockner A and B**")
    else:
        st.success(f"✅ Analyzing **Trockner {trockner_option} only**")
    st.markdown('</div>', unsafe_allow_html=True)

    products = st.multiselect(
        "🧱 Product(s):",
        ["L20", "L24", "L28", "L30", "L32", "L34", "L36", "L37", "L38", "L40", "L42", "L44", 
         "N24", "N30", "N34", "N36", "N38", "N40", "N42", "N44", 
         "Y30", "Y34", "Y38", "Y44"],
        default=["L28", "L30", "L32", "L34", "L36", "L38", "L40", "L42", "L44", 
                 "N24", "N30", "N34", "N36", "N38", "N40", "N42", "N44", 
                 "Y30", "Y34", "Y38", "Y44"],
    )

    month = st.number_input(
        "📅 Month (0 = all):",
        min_value=0,
        max_value=12,
        value=0,
    )

    st.markdown("---")
    run_button = st.button("▶️ Run Analysis", type="primary", use_container_width=True)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def create_kpi_card(title: str, value, unit: str) -> str:
    """Create KPI card with German number formatting"""
    if value is None:
        text, unit_str = "–", ""
    else:
        try:
            if np.isnan(value):
                text, unit_str = "–", ""
            else:
                text = format_german(value, 2)
                unit_str = f" {unit}"
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
    Run the complete KPI analysis with overlap period filtering.
    """
    progress = st.progress(0)
    status = st.empty()
    
    debug_container = st.container()

    try:
        # ===== PARSE ENERGY =====
        status.text("🔄 Parsing energy data...")
        progress.progress(10)
        
        try:
            e_raw = pd.read_excel(energy_path, sheet_name=CONFIG["energy_sheet"])
        except Exception as ex:
            try:
                e_raw = pd.read_excel(energy_path, sheet_name=CONFIG["energy_sheet"], engine='openpyxl')
            except:
                raise ValueError(f"Cannot read energy file: {ex}")
        
        e = parse_energy(e_raw)
        if e.empty:
            raise ValueError("Parsed energy data is empty.")
        
        energy_start = e["E_start"].min()
        energy_end = e["E_end"].max()
        energy_hours_raw = len(e)
        energy_thermal_raw = e["E_thermal_total_kWh"].sum()
        energy_electrical_raw = e["E_el_kWh"].sum()

        # ===== PARSE WAGON DATA =====
        status.text("🔄 Parsing wagon tracking data...")
        progress.progress(25)
        
        try:
            w_raw = pd.read_excel(
                wagon_path,
                sheet_name=CONFIG["wagon_sheet"],
                header=CONFIG["wagon_header_row"],
            )
        except Exception as ex:
            try:
                w_raw = pd.read_excel(
                    wagon_path,
                    sheet_name=CONFIG["wagon_sheet"],
                    header=CONFIG["wagon_header_row"],
                    engine='openpyxl'
                )
            except:
                raise ValueError(f"Cannot read wagon file: {ex}")
        
        raw_wagon_rows = len(w_raw)
        
        with debug_container:
            with st.expander("🔧 DEBUG 1: Raw Wagon File Analysis", expanded=False):
                st.write(f"**Total rows in raw file:** {format_german_int(raw_wagon_rows)}")
                st.write(f"**Total columns:** {len(w_raw.columns)}")
                
                col_list = list(w_raw.columns)[:20]
                st.write("**First 20 columns:**")
                for i, col in enumerate(col_list):
                    st.code(f"[{i:2d}] {repr(col)}")
        
        # ===== APPLY TROCKNER FILTER =====
        status.text(f"🔄 Parsing wagon data (Trockner: {trockner_filter})...")
        progress.progress(35)
        
        trockner_to_use = trockner_filter if trockner_filter != "All" else None
        w = parse_wagon(w_raw, trockner=trockner_to_use)
        
        if w.empty:
            raise ValueError("Parsed wagon data is empty after Trockner filter.")
        
        wagon_count_after_trockner = len(w)
        volume_after_trockner = w["m3"].sum()
        applied_trockner = trockner_to_use or "All"
        
        with debug_container:
            with st.expander("🔧 DEBUG 2: After Trockner Filter", expanded=False):
                st.write(f"**Trockner filter:** {applied_trockner}")
                st.write(f"**Rows after filter:** {format_german_int(wagon_count_after_trockner)}")
                st.write(f"**Volume after filter:** {format_german(volume_after_trockner, 2)} m³")
                
                if "Trockner" in w.columns:
                    st.write("**Trockner values:**")
                    st.write(w["Trockner"].value_counts())
        
        # ===== CALCULATE AND APPLY OVERLAP FILTER =====
        status.text("🔄 Calculating overlap period...")
        progress.progress(40)
        
        wagon_start = w["t0"].min()
        wagon_end = w["t0"].max()
        
        overlap_start = max(energy_start, wagon_start)
        overlap_end = min(energy_end, wagon_end)
        overlap_days = (overlap_end - overlap_start).days
        
        wagons_before_overlap = len(w)
        volume_before_overlap = w["m3"].sum()
        energy_hours_before_overlap = len(e)
        
        with debug_container:
            with st.expander("📅 TIME PERIOD OVERLAP - CRITICAL", expanded=True):
                st.markdown("### ⏰ Date Range Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📊 Energy Data**")
                    st.write(f"Start: `{energy_start}`")
                    st.write(f"End: `{energy_end}`")
                    st.write(f"Hours: {format_german_int(energy_hours_raw)}")
                    st.write(f"Thermal: {format_german_int(energy_thermal_raw)} kWh")
                    st.write(f"Electrical: {format_german_int(energy_electrical_raw)} kWh")
                
                with col2:
                    st.markdown("**🚛 Wagon Data**")
                    st.write(f"Start: `{wagon_start}`")
                    st.write(f"End: `{wagon_end}`")
                    st.write(f"Wagons: {format_german_int(wagons_before_overlap)}")
                    st.write(f"Volume: {format_german(volume_before_overlap, 2)} m³")
                
                st.markdown("---")
                st.markdown("### ✅ OVERLAP PERIOD (Used for Analysis)")
                
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.success(f"**Start:** {overlap_start}")
                with col4:
                    st.success(f"**End:** {overlap_end}")
                with col5:
                    st.success(f"**Duration:** {format_german_int(overlap_days)} days")
                
                st.info("⚠️ Only data within this overlap period is used for all KPI calculations!")
        
        # ===== FILTER WAGONS TO OVERLAP PERIOD =====
        status.text("🔄 Filtering wagons to overlap period...")
        progress.progress(45)
        
        w = w[(w["t0"] >= overlap_start) & (w["t0"] <= overlap_end)].copy()
        
        wagons_after_overlap = len(w)
        volume_after_overlap = w["m3"].sum()
        wagons_removed_overlap = wagons_before_overlap - wagons_after_overlap
        volume_removed_overlap = volume_before_overlap - volume_after_overlap
        
        e = e[(e["E_start"] >= overlap_start) & (e["E_end"] <= overlap_end)].copy()
        
        energy_hours_after_overlap = len(e)
        energy_thermal_after_overlap = e["E_thermal_total_kWh"].sum()
        energy_electrical_after_overlap = e["E_el_kWh"].sum()
        
        with debug_container:
            with st.expander("🔧 DEBUG 3: After Overlap Filter", expanded=True):
                st.markdown("### 🚛 Wagon Data")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Before Overlap", format_german_int(wagons_before_overlap))
                with col2:
                    st.metric("After Overlap", format_german_int(wagons_after_overlap))
                with col3:
                    st.metric("Removed", format_german_int(wagons_removed_overlap), 
                             delta=f"-{format_german_int(wagons_removed_overlap)}" if wagons_removed_overlap > 0 else None,
                             delta_color="inverse")
                
                st.write(f"Volume before: {format_german(volume_before_overlap, 2)} m³")
                st.write(f"Volume after: {format_german(volume_after_overlap, 2)} m³")
                st.write(f"Volume removed: {format_german(volume_removed_overlap, 2)} m³")
                
                if wagons_removed_overlap > 0:
                    st.warning(f"⚠️ {format_german_int(wagons_removed_overlap)} wagons removed (outside energy data period)")
                
                st.markdown("---")
                st.markdown("### ⚡ Energy Data")
                col4, col5 = st.columns(2)
                with col4:
                    st.write(f"Hours before: {format_german_int(energy_hours_before_overlap)}")
                    st.write(f"Hours after: {format_german_int(energy_hours_after_overlap)}")
                with col5:
                    st.write(f"Thermal after overlap: {format_german_int(energy_thermal_after_overlap)} kWh")
                    st.write(f"Electrical after overlap: {format_german_int(energy_electrical_after_overlap)} kWh")
        
        if w.empty:
            raise ValueError("No wagon data within the overlap period!")
        
        if e.empty:
            raise ValueError("No energy data within the overlap period!")
        
        wagon_count_after_overlap = len(w)
        volume_after_overlap_filter = w["m3"].sum()
        
        # ===== APPLY PRODUCT FILTER =====
        status.text("🔄 Applying product filters...")
        progress.progress(50)
        
        wagon_count_before_product_filter = len(w)
        volume_before_product_filter = w["m3"].sum()
        
        w_before_product_filter = w.copy()
        
        if products_filter:
            w = w[w["Produkt"].astype(str).isin(products_filter)]
            if w.empty:
                raise ValueError(f"No wagon records found for selected products: {products_filter}")
        
        wagon_count_after_product_filter = len(w)
        volume_after_product_filter = w["m3"].sum()
        
        with debug_container:
            with st.expander("🔧 DEBUG 4: After Product Filter", expanded=False):
                st.write(f"**Product filter:** {products_filter}")
                st.write(f"**Rows before:** {format_german_int(wagon_count_before_product_filter)}")
                st.write(f"**Rows after:** {format_german_int(wagon_count_after_product_filter)}")
                st.write(f"**Rows removed:** {format_german_int(wagon_count_before_product_filter - wagon_count_after_product_filter)}")
                st.write(f"**Volume before:** {format_german(volume_before_product_filter, 2)} m³")
                st.write(f"**Volume after:** {format_german(volume_after_product_filter, 2)} m³")
        
        total_wagons = len(w)
        total_volume = w["m3"].sum()
        
        status.text(f"📊 Found {format_german_int(total_wagons)} wagon rows with {format_german(total_volume, 2)} m³ total volume")

        # ===== APPLY MONTH FILTER =====
        status.text("🔄 Applying month filter...")
        progress.progress(55)
        
        wagon_count_before_month_filter = len(w)
        volume_before_month_filter = w["m3"].sum()
        
        if month_filter:
            e = e[e["Month"] == month_filter]
            w = w[w["Month"] == month_filter]
            if e.empty or w.empty:
                raise ValueError(f"No data found for month = {month_filter}.")
        
        wagon_count_after_month_filter = len(w)
        volume_after_month_filter = w["m3"].sum()
        
        with debug_container:
            with st.expander("🔧 DEBUG 5: After Month Filter", expanded=False):
                st.write(f"**Month filter:** {month_filter or 'None (all months)'}")
                st.write(f"**Rows before:** {format_german_int(wagon_count_before_month_filter)}")
                st.write(f"**Rows after:** {format_german_int(wagon_count_after_month_filter)}")
                st.write(f"**Volume before:** {format_german(volume_before_month_filter, 2)} m³")
                st.write(f"**Volume after:** {format_german(volume_after_month_filter, 2)} m³")

        # ===== BUILD INTERVALS =====
        status.text("🔄 Building zone intervals...")
        progress.progress(65)
        
        ivals = explode_intervals(w)
        if ivals.empty:
            raise ValueError("Zone intervals could not be created.")
        
        with debug_container:
            with st.expander("🔧 DEBUG 6: Zone Intervals", expanded=False):
                st.write(f"**Total intervals:** {format_german_int(len(ivals))}")
                st.write(f"**Zones:** {ivals['Zone'].unique().tolist()}")
                st.write("**Intervals per zone:**")
                st.write(ivals.groupby("Zone").size())

        # ===== ALLOCATE ENERGY =====
        status.text("🔄 Allocating energy to products...")
        progress.progress(80)
        
        alloc = allocate_energy(e, ivals, full_electrical_kwh=energy_electrical_raw)
        
        # ===== COMPUTE ZONE DURATION STATS =====
        status.text("🔄 Computing zone duration statistics...")
        progress.progress(85)
        
        try:
            zone_overall_stats, zone_product_stats = compute_zone_duration_stats(w)
        except Exception as ex:
            zone_overall_stats = pd.DataFrame()
            zone_product_stats = pd.DataFrame()
            print(f"Zone stats error: {ex}")
        
        if "Z2_in" in w.columns and w["Z2_in"].notna().any():
            w["Week_Energy"] = w["Z2_in"].dt.isocalendar().week
            w["Year_Energy"] = w["Z2_in"].dt.year
        else:
            w["Week_Energy"] = w["t0"].dt.isocalendar().week
            w["Year_Energy"] = w["t0"].dt.year
      
        if alloc.empty:
            raise ValueError("Energy allocation result is empty.")
        
        with debug_container:
            with st.expander("🔧 DEBUG 7: Energy Allocation Results", expanded=False):
                st.write(f"**Allocation records:** {format_german_int(len(alloc))}")
                st.write(f"**Thermal allocated:** {format_german_int(alloc['Energy_thermal_kWh'].sum())} kWh")
                st.write(f"**Electrical allocated:** {format_german_int(alloc['Energy_electrical_kWh'].sum())} kWh")
                st.write(f"**Total allocated:** {format_german_int(alloc['Energy_share_kWh'].sum())} kWh")

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
            "wagons_before_product_filter": w_before_product_filter,
            "intervals": ivals,
            "allocation": alloc,
            "summary": summary,
            "yearly": yearly,
            "product_totals": product_totals,
            "applied_trockner": applied_trockner,
            "energy_start": energy_start,
            "energy_end": energy_end,
            "wagon_start": wagon_start,
            "wagon_end": wagon_end,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "overlap_days": overlap_days,
            "raw_wagon_file_rows": raw_wagon_rows,
            "energy_hours_raw": energy_hours_raw,
            "energy_thermal_raw": energy_thermal_raw,
            "energy_electrical_raw": energy_electrical_raw,
            "wagon_count_after_trockner": wagon_count_after_trockner,
            "volume_after_trockner": volume_after_trockner,
            "wagons_before_overlap": wagons_before_overlap,
            "volume_before_overlap": volume_before_overlap,
            "wagon_count_after_overlap": wagon_count_after_overlap,
            "volume_after_overlap": volume_after_overlap_filter,
            "wagons_removed_overlap": wagons_removed_overlap,
            "volume_removed_overlap": volume_removed_overlap,
            "energy_hours_after_overlap": energy_hours_after_overlap,
            "energy_thermal_after_overlap": energy_thermal_after_overlap,
            "energy_electrical_after_overlap": energy_electrical_after_overlap,
            "wagon_count_before_product_filter": wagon_count_before_product_filter,
            "wagon_count_after_product_filter": wagon_count_after_product_filter,
            "volume_before_product_filter": volume_before_product_filter,
            "volume_after_product_filter": volume_after_product_filter,
            "wagon_count_before_month_filter": wagon_count_before_month_filter,
            "wagon_count_after_month_filter": wagon_count_after_month_filter,
            "volume_before_month_filter": volume_before_month_filter,
            "volume_after_month_filter": volume_after_month_filter,
            "products_filter_applied": products_filter,
            "month_filter_applied": month_filter,
            "zone_overall_stats": zone_overall_stats,
            "zone_product_stats": zone_product_stats,
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
            st.info(f"📄 Energy file: {energy_file.name} ({format_german_int(energy_file.size)} bytes)")
            st.info(f"📄 Wagon file: {wagon_file.name} ({format_german_int(wagon_file.size)} bytes)")
            st.info(f"🏭 Trockner filter: **{trockner_option}**")
            
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
            st.warning(
                "💡 **Troubleshooting Tips:**\n"
                "1. Try re-downloading the original Excel file\n"
                "2. Open and re-save the file in Excel\n"
                "3. Ensure the file is not password protected\n"
                "4. Check if the file opens correctly on your computer"
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
        wagons_df = results["wagons"]
        
        applied_trockner = results.get("applied_trockner", "All")
        
        raw_rows = results.get("raw_wagon_file_rows", 0)
        count_after_trockner = results.get("wagon_count_after_trockner", 0)
        count_after_product = results.get("wagon_count_after_product_filter", 0)
        count_after_month = results.get("wagon_count_after_month_filter", 0)
        products_applied = results.get("products_filter_applied", [])
        month_applied = results.get("month_filter_applied", None)

        if summary.empty:
            st.warning("⚠️ No data available after filtering.")
        else:
            # ============================================================
            #   FILTER PIPELINE SUMMARY
            # ============================================================
            st.markdown('<div class="section-header">🔍 Filter Pipeline Summary</div>', unsafe_allow_html=True)
            
            col_fp1, col_fp2, col_fp3, col_fp4 = st.columns(4)
            
            with col_fp1:
                st.metric("1️⃣ Raw File", format_german_int(raw_rows), help="Total rows in uploaded Excel file")
            with col_fp2:
                delta_trockner = count_after_trockner - raw_rows if raw_rows > 0 else 0
                st.metric(f"2️⃣ After Trockner {applied_trockner}", format_german_int(count_after_trockner), 
                         delta=format_german_int(delta_trockner) if delta_trockner != 0 else None,
                         delta_color="inverse")
            with col_fp3:
                delta_product = count_after_product - count_after_trockner
                st.metric("3️⃣ After Products", format_german_int(count_after_product),
                         delta=format_german_int(delta_product) if delta_product != 0 else None,
                         delta_color="inverse")
            with col_fp4:
                delta_month = count_after_month - count_after_product
                st.metric("4️⃣ After Month", format_german_int(count_after_month),
                         delta=format_german_int(delta_month) if delta_month != 0 else None,
                         delta_color="inverse")
            
            filter_parts = []
            if applied_trockner != "All":
                filter_parts.append(f"🏭 Trockner: **{applied_trockner}**")
            if products_applied and len(products_applied) < 12:
                filter_parts.append(f"🧱 Products: **{len(products_applied)}/12** selected")
            if month_applied:
                filter_parts.append(f"📅 Month: **{month_applied}**")
            
            if filter_parts:
                st.warning("**Active Filters:** " + " | ".join(filter_parts))
            else:
                st.success("✅ **No filters applied** - showing all data")

            # ============================================================
            #   WAGON AND VOLUME CALCULATION
            # ============================================================
            total_wagons = len(wagons_df)
            total_volume = wagons_df["m3"].sum()
            avg_volume_per_wagon = wagons_df["m3"].mean()
            unique_wagon_numbers = wagons_df["WG_Nr"].nunique()

            # ===== ENERGY TOTALS =====
            total_thermal = float(yearly["Energy_thermal_kWh"].sum())
            total_electrical = float(yearly["Energy_electrical_kWh"].sum())
            total_energy = float(yearly["Energy_kWh"].sum())
            
            expected_electrical = results.get("energy_electrical_raw", 0)
            st.write(f"DEBUG: total_electrical={format_german_int(total_electrical)}, expected={format_german_int(expected_electrical)}")
            
            if total_electrical < expected_electrical * 0.5:
                st.warning(f"⚠️ Electrical mismatch! Using raw value: {format_german_int(expected_electrical)} kWh")
                total_electrical = float(expected_electrical)
                total_energy = total_thermal + total_electrical

            # ===== WATER CALCULATION =====
            product_volume_all = wagons_df.groupby("Produkt")["m3"].sum().reset_index()
            
            water_calc_details = []
            total_water = 0.0
            
            for _, row in product_volume_all.iterrows():
                prod = row["Produkt"]
                vol = row["m3"]
                wagon_count_prod = len(wagons_df[wagons_df["Produkt"] == prod])
                
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
                    "Wagon Rows": wagon_count_prod,
                    "Volume (m³)": round(vol, 2),
                    "Water/m³ (kg)": round(water_per_m3, 1),
                    "Total Water (kg)": round(water_kg, 0),
                    "Water/Plate (kg)": round(water_per_plate_kg, 3) if water_per_plate_kg else 0,
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
                st.success(f"🏭 **Showing data for Trockner {applied_trockner} only** | After all filters: **{format_german_int(total_wagons)}** rows")
            else:
                st.info(f"🏭 **Showing data for all Trockner (A + B)** | After all filters: **{format_german_int(total_wagons)}** rows")

                         # ============================================================
            #                     SUMMARY KPIs SECTION
            # ============================================================
            st.markdown('<div class="section-header">📈 Summary KPIs</div>', unsafe_allow_html=True)

            st.subheader("🏭 Production")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(f'<div class="metric-card"><h3> Total Number of Wagon Rows </h3><h2>{format_german_int(total_wagons)}</h2></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><h3>Total Volume of the products</h3><h2>{format_german(total_volume, 2)} m³</h2></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><h3>Total Water Evaporated(in kg) </h3><h2>{format_german(total_water, 0)} kg</h2></div>', unsafe_allow_html=True)
   
            st.subheader("⚡ Energy Consumption")
            c5, c6, c7 = st.columns(3)

            with c5:
                st.markdown(f'<div class="metric-card"><h3>Thermal Energy</h3><h2>{format_german(total_thermal, 0)} kWh</h2></div>', unsafe_allow_html=True)
            with c6:
                st.markdown(f'<div class="metric-card"><h3>Electrical Energy</h3><h2>{format_german(total_electrical, 0)} kWh</h2></div>', unsafe_allow_html=True)
            with c7:
                st.markdown(f'<div class="metric-card"><h3>Total Energy</h3><h2>{format_german(total_energy, 0)} kWh</h2></div>', unsafe_allow_html=True)

            st.subheader("📊 Energy Efficiency")
            c8, c9, c10, c11 = st.columns(4)

            with c8:
                st.markdown(f'<div class="metric-card"><h3>kWh/kg water</h3><h2>{format_german(avg_kwh_per_kg, 3)} kWh/kg</h2></div>', unsafe_allow_html=True)
            with c9:
                st.markdown(f'<div class="metric-card"><h3>Thermal kWh/kg</h3><h2>{format_german(avg_kwh_thermal_per_kg, 3)} kWh/kg</h2></div>', unsafe_allow_html=True)
            with c10:
                st.markdown(f'<div class="metric-card"><h3>kWh/m³</h3><h2>{format_german(avg_kwh_per_m3, 1)} kWh/m³</h2></div>', unsafe_allow_html=True)
            with c11:
                st.markdown(f'<div class="metric-card"><h3>Thermal kWh/m³</h3><h2>{format_german(avg_kwh_thermal_per_m3, 1)} kWh/m³</h2></div>', unsafe_allow_html=True)
                


            # ===== INFO BOX =====
            st.info(
                f"🏭 **Trockner {applied_trockner}** | "
                f"⚡ **Energy Mix:** Thermal = **{format_german(thermal_pct, 1)}%** ({format_german_int(total_thermal)} kWh) | "
                f"Electrical = **{format_german(electrical_pct, 1)}%** ({format_german_int(total_electrical)} kWh) | "
                f"🚛 **Production:** {format_german_int(total_wagons)} wagon rows ({format_german_int(unique_wagon_numbers)} unique wagons) | {format_german_int(total_volume)} m³ | "
                f"💧 **Water:** {format_german_int(total_water)} kg ({format_german(total_water/1000, 1)} tons) evaporated"
            )

            # ============================================================
            #    DEBUG: FINAL VERIFICATION SECTION
            # ============================================================
            with st.expander("🔧 DEBUG 6: Final Data Verification", expanded=True):
                st.markdown("### ✅ Final Verification Checklist")
                
                st.markdown("**Wagon Count Verification:**")
                col_v1, col_v2, col_v3 = st.columns(3)
                
                with col_v1:
                    st.write(f"len(wagons_df) = **{format_german_int(len(wagons_df))}**")
                with col_v2:
                    st.write(f"total_wagons = **{format_german_int(total_wagons)}**")
                with col_v3:
                    if len(wagons_df) == total_wagons:
                        st.success("✅ Match!")
                    else:
                        st.error("❌ Mismatch!")
                
                st.markdown("**Volume Verification:**")
                col_v4, col_v5, col_v6 = st.columns(3)
                
                with col_v4:
                    st.write(f"wagons_df['m3'].sum() = **{format_german(wagons_df['m3'].sum(), 2)}**")
                with col_v5:
                    st.write(f"total_volume = **{format_german(total_volume, 2)}**")
                with col_v6:
                    if abs(wagons_df['m3'].sum() - total_volume) < 0.01:
                        st.success("✅ Match!")
                    else:
                        st.error("❌ Mismatch!")
                
                st.markdown("---")
                st.markdown("### 📊 Complete Filter Pipeline Trace")
                
                pipeline_data = [
                    {"Step": "1. Raw File", "Rows": format_german_int(raw_rows), "Change": "—"},
                    {"Step": f"2. After Trockner ({applied_trockner})", "Rows": format_german_int(count_after_trockner), 
                     "Change": f"-{format_german_int(raw_rows - count_after_trockner)}" if raw_rows > count_after_trockner else "0"},
                    {"Step": "3. After Product Filter", "Rows": format_german_int(count_after_product),
                     "Change": f"-{format_german_int(count_after_trockner - count_after_product)}" if count_after_trockner > count_after_product else "0"},
                    {"Step": "4. After Month Filter", "Rows": format_german_int(count_after_month),
                     "Change": f"-{format_german_int(count_after_product - count_after_month)}" if count_after_product > count_after_month else "0"},
                    {"Step": "5. Final (in wagons_df)", "Rows": format_german_int(len(wagons_df)),
                     "Change": "—"},
                ]
                st.dataframe(pd.DataFrame(pipeline_data), hide_index=True, use_container_width=True)
                
                if len(wagons_df) == count_after_month:
                    st.success(f"✅ Final wagon count ({format_german_int(len(wagons_df))}) matches step 4 ({format_german_int(count_after_month)})")
                else:
                    st.error(f"❌ Final wagon count ({format_german_int(len(wagons_df))}) does NOT match step 4 ({format_german_int(count_after_month)})")
                
                st.markdown("---")
                st.markdown("### 📋 Final wagons_df Sample (first 20 rows)")
                
                sample_cols = ["WG_Nr", "Produkt", "m3", "Month", "t0", "Trockner"]
                available_cols = [c for c in sample_cols if c in wagons_df.columns]
                
                sample_df = wagons_df[available_cols].head(20).copy()
                if "t0" in sample_df.columns:
                    sample_df["t0"] = sample_df["t0"].dt.strftime("%Y-%m-%d %H:%M")
                if "m3" in sample_df.columns:
                    sample_df["m3"] = sample_df["m3"].apply(lambda x: format_german(x, 4))
                
                st.dataframe(sample_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("### 📦 Product Breakdown in Final Data")
                
                product_breakdown = wagons_df.groupby("Produkt").agg({
                    "WG_Nr": "count",
                    "m3": ["sum", "mean"]
                }).round(4)
                product_breakdown.columns = ["Row Count", "Total Volume (m³)", "Avg Volume (m³)"]
                product_breakdown = product_breakdown.reset_index()
                product_breakdown = product_breakdown.sort_values("Total Volume (m³)", ascending=False)
                
                # Format for German display
                pb_display = product_breakdown.copy()
                pb_display["Row Count"] = pb_display["Row Count"].apply(format_german_int)
                pb_display["Total Volume (m³)"] = pb_display["Total Volume (m³)"].apply(lambda x: format_german(x, 2))
                pb_display["Avg Volume (m³)"] = pb_display["Avg Volume (m³)"].apply(lambda x: format_german(x, 4))
                
                st.dataframe(pb_display, use_container_width=True, hide_index=True)
                
                st.markdown(f"**Total:** {format_german_int(int(product_breakdown['Row Count'].sum()))} rows | {format_german(product_breakdown['Total Volume (m³)'].sum(), 2)} m³")

            # ============================================================
            #    ENERGY CALCULATION EXPLANATION (EXPANDABLE)
            # ============================================================
            with st.expander("⚡ Energy Consumption Calculation - Detailed Explanation"):
                st.markdown("### 📊 How is Energy Consumption Calculated?")
                
                st.markdown("""
                **Overview:**  
                Total energy consumption is read from the hourly energy file, 
                allocated to products and zones, then aggregated.
                """)
                
                st.markdown("---")
                st.markdown("### 📥 Step 1: Energy Input Data")
                
                if "energy" in results and not results["energy"].empty:
                    energy_df = results["energy"]
                    
                    input_thermal_total = energy_df["E_thermal_total_kWh"].sum()
                    input_electrical_total = energy_df["E_el_kWh"].sum()
                    input_total_energy = input_thermal_total + input_electrical_total
                    
                    total_hours = len(energy_df)
                    avg_thermal_per_hour = input_thermal_total / total_hours if total_hours > 0 else 0
                    avg_electrical_per_hour = input_electrical_total / total_hours if total_hours > 0 else 0
                    
                    col_e1, col_e2, col_e3 = st.columns(3)
                    
                    with col_e1:
                        st.metric("Input: Thermal Energy", f"{format_german_int(input_thermal_total)} kWh")
                        st.caption(f"Average: {format_german(avg_thermal_per_hour, 1)} kWh/hour")
                    
                    with col_e2:
                        st.metric("Input: Electrical Energy", f"{format_german_int(input_electrical_total)} kWh")
                        st.caption(f"Average: {format_german(avg_electrical_per_hour, 1)} kWh/hour")
                    
                    with col_e3:
                        st.metric("Input: Total Energy", f"{format_german_int(input_total_energy)} kWh")
                        st.caption(f"Over {format_german_int(total_hours)} hours")
                    
                    st.markdown("**Source:** Energy Excel file (hourly measurements)")
                    
                    st.code(f"""
Thermal Energy = Gas Consumption (m³) × 11,5 kWh/m³

Zones: Z2, Z3, Z4, Z5
Total Thermal Energy = Z2 + Z3 + Z4 + Z5

Example for one hour:
- Gas Z2: 15,2 m³ × 11,5 = 174,8 kWh
- Gas Z3: 20,3 m³ × 11,5 = 233,4 kWh
- Gas Z4: 17,3 m³ × 11,5 = 198,9 kWh
- Gas Z5: 13,6 m³ × 11,5 = 156,4 kWh
- Total Thermal: 763,5 kWh
- Electrical: 45,0 kWh (directly measured)
- Hour Total: 808,5 kWh
                    """, language="text")
                    
                    st.markdown("**Sample Energy Data:**")
                    sample_energy = energy_df.head(10).copy()
                    display_cols = ["E_start", "E_thermal_total_kWh", "E_el_kWh", "Month"]
                    available_cols = [col for col in display_cols if col in sample_energy.columns]
                    st.dataframe(sample_energy[available_cols], use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("### 🔄 Step 2: Energy Allocation to Products & Zones")
                
                if "allocation" in results and not results["allocation"].empty:
                    allocation_df = results["allocation"]
                    
                    allocated_thermal = allocation_df["Energy_thermal_kWh"].sum()
                    allocated_electrical = allocation_df["Energy_electrical_kWh"].sum()
                    allocated_total = allocation_df["Energy_share_kWh"].sum()
                    
                    num_allocation_rows = len(allocation_df)
                    unique_wagons_allocated = allocation_df["Produkt"].nunique()
                    
                    st.code(f"""
Example: Hour 01.01.2024 10:00-11:00

Energy available:
- Thermal: 850 kWh
- Electrical: 45 kWh
- Total: 895 kWh

Wagons in dryer during this hour:
- Wagon 1234 (L36) in Z2: 08:00-10:30 → overlaps 30 min (10:00-10:30)
- Wagon 1235 (L38) in Z3: 09:00-12:00 → overlaps 60 min (10:00-11:00)
- Wagon 1236 (L36) in Z4: 10:30-14:00 → overlaps 30 min (10:30-11:00)
- Wagon 1237 (L42) in Z5: 09:30-11:30 → overlaps 60 min (10:00-11:00)

Total overlap time: 30 + 60 + 30 + 60 = 180 minutes

Share calculation:
- Wagon 1234: 30/180 = 16,67%  → 850 × 0,1667 = 141,7 kWh thermal
- Wagon 1235: 60/180 = 33,33%  → 850 × 0,3333 = 283,3 kWh thermal
- Wagon 1236: 30/180 = 16,67%  → 850 × 0,1667 = 141,7 kWh thermal
- Wagon 1237: 60/180 = 33,33%  → 850 × 0,3333 = 283,3 kWh thermal

Sum: 141,7 + 283,3 + 141,7 + 283,3 = 850,0 kWh ✅

This process repeats for each hour.
                    """, language="text")
                    
                    col_a1, col_a2 = st.columns(2)
                    
                    with col_a1:
                        st.metric("Allocation Rows", format_german_int(num_allocation_rows))
                        st.caption("Number of wagon×zone×hour allocations")
                    
                    with col_a2:
                        st.metric("Unique Products", format_german_int(unique_wagons_allocated))
                        st.caption("Products with energy allocation")
                
                st.markdown("---")
                st.markdown("### 📈 Step 3: Final KPI Calculation")
                
                st.code(f"""
Summary KPI Calculation:

# Step 1: Sum from yearly DataFrame
total_thermal = yearly["Energy_thermal_kWh"].sum()
              = {format_german_int(total_thermal)} kWh

total_electrical = yearly["Energy_electrical_kWh"].sum()
                 = {format_german_int(total_electrical)} kWh

total_energy = yearly["Energy_kWh"].sum()
             = {format_german_int(total_energy)} kWh

# Step 2: Percentage shares
thermal_pct = (total_thermal / total_energy) × 100
            = ({format_german_int(total_thermal)} / {format_german_int(total_energy)}) × 100
            = {format_german(thermal_pct, 1)}%

electrical_pct = (total_electrical / total_energy) × 100
               = ({format_german_int(total_electrical)} / {format_german_int(total_energy)}) × 100
               = {format_german(electrical_pct, 1)}%
                """, language="text")
                
                st.success("✅ All energy consumption KPIs are fully documented and validated.")
            
            # ============================================================
            #    DETAILED kWh/kg CALCULATION TRANSPARENCY
            # ============================================================
            with st.expander("🔍 kWh/kg Water Calculation - Detailed Breakdown"):
                st.markdown("### 📊 How kWh/kg is Calculated")
                
                st.markdown("""
                **Formula:** `kWh/kg = Total Energy (kWh) ÷ Total Water Evaporated (kg)`
                
                **Important Notes:**
                - Water is calculated **per product** based on the formula-derived water content
                - Volume comes from wagon tracking data (ALL rows, m³ column AA)
                - Energy is the total allocated energy from all zones
                """)
                
                st.markdown("---")
                st.markdown("### 📦 Step 1: Volume from All Wagon Rows")
                st.code(f"""
Total Wagon Rows: {format_german_int(total_wagons)}
Unique Wagon Numbers: {format_german_int(unique_wagon_numbers)}
Total Volume: {format_german(total_volume, 2)} m³
Average Volume/Row: {format_german(avg_volume_per_wagon, 4)} m³
Source: m³ column (AA) from Hordenwagen file
                """, language="text")
                
                st.markdown("### 💧 Step 2: Water Calculation per Product")
                water_df = pd.DataFrame(water_calc_details)
                
                # Format water_df for German display
                water_df_display = water_df.copy()
                water_df_display["Wagon Rows"] = water_df_display["Wagon Rows"].apply(format_german_int)
                water_df_display["Volume (m³)"] = water_df_display["Volume (m³)"].apply(lambda x: format_german(x, 2))
                water_df_display["Water/m³ (kg)"] = water_df_display["Water/m³ (kg)"].apply(lambda x: format_german(x, 1))
                water_df_display["Total Water (kg)"] = water_df_display["Total Water (kg)"].apply(lambda x: format_german_int(x))
                water_df_display["Water/Plate (kg)"] = water_df_display["Water/Plate (kg)"].apply(lambda x: format_german(x, 3))
                
                st.dataframe(water_df_display, use_container_width=True, hide_index=True)
                
                st.code(f"""
Total Water = Σ(Volume × Water/m³) for each product
            = {format_german_int(total_water)} kg
            = {format_german(total_water/1000, 1)} tons
                """, language="text")
                
                st.markdown("### ⚡ Step 3: Energy Totals")
                st.code(f"""
Thermal Energy:    {format_german_int(total_thermal)} kWh
Electrical Energy: {format_german_int(total_electrical)} kWh
Total Energy:      {format_german_int(total_energy)} kWh
                """, language="text")
                
                st.markdown("### 📈 Step 4: Final KPI Calculation")
                st.code(f"""
kWh/kg (Total) = Total Energy / Total Water
               = {format_german_int(total_energy)} / {format_german_int(total_water)}
               = {format_german(avg_kwh_per_kg, 4)} kWh/kg

kWh/kg (Thermal) = Thermal Energy / Total Water
                 = {format_german_int(total_thermal)} / {format_german_int(total_water)}
                 = {format_german(avg_kwh_thermal_per_kg, 4)} kWh/kg
                """, language="text")
                
                st.markdown("### 🔗 Cross-Check: kWh/m³ vs kWh/kg")
                calculated_kwh_m3 = avg_kwh_per_kg * avg_water_per_m3
                st.code(f"""
kWh/m³ should ≈ kWh/kg × Water/m³
Calculated: {format_german(avg_kwh_per_kg, 4)} × {format_german(avg_water_per_m3, 1)} = {format_german(calculated_kwh_m3, 1)} kWh/m³
Actual:     {format_german(avg_kwh_per_m3, 1)} kWh/m³
Difference: {format_german(abs(calculated_kwh_m3 - avg_kwh_per_m3), 1)} kWh/m³
                """, language="text")
            
            # ============================================================
            #    VOLUME BREAKDOWN & VALIDATION
            # ============================================================
            with st.expander("📊 Volume Breakdown & Wagon Count Validation"):
                st.markdown("### 🚛 Wagon Count Methodology")
                
                st.markdown(f"""
                **How wagons are counted:**
                1. **Column A (WG-Nr)** in the Excel file contains wagon numbers
                2. Each **row with a valid wagon number** = **1 wagon row** (one usage/batch)
                3. One physical wagon can be used multiple times → multiple rows
                4. Trockner filter: **{applied_trockner}**
                
                **Filter Pipeline:**
                - Raw file: **{format_german_int(raw_rows)}** rows
                - After Trockner {applied_trockner}: **{format_german_int(count_after_trockner)}** rows
                - After Product filter: **{format_german_int(count_after_product)}** rows
                - After Month filter: **{format_german_int(count_after_month)}** rows
                - **Final count: {format_german_int(total_wagons)} wagon rows**
                """)
                
                st.markdown("---")
                st.markdown("### 📊 Summary Statistics")
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                
                with col_s1:
                    st.metric("Total Wagon Rows", format_german_int(total_wagons))
                    st.caption(f"{format_german_int(unique_wagon_numbers)} unique wagons")
                with col_s2:
                    st.metric("Total Volume", f"{format_german(total_volume, 2)} m³")
                    st.caption(f"Sum of m³ column (AA)")
                with col_s3:
                    st.metric("Avg Volume/Row", f"{format_german(avg_volume_per_wagon, 4)} m³")
                    st.caption("Mean of m³ column (AA)")
                with col_s4:
                    unique_products = wagons_df["Produkt"].nunique()
                    st.metric("Unique Products", format_german_int(unique_products))
                
                st.markdown("---")
                st.markdown("### 📦 Breakdown by Product (All Rows)")
                
                product_breakdown = wagons_df.groupby("Produkt").agg({
                    "WG_Nr": "count",
                    "m3": ["sum", "mean", "min", "max"]
                }).round(4)
                
                product_breakdown.columns = ["Row Count", "Total Volume (m³)", "Avg Volume (m³)", "Min Volume (m³)", "Max Volume (m³)"]
                product_breakdown = product_breakdown.reset_index()
                product_breakdown = product_breakdown.sort_values("Total Volume (m³)", ascending=False)
                
                product_breakdown["% of Rows"] = (product_breakdown["Row Count"] / total_wagons * 100).round(1)
                product_breakdown["% of Volume"] = (product_breakdown["Total Volume (m³)"] / total_volume * 100).round(1)
                
                # Format for German display
                pb_display = product_breakdown.copy()
                pb_display["Row Count"] = pb_display["Row Count"].apply(format_german_int)
                pb_display["Total Volume (m³)"] = pb_display["Total Volume (m³)"].apply(lambda x: format_german(x, 2))
                pb_display["Avg Volume (m³)"] = pb_display["Avg Volume (m³)"].apply(lambda x: format_german(x, 4))
                pb_display["Min Volume (m³)"] = pb_display["Min Volume (m³)"].apply(lambda x: format_german(x, 4))
                pb_display["Max Volume (m³)"] = pb_display["Max Volume (m³)"].apply(lambda x: format_german(x, 4))
                pb_display["% of Rows"] = pb_display["% of Rows"].apply(lambda x: format_german(x, 1) + "%")
                pb_display["% of Volume"] = pb_display["% of Volume"].apply(lambda x: format_german(x, 1) + "%")
                
                st.dataframe(pb_display, use_container_width=True, hide_index=True)
                
                st.markdown(f"""
                **Totals:** {format_german_int(total_wagons)} wagon rows | {format_german(total_volume, 2)} m³ | {format_german(avg_volume_per_wagon, 4)} m³/row (mean from AA column)
                """)
                
                st.markdown("---")
                st.markdown("### 📅 Breakdown by Month (All Rows)")
                
                monthly_breakdown = wagons_df.groupby("Month").agg({
                    "WG_Nr": "count",
                    "m3": ["sum", "mean"]
                }).round(4)
                monthly_breakdown.columns = ["Row Count", "Total Volume (m³)", "Avg Volume (m³)"]
                monthly_breakdown = monthly_breakdown.reset_index()
                
                # Format for German display
                mb_display = monthly_breakdown.copy()
                mb_display["Row Count"] = mb_display["Row Count"].apply(format_german_int)
                mb_display["Total Volume (m³)"] = mb_display["Total Volume (m³)"].apply(lambda x: format_german(x, 2))
                mb_display["Avg Volume (m³)"] = mb_display["Avg Volume (m³)"].apply(lambda x: format_german(x, 4))
                
                st.dataframe(mb_display, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("### 🔍 Sample Wagon Data (First 20 rows)")
                
                sample_cols = ["WG_Nr", "Produkt", "m3", "Month", "t0", "Trockner"]
                available_cols = [c for c in sample_cols if c in wagons_df.columns]
                
                sample_df = wagons_df[available_cols].head(20).copy()
                if "t0" in sample_df.columns:
                    sample_df["t0"] = sample_df["t0"].dt.strftime("%Y-%m-%d %H:%M")
                if "m3" in sample_df.columns:
                    sample_df["m3"] = sample_df["m3"].apply(lambda x: format_german(x, 4))
                
                st.dataframe(sample_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("### ✅ Validation")
                
                volume_from_breakdown = product_breakdown["Total Volume (m³)"].sum()
                row_count_from_breakdown = int(product_breakdown["Row Count"].sum())
                avg_from_mean = wagons_df["m3"].mean()
                
                col_v1, col_v2, col_v3 = st.columns(3)
                
                with col_v1:
                    if abs(volume_from_breakdown - total_volume) < 0.01:
                        st.success(f"✅ Volume check: {format_german(total_volume, 2)} m³")
                    else:
                        st.error(f"❌ Volume mismatch: {format_german(total_volume, 2)} vs {format_german(volume_from_breakdown, 2)}")
                
                with col_v2:
                    if row_count_from_breakdown == total_wagons:
                        st.success(f"✅ Row count: {format_german_int(total_wagons)}")
                    else:
                        st.error(f"❌ Count mismatch: {format_german_int(total_wagons)} vs {format_german_int(row_count_from_breakdown)}")
                
                with col_v3:
                    if abs(avg_from_mean - avg_volume_per_wagon) < 0.0001:
                        st.success(f"✅ Avg volume: {format_german(avg_volume_per_wagon, 4)} m³")
                    else:
                        st.error(f"❌ Avg mismatch: {format_german(avg_volume_per_wagon, 4)} vs {format_german(avg_from_mean, 4)}")
                
                st.markdown("---")
                st.markdown("### 🔧 Debug Information")
                
                st.code(f"""
Wagon Count Calculation:
- Total rows in wagons_df: {format_german_int(len(wagons_df))}
- Unique wagon numbers (WG_Nr): {format_german_int(wagons_df["WG_Nr"].nunique())}
- One wagon used multiple times = multiple rows
- Final count: {format_german_int(total_wagons)} rows

Volume Calculation:
- Total volume (sum of m³): {format_german(total_volume, 2)} m³
- Average volume (mean of m³): {format_german(avg_volume_per_wagon, 4)} m³
- Source: Column AA (m³) from Hordenwagen file

Verification:
- Sum ÷ Count = {format_german(total_volume / total_wagons, 4)} m³
- Direct mean = {format_german(wagons_df["m3"].mean(), 4)} m³
- Match: {"✅ Yes" if abs(total_volume/total_wagons - avg_volume_per_wagon) < 0.0001 else "❌ No"}
                """, language="text")

            # ===== 2. ZONE COMPARISON =====
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
                    text=[format_german_int(v) for v in zone_totals['Energy_thermal_kWh']],
                    textposition='inside'
                ))
                fig_zone_energy.add_trace(go.Bar(
                    name='Electrical',
                    x=zone_totals['Zone'],
                    y=zone_totals['Energy_electrical_kWh'],
                    marker_color='#4ECDC4',
                    text=[format_german_int(v) for v in zone_totals['Energy_electrical_kWh']],
                    textposition='inside'
                ))
                fig_zone_energy.update_layout(
                    title="Energy Consumption by Zone (kWh)",
                    yaxis_title="Energy (kWh)",
                    barmode='stack',
                    height=400,
                    plot_bgcolor="white",
                    separators=",."  # German number format for Plotly
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
                fig_pie.update_layout(height=400, separators=",.")
                st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("Zone Energy Summary")
            zone_display = zone_totals.copy()
            zone_display["Thermal %"] = (
                zone_display["Energy_thermal_kWh"] / zone_display["Energy_kWh"] * 100
            ).round(1)
            zone_display["Electrical %"] = (
                zone_display["Energy_electrical_kWh"] / zone_display["Energy_kWh"] * 100
            ).round(1)
            
            # Format for German display
            zd_display = zone_display.copy()
            zd_display["Energy_thermal_kWh"] = zd_display["Energy_thermal_kWh"].apply(lambda x: format_german(x, 0))
            zd_display["Energy_electrical_kWh"] = zd_display["Energy_electrical_kWh"].apply(lambda x: format_german(x, 0))
            zd_display["Energy_kWh"] = zd_display["Energy_kWh"].apply(lambda x: format_german(x, 0))
            zd_display["Volume_m3"] = zd_display["Volume_m3"].apply(lambda x: format_german(x, 2))
            zd_display["Water_kg"] = zd_display["Water_kg"].apply(lambda x: format_german(x, 0))
            zd_display["kWh_per_m3"] = zd_display["kWh_per_m3"].apply(lambda x: format_german(x, 1))
            zd_display["Thermal %"] = zd_display["Thermal %"].apply(lambda x: format_german(x, 1) + "%")
            zd_display["Electrical %"] = zd_display["Electrical %"].apply(lambda x: format_german(x, 1) + "%")
            
            zd_display = zd_display.rename(columns={
                "Energy_thermal_kWh": "Thermal (kWh)",
                "Energy_electrical_kWh": "Electrical (kWh)",
                "Energy_kWh": "Total (kWh)",
                "Volume_m3": "Volume (m³)",
                "Water_kg": "Water (kg)",
                "kWh_per_m3": "kWh/m³"
            })
            st.dataframe(zd_display, use_container_width=True, hide_index=True)

            # ===== ZONE DURATION STATISTICS =====
            st.markdown('<div class="section-header">⏱️ Time in Each Zone</div>', unsafe_allow_html=True)
            
            zone_overall_stats = results.get("zone_overall_stats", pd.DataFrame())
            zone_product_stats = results.get("zone_product_stats", pd.DataFrame())
            
            if zone_overall_stats is not None and not zone_overall_stats.empty:
                st.subheader("📊 Average Time in Each Zone")
                
                col_zd1, col_zd2 = st.columns(2)
                
                with col_zd1:
                    fig_zone_dur = px.bar(
                        zone_overall_stats,
                        x="Zone",
                        y="Avg_Hours",
                        title="Average Time in Each Zone (Hours)",
                        text_auto='.1f',
                        color="Zone",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_zone_dur.update_traces(textposition='outside')
                    fig_zone_dur.update_layout(height=350, plot_bgcolor="white", showlegend=False, separators=",.")
                    st.plotly_chart(fig_zone_dur, use_container_width=True)
                
                with col_zd2:
                    zone_dur_display = zone_overall_stats.copy()
                    zone_dur_display["Avg_Hours"] = zone_dur_display["Avg_Hours"].apply(lambda x: format_german(x, 2))
                    zone_dur_display["Min_Hours"] = zone_dur_display["Min_Hours"].apply(lambda x: format_german(x, 2))
                    zone_dur_display["Max_Hours"] = zone_dur_display["Max_Hours"].apply(lambda x: format_german(x, 2))
                    zone_dur_display["Std_Hours"] = zone_dur_display["Std_Hours"].apply(lambda x: format_german(x, 2))
                    zone_dur_display["Count"] = zone_dur_display["Count"].apply(format_german_int)
                    
                    zone_dur_display = zone_dur_display.rename(columns={
                        "Avg_Hours": "Avg (h)",
                        "Min_Hours": "Min (h)",
                        "Max_Hours": "Max (h)",
                        "Std_Hours": "Std Dev (h)",
                        "Count": "Samples"
                    })
                    
                    st.dataframe(zone_dur_display, use_container_width=True, hide_index=True)
                    
                    total_avg = zone_overall_stats["Avg_Hours"].sum()
                    st.metric("Total Avg Residence Time", f"{format_german(total_avg, 1)} hours ({format_german(total_avg/24, 1)} days)")
                
                if zone_product_stats is not None and not zone_product_stats.empty:
                    with st.expander("📦 Zone Duration by Product"):
                        pivot_zone_dur = zone_product_stats.pivot_table(
                            index="Produkt",
                            columns="Zone",
                            values="Avg_Hours",
                            aggfunc="mean"
                        ).round(2)
                        
                        pivot_zone_dur["Total"] = pivot_zone_dur.sum(axis=1).round(2)
                        
                        # Format for German display
                        pivot_display = pivot_zone_dur.copy()
                        for col in pivot_display.columns:
                            pivot_display[col] = pivot_display[col].apply(lambda x: format_german(x, 2) if pd.notna(x) else "–")
                        
                        st.dataframe(pivot_display, use_container_width=True)
                        
                        fig_heatmap = px.imshow(
                            pivot_zone_dur.drop(columns=["Total"], errors="ignore"),
                            title="Average Zone Duration by Product (Hours)",
                            labels=dict(x="Zone", y="Product", color="Hours"),
                            color_continuous_scale="YlOrRd",
                            aspect="auto"
                        )
                        fig_heatmap.update_layout(height=400, separators=",.")
                        st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("⏱️ Zone duration data not available.")

            # ===== 3. MONTHLY & WEEKLY TRENDS =====
            st.markdown(
                '<div class="section-header">📊 Monthly & Weekly KPI Trends</div>',
                unsafe_allow_html=True
            )

            # ===== TIMELINE FILTER =====
            st.subheader("📅 Timeline Filter")
            
            if "t0" in wagons_df.columns:
                min_date = wagons_df["t0"].min()
                max_date = wagons_df["t0"].max()
            else:
                min_date = pd.Timestamp.now() - pd.Timedelta(days=365)
                max_date = pd.Timestamp.now()
            
            col_date1, col_date2 = st.columns(2)
            
            with col_date1:
                start_date = st.date_input(
                    "Start Date",
                    value=pd.to_datetime(min_date).date(),
                    min_value=pd.to_datetime(min_date).date(),
                    max_value=pd.to_datetime(max_date).date(),
                    key="trend_start_date"
                )
            
            with col_date2:
                end_date = st.date_input(
                    "End Date",
                    value=pd.to_datetime(max_date).date(),
                    min_value=pd.to_datetime(min_date).date(),
                    max_value=pd.to_datetime(max_date).date(),
                    key="trend_end_date"
                )
            
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            date_mask = (wagons_df["t0"] >= start_datetime) & (wagons_df["t0"] <= end_datetime)
            wagons_date_filtered = wagons_df[date_mask].copy()
            
            months_in_range = wagons_date_filtered["Month"].unique().tolist() if not wagons_date_filtered.empty else []
            
            summary_date_filtered = summary[summary["Month"].isin(months_in_range)].copy()
            
            days_selected = (end_datetime - start_datetime).days + 1
            st.info(f"📅 **Selected Period:** {start_date} to {end_date} ({format_german_int(days_selected)} days) | "
                   f"**Wagons:** {format_german_int(len(wagons_date_filtered))} | "
                   f"**Volume:** {format_german(wagons_date_filtered['m3'].sum(), 2)} m³")

            available_products = sorted(summary_date_filtered["Produkt"].unique().tolist())
            
            if not available_products:
                st.warning("⚠️ No data available for the selected date range.")
            else:
                st.subheader("🎯 Filter by Product")
                col_filter1, col_filter2 = st.columns([3, 1])
                
                with col_filter1:
                    selected_products_trends = st.multiselect(
                        "Select products to display:",
                        options=available_products,
                        default=available_products,
                        key="trends_product_filter"
                    )
                
                with col_filter2:
                    if st.button("Select All", key="select_all_trends"):
                        selected_products_trends = available_products
                    if st.button("Clear All", key="clear_all_trends"):
                        selected_products_trends = []

                if not selected_products_trends:
                    st.warning("⚠️ Please select at least one product.")
                else:
                    summary_filtered = summary_date_filtered[
                        summary_date_filtered["Produkt"].isin(selected_products_trends)
                    ].copy()
                    
                    if summary_filtered.empty:
                        st.warning("⚠️ No data for selected products in this date range.")
                    else:
                        st.info(f"📊 Showing **{len(selected_products_trends)}** product(s)")

                        # Monthly by Product
                        monthly_product = summary_filtered.groupby(["Month", "Produkt"], as_index=False).agg({
                            "Energy_thermal_kWh": "sum",
                            "Energy_electrical_kWh": "sum",
                            "Energy_kWh": "sum",
                            "Volume_m3": "sum",
                            "Water_kg": "sum",
                        })
                        
                        monthly_product = monthly_product[monthly_product["Volume_m3"] > 0].copy()
                        
                        monthly_product["kWh_per_m3"] = monthly_product["Energy_kWh"] / monthly_product["Volume_m3"]
                        monthly_product["kWh_per_kg"] = np.where(
                            monthly_product["Water_kg"] > 0,
                            monthly_product["Energy_kWh"] / monthly_product["Water_kg"],
                            0
                        )
                        monthly_product["kWh_thermal_per_m3"] = monthly_product["Energy_thermal_kWh"] / monthly_product["Volume_m3"]

                        # Monthly by Zone
                        monthly_zone = summary_filtered.groupby(["Month", "Zone"], as_index=False).agg({
                            "Energy_thermal_kWh": "sum",
                            "Energy_electrical_kWh": "sum",
                            "Energy_kWh": "sum",
                            "Volume_m3": "sum",
                            "Water_kg": "sum",
                        })
                        
                        monthly_zone = monthly_zone[monthly_zone["Volume_m3"] > 0].copy()
                        
                        monthly_zone["kWh_per_m3"] = monthly_zone["Energy_kWh"] / monthly_zone["Volume_m3"]
                        monthly_zone["kWh_per_kg"] = np.where(
                            monthly_zone["Water_kg"] > 0,
                            monthly_zone["Energy_kWh"] / monthly_zone["Water_kg"],
                            0
                        )

                        # Monthly Overall
                        monthly_overall = summary_filtered.groupby(["Month"], as_index=False).agg({
                            "Energy_thermal_kWh": "sum",
                            "Energy_electrical_kWh": "sum",
                            "Energy_kWh": "sum",
                            "Volume_m3": "sum",
                            "Water_kg": "sum",
                        })
                        
                        monthly_overall["kWh_per_m3"] = np.where(
                            monthly_overall["Volume_m3"] > 0,
                            monthly_overall["Energy_kWh"] / monthly_overall["Volume_m3"],
                            0
                        )
                        monthly_overall["kWh_per_kg"] = np.where(
                            monthly_overall["Water_kg"] > 0,
                            monthly_overall["Energy_kWh"] / monthly_overall["Water_kg"],
                            0
                        )

                        # Weekly data
                        weekly_energy = None
                        if "energy" in results and not results["energy"].empty:
                            energy_df = results["energy"].copy()
                            
                            if "E_start" in energy_df.columns:
                                energy_df = energy_df[
                                    (energy_df["E_start"] >= start_datetime) & 
                                    (energy_df["E_start"] <= end_datetime)
                                ]
                            
                            if not energy_df.empty:
                                energy_df["Week"] = energy_df["E_start"].dt.isocalendar().week
                                energy_df["Year"] = energy_df["E_start"].dt.year

                                weekly_energy = energy_df.groupby(["Year", "Week"], as_index=False).agg({
                                    "E_thermal_total_kWh": "sum",
                                    "E_el_kWh": "sum"
                                })
                                weekly_energy["Total_kWh"] = weekly_energy["E_thermal_total_kWh"] + weekly_energy["E_el_kWh"]
                                weekly_energy["Week_Label"] = (
                                    weekly_energy["Year"].astype(str) + "-W" +
                                    weekly_energy["Week"].astype(str).str.zfill(2)
                                )
                                weekly_energy = weekly_energy[weekly_energy["Total_kWh"] > 0]

                        # Overall Trends
                        st.subheader("📈 Overall Performance Trends")
                        
                        filtered_thermal = monthly_overall["Energy_thermal_kWh"].sum()
                        filtered_electrical = monthly_overall["Energy_electrical_kWh"].sum()
                        filtered_energy = monthly_overall["Energy_kWh"].sum()
                        filtered_volume = monthly_overall["Volume_m3"].sum()
                        filtered_water = monthly_overall["Water_kg"].sum()
                        
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric("Total Energy", f"{format_german_int(filtered_energy)} kWh")
                        with col_m2:
                            st.metric("Total Volume", f"{format_german_int(filtered_volume)} m³")
                        with col_m3:
                            st.metric("Total Water", f"{format_german_int(filtered_water)} kg")
                        with col_m4:
                            filtered_kwh_kg = filtered_energy / filtered_water if filtered_water > 0 else 0
                            st.metric("kWh/kg", format_german(filtered_kwh_kg, 3))

                        # Monthly Charts
                        col_o1, col_o2 = st.columns(2)

                        with col_o1:
                            fig_monthly_energy = go.Figure()
                            fig_monthly_energy.add_trace(go.Bar(
                                name='Thermal',
                                x=monthly_overall["Month"],
                                y=monthly_overall["Energy_thermal_kWh"],
                                marker_color='#FF6B6B',
                                text=[format_german_int(v) for v in monthly_overall["Energy_thermal_kWh"]],
                                textposition='inside'
                            ))
                            fig_monthly_energy.add_trace(go.Bar(
                                name='Electrical',
                                x=monthly_overall["Month"],
                                y=monthly_overall["Energy_electrical_kWh"],
                                marker_color='#4ECDC4',
                                text=[format_german_int(v) for v in monthly_overall["Energy_electrical_kWh"]],
                                textposition='inside'
                            ))
                            fig_monthly_energy.update_layout(
                                title=f"Monthly Energy ({start_date} to {end_date})",
                                xaxis_title="Month",
                                yaxis_title="Energy (kWh)",
                                barmode='stack',
                                height=350,
                                plot_bgcolor="white",
                                separators=",."
                            )
                            st.plotly_chart(fig_monthly_energy, use_container_width=True)

                        with col_o2:
                            fig_monthly_volume = px.bar(
                                monthly_overall,
                                x="Month",
                                y="Volume_m3",
                                title=f"Monthly Volume ({start_date} to {end_date})",
                                labels={"Volume_m3": "Volume (m³)", "Month": "Month"},
                                text_auto='.0f'
                            )
                            fig_monthly_volume.update_traces(marker_color='#26de81', textposition='outside')
                            fig_monthly_volume.update_layout(height=350, plot_bgcolor="white", showlegend=False, separators=",.")
                            st.plotly_chart(fig_monthly_volume, use_container_width=True)

                        # Weekly Trends
                        st.subheader("📅 Weekly Energy Trends (Based on Zone 2 Entry)")
                        
                        if weekly_energy is not None and not weekly_energy.empty:
                            col_w1, col_w2 = st.columns(2)

                            with col_w1:
                                fig_weekly_total = go.Figure()
                                fig_weekly_total.add_trace(go.Bar(
                                    name='Thermal',
                                    x=weekly_energy["Week_Label"],
                                    y=weekly_energy["E_thermal_total_kWh"],
                                    marker_color='#FF6B6B'
                                ))
                                fig_weekly_total.add_trace(go.Bar(
                                    name='Electrical',
                                    x=weekly_energy["Week_Label"],
                                    y=weekly_energy["E_el_kWh"],
                                    marker_color='#4ECDC4'
                                ))
                                fig_weekly_total.update_layout(
                                    title=f"Weekly Energy ({start_date} to {end_date})",
                                    xaxis_title="Week",
                                    yaxis_title="Energy (kWh)",
                                    barmode='stack',
                                    height=350,
                                    plot_bgcolor="white",
                                    xaxis_tickangle=-45,
                                    separators=",."
                                )
                                st.plotly_chart(fig_weekly_total, use_container_width=True)

                            with col_w2:
                                fig_weekly_trend = px.line(
                                    weekly_energy,
                                    x="Week_Label",
                                    y="Total_kWh",
                                    markers=True,
                                    title=f"Weekly Total Energy Trend"
                                )
                                fig_weekly_trend.update_traces(line_color='#667eea', line_width=3)
                                fig_weekly_trend.update_layout(
                                    height=350,
                                    plot_bgcolor="white",
                                    xaxis_tickangle=-45,
                                    separators=",."
                                )
                                st.plotly_chart(fig_weekly_trend, use_container_width=True)

                            avg_weekly = weekly_energy["Total_kWh"].mean()
                            max_weekly = weekly_energy["Total_kWh"].max()
                            st.info(f"📊 **Weekly Stats:** Avg = **{format_german_int(avg_weekly)} kWh/week** | Peak = **{format_german_int(max_weekly)} kWh**")
                        else:
                            st.info("📅 No weekly energy data for the selected period")

                        # Monthly KPI Charts
                        col_o3, col_o4 = st.columns(2)
                        
                        with col_o3:
                            fig_monthly_kwh_m3 = px.line(
                                monthly_overall,
                                x="Month",
                                y="kWh_per_m3",
                                markers=True,
                                title="Monthly kWh/m³ Trend"
                            )
                            fig_monthly_kwh_m3.update_traces(line_color='#667eea', line_width=3)
                            fig_monthly_kwh_m3.update_layout(height=300, plot_bgcolor="white", separators=",.")
                            st.plotly_chart(fig_monthly_kwh_m3, use_container_width=True)

                        with col_o4:
                            fig_monthly_kwh_kg = px.line(
                                monthly_overall,
                                x="Month",
                                y="kWh_per_kg",
                                markers=True,
                                title="Monthly kWh/kg Trend"
                            )
                            fig_monthly_kwh_kg.update_traces(line_color='#f093fb', line_width=3)
                            fig_monthly_kwh_kg.update_layout(height=300, plot_bgcolor="white", separators=",.")
                            st.plotly_chart(fig_monthly_kwh_kg, use_container_width=True)

                        # Trends by Product
                        with st.expander("🧱 Trends by Product - Detailed Charts"):
                            if monthly_product.empty:
                                st.warning("No product data available.")
                            else:
                                st.markdown("### Energy Efficiency by Product")
                                
                                col_p1, col_p2 = st.columns(2)

                                with col_p1:
                                    fig_prod_efficiency = px.line(
                                        monthly_product,
                                        x="Month",
                                        y="kWh_per_m3",
                                        color="Produkt",
                                        markers=True,
                                        title="Energy Efficiency (kWh/m³)"
                                    )
                                    fig_prod_efficiency.update_layout(height=350, plot_bgcolor="white", separators=",.")
                                    st.plotly_chart(fig_prod_efficiency, use_container_width=True)

                                with col_p2:
                                    fig_prod_specific = px.line(
                                        monthly_product,
                                        x="Month",
                                        y="kWh_per_kg",
                                        color="Produkt",
                                        markers=True,
                                        title="Specific Energy (kWh/kg)"
                                    )
                                    fig_prod_specific.update_layout(height=350, plot_bgcolor="white", separators=",.")
                                    st.plotly_chart(fig_prod_specific, use_container_width=True)

                                col_p3, col_p4 = st.columns(2)

                                with col_p3:
                                    fig_prod_thermal = px.line(
                                        monthly_product,
                                        x="Month",
                                        y="kWh_thermal_per_m3",
                                        color="Produkt",
                                        markers=True,
                                        title="Thermal Efficiency (kWh/m³)"
                                    )
                                    fig_prod_thermal.update_layout(height=350, plot_bgcolor="white", separators=",.")
                                    st.plotly_chart(fig_prod_thermal, use_container_width=True)

                                with col_p4:
                                    fig_prod_volume = px.line(
                                        monthly_product,
                                        x="Month",
                                        y="Volume_m3",
                                        color="Produkt",
                                        markers=True,
                                        title="Volume by Product (m³)"
                                    )
                                    fig_prod_volume.update_layout(height=350, plot_bgcolor="white", separators=",.")
                                    st.plotly_chart(fig_prod_volume, use_container_width=True)

                        # Trends by Zone
                        with st.expander("🏭 Trends by Zone - Detailed Charts"):
                            if monthly_zone.empty:
                                st.warning("No zone data available.")
                            else:
                                col_z1, col_z2 = st.columns(2)

                                with col_z1:
                                    fig_zone_efficiency = px.line(
                                        monthly_zone,
                                        x="Month",
                                        y="kWh_per_m3",
                                        color="Zone",
                                        markers=True,
                                        title="Energy Efficiency by Zone (kWh/m³)"
                                    )
                                    fig_zone_efficiency.update_layout(height=350, plot_bgcolor="white", separators=",.")
                                    st.plotly_chart(fig_zone_efficiency, use_container_width=True)

                                with col_z2:
                                    fig_zone_specific = px.line(
                                        monthly_zone,
                                        x="Month",
                                        y="kWh_per_kg",
                                        color="Zone",
                                        markers=True,
                                        title="Specific Energy by Zone (kWh/kg)"
                                    )
                                    fig_zone_specific.update_layout(height=350, plot_bgcolor="white", separators=",.")
                                    st.plotly_chart(fig_zone_specific, use_container_width=True)

                                col_z3, col_z4 = st.columns(2)

                                with col_z3:
                                    fig_zone_thermal = px.line(
                                        monthly_zone,
                                        x="Month",
                                        y="Energy_thermal_kWh",
                                        color="Zone",
                                        markers=True,
                                        title="Thermal Energy by Zone (kWh)"
                                    )
                                    fig_zone_thermal.update_layout(height=350, plot_bgcolor="white", separators=",.")
                                    st.plotly_chart(fig_zone_thermal, use_container_width=True)

                                with col_z4:
                                    fig_zone_volume = px.line(
                                        monthly_zone,
                                        x="Month",
                                        y="Volume_m3",
                                        color="Zone",
                                        markers=True,
                                        title="Volume by Zone (m³)"
                                    )
                                    fig_zone_volume.update_layout(height=350, plot_bgcolor="white", separators=",.")
                                    st.plotly_chart(fig_zone_volume, use_container_width=True)
            
            # ===== 4. SINGLE PRODUCT DEEP DIVE =====
            st.markdown(
                '<div class="section-header">🔬 Single Product Deep Dive</div>',
                unsafe_allow_html=True
            )
            
            selected_single_product = st.selectbox(
                "Select a product for detailed analysis:",
                options=sorted(summary["Produkt"].unique().tolist()),
                key="single_product_select"
            )
            
            if selected_single_product:
                single_product_data = summary[summary["Produkt"] == selected_single_product]
                
                if single_product_data.empty:
                    st.warning(f"No data available for {selected_single_product}")
                else:
                    single_monthly = single_product_data.groupby("Month", as_index=False).agg({
                        "Energy_thermal_kWh": "sum",
                        "Energy_electrical_kWh": "sum",
                        "Energy_kWh": "sum",
                        "Volume_m3": "sum",
                        "Water_kg": "sum",
                    })
                    single_monthly["kWh_per_m3"] = safe_divide(
                        single_monthly["Energy_kWh"], single_monthly["Volume_m3"]
                    )
                    single_monthly["kWh_per_kg"] = safe_divide(
                        single_monthly["Energy_kWh"], single_monthly["Water_kg"]
                    )
                    
                    total_energy_prod = single_monthly["Energy_kWh"].sum()
                    total_volume_prod = single_monthly["Volume_m3"].sum()
                    total_water_prod = single_monthly["Water_kg"].sum()
                    avg_kwh_m3_prod = safe_divide(total_energy_prod, total_volume_prod)
                    avg_kwh_kg_prod = safe_divide(total_energy_prod, total_water_prod)
                    
                    st.subheader(f"📊 {selected_single_product} - Summary")
                    
                    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
                    with col_s1:
                        st.metric("Total Energy", f"{format_german_int(total_energy_prod)} kWh")
                    with col_s2:
                        st.metric("Total Volume", f"{format_german_int(total_volume_prod)} m³")
                    with col_s3:
                        st.metric("Total Water", f"{format_german_int(total_water_prod)} kg")
                    with col_s4:
                        st.metric("kWh/m³", format_german(avg_kwh_m3_prod, 1))
                    with col_s5:
                        st.metric("kWh/kg", format_german(avg_kwh_kg_prod, 3))
                    
                    col_sp1, col_sp2 = st.columns(2)
                    
                    with col_sp1:
                        fig_single_energy = go.Figure()
                        fig_single_energy.add_trace(go.Bar(
                            name='Thermal',
                            x=single_monthly["Month"],
                            y=single_monthly["Energy_thermal_kWh"],
                            marker_color='#FF6B6B'
                        ))
                        fig_single_energy.add_trace(go.Bar(
                            name='Electrical',
                            x=single_monthly["Month"],
                            y=single_monthly["Energy_electrical_kWh"],
                            marker_color='#4ECDC4'
                        ))
                        fig_single_energy.update_layout(
                            title=f"{selected_single_product} - Monthly Energy (kWh)",
                            barmode='stack',
                            height=300,
                            plot_bgcolor="white",
                            separators=",."
                        )
                        st.plotly_chart(fig_single_energy, use_container_width=True)
                    
                    with col_sp2:
                        fig_single_kpi = go.Figure()
                        fig_single_kpi.add_trace(go.Scatter(
                            x=single_monthly["Month"],
                            y=single_monthly["kWh_per_m3"],
                            mode='lines+markers',
                            name='kWh/m³',
                            line=dict(color='#667eea', width=3)
                        ))
                        fig_single_kpi.add_trace(go.Scatter(
                            x=single_monthly["Month"],
                            y=single_monthly["kWh_per_kg"] * 100,
                            mode='lines+markers',
                            name='kWh/kg (×100)',
                            line=dict(color='#f093fb', width=3),
                            yaxis='y2'
                        ))
                        fig_single_kpi.update_layout(
                            title=f"{selected_single_product} - Monthly KPIs",
                            yaxis=dict(title="kWh/m³"),
                            yaxis2=dict(title="kWh/kg (×100)", overlaying='y', side='right'),
                            height=300,
                            plot_bgcolor="white",
                            separators=",."
                        )
                        st.plotly_chart(fig_single_kpi, use_container_width=True)
                    
                    st.subheader(f"📋 {selected_single_product} - Monthly Details")
                    display_single = single_monthly.copy()
                    
                    # Format for German display
                    display_single["Energy_thermal_kWh"] = display_single["Energy_thermal_kWh"].apply(lambda x: format_german(x, 0))
                    display_single["Energy_electrical_kWh"] = display_single["Energy_electrical_kWh"].apply(lambda x: format_german(x, 0))
                    display_single["Energy_kWh"] = display_single["Energy_kWh"].apply(lambda x: format_german(x, 0))
                    display_single["Volume_m3"] = display_single["Volume_m3"].apply(lambda x: format_german(x, 2))
                    display_single["Water_kg"] = display_single["Water_kg"].apply(lambda x: format_german(x, 0))
                    display_single["kWh_per_m3"] = display_single["kWh_per_m3"].apply(lambda x: format_german(x, 1))
                    display_single["kWh_per_kg"] = display_single["kWh_per_kg"].apply(lambda x: format_german(x, 3))
                    
                    display_single = display_single.rename(columns={
                        "Energy_thermal_kWh": "Thermal (kWh)",
                        "Energy_electrical_kWh": "Electrical (kWh)",
                        "Energy_kWh": "Total (kWh)",
                        "Volume_m3": "Volume (m³)",
                        "Water_kg": "Water (kg)",
                        "kWh_per_m3": "kWh/m³",
                        "kWh_per_kg": "kWh/kg"
                    })
                    st.dataframe(display_single, use_container_width=True, hide_index=True)
                    
                    if selected_single_product in PRODUCT_SPECIFICATIONS:
                        spec = PRODUCT_SPECIFICATIONS[selected_single_product]
                        st.subheader(f"📐 {selected_single_product} - Specifications")
                        
                        col_spec1, col_spec2, col_spec3 = st.columns(3)
                        
                        with col_spec1:
                            st.markdown("**Physical Properties**")
                            st.write(f"- Edge Length: {format_german_int(spec['edge_length_mm'])} mm")
                            st.write(f"- Final Thickness: {format_german_int(spec['final_thickness_mm'])} mm")
                            st.write(f"- Pressed Thickness: {format_german(spec['pressed_thickness_mm'], 1)} mm")
                            st.write(f"- Volume/Plate: {format_german(spec['volume_m3'], 6)} m³")
                        
                        with col_spec2:
                            st.markdown("**Water Content**")
                            water_per_mm = spec["slope"] * SUSPENSION_KG + spec["intercept"]
                            water_per_plate = (water_per_mm * spec["pressed_thickness_mm"]) / 1000
                            water_per_m3_spec = water_per_plate / spec["volume_m3"]
                            st.write(f"- Formula: {spec['formula']}")
                            st.write(f"- Water/mm: {format_german(water_per_mm, 1)} g")
                            st.write(f"- Water/Plate: {format_german(water_per_plate, 3)} kg")
                            st.write(f"- Water/m³: {format_german(water_per_m3_spec, 1)} kg/m³")
                        
                        with col_spec3:
                            st.markdown("**Formula Parameters**")
                            st.write(f"- Slope: {format_german(spec['slope'], 4)}")
                            st.write(f"- Intercept: {format_german(spec['intercept'], 1)}")
                            st.write(f"- Product Type: {spec['product_type']}")

            # ===== 5. PRODUCT PERFORMANCE =====
            if product_totals is not None and not product_totals.empty:
                st.markdown(
                    '<div class="section-header">📊 Product Performance</div>',
                    unsafe_allow_html=True
                )

                prod_agg = product_totals.groupby("Produkt", as_index=False).agg({
                    "Energy_thermal_kWh": "sum",
                    "Energy_electrical_kWh": "sum",
                    "Energy_kWh": "sum",
                    "Volume_m3": "sum",
                    "Water_kg": "sum",
                })

                prod_agg["kWh_per_m3"] = np.where(
                    prod_agg["Volume_m3"] > 0,
                    prod_agg["Energy_kWh"] / prod_agg["Volume_m3"],
                    0
                )
                prod_agg["kWh_per_kg"] = np.where(
                    prod_agg["Water_kg"] > 0,
                    prod_agg["Energy_kWh"] / prod_agg["Water_kg"],
                    0
                )
                prod_agg["Thermal_pct"] = np.where(
                    prod_agg["Energy_kWh"] > 0,
                    (prod_agg["Energy_thermal_kWh"] / prod_agg["Energy_kWh"] * 100),
                    0
                )
                prod_agg = prod_agg.fillna(0)

                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    fig_prod_energy = go.Figure()
                    fig_prod_energy.add_trace(go.Bar(
                        name='Thermal (Gas)',
                        x=prod_agg['Produkt'],
                        y=prod_agg['Energy_thermal_kWh'],
                        marker_color='#FF6B6B',
                        text=[format_german_int(v) for v in prod_agg['Energy_thermal_kWh']],
                        textposition='auto'
                    ))
                    fig_prod_energy.add_trace(go.Bar(
                        name='Electrical',
                        x=prod_agg['Produkt'],
                        y=prod_agg['Energy_electrical_kWh'],
                        marker_color='#4ECDC4',
                        text=[format_german_int(v) for v in prod_agg['Energy_electrical_kWh']],
                        textposition='auto'
                    ))
                    fig_prod_energy.update_layout(
                        title="Total Energy Consumption by Product (kWh)",
                        xaxis_title="Product",
                        yaxis_title="Energy (kWh)",
                        barmode='stack',
                        height=400,
                        plot_bgcolor="white",
                        separators=",."
                    )
                    st.plotly_chart(fig_prod_energy, use_container_width=True)

                with col_p2:
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
                        text=[f"{format_german_int(v)} kWh" for v in prod_sorted['Energy_thermal_kWh']],
                        textposition='outside'
                    ))
                    fig_thermal.update_layout(
                        title="Thermal Energy Consumption by Product (kWh)",
                        xaxis_title="Thermal Energy (kWh)",
                        yaxis_title="Product",
                        height=400,
                        plot_bgcolor="white",
                        showlegend=False,
                        separators=",."
                    )
                    st.plotly_chart(fig_thermal, use_container_width=True)

                st.subheader("Product Energy Summary")
                prod_display = prod_agg.copy()
                
                # Format for German display
                prod_display["Energy_thermal_kWh"] = prod_display["Energy_thermal_kWh"].apply(lambda x: format_german(x, 0))
                prod_display["Energy_electrical_kWh"] = prod_display["Energy_electrical_kWh"].apply(lambda x: format_german(x, 0))
                prod_display["Energy_kWh"] = prod_display["Energy_kWh"].apply(lambda x: format_german(x, 0))
                prod_display["Thermal_pct"] = prod_display["Thermal_pct"].apply(lambda x: format_german(x, 1) + "%")
                prod_display["Volume_m3"] = prod_display["Volume_m3"].apply(lambda x: format_german(x, 2))
                prod_display["Water_kg"] = prod_display["Water_kg"].apply(lambda x: format_german(x, 0))
                prod_display["kWh_per_m3"] = prod_display["kWh_per_m3"].apply(lambda x: format_german(x, 1))
                prod_display["kWh_per_kg"] = prod_display["kWh_per_kg"].apply(lambda x: format_german(x, 3))
                
                prod_display = prod_display.rename(columns={
                    "Produkt": "Product",
                    "Energy_thermal_kWh": "Thermal (kWh)",
                    "Energy_electrical_kWh": "Electrical (kWh)",
                    "Energy_kWh": "Total (kWh)",
                    "Thermal_pct": "Thermal %",
                    "Volume_m3": "Volume (m³)",
                    "Water_kg": "Water (kg)",
                    "kWh_per_m3": "kWh/m³",
                    "kWh_per_kg": "kWh/kg"
                })
                display_cols = [
                    "Product", "Thermal (kWh)", "Electrical (kWh)", "Total (kWh)",
                    "Thermal %", "Volume (m³)", "Water (kg)", "kWh/m³", "kWh/kg"
                ]
                display_cols = [c for c in display_cols if c in prod_display.columns]
                st.dataframe(prod_display[display_cols], use_container_width=True, hide_index=True)

            # ===== 6. PRODUCT SPECIFICATIONS =====
            st.markdown(
                '<div class="section-header">📐 Product Specifications</div>',
                unsafe_allow_html=True
            )

            st.write(f"**Formula:** Water/mm (g) = Slope × Suspension ({format_german_int(SUSPENSION_KG)} kg) + Intercept")

            specs_data = []
            for prod, spec in PRODUCT_SPECIFICATIONS.items():
                slope = spec["slope"]
                intercept = spec["intercept"]
                water_per_mm_g = slope * SUSPENSION_KG + intercept
                pressed_thickness_mm = spec["pressed_thickness_mm"]
                water_per_plate_kg = (water_per_mm_g * pressed_thickness_mm) / 1000.0
                water_per_m3_kg = water_per_plate_kg / spec["volume_m3"]

                is_interpolated = spec.get("interpolated", False)
                formula_display = spec["formula"]
                if is_interpolated:
                    formula_display += " ⚠️"

                specs_data.append({
                    "Product": prod,
                    "Type": spec["product_type"],
                    "Formula": formula_display,
                    "Water/mm (g)": format_german(water_per_mm_g, 1),
                    "Pressed (mm)": format_german(pressed_thickness_mm, 1),
                    "Water/Plate (kg)": format_german(water_per_plate_kg, 3),
                    "Water/m³ (kg)": format_german(water_per_m3_kg, 1),
                })

            specs_df = pd.DataFrame(specs_data)
            st.dataframe(specs_df, use_container_width=True, hide_index=True)
            st.info("⚠️ Products marked with ⚠️ are interpolated values")

            # ===== 7. DATA TABLES =====
            with st.expander("📋 View Detailed Data Tables"):
                tab1, tab2, tab3 = st.tabs(["Monthly Summary", "Yearly Summary", "Product Totals"])
                with tab1:
                    st.dataframe(summary, use_container_width=True)
                with tab2:
                    st.dataframe(yearly, use_container_width=True)
                with tab3:
                    if product_totals is not None:
                        st.dataframe(product_totals, use_container_width=True)

            # ===== 8. WEEKLY PREDICTION =====
            st.markdown(
                '<div class="section-header">🔮 Weekly Energy Prediction</div>',
                unsafe_allow_html=True
            )

            wagon_stats = compute_product_wagon_stats(wagons_df)
            wagon_capacity = wagon_stats.get("wagon_capacity_m3", {})

            baseline_kwh_m3 = float(yearly["kWh_per_m3"].mean()) if len(yearly) > 0 else 0.0
            baseline_kwh_kg = float(yearly["kWh_per_kg"].mean()) if len(yearly) > 0 else 0.0

            st.info(
                f"📈 **Historical Baseline KPIs:** "
                f"**{format_german(baseline_kwh_kg, 3)} kWh/kg** | **{format_german(baseline_kwh_m3, 1)} kWh/m³**"
            )

            use_custom_kpis = st.checkbox("🔧 Use custom KPIs", value=False)

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
                        cap_text = f" ({format_german(cap, 2)} m³/w)" if cap > 0 else ""
                        planned_wagons[p] = st.number_input(
                            f"{p}{cap_text}", min_value=0, value=0, step=10, key=f"w_{p}"
                        )

                with col2:
                    st.write("**L-Type (Heavy)**")
                    for p in ["L38", "L42", "L44"]:
                        cap = wagon_capacity.get(p, 0)
                        cap_text = f" ({format_german(cap, 2)} m³/w)" if cap > 0 else ""
                        planned_wagons[p] = st.number_input(
                            f"{p}{cap_text}", min_value=0, value=0, step=10, key=f"w_{p}"
                        )

                with col3:
                    st.write("**N & Y-Type**")
                    for p in ["N40", "N44", "Y44"]:
                        cap = wagon_capacity.get(p, 0)
                        cap_text = f" ({format_german(cap, 2)} m³/w)" if cap > 0 else ""
                        planned_wagons[p] = st.number_input(
                            f"{p}{cap_text}", min_value=0, value=0, step=10, key=f"w_{p}"
                        )

                submitted = st.form_submit_button("🔮 Calculate Prediction", type="primary")

            if submitted:
                product_volumes = {}
                total_wagons_pred = 0

                for prod, wagons in planned_wagons.items():
                    if wagons > 0:
                        capacity = wagon_capacity.get(prod, 1.5) or 1.5
                        product_volumes[prod] = wagons * capacity
                        total_wagons_pred += wagons

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
                        st.metric("Total Wagons", format_german_int(total_wagons_pred))
                    with c2:
                        st.metric("Total Volume", f"{format_german(pred['total_volume_m3'], 1)} m³")
                    with c3:
                        st.metric("Water to Evaporate", f"{format_german_int(pred['total_water_kg'])} kg")
                    with c4:
                        if pred.get("total_energy_kwh", 0) > 0:
                            energy = pred["total_energy_kwh"]
                            st.metric("Energy Required", f"{format_german_int(energy)} kWh")

                    if pred.get("products"):
                        st.write("### 📦 Product Breakdown")
                        breakdown = pd.DataFrame(pred["products"])
                        
                        # Format for German display
                        if "volume_m3" in breakdown.columns:
                            breakdown["volume_m3"] = breakdown["volume_m3"].apply(lambda x: format_german(x, 2))
                        if "water_per_plate_kg" in breakdown.columns:
                            breakdown["water_per_plate_kg"] = breakdown["water_per_plate_kg"].apply(lambda x: format_german(x, 3))
                        if "water_kg" in breakdown.columns:
                            breakdown["water_kg"] = breakdown["water_kg"].apply(lambda x: format_german(x, 0))
                        if "energy_from_water_kwh" in breakdown.columns:
                            breakdown["energy_from_water_kwh"] = breakdown["energy_from_water_kwh"].apply(lambda x: format_german(x, 0))
                        
                        display_cols = {
                            "product": "Product",
                            "volume_m3": "Volume (m³)",
                            "water_per_plate_kg": "Water/Plate (kg)",
                            "water_kg": "Total Water (kg)",
                        }
                        if "energy_from_water_kwh" in breakdown.columns:
                            display_cols["energy_from_water_kwh"] = "Energy (kWh)"
                        breakdown = breakdown.rename(columns=display_cols)
                        cols = [c for c in display_cols.values() if c in breakdown.columns]
                        st.dataframe(breakdown[cols], use_container_width=True, hide_index=True)

                    st.success("✅ Prediction complete!")
                else:
                    st.warning("⚠️ Enter wagon counts for at least one product.")

            # ===== 9. EXPORT =====
            st.markdown(
                '<div class="section-header">📥 Export Results</div>',
                unsafe_allow_html=True
            )

            excel_data = create_excel_download(results)
            st.download_button(
                label="📥 Download Complete Excel Report",
                data=excel_data,
                file_name=f"Dryer_KPI_Analysis_Trockner_{applied_trockner}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success("✅ Analysis complete!")

    except Exception as e:
        st.error(f"❌ Display error: {e}")
        with st.expander("🔍 View Error Details"):
            st.exception(e)







