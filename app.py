import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import io
from pathlib import Path

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Spatial Query System",
    page_icon="🔬",
    layout="wide",
)

# ──────────────────────────────────────────────
# Theme — detect browser dark mode via JS
# ──────────────────────────────────────────────
try:
    from streamlit_js_eval import streamlit_js_eval
    _is_dark_raw = streamlit_js_eval(
        js_expressions="window.matchMedia('(prefers-color-scheme: dark)').matches",
        key="detect_dark_mode",
    )
    IS_DARK = bool(_is_dark_raw) if _is_dark_raw is not None else False
except Exception:
    IS_DARK = False

# ──────────────────────────────────────────────
# Custom CSS with CSS variables for theming
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ───── Light mode (default) ───── */
    :root {
        --header-bg: linear-gradient(135deg, #4f46e5, #6366f1, #818cf8);
        --header-title: #ffffff;
        --header-sub: #e0e7ff;

        --stat-bg: linear-gradient(135deg, #e0e7ff, #c7d2fe);
        --stat-border: rgba(99, 102, 241, 0.2);
        --stat-value: #4338ca;
        --stat-label: #6366f1;
        --stat-shadow: rgba(99, 102, 241, 0.1);

        --result-bg: linear-gradient(135deg, #d1fae5, #a7f3d0);
        --result-border: rgba(16, 185, 129, 0.25);
        --result-value: #065f46;
        --result-label: #047857;
        --result-shadow: rgba(16, 185, 129, 0.1);

        --sidebar-bg: linear-gradient(180deg, #f1f5f9, #e0e7ff);
        --sidebar-h2: #4338ca;
        --sidebar-h2-border: rgba(99, 102, 241, 0.2);

        --empty-bg: linear-gradient(135deg, #e0e7ff, #c7d2fe);
        --empty-border: rgba(99, 102, 241, 0.35);
        --empty-title: #4338ca;
        --empty-sub: #6366f1;

        --text-primary: #1e293b;
        --text-secondary: #334155;
    }

    /* ───── Dark mode ───── */
    @media (prefers-color-scheme: dark) {
        :root {
            --header-bg: linear-gradient(135deg, #312e81, #3730a3, #4338ca);
            --header-title: #e0e7ff;
            --header-sub: #a5b4fc;

            --stat-bg: linear-gradient(135deg, #1e1b4b, #312e81);
            --stat-border: rgba(129, 140, 248, 0.25);
            --stat-value: #a5b4fc;
            --stat-label: #818cf8;
            --stat-shadow: rgba(99, 102, 241, 0.15);

            --result-bg: linear-gradient(135deg, #064e3b, #065f46);
            --result-border: rgba(52, 211, 153, 0.25);
            --result-value: #6ee7b7;
            --result-label: #34d399;
            --result-shadow: rgba(16, 185, 129, 0.15);

            --sidebar-bg: linear-gradient(180deg, #0f172a, #1e1b4b);
            --sidebar-h2: #a5b4fc;
            --sidebar-h2-border: rgba(129, 140, 248, 0.25);

            --empty-bg: linear-gradient(135deg, #1e1b4b, #312e81);
            --empty-border: rgba(129, 140, 248, 0.35);
            --empty-title: #a5b4fc;
            --empty-sub: #818cf8;

            --text-primary: #e2e8f0;
            --text-secondary: #cbd5e1;
        }
    }

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: var(--header-bg);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
    }
    .main-header h1 {
        color: var(--header-title);
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: var(--header-sub);
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
        font-weight: 300;
    }

    /* Stat cards */
    .stat-card {
        background: var(--stat-bg);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--stat-border);
        text-align: center;
        box-shadow: 0 4px 16px var(--stat-shadow);
        transition: transform 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
    }
    .stat-card .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--stat-value);
        margin: 0;
    }
    .stat-card .stat-label {
        font-size: 0.8rem;
        color: var(--stat-label);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0.2rem 0 0 0;
    }

    /* Result card */
    .result-card {
        background: var(--result-bg);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--result-border);
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 16px var(--result-shadow);
    }
    .result-card .result-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--result-value);
        margin: 0;
    }
    .result-card .result-label {
        font-size: 0.75rem;
        color: var(--result-label);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0.2rem 0 0 0;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: var(--sidebar-bg);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: var(--sidebar-h2);
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 1px solid var(--sidebar-h2-border);
        padding-bottom: 0.5rem;
    }

    /* Empty state card */
    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
        background: var(--empty-bg);
        border-radius: 16px;
        border: 1px dashed var(--empty-border);
        margin: 2rem 0;
    }
    .empty-state .empty-icon { font-size: 3rem; margin: 0; }
    .empty-state .empty-title {
        font-size: 1.2rem;
        color: var(--empty-title);
        font-weight: 500;
        margin: 0.5rem 0 0.2rem 0;
    }
    .empty-state .empty-sub {
        font-size: 0.85rem;
        color: var(--empty-sub);
        margin: 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 Spatial Query System</h1>
    <p>Upload spatial cell-type data · Visualize · Query nearest neighbours</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper: build color map
# ──────────────────────────────────────────────
PALETTE = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.D3
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Pastel
)

def build_color_map(cell_types):
    """Return a dict mapping each cell type to a fixed colour."""
    unique = sorted(cell_types.dropna().unique())
    return {ct: PALETTE[i % len(PALETTE)] for i, ct in enumerate(unique)}


# ──────────────────────────────────────────────
# Plotly theme helper
# ──────────────────────────────────────────────
def get_plotly_layout():
    """Return Plotly layout kwargs that adapt to dark / light mode."""
    if IS_DARK:
        return dict(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e1b4b",
            font=dict(family="Inter, sans-serif", color="#e2e8f0"),
            legend=dict(
                title="Cell Type",
                bgcolor="rgba(30,27,75,0.9)",
                bordercolor="rgba(129,140,248,0.20)",
                borderwidth=1,
                font=dict(size=11, color="#cbd5e1"),
            ),
        )
    return dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font=dict(family="Inter, sans-serif", color="#1e293b"),
        legend=dict(
            title="Cell Type",
            bgcolor="rgba(248,250,252,0.9)",
            bordercolor="rgba(99,102,241,0.15)",
            borderwidth=1,
            font=dict(size=11, color="#334155"),
        ),
    )

def get_plotly_title_color():
    """Title font colour for Plotly charts."""
    return "#e2e8f0" if IS_DARK else "#1e293b"

def get_plotly_grid_color():
    """Axis grid colour for Plotly charts."""
    return "rgba(129,140,248,0.12)" if IS_DARK else "rgba(99,102,241,0.08)"

def get_marker_outline_color():
    """Marker outline colour."""
    return "#312e81" if IS_DARK else "#e0e7ff"

def get_connection_line_color():
    """Connection line colour for neighbour queries."""
    return "rgba(129, 140, 248, 0.40)" if IS_DARK else "rgba(99, 102, 241, 0.30)"

def get_contour_label_color():
    """Contour label font colour."""
    return "#cbd5e1" if IS_DARK else "#334155"

def get_density_scatter_color():
    """Overlay scatter point colour for density heatmap."""
    return "#818cf8" if IS_DARK else "#4338ca"


# ──────────────────────────────────────────────
# 1. DATA UPLOAD
# ──────────────────────────────────────────────
SAMPLE_FILE = Path(__file__).parent / "sample.xlsx"

with st.sidebar:
    st.markdown("## 📂 Data Upload")
    uploaded = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="File must contain columns for X, Y coordinates and cell type.",
    )
    if SAMPLE_FILE.exists():
        with open(SAMPLE_FILE, "rb") as f:
            st.download_button(
                "⬇️ Download Sample Data (sample.xlsx)",
                data=f,
                file_name="sample.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch',
            )

df = None

if uploaded is not None:
    # Read file
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

if df is not None:
    with st.sidebar:
        st.markdown("## 🗂 Column Mapping")
        cols = list(df.columns)

        def _default_idx(candidates):
            """Return index of first matching column name (case-insensitive)."""
            lower = [c.lower() for c in cols]
            for c in candidates:
                if c.lower() in lower:
                    return lower.index(c.lower())
            return 0

        x_col = st.selectbox("X coordinate column", cols, index=_default_idx(["X", "x_coord", "x_position"]))
        y_col = st.selectbox("Y coordinate column", cols, index=_default_idx(["Y", "y_coord", "y_position"]))
        type_col = st.selectbox("Cell type column", cols, index=_default_idx(["type", "cell_type", "celltype", "type_orig", "ClusterName"]))

    # Validate numeric
    for label, col in [("X", x_col), ("Y", y_col)]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error(f"Column **{col}** chosen for {label} is not numeric.")
            st.stop()

    # ── Stat cards ────────────────────────────
    n_cells = len(df)
    n_types = df[type_col].nunique()
    x_range = f"{df[x_col].min():.0f} – {df[x_col].max():.0f}"
    y_range = f"{df[y_col].min():.0f} – {df[y_col].max():.0f}"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="stat-card"><p class="stat-value">{n_cells:,}</p><p class="stat-label">Total Cells</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card"><p class="stat-value">{n_types}</p><p class="stat-label">Cell Types</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-card"><p class="stat-value">{x_range}</p><p class="stat-label">X Range</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="stat-card"><p class="stat-value">{y_range}</p><p class="stat-label">Y Range</p></div>', unsafe_allow_html=True)

    st.markdown("")  # spacer

    # ── Data preview (expander) ───────────────
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df.head(50), width='stretch')

    # ── Color map ─────────────────────────────
    color_map = build_color_map(df[type_col])

    # ──────────────────────────────────────────
    # 2. SPATIAL VISUALIZATION
    # ──────────────────────────────────────────
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=type_col,
        color_discrete_map=color_map,
        hover_data={type_col: True, x_col: True, y_col: True},
        title="Spatial Distribution of Cells",
        labels={x_col: "X", y_col: "Y", type_col: "Cell Type"},
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0.3, color=get_marker_outline_color())))
    fig.update_layout(
        **get_plotly_layout(),
        title=dict(font=dict(size=18, color=get_plotly_title_color())),
        xaxis=dict(gridcolor=get_plotly_grid_color(), zeroline=False),
        yaxis=dict(gridcolor=get_plotly_grid_color(), zeroline=False, scaleanchor="x"),
        height=650,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    # ──────────────────────────────────────────
    # 3. NEAREST NEIGHBOUR QUERY PANEL
    # ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("## 🔍 Nearest Neighbour Query")

        unique_types = sorted(df[type_col].dropna().unique())

        query_type = st.selectbox("Query cell type (source)", unique_types, key="query_type")
        target_options = ["All Types"] + unique_types
        target_type = st.selectbox("Target cell type (neighbours)", target_options, key="target_type")

        mode = st.radio("Query mode", ["Radius", "KNN"], horizontal=True)

        if mode == "Radius":
            radius = st.number_input("Radius", min_value=1.0, value=100.0, step=10.0)
        else:
            k_val = st.number_input("K (number of neighbours)", min_value=1, value=5, step=1)

        run_query = st.button("🚀 Run Query", width='stretch', type="primary")

    # ── Execute query ─────────────────────────
    if run_query:
        query_mask = df[type_col] == query_type
        query_df = df[query_mask].copy()

        if target_type == "All Types":
            target_mask = ~query_mask  # everything except the query type itself
        else:
            target_mask = df[type_col] == target_type
        target_df = df[target_mask].copy()

        if len(query_df) == 0:
            st.warning("No cells found for the selected query type.")
        elif len(target_df) == 0:
            st.warning("No cells found for the selected target type.")
        else:
            query_coords = query_df[[x_col, y_col]].values
            target_coords = target_df[[x_col, y_col]].values

            tree = cKDTree(target_coords)

            all_neighbour_rows = []
            lines_x = []
            lines_y = []

            if mode == "Radius":
                indices_list = tree.query_ball_point(query_coords, r=radius)
                for qi, indices in enumerate(indices_list):
                    if len(indices) == 0:
                        continue
                    qx, qy = query_coords[qi]
                    for idx in indices:
                        nb = target_df.iloc[idx]
                        dist = np.sqrt((qx - nb[x_col])**2 + (qy - nb[y_col])**2)
                        all_neighbour_rows.append({
                            "Query_Cell_Type": query_type,
                            "Query_X": qx,
                            "Query_Y": qy,
                            "Neighbour_Cell_Type": nb[type_col],
                            "Neighbour_X": nb[x_col],
                            "Neighbour_Y": nb[y_col],
                            "Distance": round(dist, 2),
                        })
                        lines_x += [qx, nb[x_col], None]
                        lines_y += [qy, nb[y_col], None]
            else:
                k = min(k_val, len(target_df))
                dists, indices = tree.query(query_coords, k=k)
                if k == 1:
                    dists = dists.reshape(-1, 1)
                    indices = indices.reshape(-1, 1)
                for qi in range(len(query_coords)):
                    qx, qy = query_coords[qi]
                    for j in range(k):
                        idx = indices[qi, j]
                        d = dists[qi, j]
                        nb = target_df.iloc[idx]
                        all_neighbour_rows.append({
                            "Query_Cell_Type": query_type,
                            "Query_X": qx,
                            "Query_Y": qy,
                            "Neighbour_Cell_Type": nb[type_col],
                            "Neighbour_X": nb[x_col],
                            "Neighbour_Y": nb[y_col],
                            "Distance": round(d, 2),
                        })
                        lines_x += [qx, nb[x_col], None]
                        lines_y += [qy, nb[y_col], None]

            results_df = pd.DataFrame(all_neighbour_rows)

            if len(results_df) == 0:
                st.warning("No neighbours found with the given parameters.")
            else:
                # ── Add connecting lines to figure ──
                fig.add_trace(go.Scatter(
                    x=lines_x, y=lines_y,
                    mode="lines",
                    line=dict(color=get_connection_line_color(), width=0.8),
                    hoverinfo="skip",
                    showlegend=False,
                    name="connections",
                ))

                # Highlight query cells
                fig.add_trace(go.Scatter(
                    x=query_df[x_col], y=query_df[y_col],
                    mode="markers",
                    marker=dict(size=9, color="#facc15", symbol="diamond",
                                line=dict(width=1.5, color="#ffffff")),
                    name=f"Query: {query_type}",
                    hovertemplate=f"<b>Query: {query_type}</b><br>X: %{{x}}<br>Y: %{{y}}<extra></extra>",
                ))

                # Highlight neighbour cells
                nb_x = results_df["Neighbour_X"].values
                nb_y = results_df["Neighbour_Y"].values
                fig.add_trace(go.Scatter(
                    x=nb_x, y=nb_y,
                    mode="markers",
                    marker=dict(size=7, color="#f472b6", symbol="circle",
                                line=dict(width=1, color="#ffffff")),
                    name="Neighbours",
                    hovertemplate="<b>Neighbour</b><br>X: %{x}<br>Y: %{y}<extra></extra>",
                ))

                # ── Summary statistics ────────────
                st.markdown("### 📊 Query Results")
                mode_label = f"Radius = {radius}" if mode == "Radius" else f"K = {k_val}"
                st.caption(f"**{query_type}** → **{target_type}** | Mode: {mode} ({mode_label})")

                r1, r2, r3 = st.columns(3)
                r1.markdown(
                    f'<div class="result-card"><p class="result-value">{len(results_df):,}</p>'
                    f'<p class="result-label">Total Neighbour Pairs</p></div>',
                    unsafe_allow_html=True,
                )
                r2.markdown(
                    f'<div class="result-card"><p class="result-value">{results_df["Distance"].mean():.1f}</p>'
                    f'<p class="result-label">Mean Distance</p></div>',
                    unsafe_allow_html=True,
                )
                r3.markdown(
                    f'<div class="result-card"><p class="result-value">{results_df["Distance"].median():.1f}</p>'
                    f'<p class="result-label">Median Distance</p></div>',
                    unsafe_allow_html=True,
                )

                # ── Downloadable results ──────────
                with st.expander("📄 Results Table", expanded=True):
                    st.dataframe(results_df, width='stretch')

                    csv_buf = io.StringIO()
                    results_df.to_csv(csv_buf, index=False)
                    st.download_button(
                        "⬇️ Download Results CSV",
                        csv_buf.getvalue(),
                        file_name="spatial_query_results.csv",
                        mime="text/csv",
                        width='stretch',
                    )

    # ── Render the (possibly updated) figure ──
    st.plotly_chart(fig, width='stretch', key="main_scatter")

    # ══════════════════════════════════════════
    # 4. ANALYSIS & STATISTICS
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📈 Analysis & Statistics")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Composition",
        "🌡️ Density Heatmap",
        "🧮 Spatial Autocorrelation",
        "📏 Pairwise Distances",
        "🔗 Co-occurrence",
    ])

    # Clean data for analysis (drop NaN in type col)
    df_clean = df.dropna(subset=[type_col]).copy()
    coords = df_clean[[x_col, y_col]].values
    types_series = df_clean[type_col]
    unique_types_sorted = sorted(types_series.unique())

    # ──────────────────────────────────────────
    # TAB 1: Cell Type Composition
    # ──────────────────────────────────────────
    with tab1:
        counts = types_series.value_counts().sort_values(ascending=True)
        fig_comp = px.bar(
            x=counts.values,
            y=counts.index,
            orientation="h",
            labels={"x": "Count", "y": "Cell Type"},
            title="Cell Type Composition",
            color=counts.index,
            color_discrete_map=color_map,
        )
        fig_comp.update_layout(
            **get_plotly_layout(),
            height=max(350, len(counts) * 30 + 100),
            showlegend=False,
            margin=dict(l=10, r=20, t=50, b=40),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_comp, width='stretch', key="composition")

        # Percentage table
        comp_df = pd.DataFrame({
            "Cell Type": counts.index,
            "Count": counts.values,
            "Percentage": (counts.values / counts.values.sum() * 100).round(1),
        }).reset_index(drop=True)
        st.dataframe(comp_df, width='stretch', hide_index=True)

    # ──────────────────────────────────────────
    # TAB 2: Density Heatmap (2D Histogram)
    # ──────────────────────────────────────────
    with tab2:
        density_type = st.selectbox(
            "Show density for",
            ["All Types"] + unique_types_sorted,
            key="density_type",
        )
        if density_type == "All Types":
            dens_df = df_clean
        else:
            dens_df = df_clean[df_clean[type_col] == density_type]

        fig_dens = go.Figure(
            go.Histogram2dContour(
                x=dens_df[x_col],
                y=dens_df[y_col],
                colorscale="Blues",
                contours=dict(showlabels=True, labelfont=dict(size=10, color=get_contour_label_color())),
                ncontours=15,
                showscale=True,
            )
        )
        # Overlay scatter points
        fig_dens.add_trace(go.Scatter(
            x=dens_df[x_col], y=dens_df[y_col],
            mode="markers",
            marker=dict(size=3, color=get_density_scatter_color(), opacity=0.4),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig_dens.update_layout(
            **get_plotly_layout(),
            title=f"Density Heatmap — {density_type}",
            height=600,
            margin=dict(l=40, r=20, t=50, b=40),
            yaxis=dict(scaleanchor="x"),
        )
        st.plotly_chart(fig_dens, width='stretch', key="density")
        st.caption(f"Showing **{len(dens_df):,}** cells. Contour lines represent regions of equal density.")

    # ──────────────────────────────────────────
    # TAB 3: Spatial Autocorrelation
    # ──────────────────────────────────────────
    with tab3:
        st.markdown("#### Moran's I (per cell type)")
        st.caption(
            "Measures spatial autocorrelation. **I > 0** = clustered, **I ≈ 0** = random, **I < 0** = dispersed. "
            "Uses a binary spatial weights matrix with the specified radius."
        )

        morans_radius = st.number_input(
            "Neighbourhood radius for weights matrix",
            min_value=1.0, value=50.0, step=10.0, key="morans_radius",
        )

        if st.button("🧮 Compute Moran's I", key="run_morans", width='stretch'):
            tree_all = cKDTree(coords)
            # Build sparse binary weight matrix via ball-tree query
            pairs = tree_all.query_pairs(r=morans_radius, output_type="ndarray")

            moran_rows = []
            for ct in unique_types_sorted:
                # Binary variable: 1 if cell is this type, 0 otherwise
                z = (types_series.values == ct).astype(float)
                n = len(z)
                z_bar = z.mean()
                z_dev = z - z_bar
                denom = np.sum(z_dev ** 2)

                if denom == 0 or len(pairs) == 0:
                    moran_rows.append({"Cell Type": ct, "Moran's I": np.nan,
                                       "Interpretation": "—", "N cells": int(z.sum())})
                    continue

                # Numerator: sum of w_ij * z_i * z_j
                numer = 0.0
                for i, j in pairs:
                    numer += z_dev[i] * z_dev[j]
                numer *= 2  # since pairs are unordered, count both (i,j) and (j,i)

                W = len(pairs) * 2  # total weight sum
                I = (n / W) * (numer / denom)

                if I > 0.1:
                    interp = "🟢 Clustered"
                elif I < -0.1:
                    interp = "🔴 Dispersed"
                else:
                    interp = "⚪ Random"

                moran_rows.append({"Cell Type": ct, "Moran's I": round(I, 4),
                                   "Interpretation": interp, "N cells": int(z.sum())})

            moran_df = pd.DataFrame(moran_rows)
            st.dataframe(moran_df, width='stretch', hide_index=True)

        st.markdown("---")
        st.markdown("#### Ripley's K Function")
        st.caption(
            "Ripley's K estimates the expected number of points within distance *r* of a typical point. "
            "**K(r) > πr²** indicates clustering; **K(r) < πr²** indicates dispersion."
        )

        ripley_type = st.selectbox(
            "Cell type for Ripley's K",
            unique_types_sorted,
            key="ripley_type",
        )
        ripley_max_r = st.number_input(
            "Max radius", min_value=10.0, value=200.0, step=20.0, key="ripley_max_r",
        )
        ripley_steps = 30

        if st.button("🧮 Compute Ripley's K", key="run_ripley", width='stretch'):
            ct_mask = types_series.values == ripley_type
            ct_coords = coords[ct_mask]
            n_ct = len(ct_coords)

            if n_ct < 2:
                st.warning("Need at least 2 cells of this type.")
            else:
                # Study area bounding box
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0)
                area = (x_max - x_min) * (y_max - y_min)
                lam = n_ct / area  # intensity

                tree_ct = cKDTree(ct_coords)
                r_vals = np.linspace(0, ripley_max_r, ripley_steps + 1)[1:]
                K_vals = []

                for r in r_vals:
                    count = tree_ct.query_pairs(r=r, output_type="ndarray").shape[0]
                    K = (area / (n_ct ** 2)) * 2 * count
                    K_vals.append(K)

                K_vals = np.array(K_vals)
                K_csr = np.pi * r_vals ** 2  # expected K under CSR

                fig_rip = go.Figure()
                fig_rip.add_trace(go.Scatter(
                    x=r_vals, y=K_vals, mode="lines+markers",
                    name=f"K — {ripley_type}",
                    line=dict(color="#4f46e5", width=2),
                    marker=dict(size=4),
                ))
                fig_rip.add_trace(go.Scatter(
                    x=r_vals, y=K_csr, mode="lines",
                    name="CSR (πr²)",
                    line=dict(color="#94a3b8", width=1.5, dash="dash"),
                ))
                fig_rip.update_layout(
                    **get_plotly_layout(),
                    title=f"Ripley's K — {ripley_type} ({n_ct} cells)",
                    xaxis_title="r (radius)",
                    yaxis_title="K(r)",
                    height=450,
                    margin=dict(l=40, r=20, t=50, b=40),
                    yaxis=dict(scaleanchor=None),
                )
                st.plotly_chart(fig_rip, width='stretch', key="ripley")
                st.caption(
                    "If the **blue line** is above the **dashed CSR line**, "
                    "the cell type is **clustered**. Below = **dispersed**."
                )

    # ──────────────────────────────────────────
    # TAB 4: Pairwise Distance Matrix
    # ──────────────────────────────────────────
    with tab4:
        st.caption(
            "Mean Euclidean distance between all pairs of cell types. "
            "Diagonal = mean intra-type distance."
        )

        if st.button("📏 Compute Pairwise Distances", key="run_pairwise", width='stretch'):
            n_types_pw = len(unique_types_sorted)
            dist_matrix = np.zeros((n_types_pw, n_types_pw))

            # Pre-compute per-type coordinate arrays
            type_coords = {}
            for ct in unique_types_sorted:
                mask = types_series.values == ct
                type_coords[ct] = coords[mask]

            for i, ct_a in enumerate(unique_types_sorted):
                for j, ct_b in enumerate(unique_types_sorted):
                    if j < i:
                        dist_matrix[i, j] = dist_matrix[j, i]
                        continue
                    ca = type_coords[ct_a]
                    cb = type_coords[ct_b]
                    if len(ca) == 0 or len(cb) == 0:
                        dist_matrix[i, j] = np.nan
                        continue
                    if ct_a == ct_b and len(ca) < 2:
                        dist_matrix[i, j] = np.nan
                        continue
                    dists = cdist(ca, cb, metric="euclidean")
                    if ct_a == ct_b:
                        # Exclude self-distances (diagonal of the sub-matrix)
                        np.fill_diagonal(dists, np.nan)
                    dist_matrix[i, j] = np.nanmean(dists)

            fig_pw = px.imshow(
                dist_matrix,
                x=unique_types_sorted,
                y=unique_types_sorted,
                color_continuous_scale="Viridis",
                labels=dict(color="Mean Distance"),
                title="Pairwise Mean Distance Matrix",
                text_auto=".0f",
                aspect="equal",
            )
            fig_pw.update_layout(
                **get_plotly_layout(),
                height=max(450, n_types_pw * 40 + 100),
                margin=dict(l=10, r=20, t=50, b=10),
                yaxis=dict(scaleanchor=None),
            )
            st.plotly_chart(fig_pw, width='stretch', key="pairwise")

    # ──────────────────────────────────────────
    # TAB 5: Co-occurrence Analysis
    # ──────────────────────────────────────────
    with tab5:
        st.caption(
            "Co-occurrence counts how often pairs of cell types appear within "
            "a given radius. The matrix is normalised by the geometric mean "
            "of each type's count to remove abundance bias."
        )

        coocc_radius = st.number_input(
            "Co-occurrence radius",
            min_value=1.0, value=50.0, step=10.0, key="coocc_radius",
        )

        if st.button("🔗 Compute Co-occurrence", key="run_coocc", width='stretch'):
            tree_all = cKDTree(coords)
            pairs = tree_all.query_pairs(r=coocc_radius, output_type="ndarray")

            type_labels = types_series.values
            type_to_idx = {ct: i for i, ct in enumerate(unique_types_sorted)}
            n_t = len(unique_types_sorted)

            # Raw co-occurrence count matrix
            coocc_raw = np.zeros((n_t, n_t), dtype=int)
            for i_idx, j_idx in pairs:
                ti = type_to_idx.get(type_labels[i_idx])
                tj = type_to_idx.get(type_labels[j_idx])
                if ti is not None and tj is not None:
                    coocc_raw[ti, tj] += 1
                    coocc_raw[tj, ti] += 1

            # Normalise by geometric mean of counts
            counts_per_type = np.array([np.sum(type_labels == ct) for ct in unique_types_sorted], dtype=float)
            geo_mean = np.sqrt(np.outer(counts_per_type, counts_per_type))
            geo_mean[geo_mean == 0] = 1  # avoid division by zero
            coocc_norm = coocc_raw / geo_mean

            # Tabs for raw and normalised
            co_raw_tab, co_norm_tab = st.tabs(["Raw Counts", "Normalised"])

            with co_raw_tab:
                fig_co_raw = px.imshow(
                    coocc_raw,
                    x=unique_types_sorted,
                    y=unique_types_sorted,
                    color_continuous_scale="YlOrRd",
                    labels=dict(color="Count"),
                    title=f"Co-occurrence (raw) — radius {coocc_radius}",
                    text_auto=True,
                    aspect="equal",
                )
                fig_co_raw.update_layout(
                    **get_plotly_layout(),
                    height=max(450, n_t * 40 + 100),
                    margin=dict(l=10, r=20, t=50, b=10),
                    yaxis=dict(scaleanchor=None),
                )
                st.plotly_chart(fig_co_raw, width='stretch', key="coocc_raw")

            with co_norm_tab:
                fig_co_norm = px.imshow(
                    np.round(coocc_norm, 2),
                    x=unique_types_sorted,
                    y=unique_types_sorted,
                    color_continuous_scale="Purples",
                    labels=dict(color="Normalised"),
                    title=f"Co-occurrence (normalised) — radius {coocc_radius}",
                    text_auto=".2f",
                    aspect="equal",
                )
                fig_co_norm.update_layout(
                    **get_plotly_layout(),
                    height=max(450, n_t * 40 + 100),
                    margin=dict(l=10, r=20, t=50, b=10),
                    yaxis=dict(scaleanchor=None),
                )
                st.plotly_chart(fig_co_norm, width='stretch', key="coocc_norm")

            st.caption(
                "High normalised values indicate cell types that co-locate "
                "more than expected by their abundance alone."
            )

else:
    # ── No file uploaded yet ──────────────────
    st.markdown("""
    <div class="empty-state">
        <p class="empty-icon">📤</p>
        <p class="empty-title">Upload a file to get started</p>
        <p class="empty-sub">Supported formats: CSV, XLSX &nbsp;|&nbsp; Required columns: X, Y, Cell Type</p>
    </div>
    """, unsafe_allow_html=True)
