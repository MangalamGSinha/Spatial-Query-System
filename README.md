# 🔬 Spatial Query System

An interactive Streamlit application for visualizing and analyzing spatial cell-type data from multiplexed imaging experiments (e.g., CODEX, IMC, MERFISH).

## Features

### 📊 Visualization

- Interactive scatter plot of cell positions, color-coded by cell type
- Filterable 2D density heatmap with contour overlays

### 🔍 Nearest Neighbour Queries

- **Radius search** — find all neighbours within a user-defined radius
- **KNN search** — find K nearest neighbours
- Visual highlighting of query/neighbour cells with connecting lines
- Downloadable results as CSV

### 📈 Spatial Analysis & Statistics

- **Cell Type Composition** — bar chart and percentage breakdown
- **Density Heatmap** — 2D kernel density estimation per cell type
- **Spatial Autocorrelation** — Moran's I (clustered/random/dispersed) and Ripley's K function vs CSR
- **Pairwise Distance Matrix** — mean distance heatmap between all cell-type pairs
- **Co-occurrence Analysis** — raw and normalised co-occurrence matrices

## Installation

```bash
git clone https://github.com/MangalamGSinha>/spatial-query-system.git
cd spatial-query-system
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Upload a CSV or Excel file with columns for **X**, **Y** coordinates and **Cell Type**. The app auto-detects common column names.

## Requirements

- Python 3.8+
- streamlit
- pandas
- plotly
- scipy
- numpy
- openpyxl

## Tech Stack

| Component        | Library               |
| ---------------- | --------------------- |
| UI Framework     | Streamlit             |
| Plotting         | Plotly                |
| Spatial Indexing | scipy.spatial.cKDTree |
| Data Handling    | pandas, numpy         |

## License

MIT
