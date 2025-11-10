# Intensifying Freezing Rain Shifts Southward in Warming US

## Overview

This repository contains code and data supporting the findings of our study "Intensifying Freezing Rain Shifts Southward in Warming US". 

This study presents observational evidence that freezing rain—traditionally associated with colder climates—is intensifying and migrating southward under global warming.
Using a 30-year NOAA Storm Events dataset (1996–2025), enhanced with large language models to quantify event severity, we identify a >200% increase in severe February ice storms along a Texas-to-Pennsylvania corridor, coinciding with a temporal shift from December–January to February. These findings challenge the assumption that warming reduces cold-season hazards, instead revealing redistribution toward mid-latitude regions with limited adaptive capacity.

**Key Findings:**
- Southward shift in freezing rain despite winter warming
- A pivot toward February that amplifies threats to agriculture and ecosystems
- Intensified severity in emerging hotspots
---

## Requirements

### Python Environment
This study uses **Python 3.12.3**. We recommend creating a dedicated conda environment:

```bash
conda create -n freezing_rain python=3.12.3
conda activate freezing_rain
```

### Dependencies
Install all required packages via conda-forge:

```bash
conda install -c conda-forge pandas requests numpy scipy matplotlib seaborn \
    geopandas xarray pygrib scikit-learn shapely pathlib
```

**Required Python packages:**
- Data processing: `pandas`, `numpy`, `scipy`
- Spatial analysis: `geopandas`, `shapely`, `scikit-learn`
- Climate data: `xarray`, `pygrib`
- Visualization: `matplotlib`, `seaborn`
- Utilities: `requests`, `json`, `time`, `os`, `regex`, `datetime`, `typing`, `concurrent.futures`, `pathlib`
---

## Data

### Included in Repository

The `data/` folder contains raw and processed datasets ready for analysis:

| File | Description |
|------|-------------|
| `freezing_rain_events.xlsx` | Raw freezing rain data from NOAA Storm Events Database (1996–2025) including state, county, date, duration, and narrative text |
| `freezing_rain_events_county_llm.xlsx` | Processed data with LLM-extracted ice thickness (in inches) and damage severity classifications (Low, Medium, High) |
| `freezing_rain_random_sample_500.xlsx` | Random sample of 500 events from the full dataset, and conducted independent manual measurements for LLM validation |
| `freezing_rain_stratified_sample_414_counties.xlsx` | Stratified sample from 414 February emerging hotspot counties for validation |
| `february_emerging_hotspots.xlsx` | State and county names of February emerging hotspot counties |
| `february_baseline.xlsx` | State and county names of February high baseline counties |
| `co-est2024-alldata.csv` | 2024 county-level U.S. population data from U.S. Census Bureau |
| `LAI_1996_2025_Feb.nc` | February U.S. Leaf Area Index (1996–2025) and vegetation type data from ERA5 reanalysis |
| `crop/` | U.S. harvest area data for key crops from 2022 USDA Census of Agriculture |

### External Data Sources

Due to file size limitations, some datasets must be downloaded separately:

#### ERA5 Reanalysis Data
- **Description**: Hourly 2-meter temperature data for December–February (1996–2025)
- **Source**: [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)
- **Required files**: 
  - `2m_tem_DJF_1996_2010.grib`
  - `2m_tem_DJF_2011_2025.grib`
- **Format**: GRIB
- **Placement**: Save in `data/` folder
- **Note**: Required for Main Figure 1d Extended Data Figures S2 and S5

#### NOAA nClimGrid Data
- **Description**: Gridded monthly temperature data
- **Source**: [NOAA NCEI](https://www.drought.gov/data-maps-tools/gridded-climate-datasets-noaas-nclimgrid-monthly)
- **Required files**: 'nclimgrid-tavg.nc'
- **Format**: NETCDF 
- **Placement**: Save in `data/` folder
- **Note**: Required for Extended Data Figures S2 and S5
---

## Code

### LLM Extraction (`code/LLM/`)

#### `llm.py`
Extracts quantitative information (ice thickness in inches, damage severity classifications) from unstructured narrative text in the NOAA Storm Events Database using large language models.

**Setup:**
1. Register an account at [OpenRouter](https://openrouter.ai)
2. Obtain API access key
3. Purchase credits for API usage

**Usage:**
```bash
cd code/LLM
python llm.py
```

**Expected runtime**: ~12 hours for complete dataset (54,708 records)

**Output**: `freezing_rain_events_county_llm.xlsx` with extracted ice thickness (inches) and damage severity (Low/Medium/High)

**Note on LLM Usage**: 
Large language models (GPT-4 via OpenRouter API) were used solely for extracting quantitative metrics from unstructured text. All scientific interpretations and conclusions were made by the authors. Model performance was rigorously validated against manual annotations (see Validation section).

---

### Model Validation (`code/LLM/Evaluation/`)

#### Random Sampling Scripts
- `random_select_case.py`: Randomly selects 500 events for validation
- `random_select_case_414county.py`: Randomly selects one event per emerging hotspot county (414 counties)

#### Validation Analysis
- `FigureS10ab.py`: Compares LLM-extracted vs. manual measurements for 500 random events
  - **Output**: Extended Data Figure 10a (ice thickness R²), 10b (damage severity accuracy)
  
- `FigureS10cd.py`: Compares LLM-extracted vs. manual measurements for 414 county samples
  - **Output**: Extended Data Figure 10c (ice thickness R²), 10d (damage severity accuracy)

**Validation Results**: High agreement between LLM extraction and manual annotation (R² > 0.95 for continuous variables, accuracy > 94% for categorical classification)

---

### Analysis and Figures (`code/Analysis/`)

#### Main Text Figures 

**Figure 1: Spatial and temporal patterns of freezing rain events across the continental U.S.**
- `Figure1a.py`: Change in annual freezing rain frequency (2011–2025 minus 1996–2010), events yr⁻¹
- `Figure1b.py`: Annual trend of U.S. freezing rain event counts (1996–2025)
- `Figure1c.py`: Annual trend of longitude and latitude of county-level event centroid
- `Figure1d.py`: Annual trend of winter temperature (DJF, 1996–2025) from ERA5 data
  - *Requires external ERA5 GRIB files*

**Figure 2: February emergence and seasonal timing shifts of freezing rain events**
- `Figure2a.py`: Change in February freezing rain frequency (2011–2025 minus 1996–2010), events yr⁻¹
- `Figure2b.py`: Annual trend of February event counts (1996–2025)
- `Figure2c.py`: Monthly distribution comparison between periods (1996–2010 vs. 2011–2025)
- `Figure2d.py`: Change in peak month of freezing rain occurrence at the county level

**Figure 3: Intensification of severe February freezing rain events across the continental U.S.**
- `Figure3a.py`: Change in February long-duration events (>12 hr), events yr⁻¹
- `Figure3b.py`: Change in February thick-ice events (>0.25 inches), events yr⁻¹
- `Figure3cd.py`: Change in February Medium and High damage severity events, events yr⁻¹

#### Extended Data Figures
**Distribution of Temporal trends:**
- `FigureS1.py`: Distribution of trends in annual event counts (1996–2025), events yr⁻²
- `FigureS3.py`: Distribution of trends in February event counts (1996–2025), events yr⁻²

**Temporal trends:**
- `FigureS4.py`: Annual trends for December and January events
- `FigureS6a.py`: Annual trend of February long-duration events (>12 hr)
- `FigureS6b.py`: Annual trend of February thick-ice events (>0.25 inches)
- `FigureS6cd.py`: Annual trends of February Medium and High damage events
- `FigureS7a.py`: Annual trend of mean event duration (1996–2025)
- `FigureS7b.py`: Annual trend of mean ice thickness (1996–2025)

**Winter temperature patterns:**
- `FigureS2a-d.py`: Winter (DJF) warming trends and patterns from ERA5 and NOAA nClimGrid
  - *Requires external ERA5 and nClimGrid data*
- `FigureS5a-d.py`: February warming trends and patterns from ERA5 and NOAA nClimGrid
  - *Requires external ERA5 and nClimGrid data*

**Exposure and vulnerability:**
- `FigureS8a.py`: County-level population distribution (2024 U.S. Census)
- `FigureS8b.py`: Dominant vegetation types in February
- `FigureS8c.py`: Mean February Leaf Area Index (LAI, 1996–2025)
- `FigureS8d.py`: Dominant crop types by county (2022 USDA Census) and
  - Harvested area (acres) for major crops in February emerging hotspot counties compared to historical high-baseline counties
  - Also generates data for Extended Data Table 1

---

## Reproducibility Guide

### Quick Start

1. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/Freezing_Rain.git
   cd Freezing_Rain
   ```

2. **Set up environment**
   ```bash
   conda install conda-forge::pandas numpy scipy
   # ... continue for other Python packages
   ```

3. **Download external datasets**
   - ERA5 temperature data (for Figure 1d, Figures S2, S5)
   - NOAA nClimGrid data (for Figures S2, S5)
   - Place in `data/` folder

4. **Run analysis**
   ```bash
   cd code/Analysis/
   python Figure1a.py  # Generates Figure 1a
   python Figure1b.py  # Generates Figure 1b
   # ... continue for other figures
   ```

### Complete Workflow

#### Step 1: LLM Extraction (Optional - processed data provided)
```bash
cd code/LLM
# Configure OpenRouter API key in script
python llm.py
# Expected runtime: ~12 hours
```

#### Step 2: Model Validation (Optional - validation results provided)
```bash
cd code/Evaluation
python FigureS10ab.py
python FigureS10cd.py
```

#### Step 3: Generate Figures
```bash
cd code/Analysis/
# Main text figures
for script in Figure*.py; do python $script; done

# Extended data figures
for script in FigureS*.py; do python $script; done
```

### Expected Runtimes
- **LLM extraction**: ~12 hours (requires API credits)
- **All figure generation**: ~30 minutes (excluding external data download)
- **Individual figure**: 10 seconds – 10 minutes

### Troubleshooting

**Missing external data**: If you encounter errors related to missing GRIB or nClimGrid files, ensure you've downloaded the external datasets listed in the Data Availability section.

**Memory issues**: Some scripts processing large climate datasets may require 16+ GB RAM. Consider processing data in chunks if memory is limited.

**API errors**: For LLM extraction, ensure your OpenRouter API key is valid and has sufficient credits.

---

## Data and Code Availability

All data and code necessary to reproduce the findings of this study are available in this repository, except for large external datasets which can be freely downloaded from the sources specified above. Processed datasets are provided to enable immediate reproduction of all figures and analyses.

Raw storm event data were obtained from the NOAA Storm Events Database (https://www.ncei.noaa.gov/stormevents/ftp.jsp). U.S. state and county boundary shapefiles were obtained from the U.S. Census Bureau (https://www2.census.gov/geo/tiger/GENZ2020/shp/ cb_2020_us_state_20m.zip) and (https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip). Winter temperature from 1996 to 2025 is obtained from ERA5 hourly single levels data (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=download) and NOAA nClimGrid data (https://www.drought.gov/data-maps-tools/gridded-climate-datasets-noaas-nclimgrid-monthly). 2024 U.S. county-level population data was obtained from United States Census Bureau (https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/). The February forest type and leaf area index is available from ERA5 monthly single levels data (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download). Crop harvest area data was obtained from 2022 Census of Agriculture Data from United States Department of Agriculture (https://quickstats.nass.usda.gov/). 

All analysis code is provided under the MIT License. Questions about data or code should be directed to the corresponding author.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Version History

- **v1.0.0** (2025-11-10): Initial release
  - Complete dataset (1996–2025)
  - All analysis code for main and extended data figures
  - LLM extraction and validation scripts

---

*Last updated: Nov/10/2025
