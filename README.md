# Freezing_Rain
Code and data supporting the findings of the study "Intensifying Freezing Rain Shifts Southward in Warming US"

For this study, we used Python Version 3.12.3 to run all code. Before running the code, please make sure the Python packages listed below are installed:
pandas; requests;Json; Time; Os; regex; Datetime; Typing; Futures; cothread; pathlib; geopandas; matplotlib; numpy; Scipy; Shapely; scikit-learn; Seaborn; pygrib; xarray

We installed all these Python packages via conda: https://anaconda.org/conda-forge/repo

# Data
The data folder provides raw data and processed data for plotting and analysis to support this study.
freezing_rain_events.xlsx: Freezing rain raw data from the NOAA Storm Events Database, including information like: STATE, COUNTY, BEGIN_YEAR, BEGIN_MONTH, DURATION_HOURS, EPISODE_NARRATIVE,  EVENT_NARRATIVE
freezing_rain_events_county_llm.xlsx: Freezing rain processed data with LLM-identified ice thickness and damage severity.
freezing_rain_random_sample_500.xlsx: Randomly selected 500 events (without replacement) from the full dataset and conducted independent manual measurements for LLM performance validation.
freezing_rain_stratified_sample_414_counties.xlsx: Randomly selected one event from each of the 414 February emerging hotspot counties for LLM performance validation.
february_emerging_hotspots.xlsx: State and County name of February emerging hotspot counties.
february_baseline.xlsx: State and County name of February high baseline counties.
co-est2024-alldata.csv: 2024 County-level U.S. population data from the United States Census Bureau.
LAI_1996_2025_Feb.nc: February U.S. LAI from 1996 to 2025 from ERA5 reanalysis data.
crop folder: U.S. Harvest area data for several key crops from the 2022 Census of Agriculture Data from the United States Department of Agriculture.

# Code
The code folder provides code for plotting and analysis that support the key findings of this study.
LLM folder:
llm.py: This script uses LLM to identify ice thickness and quantify damage severity of freezing rain from unstructured narrative text in the NOAA Storm Events Database. Before running this code, you need to register an account in OpenRouter (https://openrouter.ai) to get access to the LLM API and buy credit to use it. The entire task will take approximately 12 hours to complete. You will get an Excel file with identified ice thickness (in inches) and damage severity (Low, Medium, High) for all 54,708 county-level records.

Evaluation folder:
random_select_case.py and random_select_case_414county.py are used to randomly select 500 events and 414 February emerging hotspot counties events from freezing_rain_events_county_llm.xlsx for LLM performance validation.
FigureS10ab.py: This code compared extracted ice thickness values and damage severity classifications for randomly selected 500 events against manual measurements from narrative text (R² for continuous agreement, accuracy for categorical agreement). Extended Data Figure 10a,b are the outputs.
FigureS10cd.py: This code compared extracted ice thickness values and damage severity classifications for randomly selected 414 February emerging hotspot counties events against manual measurements from narrative text (R² for continuous agreement, accuracy for categorical agreement). Extended Data Figure 10c,d are the outputs.

Analysis folder:
This folder includes all code analysis and plot figures in the main text and the Extended Data.
Main:
Figure1a.py: This code plots the change in annual freezing rain frequency (2011–2025 minus 1996–2010), in events yr⁻¹, generating Figure 1a in the main text.
Figure1b.py: This code plots the annual trend of U.S. freezing rain event counts (1996–2025), generating Figure 1b in the main text.
Figure1c.py: This code plots the annual trend of longitude and latitude of the county-level event centroid (1996–2025), generating Figure 1c in the main text.
Figure1d.py: This code plots the annual trend of winter temperature (December to February) from 1996 to 2025 using hourly ERA5 temperature reanalysis data. The hourly data is too large to upload; you need to download 2m temperature in DJF from (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=download) in grib version, and save it as 2m_tem_DJF_1996_2010.grib and 2m_tem_DJF_2011_2025.grib in the data folder.
Figure2a.py: This code plots the change in mean February freezing rain frequency (2011–2025 minus 1996–2010), in events yr⁻¹, generating Figure 2a in the main text.
Figure2b.py: This code plots the annual trend of U.S. February freezing rain event counts (1996–2025), generating Figure 2b in the main text.
Figure2c.py: This code plots the monthly distribution of freezing rain events averaged over 1996–2010 and 2011–2025 in events yr⁻¹, generating Figure 2c in the main text.
Figure2d.py: This code plots the change in peak month of freezing rain occurrence at the county level between periods, generating Figure 2d in the main text.
Figure3a.py: This code plots the change in mean February freezing rain frequency (2011–2025 minus 1996–2010) for events with duration >12 hr, in events yr⁻¹, generating Figure 3a in the main text.
Figure3b.py: This code plots the change in mean February freezing rain frequency (2011–2025 minus 1996–2010) for events with ice thickness >0.25 inches (6.4 mm), in events yr⁻¹, generating Figure 3b in the main text.
Figure3cd.py: This code plots the change in mean February freezing rain frequency (2011–2025 minus 1996–2010) for events classified as Medium and High damage severity, in events yr⁻¹, generating Figure 3c and 3d in the main text.

Extended Data:
FigureS1.py: This code plots linear trends in annual freezing rain event counts from 1996 to 2025, expressed as events yr⁻², generating Figure 1 in the Extended Data.
FigureS2a-d.py: These codes plot the temporal and spatial patterns of winter (December to February) warming across the contiguous U.S from ERA5 and NOAA nClimGrid data. The NOAA nClimGrid data can be downloaded from (https://www.ncei.noaa.gov/data/nclimgrid-monthly/access/). Please put it in the data folder.
FigureS3.py: This code plots the linear trends in annual February freezing rain event counts from 1996 to 2025, expressed as events yr⁻², generating Figure 3 in the Extended Data.
FigureS4.py: This code plots the annual trend of December and January freezing rain event counts from 1996 to 2025, generating Figure 4a and 4b in the Extended Data.
FigureS5a-d.py: These codes plot the temporal and spatial patterns of February warming across the contiguous U.S from ERA5 and NOAA nClimGrid data, generating Figure 5a-5d in the Extended Data.
FigureS6a.py: This code plots the annual trend of U.S February long-duration events (> 12 hr) counts from 1996 to 2025, generating Figure 6a in the Extended Data.
FigureS6b.py: This code plots the annual trend of U.S February events (Ice thickness > 0.25 inches) counts from 1996 to 2025, generating Figure 6b in the Extended Data.
FigureS6cd.py: This code plots the annual trend of U.S February Medium and High damage event counts from 1996 to 2025, generating Figure 6c and 6d in the Extended Data.
FigureS7a.py: This code plots the annual trend of mean duration from 1996 to 2025, generating Figure 7a in the Extended Data.
FigureS7b.py: This code plots the annual trend of mean ice thickness from 1996 to 2025, generating Figure 7b in the Extended Data.
FigureS8a.py: This code plots county-level population (2024 U.S. Census Bureau) distribution, generating Figure 8a in the Extended Data.
FigureS8b.py: This code plots the distribution of dominant vegetation types in each county in February, generating Figure 8b in the Extended Data.
FigureS8c.py: This code plots the mean February leaf area index (LAI) from 1996 to 2025, generating Figure 8c in the Extended Data.
FigureS8d.py: This code plots the distribution of dominant major crop types (2022 USDA Census of Agriculture) in February, generating Figure 8d in the Extended Data. It also outputs information for Extended Data Table 1.
