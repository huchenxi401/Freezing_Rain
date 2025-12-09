import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
from datetime import datetime
import warnings
from scipy import stats
from scipy.stats import bootstrap
warnings.filterwarnings('ignore')

STATE_ABBREV = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
    'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
    'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA',
    'KANSAS': 'KS', 'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
    'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH',
    'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC',
    'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK', 'OREGON': 'OR', 'PENNSYLVANIA': 'PA',
    'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC', 'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN',
    'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA',
    'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY', 'DISTRICT OF COLUMBIA': 'DC'
}

def standardize_county_name(county_name):
    if pd.isna(county_name):
        return None
    
    county_name = str(county_name).strip().upper()

    suffixes_to_remove = [' COUNTY', ' PARISH', ' BOROUGH', ' CENSUS AREA', ' CITY', ' CITY AND BOROUGH']
    for suffix in suffixes_to_remove:
        if county_name.endswith(suffix):
            county_name = county_name[:-len(suffix)].strip()
            break
    
    return county_name

def load_emerging_hotspots(hotspot_file):
    try:
        hotspots_df = pd.read_excel(hotspot_file)
        return hotspots_df
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def download_boundary_data():
    try:
        counties_gdf_full = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip")

        states_gdf_full = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_20m.zip")
        
        exclude_states = ['02', '15', '60', '66', '69', '72', '78'] 
        states_gdf = states_gdf_full[~states_gdf_full['STATEFP'].isin(exclude_states)]
        counties_gdf = counties_gdf_full[~counties_gdf_full['STATEFP'].isin(exclude_states)]

        counties_gdf['COUNTY_CLEAN'] = counties_gdf['NAME'].apply(standardize_county_name)

        return counties_gdf, states_gdf
        
    except Exception as e:
        return None, None

def read_era5_vwind_data(file_path):
    try:
        ds = xr.open_dataset(file_path)
        
        ds['valid_time'] = xr.decode_cf(ds).valid_time

        lats = ds['latitude'].values
        lons = ds['longitude'].values

        v = ds['v'].squeeze()

        time_values = pd.to_datetime(ds['valid_time'].values)
        
        return v, lats, lons, time_values, ds
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def calculate_regional_vwind_anomaly(v_data, lats, lons, time_values):
    lat_mask = (lats >= 40) & (lats <= 50)  
    lon_mask = (lons >= -120) & (lons <= -90) 

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    feb_mask = time_values.month == 2
    feb_years = time_values[feb_mask].year
    
    v_region = v_data[:, lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]

    v_regional_mean = []
    for i in range(len(time_values)):
        if feb_mask[i]:
            regional_avg = np.nanmean(v_region[i, :, :])
            v_regional_mean.append(regional_avg)
    
    v_regional_mean = np.array(v_regional_mean)
    
    climatology = np.nanmean(v_regional_mean)

    yearly_vwind_anomaly = {}
    year_means = []
    
    for year, v_val in zip(feb_years, v_regional_mean):
        if year not in yearly_vwind_anomaly:
            yearly_vwind_anomaly[year] = []
        yearly_vwind_anomaly[year].append(v_val)

    for year in yearly_vwind_anomaly:
        yearly_mean = np.nanmean(yearly_vwind_anomaly[year])
        anomaly = yearly_mean - climatology
        yearly_vwind_anomaly[year] = anomaly
    
    return yearly_vwind_anomaly

def load_freezing_rain_events(excel_file, hotspot_counties):

    df = pd.read_excel(excel_file, usecols=['STATE', 'COUNTY', 'BEGIN_YEAR', 'BEGIN_MONTH', 'DURATION_HOURS', 'EVENT_ID'])

    df['STATE_CLEAN'] = df['STATE'].str.strip().str.upper()
    df['COUNTY_CLEAN'] = df['COUNTY'].apply(standardize_county_name)
    df['STATE_ABBREV'] = df['STATE_CLEAN'].map(STATE_ABBREV)

    df_clean = df.dropna(subset=['STATE_ABBREV', 'COUNTY_CLEAN', 'BEGIN_YEAR', 'BEGIN_MONTH'])
    df_clean = df_clean[df_clean['DURATION_HOURS'] < 144]

    df_clean = df_clean[(df_clean['BEGIN_YEAR'] >= 1996) & (df_clean['BEGIN_YEAR'] <= 2025)]

    df_feb = df_clean[df_clean['BEGIN_MONTH'] == 2]
    
    hotspot_list = set()
    for idx, row in hotspot_counties.iterrows():
        state = row['STUSPS']
        county = row['COUNTY_CLEAN']
        hotspot_list.add(f"{state}_{county}")

    df_feb['COUNTY_KEY'] = df_feb['STATE_ABBREV'] + '_' + df_feb['COUNTY_CLEAN']
    df_hotspot = df_feb[df_feb['COUNTY_KEY'].isin(hotspot_list)]

    county_year_counts = df_hotspot.groupby(['COUNTY_KEY', 'BEGIN_YEAR']).size().reset_index(name='EVENT_COUNT')
    
    return county_year_counts

def create_vwind_freezing_rain_dataset(yearly_vwind_anomaly, county_year_counts, hotspot_counties):

    county_keys = []
    for idx, row in hotspot_counties.iterrows():
        state = row['STUSPS']
        county = row['COUNTY_CLEAN']
        county_keys.append(f"{state}_{county}")
    
    data_list = []
    
    for year in range(1996, 2026):
        if year in yearly_vwind_anomaly:
            vwind_anomaly = yearly_vwind_anomaly[year]

            for county_key in county_keys:
                event_count = county_year_counts[
                    (county_year_counts['COUNTY_KEY'] == county_key) & 
                    (county_year_counts['BEGIN_YEAR'] == year)
                ]
                
                if len(event_count) > 0:
                    count = event_count.iloc[0]['EVENT_COUNT']
                else:
                    count = 0
                
                data_list.append({
                    'COUNTY_KEY': county_key,
                    'YEAR': year,
                    'VWIND_ANOMALY': vwind_anomaly,
                    'EVENT_COUNT': count
                })
    
    merged_df = pd.DataFrame(data_list)

    return merged_df

def calculate_bootstrap_ci(x, y, n_bootstrap=10000, confidence=0.95):

    def fit_line(x_data, y_data):
        slope, intercept = np.polyfit(x_data, y_data, 1)
        return slope * x_data + intercept

    data = (x, y)

    rng = np.random.default_rng(42)

    x_range = np.linspace(x.min(), x.max(), 100)
    
    def statistic(x_sample, y_sample):
        slope, intercept = np.polyfit(x_sample, y_sample, 1)
        return slope * x_range + intercept
    
    res = bootstrap(data, statistic, n_resamples=n_bootstrap, 
                   confidence_level=confidence, random_state=rng,
                   paired=True)
    
    return x_range, res.confidence_interval.low, res.confidence_interval.high


def plot_scatter_with_regression(merged_df, output_file):
    x = merged_df['VWIND_ANOMALY'].values
    y = merged_df['EVENT_COUNT'].values
    
    corr_coef, p_value = stats.pearsonr(x, y)

    slope, intercept, r_value, p_value_reg, std_err = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.1, s=10, color='lightgray', edgecolors='none')

    n_bins = 15 
    x_min, x_max = x.min(), x.max()
    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if np.sum(mask) > 0:
            bin_means.append(np.mean(y[mask]))
            bin_stds.append(np.std(y[mask]) / np.sqrt(np.sum(mask)))  
            bin_counts.append(np.sum(mask))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_counts.append(0)
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, 
                fmt='o', markersize=5, capsize=2, capthick=1,
                color='#58539f', ecolor='#58539f',
                label='Binned mean', zorder=3)

    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='#d86967', linewidth=2, 
            label=f'R = {corr_coef:.2f}**')

    x_range, ci_low, ci_high = calculate_bootstrap_ci(x, y, n_bootstrap=10000, confidence=0.95)
    
    ax.fill_between(x_range, ci_low, ci_high, color='#d86967', alpha=0.2, 
                    label='95% CI')
    
    ax.set_xlabel('250 hPa V-wind Anomaly [m/s]', fontsize=10)
    ax.set_ylabel('February Freezing Rain Events Yr⁻¹', fontsize=10)
    
    
    ax.legend(fontsize=10, loc='upper left')
    

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    era5_file = "../../data/250hPa_monthly_wind_DJF_1996_2025.nc"
    hotspot_file = "../../data/february_emerging_hotspots.xlsx"
    freezing_rain_file = "../../data/freezing_rain_events_county_llm.xlsx"
    output_file = './figureS10a.png'

    hotspots_df = load_emerging_hotspots(hotspot_file)
    if hotspots_df is None:
        print("错误: 无法加载hotspot counties数据")
        return
    
    counties_gdf, states_gdf = download_boundary_data()
    if counties_gdf is None:
        return

    hotspots_df['COUNTY_NAME_TITLE'] = hotspots_df['COUNTY_NAME'].str.title()
    
    hotspot_counties = counties_gdf.merge(
        hotspots_df,
        left_on=['STUSPS', 'COUNTY_CLEAN'],
        right_on=['STATE_ABBREV', 'COUNTY_NAME'],
        how='inner'
    )

    v_data, lats, lons, time_values, ds = read_era5_vwind_data(era5_file)
    
    yearly_vwind_anomaly = calculate_regional_vwind_anomaly(
        v_data, lats, lons, time_values
    )
    
    ds.close()

    county_year_counts = load_freezing_rain_events(freezing_rain_file, hotspot_counties)

    merged_df = create_vwind_freezing_rain_dataset(
        yearly_vwind_anomaly, county_year_counts, hotspot_counties
    )

    plot_scatter_with_regression(merged_df, output_file)
    
if __name__ == "__main__":
    main()