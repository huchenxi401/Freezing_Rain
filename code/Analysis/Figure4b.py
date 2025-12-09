import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
from datetime import datetime
import warnings
from shapely.geometry import Point
from scipy import stats
warnings.filterwarnings('ignore')

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
        
        return counties_gdf, states_gdf
        
    except Exception as e:
        return None, None

def create_hotspot_mask(lats, lons, hotspot_counties_gdf):
    
    if hotspot_counties_gdf is None or len(hotspot_counties_gdf) == 0:
        return np.zeros(lats.shape, dtype=bool)
    
    try:
        hotspot_boundary = hotspot_counties_gdf.geometry.unary_union
        
        mask = np.zeros(lats.shape, dtype=bool)
        
        total_points = lats.size
        points_in_hotspots = 0
        
        for i in range(lats.shape[0]):
            for j in range(lats.shape[1]):
                point = Point(lons[i, j], lats[i, j])
                if hotspot_boundary.contains(point) or hotspot_boundary.touches(point):
                    mask[i, j] = True
                    points_in_hotspots += 1
            
            if (i + 1) % 10 == 0:
                progress = (i + 1) / lats.shape[0] * 100
        return mask
        
    except Exception as e:
        return np.zeros(lats.shape, dtype=bool)

def read_era5_temperature_data(file_path):
    try:
        ds = xr.open_dataset(file_path)
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        pressure_levels = ds['pressure_level'].values

        time_values = pd.to_datetime(ds['valid_time'].values, unit='s')

        temp_data = ds['t'].values

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        return temp_data, lat_grid, lon_grid, pressure_levels, time_values, ds
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

def bootstrap_confidence_interval(data, n_bootstrap=10000, confidence=0.95, random_seed=42):
    np.random.seed(random_seed)

    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    n = len(data)
    bootstrap_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    mean = np.mean(data)
    
    return mean, ci_lower, ci_upper

def calculate_period_vertical_profile(temp_data, hotspot_mask, pressure_levels, 
                                       time_values, start_year, end_year, n_bootstrap=10000):

    time_mask = (time_values.year >= start_year) & (time_values.year <= end_year) & (time_values.month == 2)
    filtered_indices = np.where(time_mask)[0]

    profile_temps = []
    ci_lower_list = []
    ci_upper_list = []
    all_level_temps = []
    
    for level_idx, pressure in enumerate(pressure_levels):
        level_temps = []
        
        for time_idx in filtered_indices:
            temp_field = temp_data[time_idx, level_idx, :, :]

            if hotspot_mask is not None and np.any(hotspot_mask):
                hotspot_temps = temp_field[hotspot_mask]
                hotspot_mean = np.nanmean(hotspot_temps)
                level_temps.append(hotspot_mean)

        level_temps_celsius = np.array(level_temps) - 273.15

        if len(level_temps_celsius) > 0:
            mean, ci_lower, ci_upper = bootstrap_confidence_interval(
                level_temps_celsius, n_bootstrap=n_bootstrap
            )
            profile_temps.append(mean)
            ci_lower_list.append(ci_lower)
            ci_upper_list.append(ci_upper)
            all_level_temps.append(level_temps_celsius)

        else:
            profile_temps.append(np.nan)
            ci_lower_list.append(np.nan)
            ci_upper_list.append(np.nan)
            all_level_temps.append(np.array([]))
    
    return np.array(profile_temps), np.array(ci_lower_list), np.array(ci_upper_list), all_level_temps

def calculate_significance(temps_period1, temps_period2, pressure_levels):
    p_values = []
    significant_mask = []
    
    for level_idx, pressure in enumerate(pressure_levels):
        series1 = temps_period1[level_idx]
        series2 = temps_period2[level_idx]
        valid_mask1 = ~np.isnan(series1)
        valid_mask2 = ~np.isnan(series2)
        
        if np.sum(valid_mask1) > 2 and np.sum(valid_mask2) > 2:
            t_stat, p_val = stats.ttest_ind(
                series1[valid_mask1], 
                series2[valid_mask2],
                equal_var=False
            )
            p_values.append(p_val)
            is_significant = p_val < 0.05
            significant_mask.append(is_significant)
            
            sig_marker = "**" if is_significant else ""
        else:
            p_values.append(np.nan)
            significant_mask.append(False)
    
    return np.array(p_values), np.array(significant_mask)

def plot_vertical_profiles(pressure_levels, 
                           profile_1996_2010, ci_lower_1, ci_upper_1,
                           profile_2011_2025, ci_lower_2, ci_upper_2,
                           significant_mask, output_file):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    
   
    ax1.fill_betweenx(pressure_levels, ci_lower_1, ci_upper_1, 
                      color='blue', alpha=0.15, linewidth=0)
    ax1.fill_betweenx(pressure_levels, ci_lower_2, ci_upper_2, 
                      color='red', alpha=0.15, linewidth=0)

    line1 = ax1.plot(profile_1996_2010, pressure_levels, '-o', color='blue', 
            linewidth=2, markersize=6, label='1996-2010', alpha=0.8)
    
    line2 = ax1.plot(profile_2011_2025, pressure_levels, '-s', color='red', 
            linewidth=2, markersize=6, label='2011-2025', alpha=0.8)

    ax1.invert_yaxis()
    
    ax1.set_xlabel('Temperature (°C)', fontsize=12, color='black')
    ax1.set_ylabel('Pressure Level (hPa)', fontsize=12)
    ax1.tick_params(axis='x', labelcolor='black', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = ax1.twiny()

    temp_diff = profile_2011_2025 - profile_1996_2010

    line3 = ax2.plot(temp_diff, pressure_levels, '-', color='darkgreen', 
            linewidth=2.5, alpha=0.8, label='Difference (* p<0.05)')
 
    for i, (diff, pressure, is_sig) in enumerate(zip(temp_diff, pressure_levels, significant_mask)):
        if is_sig:
            ax2.plot(diff, pressure, '*', color='darkgreen', 
                    markersize=12, markeredgewidth=0)
        else:
            ax2.plot(diff, pressure, '^', color='darkgreen', 
                    markersize=7, markeredgewidth=0)

    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

    ax2.set_xlabel('Temperature Difference (°C)', fontsize=12, color='darkgreen')
    ax2.tick_params(axis='x', labelcolor='darkgreen', labelsize=10)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    legend1 = ax1.legend(lines, labels, fontsize=10, loc='upper center')
    ax1.add_artist(legend1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    for i, pressure in enumerate(pressure_levels):
        diff = profile_2011_2025[i] - profile_1996_2010[i]
        sig_marker = "**" if significant_mask[i] else ""

def main():

    era5_file = "../../data/Tem_monthly_DJF_1996-2025.nc"
    hotspot_file = "../../data/february_emerging_hotspots.xlsx"
    output_file = './figure4b.png'

    N_BOOTSTRAP = 10000  

    hotspots_df = load_emerging_hotspots(hotspot_file)
    if hotspots_df is None:
        return

    counties_gdf, states_gdf = download_boundary_data()
    if counties_gdf is None:
        return

    hotspots_df['COUNTY_NAME_TITLE'] = hotspots_df['COUNTY_NAME'].str.title()
    
    hotspot_counties = counties_gdf.merge(
        hotspots_df,
        left_on=['STUSPS', 'NAME'],
        right_on=['STATE_ABBREV', 'COUNTY_NAME_TITLE'],
        how='inner'
    )

    
    temp_data, lats, lons, pressure_levels, time_values, ds = read_era5_temperature_data(era5_file)
    
    if temp_data is None:
        return

    hotspot_mask = create_hotspot_mask(lats, lons, hotspot_counties)
    
    if not np.any(hotspot_mask):
        ds.close()
        return

    profile_1996_2010, ci_lower_1, ci_upper_1, temps_1996_2010 = calculate_period_vertical_profile(
        temp_data, hotspot_mask, pressure_levels, time_values, 1996, 2010, n_bootstrap=N_BOOTSTRAP
    )

    profile_2011_2025, ci_lower_2, ci_upper_2, temps_2011_2025 = calculate_period_vertical_profile(
        temp_data, hotspot_mask, pressure_levels, time_values, 2011, 2025, n_bootstrap=N_BOOTSTRAP
    )
    
    ds.close()

    p_values, significant_mask = calculate_significance(
        temps_1996_2010, temps_2011_2025, pressure_levels
    )

    plot_vertical_profiles(pressure_levels, 
                          profile_1996_2010, ci_lower_1, ci_upper_1,
                          profile_2011_2025, ci_lower_2, ci_upper_2,
                          significant_mask, output_file)
    

if __name__ == "__main__":

    main()
