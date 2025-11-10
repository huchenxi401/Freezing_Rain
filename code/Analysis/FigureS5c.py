import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
from datetime import datetime
import warnings
from scipy.stats import bootstrap
from scipy import stats
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

def calculate_confidence_interval(x, y, confidence=0.95):
    n = len(x)
    if n < 3:
        return None, None, None, None
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred

    mse = np.sum(residuals**2) / (n - 2)  
 
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean)**2)

    slope_std_err = np.sqrt(mse / sxx)

    alpha = 1 - confidence
    t_val = stats.t.ppf(1 - alpha/2, n - 2)

    slope_ci_lower = slope - t_val * slope_std_err
    slope_ci_upper = slope + t_val * slope_std_err
    
    return slope, slope_ci_lower, slope_ci_upper, p_value

def calculate_bootstrap_ci(years, temps_smoothed, unique_years, n_bootstrap=10000, confidence=0.95):
    
    def trend_func(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        return slope * unique_years + intercept

    data = (years, temps_smoothed)

    rng = np.random.default_rng(42)  
    res = bootstrap(data, trend_func, n_resamples=n_bootstrap, 
                   confidence_level=confidence, random_state=rng,
                   paired=True)
    
    return res.confidence_interval.low, res.confidence_interval.high

def read_nclimgrid_data_with_years(file_path, start_year, end_year, filter_month=None):    
    try:
        ds = xr.open_dataset(file_path)

        lats = ds['lat'].values
        lons = ds['lon'].values

        time_values = pd.to_datetime(ds['time'].values)

        yearly_data = {}

        filtered_count = 0
        for idx, t in enumerate(time_values):
            year = t.year
            month = t.month

            is_in_range = start_year <= year <= end_year

            if filter_month is not None:
                is_month_match = (month == filter_month)
            else:
                is_month_match = True
            
            if is_in_range and is_month_match:
                if year not in yearly_data:
                    yearly_data[year] = []

                temp_data = ds['tavg'].values[idx, :, :]
                yearly_data[year].append(temp_data)
                filtered_count += 1

        ds.close()

        for year in yearly_data:
            yearly_data[year] = np.array(yearly_data[year])
        
        years = sorted(yearly_data.keys())

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        return yearly_data, lat_grid, lon_grid, years
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None

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
        from shapely.geometry import Point

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
            
        return mask
        
    except Exception as e:
        return np.zeros(lats.shape, dtype=bool)

def calculate_annual_hotspot_mean(yearly_data, hotspot_mask):
    years = sorted(yearly_data.keys())
    annual_means = []
    
    for year in years:
        temp_celsius = yearly_data[year]

        year_mean_field = np.nanmean(temp_celsius, axis=0)

        if hotspot_mask is not None and np.any(hotspot_mask):
            hotspot_temps = year_mean_field[hotspot_mask]
            hotspot_mean = np.nanmean(hotspot_temps)
        else:
            hotspot_mean = np.nan
        
        annual_means.append(hotspot_mean)
    return np.array(years), np.array(annual_means)

def moving_average(data, window=5):
    if len(data) < window:
        return data
    smoothed = np.full_like(data, np.nan)
    half_window = window // 2
    
    for i in range(len(data)):
        if i <= half_window - 1: 
            start_idx = 0
            end_idx = min(i + half_window + 1, len(data))
        elif i >= len(data) - half_window: 
            start_idx = max(i - half_window, 0)
            end_idx = len(data)
        else:  
            start_idx = i - half_window
            end_idx = i + half_window + 1
        window_data = data[start_idx:end_idx]
        if len(window_data) > 0:
            smoothed[i] = np.nanmean(window_data)
        else:
            smoothed[i] = np.nan
    
    return smoothed

def plot_temperature_timeseries(years, temps, temps_smoothed, output_file):
    from scipy import stats
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    model = LinearRegression()
    model.fit(years.reshape(-1, 1), temps_smoothed)
    trend_line = model.predict(years.reshape(-1, 1))
    slope_weak, intercept_weak, r_weak, p_weak, _ = stats.linregress(years, temps_smoothed)
    slope, ci_lower_weak, ci_upper_weak, p_value_weak = calculate_confidence_interval(years, temps_smoothed)
    print(f"trend: {slope:.3f} °C/yr (95% CI: {ci_lower_weak:.3f} to {ci_upper_weak:.3f}, P={p_value_weak:.4f})")    
    ax.plot(years, temps_smoothed, '-', color='black', markersize=6, alpha=0.8)

    trend_line = slope_weak * years + intercept_weak
    
    ax.plot(years, trend_line, '--', color='black', label=f'R:{r_weak:.2f}')
    
    ci_low, ci_high = calculate_bootstrap_ci(years, temps_smoothed, years, 
                                           n_bootstrap=10000, confidence=0.95)
    ax.plot(years, ci_low, '--', color='.7', alpha=0.8)
    ax.plot(years, ci_high, '--', color='.7', alpha=0.8)
    years_late = years[15:]
    temps_late = temps_smoothed[15:]
    slope_late, intercept_late, r_late, _, _ = stats.linregress(years_late, temps_late)
    trend_line_late = slope_late * years_late + intercept_late
    ax.plot(years_late, trend_line_late, '--', color='red', label=f'R:{r_late:.2f}**')
    ax.set_xlabel('Year')
    ax.set_ylabel('February Temperature (°C)')
    plt.legend(fontsize=10, loc='upper left') 
    ax.set_ylim(3, 7)
    ax.set_xlim(years[0] - 1, years[-1] + 1)
    ax.set_xticks(range(years[0], years[-1] + 1, 5))
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    data_file = "../../data/nclimgrid-tavg.nc"
    hotspot_file = "../../data/february_emerging_hotspots.xlsx"

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

    yearly_data, lats, lons, years = read_nclimgrid_data_with_years(
        data_file,
        start_year=1996,
        end_year=2025,
        filter_month=2 
    )
    
    if yearly_data is None:
        return
    
    all_years = sorted(yearly_data.keys())
    hotspot_mask = create_hotspot_mask(lats, lons, hotspot_counties)
    
    if not np.any(hotspot_mask):
        return

    years_array, annual_temps = calculate_annual_hotspot_mean(yearly_data, hotspot_mask)
    temps_smoothed = moving_average(annual_temps, window=5)

    output_file = './figure/nclimgrid_hotspot_February_temperature_timeseries.png'
    
    plot_temperature_timeseries(years_array, annual_temps, temps_smoothed, output_file)
    
if __name__ == "__main__":
    main()