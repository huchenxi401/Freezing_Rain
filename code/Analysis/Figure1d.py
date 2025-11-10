import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pygrib
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

def kelvin_to_celsius(temp_k):
    return temp_k - 273.15

def read_grib_data_with_years(file_path):
    try:
        grbs = pygrib.open(file_path)

        first_grb = grbs[1]
        lats, lons = first_grb.latlons()
        yearly_data = {}
        
        grbs.rewind()
        total_messages = grbs.messages
        for i, grb in enumerate(grbs, 1):
            try:
                valid_date = grb['validityDate']
                valid_time = grb['validityTime']
                dt = datetime.strptime(f"{valid_date}{valid_time:04d}", "%Y%m%d%H%M")

                if dt.month == 12:
                    year = dt.year + 1 
                else:
                    year = dt.year

                temp_data = grb.values
                
                if year not in yearly_data:
                    yearly_data[year] = []
                yearly_data[year].append(temp_data)
                
            except Exception as e:
                continue
        
        grbs.close()
        
        for year in yearly_data:
            yearly_data[year] = np.array(yearly_data[year])
        
        years = sorted(yearly_data.keys())
        
        return yearly_data, lats, lons, years
        
    except Exception as e:
        return None, None, None, None

def download_boundary_data():    
    try:
        states_gdf_full = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_20m.zip")

        exclude_states = ['02', '15', '60', '66', '69', '72', '78'] 
        states_gdf = states_gdf_full[~states_gdf_full['STATEFP'].isin(exclude_states)]

        return states_gdf
        
    except Exception as e:
        return None

def create_us_mask(lats, lons, states_gdf, lon_limit=-100):
    
    if states_gdf is None:
        return np.ones(lats.shape, dtype=bool)
    
    try:
        from shapely.geometry import Point
        
        us_boundary = states_gdf.geometry.unary_union
        mask = np.zeros(lats.shape, dtype=bool)
        
        total_points = lats.size
        points_in_us = 0
        
        for i in range(lats.shape[0]):
            for j in range(lats.shape[1]):
                if lons[i, j] < lon_limit: 
                    continue
                
                point = Point(lons[i, j], lats[i, j])
                if us_boundary.contains(point) or us_boundary.touches(point):
                    mask[i, j] = True
                    points_in_us += 1

        
        return mask
        
    except Exception as e:
        return np.ones(lats.shape, dtype=bool)

def calculate_annual_us_mean(yearly_data, us_mask):

    years = sorted(yearly_data.keys())
    annual_means = []
    
    for year in years:
        temp_celsius = kelvin_to_celsius(yearly_data[year])
        
        year_mean_field = np.nanmean(temp_celsius, axis=0)

        if us_mask is not None:
            us_temps = year_mean_field[us_mask]
            us_mean = np.nanmean(us_temps)
        else:
            us_mean = np.nanmean(year_mean_field)
        
        annual_means.append(us_mean)
    
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
    print(f"temperature trend: {slope:.3f} h/yr (95% CI: {ci_lower_weak:.3f} to {ci_upper_weak:.3f}, P={p_value_weak:.4f})")    
    ax.plot(years, temps_smoothed, '-', color='black', markersize=6, alpha=0.8)
    trend_line = slope_weak * years + intercept_weak
    
    ax.plot(years, trend_line, '--', color='black', label=f'R:{r_weak:.2f}**')
    
    ci_low, ci_high = calculate_bootstrap_ci(years, temps_smoothed, years, 
                                           n_bootstrap=10000, confidence=0.95)
    ax.plot(years, ci_low, '--', color='.7', alpha=0.8)
    ax.plot(years, ci_high, '--', color='.7', alpha=0.8)
    years_late = years[10:]
    temps_late = temps_smoothed[10:]
    slope_late, intercept_late, r_late, _, _ = stats.linregress(years_late, temps_late)
    trend_line_late = slope_late * years_late + intercept_late
    ax.plot(years_late, trend_line_late, '--', color='red', label=f'R:{r_late:.2f}**')
    slope, ci_lower_weak, ci_upper_weak, p_value_weak = calculate_confidence_interval(years_late, temps_late)
    print(f"temperature trend: {slope:.3f} h/yr (95% CI: {ci_lower_weak:.3f} to {ci_upper_weak:.3f}, P={p_value_weak:.4f})") 
    ax.set_xlabel('Year')
    ax.set_ylabel('Winter Temperature (°C)')
    plt.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0.5, 3.5)
    ax.set_xlim(years[0] - 1, years[-1] + 1)
    ax.set_xticks(range(years[0], years[-1] + 1, 5))
   
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():

    data_dir = "../data/"
    files = {
        'period1': '2m_tem_DJF_1996_2010.grib',
        'period2': '2m_tem_DJF_2011_2025.grib'
    }

    missing_files = []
    for period in files:
        file_path = os.path.join(data_dir, files[period])
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        return

    states_gdf = download_boundary_data()

    
    all_yearly_data = {}
    lats, lons = None, None
    
    for period_key in ['period1', 'period2']:
        file_path = os.path.join(data_dir, files[period_key])
        
        yearly_data, period_lats, period_lons, years = read_grib_data_with_years(file_path)
        
        if yearly_data is None:
            continue
        if lats is None:
            lats, lons = period_lats, period_lons

        all_yearly_data.update(yearly_data)
    
    if not all_yearly_data:
        return

    us_mask = None
    if states_gdf is not None:
        us_mask = create_us_mask(lats, lons, states_gdf)

    years_array, annual_temps = calculate_annual_us_mean(all_yearly_data, us_mask)

    temps_smoothed = moving_average(annual_temps, window=5)
    
    
    output_file = './figure/US_winter_temperature_timeseries_ERA5_DJF.png'
    
    plot_temperature_timeseries(years_array, annual_temps, temps_smoothed, output_file)
    

if __name__ == "__main__":
    main()