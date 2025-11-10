import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def read_nclimgrid_data(file_path, start_year, end_year):
    try:
        ds = xr.open_dataset(file_path)

        lats = ds['lat'].values
        lons = ds['lon'].values
        
        time_values = pd.to_datetime(ds['time'].values)

        djf_mask = []
        for t in time_values:
            year = t.year
            month = t.month

            if month == 12:
                season_year = year + 1
            else:
                season_year = year

            is_djf = month in [12, 1, 2]
            is_in_range = start_year <= season_year <= end_year
            
            djf_mask.append(is_djf and is_in_range)
        
        djf_mask = np.array(djf_mask)
        
        temp_data = ds['tavg'].values[djf_mask, :, :] 
        time_stamps = time_values[djf_mask]
        
        ds.close()

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        return temp_data, lat_grid, lon_grid, time_stamps
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None

def calculate_djf_mean(temp_data):

    djf_mean = np.nanmean(temp_data, axis=0)

    return djf_mean

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

def create_us_mask(lats, lons, states_gdf):
    if states_gdf is None:
        return np.ones(lats.shape, dtype=bool)
    
    try:
        from shapely.geometry import Point
        import geopandas as gpd

        us_boundary = states_gdf.geometry.unary_union

        mask = np.zeros(lats.shape, dtype=bool)

        total_points = lats.size
        points_in_us = 0
        
        for i in range(lats.shape[0]):
            for j in range(lats.shape[1]):
                point = Point(lons[i, j], lats[i, j])
                if us_boundary.contains(point) or us_boundary.touches(point):
                    mask[i, j] = True
                    points_in_us += 1

        return mask
        
    except Exception as e:
        return np.ones(lats.shape, dtype=bool)

def calculate_ttest_significance(temp_data1, temp_data2):
    from scipy import stats
    p_values = np.ones(temp_data1.shape[1:]) 
    for i in range(temp_data1.shape[1]):
        for j in range(temp_data1.shape[2]):
            series1 = temp_data1[:, i, j]
            series2 = temp_data2[:, i, j]

            valid_mask1 = ~np.isnan(series1)
            valid_mask2 = ~np.isnan(series2)
            
            if np.sum(valid_mask1) > 2 and np.sum(valid_mask2) > 2:
                t_stat, p_val = stats.ttest_ind(
                    series1[valid_mask1], 
                    series2[valid_mask2],
                    equal_var=False  
                )
                p_values[i, j] = p_val
        
    significant_mask = p_values < 0.05
    
    total_points = p_values.size
    significant_points = np.sum(significant_mask)
    
    return p_values, significant_mask

def create_difference_map(diff_data, lats, lons, counties_gdf, states_gdf, 
                         title, output_file, us_mask=None,
                         p_values=None, significant_mask=None):

    if us_mask is not None:
        diff_data_masked = np.ma.masked_where(~us_mask, diff_data)
        valid_data = diff_data[us_mask]
    else:
        diff_data_masked = diff_data
        valid_data = diff_data
    
    fig_size = (6, 4)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    
    vmin = -1.5
    vmax = 1.5

    levels = np.linspace(vmin, vmax, 21)
    cs = ax.contourf(lons, lats, diff_data_masked, 
                     levels=levels, 
                     cmap='RdBu_r', 
                     vmin=vmin, 
                     vmax=vmax,
                     extend='both')
    if significant_mask is not None:
        if us_mask is not None:
            plot_significant_mask = significant_mask & us_mask
        else:
            plot_significant_mask = significant_mask

        sig_count = np.sum(plot_significant_mask)
        
        if sig_count > 0:
            step = 18  
            
            sig_indices = np.where(plot_significant_mask)

            for idx in range(0, len(sig_indices[0]), step):
                i = sig_indices[0][idx]
                j = sig_indices[1][idx]
                ax.plot(lons[i, j], lats[i, j], 'k.', markersize=2, markerfacecolor='black', 
                markeredgewidth=0)
    cbar = plt.colorbar(cs, ax=ax, shrink=0.8, orientation='horizontal', 
                       pad=0.15, aspect=30)
    cbar.set_label('Winter Temperature Difference (°C)', fontsize=12)

    if states_gdf is not None:
        states_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)

    if counties_gdf is not None:
        counties_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.2)
    
    ax.set_xlim(-125, -65)
    ax.set_ylim(25, 50)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
def main():
    data_file = "../../data/nclimgrid-tavg.nc"

    periods = {
        'period1': {'name': '1996-2010', 'start': 1996, 'end': 2010},
        'period2': {'name': '2011-2025', 'start': 2011, 'end': 2025}
    }

    if not os.path.exists(data_file):
        return

    counties_gdf, states_gdf = download_boundary_data()
    if counties_gdf is None or states_gdf is None:
        print("warning")

    results = {}
    us_mask = None 
    
    for period_key in ['period1', 'period2']:
        period_info = periods[period_key]
        period_name = period_info['name']

        temp_data, lats, lons, time_stamps = read_nclimgrid_data(
            data_file, 
            period_info['start'], 
            period_info['end']
        )
        
        if temp_data is None:
            continue

        if us_mask is None and states_gdf is not None:
            us_mask = create_us_mask(lats, lons, states_gdf)

        djf_mean = calculate_djf_mean(temp_data)

        results[period_name] = {
            'djf_mean': djf_mean,
            'temp_data': temp_data,
            'lats': lats,
            'lons': lons
        }
        
    
    if len(results) > 0:
        all_temps = []
        for period in results:
            if us_mask is not None:
                valid_temps = results[period]['djf_mean'][us_mask]
                all_temps.extend(valid_temps)
            else:
                all_temps.extend(results[period]['djf_mean'].flatten())
        
        global_vmin = np.nanmin(all_temps)
        global_vmax = np.nanmax(all_temps)
        
        if len(results) == 2:
            period1_temp = results['1996-2010']['djf_mean']
            period2_temp = results['2011-2025']['djf_mean']
            
            temperature_change = period2_temp - period1_temp
            
            temp_data1 = results['1996-2010']['temp_data']  
            temp_data2 = results['2011-2025']['temp_data']
            
            p_values, significant_mask = calculate_ttest_significance(temp_data1, temp_data2)

            title = 'Change in DJF Temperature\n(2011-2025) - (1996-2010)'
            output_file = './figure/nclimgrid_DJF_temp_change.png'
            
            create_difference_map(
                temperature_change,
                results['1996-2010']['lats'],
                results['1996-2010']['lons'],
                counties_gdf,
                states_gdf,
                title,
                output_file,
                us_mask=us_mask,
                p_values=p_values,            
                significant_mask=significant_mask
            )
    
if __name__ == "__main__":
    main()