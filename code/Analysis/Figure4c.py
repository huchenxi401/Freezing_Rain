import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def read_era5_humidity_data(file_path, start_year, end_year, filter_month=None):
    try:
        ds = xr.open_dataset(file_path)

        lats = ds['latitude'].values
        lons = ds['longitude'].values
        pressure_levels = ds['pressure_level'].values

        time_values = pd.to_datetime(ds['valid_time'].values, unit='s')
        month_mask = []
        for t in time_values:
            year = t.year
            month = t.month

            is_in_range = start_year <= year <= end_year

            if filter_month is not None:
                is_month_match = (month == filter_month)
            else:
                is_month_match = True
            
            month_mask.append(is_in_range and is_month_match)
        
        month_mask = np.array(month_mask)

        hum_data = ds['q'].values[month_mask, :, :, :]*1000
        time_stamps = time_values[month_mask]

        level_mask = (pressure_levels >= 500) & (pressure_levels <= 850)
        selected_levels = pressure_levels[level_mask]

        hum_data_subset = hum_data[:, level_mask, :, :]   

        hum_data_avg = np.nanmean(hum_data_subset, axis=1) 
 
        hum_data_avg = hum_data_avg 

        ds.close()

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        return hum_data_avg, lat_grid, lon_grid, time_stamps
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None


def calculate_period_mean(hum_data):
    period_mean = np.nanmean(hum_data, axis=0)
    return period_mean


def calculate_ttest_significance(hum_data1, hum_data2):
    from scipy import stats
    p_values = np.ones(hum_data1.shape[1:])

    for i in range(hum_data1.shape[1]):
        for j in range(hum_data1.shape[2]):
            series1 = hum_data1[:, i, j]
            series2 = hum_data2[:, i, j]
   
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


def download_boundary_data():
    try:
        counties_gdf_full = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip")

        states_gdf_full = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_20m.zip")
 
        exclude_states = ['02', '15', '60', '66', '69', '72', '78'] 
        states_gdf = states_gdf_full[~states_gdf_full['STATEFP'].isin(exclude_states)]
        counties_gdf = counties_gdf_full[~counties_gdf_full['STATEFP'].isin(exclude_states)]

        return counties_gdf, states_gdf
        
    except Exception as e:
        print(f"  错误: 下载边界数据失败 - {e}")
        return None, None


def create_us_mask(lats, lons, states_gdf):

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
                point = Point(lons[i, j], lats[i, j])
                if us_boundary.contains(point) or us_boundary.touches(point):
                    mask[i, j] = True
                    points_in_us += 1

        return mask
        
    except Exception as e:
        return np.ones(lats.shape, dtype=bool)


def create_difference_map(difference, lats, lons, counties_gdf, states_gdf, 
                          title, output_file, us_mask=None, hotspots_df=None,
                          p_values=None, significant_mask=None):

    plot_difference = difference.copy()
    if us_mask is not None:
        plot_difference[~us_mask] = np.nan

    plot_significant_mask = None
    if significant_mask is not None and us_mask is not None:
        plot_significant_mask = significant_mask & us_mask
    elif significant_mask is not None:
        plot_significant_mask = significant_mask
    
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': None})

    vmin = -0.8
    vmax = 0.8

    cs = ax.contourf(lons, lats, plot_difference, 
                     levels=np.linspace(vmin, vmax, 21),
                     cmap='BrBG', extend='both')
    
    if plot_significant_mask is not None:

        sig_count = np.sum(plot_significant_mask)

        if sig_count > 0:
            step = 2 
            
            sig_indices = np.where(plot_significant_mask)

            for idx in range(0, len(sig_indices[0]), step):
                i = sig_indices[0][idx]
                j = sig_indices[1][idx]
                ax.plot(lons[i, j], lats[i, j], 'k.', markersize=2, markerfacecolor='black', 
                        markeredgewidth=0)

    cbar = plt.colorbar(cs, ax=ax, shrink=0.8, orientation='horizontal', 
                       pad=0.15, aspect=30)
    cbar.set_label('Difference in February Specific Humidity (g/kg)', fontsize=12)
    if counties_gdf is not None:
        counties_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.2)
    if states_gdf is not None:
        states_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)

    if hotspots_df is not None and len(hotspots_df) > 0:
        hotspots_df['COUNTY_NAME_TITLE'] = hotspots_df['COUNTY_NAME'].str.title()
        
        hotspot_counties = counties_gdf.merge(
            hotspots_df,
            left_on=['STUSPS', 'NAME'],
            right_on=['STATE_ABBREV', 'COUNTY_NAME_TITLE'],
            how='inner'
        )

        if len(hotspot_counties) > 0:
            hotspot_counties.boundary.plot(ax=ax, color='gold', linewidth=0.3)

    ax.set_xlim(-125, -65)
    ax.set_ylim(25, 50)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"  差异地图已保存: {output_file}")
    plt.close()


def load_emerging_hotspots(hotspot_file):
    try:
        hotspots_df = pd.read_excel(hotspot_file)
        return hotspots_df
    except FileNotFoundError:
        return None
    except Exception as e:
        return None


def main():
    data_file = "../..//Q_monthly_DJF_1996-2025.nc"

    periods = {
        'period1': {'name': '1996-2010', 'start': 1996, 'end': 2010},
        'period2': {'name': '2011-2025', 'start': 2011, 'end': 2025}
    }
    
    if not os.path.exists(data_file):
        return
    
    counties_gdf, states_gdf = download_boundary_data()

    hotspot_file = "../../data/february_emerging_hotspots.xlsx"
    hotspots_df = load_emerging_hotspots(hotspot_file)

    results = {}
    us_mask = None
    
    for period_key in ['period1', 'period2']:
        period_info = periods[period_key]
        period_name = period_info['name']

        hum_data, lats, lons, time_stamps = read_era5_humidity_data(
            data_file,
            period_info['start'],
            period_info['end'],
            filter_month=2 
        )
        
        if hum_data is None:
            continue

        if us_mask is None and states_gdf is not None:
            us_mask = create_us_mask(lats, lons, states_gdf)

        period_mean = calculate_period_mean(hum_data)

        results[period_name] = {
            'period_mean': period_mean,
            'hum_data': hum_data,
            'lats': lats,
            'lons': lons
        }

    if len(results) == 2:
        period1_hum = results['1996-2010']['period_mean']
        period2_hum = results['2011-2025']['period_mean']
        
        humidity_change = period2_hum - period1_hum

        hum_data1 = results['1996-2010']['hum_data']
        hum_data2 = results['2011-2025']['hum_data']
        
        p_values, significant_mask = calculate_ttest_significance(hum_data1, hum_data2)

        title = 'Change in February Mid-level Humidity (850-500 hPa)\n(2011-2025) - (1996-2010)'
        output_file = './Figure4c'
        
        create_difference_map(
            humidity_change,
            results['1996-2010']['lats'],
            results['1996-2010']['lons'],
            counties_gdf,
            states_gdf,
            title,
            output_file,
            us_mask=us_mask,
            hotspots_df=hotspots_df,
            p_values=p_values,
            significant_mask=significant_mask
        )


if __name__ == "__main__":
    main()