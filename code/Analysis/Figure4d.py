import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pygrib
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def kelvin_to_celsius(temp_k):
    return temp_k - 273.15

def read_grib_data_with_years_by_month(file_path):
    try:
        grbs = pygrib.open(file_path)

        first_grb = grbs[1]
        lats, lons = first_grb.latlons()
        monthly_data = {2: {}}  
        
        grbs.rewind()
        total_messages = grbs.messages

        for i, grb in enumerate(grbs, 1):
            try:
                valid_date = grb['validityDate']
                valid_time = grb['validityTime']
                dt = datetime.strptime(f"{valid_date}{valid_time:04d}", "%Y%m%d%H%M")
                year = dt.year
                month = dt.month

                if month not in [2]:
                    continue

                temp_data = grb.values
                
                if year not in monthly_data[month]:
                    monthly_data[month][year] = []
                monthly_data[month][year].append(temp_data)
                
            except Exception as e:
                continue

        grbs.close()
        
        for month in monthly_data:
            for year in monthly_data[month]:
                monthly_data[month][year] = np.array(monthly_data[month][year])

        for month in [2]:
            years = sorted(monthly_data[month].keys())
    
        return monthly_data, lats, lons
        
    except Exception as e:
        return None, None, None

def load_emerging_hotspots(hotspot_file):
    try:
        hotspots_df = pd.read_excel(hotspot_file)
        return hotspots_df
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def calculate_annual_freezing_hours(yearly_data_2m, yearly_data_850hPa):
    years = sorted(yearly_data_2m.keys())
    first_year_data = yearly_data_2m[years[0]]
    spatial_shape = first_year_data.shape[1:]  
    annual_freezing_hours = np.zeros((len(years), spatial_shape[0], spatial_shape[1]))
    
    for i, year in enumerate(years):
        temp_2m_celsius = kelvin_to_celsius(yearly_data_2m[year])
        temp_850_celsius = kelvin_to_celsius(yearly_data_850hPa[year])
        
        freezing_rain_mask = (temp_2m_celsius < -1) & (temp_850_celsius > 1)
        annual_freezing_hours[i] = np.sum(freezing_rain_mask, axis=0)
        
    return annual_freezing_hours, years



def calculate_period_averages_and_significance(yearly_data_2m_p1, yearly_data_850_p1, 
                                             yearly_data_2m_p2, yearly_data_850_p2):
    annual_hours_p1, years_p1 = calculate_annual_freezing_hours(
        yearly_data_2m_p1, yearly_data_850_p1
    )
    period1_mean = np.mean(annual_hours_p1, axis=0)

    annual_hours_p2, years_p2 = calculate_annual_freezing_hours(
        yearly_data_2m_p2, yearly_data_850_p2
    )
    period2_mean = np.mean(annual_hours_p2, axis=0)

    difference = period2_mean - period1_mean

    spatial_shape = annual_hours_p1.shape[1:]
    t_statistics = np.zeros(spatial_shape)
    p_values = np.ones(spatial_shape)
    
    for i in range(spatial_shape[0]):
        for j in range(spatial_shape[1]):
            series1 = annual_hours_p1[:, i, j] 
            series2 = annual_hours_p2[:, i, j] 

            if np.all(np.isfinite(series1)) and np.all(np.isfinite(series2)):
                t_stat, p_val = stats.ttest_ind(series2, series1)
                t_statistics[i, j] = t_stat
                p_values[i, j] = p_val


    return period1_mean, period2_mean, difference, t_statistics, p_values

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



def create_fraction_map(diff_data, p_values, lats, lons, counties_gdf, states_gdf, 
                         title, output_file, us_mask=None, significance_level=0.05, hotspots_df=None,):

    if us_mask is not None:
        diff_data_masked = np.ma.masked_where(~us_mask, diff_data)
        p_values_masked = np.ma.masked_where(~us_mask, p_values)
        valid_data = diff_data[us_mask]
    else:
        diff_data_masked = diff_data
        p_values_masked = p_values
        valid_data = diff_data
    
    fig_size = (6, 4)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    
    max_abs_diff = np.max(np.abs(valid_data))
    vmin = -80
    vmax = 80

    levels = np.linspace(vmin, vmax, 21)
    cs = ax.contourf(lons, lats, diff_data_masked, 
                     levels=levels, 
                     cmap='coolwarm',  
                     vmin=vmin, 
                     vmax=vmax,
                     extend='both')

    if us_mask is not None:
        significant_mask = (p_values < significance_level) & us_mask
    else:
        significant_mask = p_values < significance_level
    
    significant_points = np.sum(significant_mask)
    
    if significant_points > 0:
        step = 2  
            
        sig_indices = np.where(significant_mask)
        
        for idx in range(0, len(sig_indices[0]), step):
                i = sig_indices[0][idx]
                j = sig_indices[1][idx]
                ax.plot(lons[i, j], lats[i, j], 'k.', markersize=2, markerfacecolor='black', 
                markeredgewidth=0)

    cbar = plt.colorbar(cs, ax=ax, shrink=0.8, orientation='horizontal', 
                       pad=0.15, aspect=30)
    cbar.set_label('Percentage Change in February FRFH (%)', fontsize=12)
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

def main():

    data_dir = "../../data/"
    files = {
        'period1_2m': '2m_tem_DJF_1996_2010.grib',
        'period2_2m': '2m_tem_DJF_2011_2025.grib',
        'period1_850': '850hPa_tem_DJF_1996_2010.grib',
        'period2_850': '850hPa_tem_DJF_2011_2025.grib'
    }
    
    missing_files = []
    for period in files:
        file_path = os.path.join(data_dir, files[period])
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        for f in missing_files:
            print(f"  {f}")
        return

    
    counties_gdf, states_gdf = download_boundary_data()
    hotspot_file = "../../data/february_emerging_hotspots.xlsx"
    hotspots_df = load_emerging_hotspots(hotspot_file)

    file_path1_2m = os.path.join(data_dir, files['period1_2m'])
    monthly_data_2m_p1, lats, lons = read_grib_data_with_years_by_month(file_path1_2m)

    file_path1_850 = os.path.join(data_dir, files['period1_850'])
    monthly_data_850_p1, _, _ = read_grib_data_with_years_by_month(file_path1_850)

    file_path2_2m = os.path.join(data_dir, files['period2_2m'])
    monthly_data_2m_p2, _, _ = read_grib_data_with_years_by_month(file_path2_2m)

    file_path2_850 = os.path.join(data_dir, files['period2_850'])
    monthly_data_850_p2, _, _ = read_grib_data_with_years_by_month(file_path2_850)

    us_mask = None
    if states_gdf is not None:
        us_mask = create_us_mask(lats, lons, states_gdf)

    month_names = {2: 'February'}
    month_abbr = {2: 'Feb'}
    
    for month in [2]:

        yearly_data_2m_p1_month = monthly_data_2m_p1[month]
        yearly_data_850_p1_month = monthly_data_850_p1[month]
        yearly_data_2m_p2_month = monthly_data_2m_p2[month]
        yearly_data_850_p2_month = monthly_data_850_p2[month]
        
        all_yearly_data_2m = {}
        all_yearly_data_850 = {}
        all_yearly_data_2m.update(yearly_data_2m_p1_month)
        all_yearly_data_2m.update(yearly_data_2m_p2_month)
        all_yearly_data_850.update(yearly_data_850_p1_month)
        all_yearly_data_850.update(yearly_data_850_p2_month)
        
        all_years = sorted(all_yearly_data_2m.keys())


        period1_mean, period2_mean, difference, t_statistics, p_values_diff = calculate_period_averages_and_significance(
            yearly_data_2m_p1_month, yearly_data_850_p1_month,
            yearly_data_2m_p2_month, yearly_data_850_p2_month
        )
        
        

        title = f'Change in Freezing Rain Favorable Hours - {month_names[month]}\n(2m T<-1°C & 850hPa T>1°C)\n(2011-2025) - (1996-2010) Average'
        output_file4 = f'./figure4d.png'
        fraction = difference/(period1_mean+period2_mean)*200
        create_fraction_map(
            fraction,
            p_values_diff,  
            lats,
            lons,
            counties_gdf,
            states_gdf,
            title,
            output_file4,
            us_mask=us_mask,
            significance_level=0.05,
            hotspots_df=hotspots_df,
        )

    

if __name__ == "__main__":
    main()