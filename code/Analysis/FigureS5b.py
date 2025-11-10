import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pygrib
from datetime import datetime
import warnings
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
    suffixes_to_remove = [' COUNTY', ' PARISH', ' BOROUGH', ' CENSUS AREA', ' CITY', ' CITY COUNTY', ' ', ' CITY AND BOROUGH']
    for suffix in suffixes_to_remove:
        if county_name.endswith(suffix):
            county_name = county_name[:-len(suffix)].strip()
            break
    
    return county_name

def kelvin_to_celsius(temp_k):
    return temp_k - 273.15

def read_grib_data(file_path, filter_month=None):
    try:
        grbs = pygrib.open(file_path)
        first_grb = grbs[1]
        lats, lons = first_grb.latlons()
        temperature_data = []
        time_stamps = []
        
        grbs.rewind()
        total_messages = grbs.messages

        filtered_count = 0
        for i, grb in enumerate(grbs, 1):
            try:
                valid_date = grb['validityDate']
                valid_time = grb['validityTime']
                dt = datetime.strptime(f"{valid_date}{valid_time:04d}", "%Y%m%d%H%M")
                
                if filter_month is not None and dt.month != filter_month:
                    continue
                
                time_stamps.append(dt)
            except:
                continue

            temp_data = grb.values
            temperature_data.append(temp_data)
            filtered_count += 1

        grbs.close()
        
        temperature_data = np.array(temperature_data) 
        
        return temperature_data, lats, lons, time_stamps
        
    except Exception as e:
        return None, None, None, None

def calculate_Feb_percentile(temp_data):
    temp_celsius = kelvin_to_celsius(temp_data)
    
    Feb = np.nanmean(temp_celsius, axis=0)
    
    return Feb

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



def create_difference_map(diff_data, lats, lons, counties_gdf, states_gdf, hotspots_df,
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
            step = 3 
            
            sig_indices = np.where(plot_significant_mask)

            for idx in range(0, len(sig_indices[0]), step):
                i = sig_indices[0][idx]
                j = sig_indices[1][idx]
                ax.plot(lons[i, j], lats[i, j], 'k.', markersize=2, markerfacecolor='black', 
                markeredgewidth=0)

    cbar = plt.colorbar(cs, ax=ax, shrink=0.8, orientation='horizontal', 
                       pad=0.15, aspect=30)
    cbar.set_label('February Temperature Difference (°C)', fontsize=12)

    if states_gdf is not None:
        states_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)

    if counties_gdf is not None:
        counties_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.2)
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

def load_emerging_hotspots(hotspot_file):

    try:
        hotspots_df = pd.read_excel(hotspot_file)
        return hotspots_df
    except FileNotFoundError:
        return None
    except Exception as e:
        return None


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
        for f in missing_files:
            print(f"  {f}")
        return

    counties_gdf, states_gdf = download_boundary_data()
    if counties_gdf is None or states_gdf is None:
        print("warning")
    hotspot_file = "../../data/february_emerging_hotspots.xlsx"
    hotspots_df = load_emerging_hotspots(hotspot_file)
    periods = ['1996-2010', '2011-2025']
    results = {}
    us_mask = None 
    
    for i, period_key in enumerate(['period1', 'period2']):
        period_name = periods[i]
        temp_file = os.path.join(data_dir, files[period_key])
        temp_data, lats, lons, time_stamps = read_grib_data(temp_file, filter_month=2)
        
        if temp_data is None:
            continue

        if us_mask is None and states_gdf is not None:
            us_mask = create_us_mask(lats, lons, states_gdf)
        
        Feb = calculate_Feb_percentile(temp_data)

        results[period_name] = {
            'Feb': Feb,
            'temp_data': temp_data,
            'lats': lats,
            'lons': lons
        }
           
    if len(results) > 0:
        all_temps = []
        for period in results:
            if us_mask is not None:
                valid_temps = results[period]['Feb'][us_mask]
                all_temps.extend(valid_temps)
            else:
                all_temps.extend(results[period]['Feb'].flatten())
        
        global_vmin = np.min(all_temps)
        global_vmax = np.max(all_temps)
        
        if len(results) == 2:
            period1_temp = results['1996-2010']['Feb']
            period2_temp = results['2011-2025']['Feb']
            
            temperature_change = period2_temp - period1_temp

            temp_data1 = results['1996-2010']['temp_data']  
            temp_data2 = results['2011-2025']['temp_data']
            
            p_values, significant_mask = calculate_ttest_significance(temp_data1, temp_data2)
            title = 'Change in Temperature (February)\n(2011-2025) - (1996-2010)'
            output_file = './figure/2mtemperature_change_ERA5_february.png'           
            create_difference_map(
                temperature_change,
                results['1996-2010']['lats'],
                results['1996-2010']['lons'],
                counties_gdf,
                states_gdf,  hotspots_df,
                title,
                output_file,
                us_mask=us_mask,
                p_values=p_values,          
                significant_mask=significant_mask
            )

 
if __name__ == "__main__":
    main()