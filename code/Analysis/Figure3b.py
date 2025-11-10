import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
from scipy import stats
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

def load_and_process_data(excel_file="../../data/freezing_rain_events_county_llm.xlsx"):
    df = pd.read_excel(excel_file, usecols=['STATE', 'COUNTY', 'BEGIN_YEAR', 'BEGIN_MONTH', 'DURATION_HOURS', 'EVENT_ID', 'ICE_THICKNESS_INCHES'])
    
    df['STATE_CLEAN'] = df['STATE'].str.strip().str.upper()
    df['COUNTY_CLEAN'] = df['COUNTY'].apply(standardize_county_name)
    df['STATE_ABBREV'] = df['STATE_CLEAN'].map(STATE_ABBREV)
    
    df['STATE_COUNTY'] = df['STATE_ABBREV'] + '_' + df['COUNTY_CLEAN']
    
    df_clean = df.dropna(subset=['STATE_ABBREV', 'COUNTY_CLEAN', 'BEGIN_MONTH', 'ICE_THICKNESS_INCHES'])
    df_clean = df_clean[df_clean['DURATION_HOURS'] < 144]

    df_clean['TIME_PERIOD'] = df_clean['BEGIN_YEAR'].apply(
        lambda year: '1996-2010' if 1996 <= year <= 2010 else '2011-2025'
    )
    
    month_dist = df_clean['BEGIN_MONTH'].value_counts().sort_index()
    return df_clean

def calculate_county_statistics_by_year(df, time_period=None, month=None, threshold=0.25):
    filter_desc = []
    df_filtered = df.copy()
    
    if time_period:
        df_filtered = df_filtered[df_filtered['TIME_PERIOD'] == time_period]
        filter_desc.append(time_period)
    
    if month is not None:
        df_filtered = df_filtered[df_filtered['BEGIN_MONTH'] == month]
        filter_desc.append(f"{month}")
    
    df_filtered = df_filtered[df_filtered['ICE_THICKNESS_INCHES'] > threshold]
    filter_desc.append(f">0.25inch")
    
    filter_text = " - ".join(filter_desc)

    yearly_stats = df_filtered.groupby(['STATE_ABBREV', 'COUNTY_CLEAN', 'BEGIN_YEAR']).agg({
        'EVENT_ID': 'count', 
    }).reset_index()

    yearly_stats.columns = ['STATE_ABBREV', 'COUNTY_NAME', 'YEAR', 'EVENT_COUNT']
    
    return yearly_stats

def calculate_county_statistics(df, time_period=None, month=None, threshold=0.25):
    filter_desc = []
    df_filtered = df.copy()
    
    if time_period:
        df_filtered = df_filtered[df_filtered['TIME_PERIOD'] == time_period]
        filter_desc.append(time_period)
    
    if month is not None:
        df_filtered = df_filtered[df_filtered['BEGIN_MONTH'] == month]
        filter_desc.append(f"{month}")
    
    filter_text = " - ".join(filter_desc) if filter_desc else ""
    
    df_above_threshold = df_filtered[df_filtered['ICE_THICKNESS_INCHES'] > threshold]
    
    county_stats = df_above_threshold.groupby(['STATE_ABBREV', 'COUNTY_CLEAN']).size().reset_index()
    county_stats.columns = ['STATE_ABBREV', 'COUNTY_NAME', 'EVENT_COUNT']
    county_stats['EVENT_COUNT'] = county_stats['EVENT_COUNT'] / 15.0

    return county_stats

def perform_ttest_by_county(yearly_data_dict, month=None):
    month_names = {2: 'February'}
    month_text = f" ({month_names[month]})" if month is not None else ""
    
    
    period1_data = yearly_data_dict['1996-2010']
    period2_data = yearly_data_dict['2011-2025']
    
    all_counties_p1 = set(period1_data[['STATE_ABBREV', 'COUNTY_NAME']].apply(
        lambda x: f"{x['STATE_ABBREV']}_{x['COUNTY_NAME']}", axis=1))
    all_counties_p2 = set(period2_data[['STATE_ABBREV', 'COUNTY_NAME']].apply(
        lambda x: f"{x['STATE_ABBREV']}_{x['COUNTY_NAME']}", axis=1))

    common_counties = all_counties_p1.union(all_counties_p2)
    
    ttest_results = []
    
    for county_key in common_counties:
        state_abbrev, county_name = county_key.split('_', 1)

        mask1 = (period1_data['STATE_ABBREV'] == state_abbrev) & (period1_data['COUNTY_NAME'] == county_name)
        mask2 = (period2_data['STATE_ABBREV'] == state_abbrev) & (period2_data['COUNTY_NAME'] == county_name)
        
        data1 = period1_data[mask1]['EVENT_COUNT'].values
        data2 = period2_data[mask2]['EVENT_COUNT'].values

        if len(data1) == 0:
            data1 = np.zeros(15)  
        else:
            years1 = period1_data[mask1]['YEAR'].values
            full_data1 = np.zeros(15)
            for i, year in enumerate(range(1996, 2011)):
                if year in years1:
                    idx = np.where(years1 == year)[0][0]
                    full_data1[i] = data1[idx]
            data1 = full_data1
            
        if len(data2) == 0:
            data2 = np.zeros(15) 
        else:
            years2 = period2_data[mask2]['YEAR'].values
            full_data2 = np.zeros(15)
            for i, year in enumerate(range(2011, 2026)):
                if year in years2:
                    idx = np.where(years2 == year)[0][0]
                    full_data2[i] = data2[idx]
            data2 = full_data2
        
        try:
            #Welch's t-test
            t_stat, p_value = stats.ttest_ind(data2, data1, equal_var=False)
            mean_diff = np.mean(data2) - np.mean(data1)
            
            ttest_results.append({
                'STATE_ABBREV': state_abbrev,
                'COUNTY_NAME': county_name,
                't_statistic': t_stat,
                'p_value': p_value,
                'mean_difference': mean_diff,
                'significant': p_value < 0.05
            })
        except:
            ttest_results.append({
                'STATE_ABBREV': state_abbrev,
                'COUNTY_NAME': county_name,
                't_statistic': np.nan,
                'p_value': 1.0,
                'mean_difference': np.mean(data2) - np.mean(data1),
                'significant': False
            })
    
    ttest_df = pd.DataFrame(ttest_results)
    
    significant_counties = ttest_df[ttest_df['significant']].shape[0]
    total_counties = len(ttest_df)
    return ttest_df

def download_geographic_data():
    try:
        counties_gdf_full = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip")
        
        states_gdf_full = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_20m.zip")
        
        exclude_states = ['15', '60', '66', '69', '72', '78'] 
        
        counties_gdf_main = counties_gdf_full[~counties_gdf_full['STATEFP'].isin(['02'] + exclude_states)]
        states_gdf_main = states_gdf_full[~states_gdf_full['STATEFP'].isin(['02'] + exclude_states)]

        counties_gdf_alaska = counties_gdf_full[counties_gdf_full['STATEFP'] == '02']
        states_gdf_alaska = states_gdf_full[states_gdf_full['STATEFP'] == '02']
        
        from shapely.geometry import box

        alaska_bbox = box(-180, 54, -130, 72) 

        counties_gdf_alaska = counties_gdf_alaska.clip(alaska_bbox)
        states_gdf_alaska = states_gdf_alaska.clip(alaska_bbox)

        counties_gdf_alaska = counties_gdf_alaska[~counties_gdf_alaska.geometry.is_empty]
        states_gdf_alaska = states_gdf_alaska[~states_gdf_alaska.geometry.is_empty]

        counties_gdf_main['COUNTY_CLEAN'] = counties_gdf_main['NAME'].apply(standardize_county_name)
        counties_gdf_alaska['COUNTY_CLEAN'] = counties_gdf_alaska['NAME'].apply(standardize_county_name)

        return counties_gdf_main, states_gdf_main, counties_gdf_alaska, states_gdf_alaska
    
    except Exception as e:
        return None, None, None, None

def merge_data_with_geography(county_stats, counties_gdf, ttest_results=None):
    merged_gdf = counties_gdf.merge(
        county_stats,
        left_on=['STUSPS', 'COUNTY_CLEAN'],
        right_on=['STATE_ABBREV', 'COUNTY_NAME'],
        how='left'
    )
    
    merged_gdf['EVENT_COUNT'] = merged_gdf['EVENT_COUNT'].fillna(0)

    if ttest_results is not None:
        merged_gdf = merged_gdf.merge(
            ttest_results[['STATE_ABBREV', 'COUNTY_NAME', 'p_value', 'significant']],
            left_on=['STUSPS', 'COUNTY_CLEAN'],
            right_on=['STATE_ABBREV', 'COUNTY_NAME'],
            how='left'
        )
        merged_gdf['significant'] = merged_gdf['significant'].fillna(False)
        merged_gdf['p_value'] = merged_gdf['p_value'].fillna(1.0)
    
    return merged_gdf

def merge_alaska_data(county_stats, counties_alaska_gdf, ttest_results=None):
    alaska_merged = counties_alaska_gdf.merge(
        county_stats,
        left_on=['STUSPS', 'COUNTY_CLEAN'],
        right_on=['STATE_ABBREV', 'COUNTY_NAME'],
        how='left'
    )
    
    alaska_merged['EVENT_COUNT'] = alaska_merged['EVENT_COUNT'].fillna(0)

    if ttest_results is not None:
        alaska_merged = alaska_merged.merge(
            ttest_results[['STATE_ABBREV', 'COUNTY_NAME', 'p_value', 'significant']],
            left_on=['STUSPS', 'COUNTY_CLEAN'],
            right_on=['STATE_ABBREV', 'COUNTY_NAME'],
            how='left'
        )
        alaska_merged['significant'] = alaska_merged['significant'].fillna(False)
        alaska_merged['p_value'] = alaska_merged['p_value'].fillna(1.0)
    
    return alaska_merged

def create_maps_diff(merged_gdf, states_gdf, counties_alaska, states_alaska, alaska_data, 
                     ttest_results, month=None, output_dir="./figure"):
    month_names = {2: 'February'}
    month_suffix = f"_{month}month" if month is not None else ""
    month_text = month_names[month] if month is not None else "All months"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_size = (6, 4)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    
    counties_with_events = merged_gdf
    
    if len(counties_with_events) > 0:
        counties_with_events.plot(column='EVENT_COUNT', 
                                   ax=ax, 
                                   cmap='RdBu_r', 
                                   legend=True,
                                   vmin=-0.3,
                                   vmax=0.3, 
                                   legend_kwds={'shrink': 0.8, 
                                               'label': 'Difference in February Events Yr⁻¹ (Ice > 0.25 inches)',
                                               'orientation': 'horizontal'})
    
    if 'significant' in merged_gdf.columns:
        significant_counties = merged_gdf[merged_gdf['significant'] == True]
        if len(significant_counties) > 0:
            significant_counties_copy = significant_counties.copy()
            significant_counties_copy['centroid'] = significant_counties_copy.geometry.centroid
            
            centroids = significant_counties_copy['centroid']
            x_coords = [point.x for point in centroids]
            y_coords = [point.y for point in centroids]

            ax.plot(x_coords, y_coords, 'k.', markersize=2, markerfacecolor='black', 
                markeredgewidth=0)
    
    states_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
    merged_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.2)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(-125, -65)
    ax.set_ylim(25, 50)

    inset_ax = fig.add_axes([0.75, 0.35, 0.15, 0.15])

    if len(alaska_data) > 0:
        alaska_data.plot(column='EVENT_COUNT',
                        ax=inset_ax,
                        cmap='RdBu_r',
                        vmin=-0.3,
                        vmax=0.3, 
                        legend=False)

        if 'significant' in alaska_data.columns:
            alaska_significant = alaska_data[alaska_data['significant'] == True]
            if len(alaska_significant) > 0:
                alaska_significant_copy = alaska_significant.copy()
                alaska_significant_copy['centroid'] = alaska_significant_copy.geometry.centroid
                
                centroids_ak = alaska_significant_copy['centroid']
                x_coords_ak = [point.x for point in centroids_ak]
                y_coords_ak = [point.y for point in centroids_ak]
                
                inset_ax.plot(x_coords_ak, y_coords_ak, 'k.', markersize=2, markerfacecolor='black', 
                markeredgewidth=0)

    states_alaska.boundary.plot(ax=inset_ax, color='black', linewidth=0.6)
    counties_alaska.boundary.plot(ax=inset_ax, color='gray', linewidth=0.2)
    
    inset_ax.set_title('Alaska', fontsize=8)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    
    plt.tight_layout()
    output_file = output_path / f'freezing_rain_event_count_ice_thickness_larger0.25_map_diff{month_suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'event_count_diff': output_file
    }


def main():
    try:
        excel_file = "../../data/freezing_rain_events_county_llm.xlsx"
        output_dir = "./figure"

        df = load_and_process_data(excel_file)
        counties_gdf, states_gdf, counties_alaska, states_alaska = download_geographic_data()
 
        months = [2]
        month_names = {2: 'February'}
        time_periods = ['1996-2010', '2011-2025']
        
        all_map_files = {}
        all_ttest_results = {}

        for month in months:

            yearly_data_dict = {}
            for period in time_periods:
                yearly_stats = calculate_county_statistics_by_year(df, period, month, threshold=0.25)
                yearly_data_dict[period] = yearly_stats

            ttest_results = perform_ttest_by_county(yearly_data_dict, month)
            all_ttest_results[month] = ttest_results

            period1_stats = calculate_county_statistics(df[df['TIME_PERIOD'] == '1996-2010'], month=month, threshold=0.25)
            period2_stats = calculate_county_statistics(df[df['TIME_PERIOD'] == '2011-2025'], month=month, threshold=0.25)

            all_counties = set()
            for stats in [period1_stats, period2_stats]:
                for _, row in stats.iterrows():
                    all_counties.add((row['STATE_ABBREV'], row['COUNTY_NAME']))
            
            diff_stats = []
            for state_abbrev, county_name in all_counties:
                p1_count = 0
                p2_count = 0
                
                p1_match = period1_stats[(period1_stats['STATE_ABBREV'] == state_abbrev) & 
                                       (period1_stats['COUNTY_NAME'] == county_name)]
                if len(p1_match) > 0:
                    p1_count = p1_match.iloc[0]['EVENT_COUNT']
                
                p2_match = period2_stats[(period2_stats['STATE_ABBREV'] == state_abbrev) & 
                                       (period2_stats['COUNTY_NAME'] == county_name)]
                if len(p2_match) > 0:
                    p2_count = p2_match.iloc[0]['EVENT_COUNT']
                
                diff_stats.append({
                    'STATE_ABBREV': state_abbrev,
                    'COUNTY_NAME': county_name,
                    'EVENT_COUNT': p2_count - p1_count
                })
            
            diff_stats_df = pd.DataFrame(diff_stats)
            
            diff_merged = merge_data_with_geography(diff_stats_df, counties_gdf, ttest_results)
            diff_alaska = merge_alaska_data(diff_stats_df, counties_alaska, ttest_results)

            diff_map_files = create_maps_diff(diff_merged, states_gdf, counties_alaska, states_alaska, 
                                             diff_alaska, ttest_results, month, output_dir)
            
            all_map_files[month] = diff_map_files

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()