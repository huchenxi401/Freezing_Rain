import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
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

    df = pd.read_excel(excel_file, usecols=['STATE', 'COUNTY', 'BEGIN_YEAR', 'BEGIN_MONTH', 'DURATION_HOURS'])

    df['STATE_CLEAN'] = df['STATE'].str.strip().str.upper()
    df['COUNTY_CLEAN'] = df['COUNTY'].apply(standardize_county_name)
    df['STATE_ABBREV'] = df['STATE_CLEAN'].map(STATE_ABBREV)

    df['STATE_COUNTY'] = df['STATE_ABBREV'] + '_' + df['COUNTY_CLEAN']

    df_clean = df.dropna(subset=['STATE_ABBREV', 'COUNTY_CLEAN', 'BEGIN_MONTH'])
    df_clean = df_clean[df_clean['DURATION_HOURS'] <= 144] 
    df_clean['TIME_PERIOD'] = df_clean['BEGIN_YEAR'].apply(
        lambda year: '1996-2010' if 1996 <= year <= 2010 else '2011-2025'
    )

    return df_clean

def month_to_sequence(month):

    month_order = {10: 0, 11: 1, 12: 2, 1: 3, 2: 4, 3: 5, 4: 6}
    return month_order.get(month, -1)

def sequence_to_month(seq):

    seq_to_month = {0: 10, 1: 11, 2: 12, 3: 1, 4: 2, 5: 3, 6: 4}
    return seq_to_month.get(seq, -1)

def calculate_month_difference(month1, month2):

    if month1 == -1 or month2 == -1:
        return np.nan
    
    seq1 = month_to_sequence(month1)
    seq2 = month_to_sequence(month2)
    
    if seq1 == -1 or seq2 == -1:
        return np.nan

    forward_diff = (seq2 - seq1) % 7
    backward_diff = (seq1 - seq2) % 7

    if forward_diff <= backward_diff:
        return forward_diff
    else:
        return -backward_diff

def calculate_peak_month_by_county(df, time_period=None):

    if time_period:
        df_filtered = df[df['TIME_PERIOD'] == time_period]
    else:
        df_filtered = df
    
    monthly_counts = df_filtered.groupby(['STATE_ABBREV', 'COUNTY_CLEAN', 'BEGIN_MONTH']).size().reset_index(name='EVENT_COUNT')

    peak_months = monthly_counts.loc[monthly_counts.groupby(['STATE_ABBREV', 'COUNTY_CLEAN'])['EVENT_COUNT'].idxmax()]
    peak_months = peak_months[['STATE_ABBREV', 'COUNTY_CLEAN', 'BEGIN_MONTH', 'EVENT_COUNT']]
    peak_months.columns = ['STATE_ABBREV', 'COUNTY_NAME', 'PEAK_MONTH', 'MAX_EVENTS']
    
    period_text = f"({time_period}) " if time_period else ""
    
    return peak_months

def calculate_peak_month_changes(df):

    period1_peaks = calculate_peak_month_by_county(df, '1996-2010')
    period2_peaks = calculate_peak_month_by_county(df, '2011-2025')

    all_counties_p1 = set(period1_peaks[['STATE_ABBREV', 'COUNTY_NAME']].apply(
        lambda x: f"{x['STATE_ABBREV']}_{x['COUNTY_NAME']}", axis=1))
    all_counties_p2 = set(period2_peaks[['STATE_ABBREV', 'COUNTY_NAME']].apply(
        lambda x: f"{x['STATE_ABBREV']}_{x['COUNTY_NAME']}", axis=1))

    all_counties = all_counties_p1.union(all_counties_p2)
    
    change_results = []
    
    for county_key in all_counties:
        state_abbrev, county_name = county_key.split('_', 1)

        p1_match = period1_peaks[(period1_peaks['STATE_ABBREV'] == state_abbrev) & 
                                (period1_peaks['COUNTY_NAME'] == county_name)]
        p2_match = period2_peaks[(period2_peaks['STATE_ABBREV'] == state_abbrev) & 
                                (period2_peaks['COUNTY_NAME'] == county_name)]

        p1_month = 12 if len(p1_match) == 0 else p1_match.iloc[0]['PEAK_MONTH']
        p2_month = 2 if len(p2_match) == 0 else p2_match.iloc[0]['PEAK_MONTH']

        month_change = calculate_month_difference(p1_month, p2_month)

        has_p1_data = len(p1_match) > 0
        has_p2_data = len(p2_match) > 0
        
        change_results.append({
            'STATE_ABBREV': state_abbrev,
            'COUNTY_NAME': county_name,
            'PERIOD1_MONTH': p1_month,
            'PERIOD2_MONTH': p2_month,
            'MONTH_CHANGE': month_change,
            'HAS_P1_DATA': has_p1_data,
            'HAS_P2_DATA': has_p2_data
        })
    
    change_df = pd.DataFrame(change_results)
    
    return change_df

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

def merge_data_with_geography(change_data, counties_gdf):
    merged_gdf = counties_gdf.merge(
        change_data,
        left_on=['STUSPS', 'COUNTY_CLEAN'],
        right_on=['STATE_ABBREV', 'COUNTY_NAME'],
        how='left'
    )
    
    merged_gdf['MONTH_CHANGE'] = merged_gdf['MONTH_CHANGE'].fillna(np.nan)
    
    return merged_gdf

def merge_alaska_data(change_data, counties_alaska_gdf):

    alaska_merged = counties_alaska_gdf.merge(
        change_data,
        left_on=['STUSPS', 'COUNTY_CLEAN'],
        right_on=['STATE_ABBREV', 'COUNTY_NAME'],
        how='left'
    )
    
    alaska_merged['MONTH_CHANGE'] = alaska_merged['MONTH_CHANGE'].fillna(np.nan)
    
    return alaska_merged

def create_month_change_map(merged_gdf, states_gdf, counties_alaska, states_alaska, alaska_data, output_dir="D:/py/freezing rain/freezing rain analysis/figure"):
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_size = (6, 4)
    
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    counties_with_data = merged_gdf[~merged_gdf['MONTH_CHANGE'].isna()]
    
    if len(counties_with_data) > 0:
        counties_with_data.plot(column='MONTH_CHANGE', 
                               ax=ax, 
                               cmap='BrBG',  
                               legend=True,
                               vmin=-3, 
                               vmax=3,
                               legend_kwds={'shrink': 0.8, 'label': 'Peak Month Change (months)',
                                          'orientation': 'horizontal'})
    
    counties_no_data = merged_gdf[merged_gdf['MONTH_CHANGE'].isna()]
    if len(counties_no_data) > 0:
        counties_no_data.plot(ax=ax, color='#f0f0f0')
    
    states_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
    merged_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.2)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(-125, -65)
    ax.set_ylim(25, 50)

    inset_ax = fig.add_axes([0.75, 0.35, 0.15, 0.15])
    
    if len(alaska_data) > 0:
        alaska_with_data = alaska_data[~alaska_data['MONTH_CHANGE'].isna()]
        alaska_no_data = alaska_data[alaska_data['MONTH_CHANGE'].isna()]
        
        if len(alaska_with_data) > 0:
            alaska_with_data.plot(column='MONTH_CHANGE',
                                 ax=inset_ax,
                                 cmap='BrBG',
                                 vmin=-3,
                                 vmax=3,
                                 legend=False)
        
        if len(alaska_no_data) > 0:
            alaska_no_data.plot(ax=inset_ax, color='#f0f0f0')
    
    states_alaska.boundary.plot(ax=inset_ax, color='black', linewidth=0.6)
    counties_alaska.boundary.plot(ax=inset_ax, color='gray', linewidth=0.2)
    
    inset_ax.set_title('Alaska', fontsize=8)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path / 'freezing_rain_peak_month_change_map.png', dpi=300, bbox_inches='tight')
    plt.close()
        
    return {
        'peak_month_change': output_path / 'freezing_rain_peak_month_change_map.png'
    }


def main():
    try:
        excel_file = "../../data/freezing_rain_events_county_llm.xlsx"
        output_dir = "./figure"
        
        df = load_and_process_data(excel_file)
        counties_gdf, states_gdf, counties_alaska, states_alaska = download_geographic_data()

        change_data = calculate_peak_month_changes(df)

        merged_gdf = merge_data_with_geography(change_data, counties_gdf)

        alaska_merged = merge_alaska_data(change_data, counties_alaska)

        map_files = create_month_change_map(merged_gdf, states_gdf, counties_alaska, states_alaska, alaska_merged, output_dir)
        
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()