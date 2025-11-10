import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
from scipy import stats
from shapely.geometry import Point
import shapely.ops
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
    df = pd.read_excel(excel_file, usecols=['STATE', 'COUNTY', 'BEGIN_YEAR', 'BEGIN_MONTH', 'DURATION_HOURS', 'EVENT_ID'])

    df['STATE_CLEAN'] = df['STATE'].str.strip().str.upper()
    df['COUNTY_CLEAN'] = df['COUNTY'].apply(standardize_county_name)
    df['STATE_ABBREV'] = df['STATE_CLEAN'].map(STATE_ABBREV)

    df['STATE_COUNTY'] = df['STATE_ABBREV'] + '_' + df['COUNTY_CLEAN']

    df_clean = df.dropna(subset=['STATE_ABBREV', 'COUNTY_CLEAN', 'BEGIN_YEAR', 'BEGIN_MONTH'])
    df_clean = df_clean[df_clean['DURATION_HOURS'] < 144]
    
    df_clean = df_clean[(df_clean['BEGIN_YEAR'] >= 1996) & (df_clean['BEGIN_YEAR'] <= 2025)]

    month_dist = df_clean['BEGIN_MONTH'].value_counts().sort_index()
    
    return df_clean

def calculate_county_annual_trends(df, month=None):
    month_name = {12: 'December', 1: 'January', 2: 'February'}
    
    if month is not None:
        df_filtered = df[df['BEGIN_MONTH'] == month]
    else:
        df_filtered = df
    
    annual_counts = df_filtered.groupby(['STATE_ABBREV', 'COUNTY_CLEAN', 'BEGIN_YEAR']).agg({
        'EVENT_ID': 'count'
    }).reset_index()
    annual_counts.columns = ['STATE_ABBREV', 'COUNTY_NAME', 'YEAR', 'EVENT_COUNT']

    years = range(1996, 2026)

    all_counties = annual_counts[['STATE_ABBREV', 'COUNTY_NAME']].drop_duplicates()
    
    trend_results = []
    
    for _, county_row in all_counties.iterrows():
        state_abbrev = county_row['STATE_ABBREV']
        county_name = county_row['COUNTY_NAME']

        county_data = annual_counts[
            (annual_counts['STATE_ABBREV'] == state_abbrev) & 
            (annual_counts['COUNTY_NAME'] == county_name)
        ]

        full_series = []
        for year in years:
            year_data = county_data[county_data['YEAR'] == year]
            if len(year_data) > 0:
                full_series.append(year_data.iloc[0]['EVENT_COUNT'])
            else:
                full_series.append(0)

        x = np.array(list(years))
        y = np.array(full_series)

        if np.sum(y) > 0:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                trend_slope = slope
                trend_significance = p_value < 0.05
                
                trend_results.append({
                    'STATE_ABBREV': state_abbrev,
                    'COUNTY_NAME': county_name,
                    'TREND_SLOPE': trend_slope,
                    'P_VALUE': p_value,
                    'R_SQUARED': r_value**2,
                    'SIGNIFICANT': trend_significance,
                    'TOTAL_EVENTS': np.sum(y),
                    'MEAN_ANNUAL_EVENTS': np.mean(y)
                })
            except:
                trend_results.append({
                    'STATE_ABBREV': state_abbrev,
                    'COUNTY_NAME': county_name,
                    'TREND_SLOPE': 0,
                    'P_VALUE': 1.0,
                    'R_SQUARED': 0,
                    'SIGNIFICANT': False,
                    'TOTAL_EVENTS': np.sum(y),
                    'MEAN_ANNUAL_EVENTS': np.mean(y)
                })
    
    trend_df = pd.DataFrame(trend_results)
    
    return trend_df

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

def merge_data_with_geography(trend_data, counties_gdf):
    merged_gdf = counties_gdf.merge(
        trend_data,
        left_on=['STUSPS', 'COUNTY_CLEAN'],
        right_on=['STATE_ABBREV', 'COUNTY_NAME'],
        how='left'
    )

    merged_gdf['TREND_SLOPE'] = merged_gdf['TREND_SLOPE'].fillna(0)
    merged_gdf['SIGNIFICANT'] = merged_gdf['SIGNIFICANT'].fillna(False)
    merged_gdf['P_VALUE'] = merged_gdf['P_VALUE'].fillna(1.0)
    
    return merged_gdf

def merge_alaska_data(trend_data, counties_alaska_gdf):
    alaska_merged = counties_alaska_gdf.merge(
        trend_data,
        left_on=['STUSPS', 'COUNTY_CLEAN'],
        right_on=['STATE_ABBREV', 'COUNTY_NAME'],
        how='left'
    )
    
    alaska_merged['TREND_SLOPE'] = alaska_merged['TREND_SLOPE'].fillna(0)
    alaska_merged['SIGNIFICANT'] = alaska_merged['SIGNIFICANT'].fillna(False)
    alaska_merged['P_VALUE'] = alaska_merged['P_VALUE'].fillna(1.0)
    
    return alaska_merged

def create_trend_map(merged_gdf, states_gdf, counties_alaska, states_alaska, alaska_data, month=None, output_dir="./figure"):
    month_names = {2: 'February'}
    month_suffix = f"_{month}month" if month is not None else ""
    month_text = month_names[month] if month is not None else "All months"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig_size = (6, 4)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    counties_with_data = merged_gdf   
    if month==2:
        label_index = 'February Freezing Rain Trend (Events Yr⁻²)' 
    if len(counties_with_data) > 0:
        counties_with_data.plot(column='TREND_SLOPE', 
                               ax=ax, 
                               cmap='RdBu_r', 
                               legend=True,
                               vmin=-0.05, 
                               vmax=0.05, 
                               legend_kwds={'shrink': 0.8, 'label': label_index,
                                          'orientation': 'horizontal'})
    if 'SIGNIFICANT' in merged_gdf.columns:
        significant_counties = merged_gdf[merged_gdf['SIGNIFICANT'] == True]
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
    
    ax.set_xlim(-125, -65)
    ax.set_ylim(25, 50)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    inset_ax = fig.add_axes([0.75, 0.35, 0.15, 0.15])

    if len(alaska_data) > 0:
        alaska_data.plot(column='TREND_SLOPE',
                        ax=inset_ax,
                        cmap='RdBu_r',
                        vmin=-0.05,
                        vmax=0.05, 
                        legend=False)
        if 'SIGNIFICANT' in alaska_data.columns:
            alaska_significant = alaska_data[alaska_data['SIGNIFICANT'] == True]
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
    output_file = output_path / f'freezing_rain_annual_trend_map{month_suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()   
    return {
        'annual_trend': output_file
    }


def main():
    """
    主函数
    """
    try:
        excel_file = "../../data/freezing_rain_events_county_llm.xlsx"
        output_dir = "./figure"

        df = load_and_process_data(excel_file)
        counties_gdf, states_gdf, counties_alaska, states_alaska = download_geographic_data()
        
        months = [2]
        month_names = {2: 'February'}
        
        all_map_files = {}
        all_trend_data = {}

        for month in months:
            trend_data = calculate_county_annual_trends(df, month)
            all_trend_data[month] = trend_data

            merged_gdf = merge_data_with_geography(trend_data, counties_gdf)

            alaska_merged = merge_alaska_data(trend_data, counties_alaska)

            map_files = create_trend_map(merged_gdf, states_gdf, counties_alaska, states_alaska, alaska_merged, month, output_dir)
            all_map_files[month] = map_files

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()