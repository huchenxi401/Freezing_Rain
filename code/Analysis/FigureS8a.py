import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
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

def load_population_data(csv_file="../../data/co-est2024-alldata.csv"):
    df = pd.read_csv(csv_file, encoding='latin1')

    df = df[['STNAME', 'CTYNAME', 'POPESTIMATE2024']].copy()

    df['STATE_CLEAN'] = df['STNAME'].str.strip().str.upper()
    df['COUNTY_ORIGINAL'] = df['CTYNAME'].str.strip().str.upper()

    df = df[df['STATE_CLEAN'] != df['COUNTY_ORIGINAL']].copy()

    df['COUNTY_CLEAN'] = df['COUNTY_ORIGINAL'].apply(standardize_county_name)

    df['STATE_ABBREV'] = df['STATE_CLEAN'].map(STATE_ABBREV)

    df = df[~df['STATE_ABBREV'].isin(['HI'])].copy()

    df_clean = df.dropna(subset=['STATE_ABBREV', 'COUNTY_CLEAN', 'POPESTIMATE2024'])

    df_clean = df_clean.rename(columns={
        'COUNTY_CLEAN': 'COUNTY_NAME',
        'POPESTIMATE2024': 'POPULATION'
    })

    df_clean = df_clean[['STATE_ABBREV', 'COUNTY_NAME', 'POPULATION']].copy()
    
    return df_clean


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

def check_county_matching(population_data, counties_gdf):
    geo_counties = set()
    for _, row in counties_gdf.iterrows():
        geo_counties.add((row['STUSPS'], row['COUNTY_CLEAN']))

    matched = []
    unmatched = []
    
    for _, row in population_data.iterrows():
        county_key = (row['STATE_ABBREV'], row['COUNTY_NAME'])
        if county_key in geo_counties:
            matched.append(county_key)
        else:
            unmatched.append(county_key)
    
    return matched, unmatched

def merge_data_with_geography(population_data, counties_gdf):
    merged_gdf = counties_gdf.merge(
        population_data,
        left_on=['STUSPS', 'COUNTY_CLEAN'],
        right_on=['STATE_ABBREV', 'COUNTY_NAME'],
        how='left'
    )

    merged_gdf['POPULATION'] = merged_gdf['POPULATION'].fillna(0)
    return merged_gdf

def merge_alaska_data(population_data, counties_alaska):
    alaska_merged = counties_alaska.merge(
        population_data,
        left_on=['STUSPS', 'COUNTY_CLEAN'],
        right_on=['STATE_ABBREV', 'COUNTY_NAME'],
        how='left'
    )
    
    alaska_merged['POPULATION'] = alaska_merged['POPULATION'].fillna(0)
    return alaska_merged


def load_emerging_hotspots(hotspot_file):
    try:
        hotspots_df = pd.read_excel(hotspot_file)
        return hotspots_df
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def create_population_map(merged_gdf, states_gdf, counties_alaska, states_alaska, alaska_merged, 
                         hotspots_df=None, 
                         output_dir="./figure"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_size = (6, 4)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    counties_with_population = merged_gdf

    if len(counties_with_population) > 0:
        counties_with_population.plot(column='POPULATION', 
                                     ax=ax, 
                                     cmap='Reds', 
                                     legend=True,
                                     vmin=0,       
                                     vmax=500000, 
                                     legend_kwds={'shrink': 0.8, 'label': 'Population',
                                        'orientation': 'horizontal'})

    states_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)

    merged_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.2)
    if hotspots_df is not None and len(hotspots_df) > 0:
        hotspot_counties = merged_gdf.merge(
            hotspots_df,
            left_on=['STUSPS', 'COUNTY_CLEAN'],
            right_on=['STATE_ABBREV', 'COUNTY_NAME'],
            how='inner'
        )
        
        if len(hotspot_counties) > 0:
            hotspot_counties.boundary.plot(ax=ax, color='gold', linewidth=0.3)
            total_hotspot_population = hotspot_counties['POPULATION'].sum()
            num_hotspot_counties = len(hotspot_counties)

            hotspot_list = hotspot_counties[['STUSPS', 'COUNTY_CLEAN', 'POPULATION']].copy()
            hotspot_list = hotspot_list.sort_values('POPULATION', ascending=False)

    ax.set_xlim(-125, -65)
    ax.set_ylim(25, 50)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    inset_ax = fig.add_axes([0.75, 0.35, 0.15, 0.15])

    if len(alaska_merged) > 0:
        alaska_merged.plot(column='POPULATION',
                          ax=inset_ax,
                          cmap='Reds',
                          vmin=0,
                          vmax=500000,
                          legend=False)

    states_alaska.boundary.plot(ax=inset_ax, color='black', linewidth=0.6)
    counties_alaska.boundary.plot(ax=inset_ax, color='gray', linewidth=0.2)
    
    inset_ax.set_title('Alaska', fontsize=8)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path / 'us_county_population_2024.png', dpi=300, bbox_inches='tight')
    plt.close()

    return output_path / 'us_county_population_2024.png'


def main():
    try:
        csv_file = "../../data/co-est2024-alldata.csv"
        output_dir = "./figure"
        population_data = load_population_data(csv_file)

        counties_gdf, states_gdf, counties_alaska, states_alaska = download_geographic_data()
        
        matched, unmatched = check_county_matching(
            population_data[population_data['STATE_ABBREV'] != 'AK'], 
            counties_gdf
        )
        
        merged_gdf = merge_data_with_geography(population_data, counties_gdf)
        alaska_merged = merge_alaska_data(population_data, counties_alaska)
        hotspot_file = "../../data/february_emerging_hotspots.xlsx"
        hotspots_df = load_emerging_hotspots(hotspot_file)
        map_file = create_population_map(merged_gdf, states_gdf, counties_alaska, states_alaska, alaska_merged, 
                                hotspots_df, output_dir)
        

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()