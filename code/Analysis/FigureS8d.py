import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
from matplotlib.colors import ListedColormap, BoundaryNorm
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

CROPS = ['CORN', 'PEACHES',  'WHEAT', 'RICE', 'SOYBEAN',  'APPLE'] #'HAY', 

CROP_COLORS = {
    'CORN': 'orange',      
    #'HAY': '#90EE90',       
    'PEACHES': '#FFB6C1',  
    'WHEAT': '#DEB887',     
    'RICE': '#F0E68C',      
    'SOYBEAN': 'darkgreen', 
    'APPLE': 'red'

}

def standardize_county_name(county_name):
    if pd.isna(county_name):
        return None
    
    county_name = str(county_name).strip().upper()
    suffixes_to_remove = [
        ' CITY AND BOROUGH',  
        ' CENSUS AREA',
        ' CITY COUNTY', 
        ' COUNTY',
        ' PARISH',
        ' BOROUGH',
        ' CITY'
    ]
    
    for suffix in suffixes_to_remove:
        if county_name.endswith(suffix):
            county_name = county_name[:-len(suffix)].strip()
            break
    
    return county_name

def load_crop_data(crop_dir="../../data/crop"):
    crop_dir = Path(crop_dir)
    all_crop_data = {}
    
    for crop in CROPS:
        csv_file = crop_dir / f"{crop}.csv"
        
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file, encoding='latin1')

        df = df[['State', 'County', 'Value']].copy()

        df['State'] = df['State'].str.strip().str.upper()
        df['County'] = df['County'].str.strip().str.upper()

        df['COUNTY_CLEAN'] = df['County'].apply(standardize_county_name)

        df['STATE_ABBREV'] = df['State'].map(STATE_ABBREV)

        df['Value'] = df['Value'].astype(str).str.replace(',', '')  
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')  

        df = df.dropna(subset=['STATE_ABBREV', 'COUNTY_CLEAN', 'Value'])

        df = df[['STATE_ABBREV', 'COUNTY_CLEAN', 'Value']].copy()
        df['CROP'] = crop
        
        all_crop_data[crop] = df
    return all_crop_data

def find_dominant_crop(all_crop_data):
    all_data = []
    for crop, df in all_crop_data.items():
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    dominant_crop = combined_df.loc[combined_df.groupby(['STATE_ABBREV', 'COUNTY_CLEAN'])['Value'].idxmax()]
    for crop in CROPS:
        count = len(dominant_crop[dominant_crop['CROP'] == crop])
        percentage = count / len(dominant_crop) * 100
    
    return dominant_crop

def download_geographic_data():
    counties_url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip"
    counties_gdf = gpd.read_file(counties_url)
    states_url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_20m.zip"
    states_gdf = gpd.read_file(states_url)
    counties_gdf['COUNTY_CLEAN'] = counties_gdf['NAME'].apply(standardize_county_name)
    counties_gdf = counties_gdf[~counties_gdf['STATEFP'].isin(['02', '15'])]
    states_gdf = states_gdf[~states_gdf['STATEFP'].isin(['02', '15'])]
    return counties_gdf, states_gdf

def load_emerging_hotspots(hotspot_file):
    try:
        hotspots_df = pd.read_excel(hotspot_file)
        return hotspots_df
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def merge_crop_with_geography(dominant_crop, counties_gdf):
    merged_gdf = counties_gdf.merge(
        dominant_crop,
        left_on=['STUSPS', 'COUNTY_CLEAN'],
        right_on=['STATE_ABBREV', 'COUNTY_CLEAN'],
        how='left'
    )

    return merged_gdf

def create_crop_map(merged_gdf, states_gdf, hotspots_df=None, 
                   output_dir="./figure"):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_size = (6, 4)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    crop_to_code = {crop: i+1 for i, crop in enumerate(CROPS)}
    merged_gdf['CROP_CODE'] = merged_gdf['CROP'].map(crop_to_code)

    colors = ['#CCCCCC']  
    colors.extend([CROP_COLORS[crop] for crop in CROPS]) 

    cmap = ListedColormap(colors)
    boundaries = np.arange(-0.5, len(CROPS) + 1, 1) 
    norm = BoundaryNorm(boundaries, cmap.N)

    merged_gdf['CROP_CODE'] = merged_gdf['CROP_CODE'].fillna(0)

    crop_counts = merged_gdf[merged_gdf['CROP'].notna()]['CROP'].value_counts()
    total_counties_with_crop = crop_counts.sum()



    no_data_counties = merged_gdf['CROP'].isna().sum()
    merged_gdf.plot(
        column='CROP_CODE',
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor='none',
        legend=False
    )
    
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color='orange', label='Corn'),
        mpatches.Patch(color='#DEB887', label='Wheat'),
        mpatches.Patch(color='#F0E68C', label='Rice'),
        mpatches.Patch(color='darkgreen', label='Soybean'),
        mpatches.Patch(color='#FFB6C1', label='Peaches'),
        mpatches.Patch(color='red', label='Apple')
    ]


    ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.2), loc='upper center',
         ncol=3, fontsize=8, title='Crop Type', title_fontsize=9)
    states_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
    merged_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.2)

    if hotspots_df is not None and len(hotspots_df) > 0:
        hotspot_counties = merged_gdf.merge(
            hotspots_df,
            left_on=['STUSPS', 'COUNTY_CLEAN'],
            right_on=['STATE_ABBREV', 'COUNTY_NAME'],
            how='inner',
            suffixes=('', '_hotspot')
        )
        
        if len(hotspot_counties) > 0:
            hotspot_counties.boundary.plot(ax=ax, color='gold', linewidth=0.5)

    ax.set_xlim(-125, -65)
    ax.set_ylim(25, 50)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.tight_layout()

    output_file = output_path / 'us_county_dominant_crop.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def calculate_hotspot_crop_statistics(all_crop_data, baseline_df, hotspots_df, merged_gdf, 
                                     output_dir="./figure"):

    output_path = Path(output_dir)

    all_data = []
    for crop, df in all_crop_data.items():
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)

    hotspot_keys = set()
    if hotspots_df is not None:
        for _, row in hotspots_df.iterrows():
            hotspot_keys.add((row['STATE_ABBREV'], row['COUNTY_NAME']))
 
    combined_df['IS_HOTSPOT'] = combined_df.apply(
        lambda x: (x['STATE_ABBREV'], x['COUNTY_CLEAN']) in hotspot_keys, 
        axis=1
    )
    baseline_keys = set()
    if baseline_df is not None:
        for _, row in baseline_df.iterrows():
            baseline_keys.add((row['STATE_ABBREV'], row['COUNTY_NAME']))

    combined_df['IS_BASELINE'] = combined_df.apply(
        lambda x: (x['STATE_ABBREV'], x['COUNTY_CLEAN']) in baseline_keys, 
        axis=1
    )
    statistics = []

    for crop in CROPS:
        crop_data = combined_df[combined_df['CROP'] == crop]
        total_area = crop_data['Value'].sum()
        num_counties = len(crop_data)
        
        statistics.append({
            'Region': 'All Counties',
            'Crop': crop,
            'Total_Area_Acres': total_area,
            'Number_of_Counties': num_counties,
            'Average_Area_per_County': total_area / num_counties if num_counties > 0 else 0
        })

    all_total = combined_df['Value'].sum()
    all_counties = len(combined_df.groupby(['STATE_ABBREV', 'COUNTY_CLEAN']))
    statistics.append({
        'Region': 'All Counties',
        'Crop': 'TOTAL',
        'Total_Area_Acres': all_total,
        'Number_of_Counties': all_counties,
        'Average_Area_per_County': all_total / all_counties if all_counties > 0 else 0
    })

    hotspot_data = combined_df[combined_df['IS_HOTSPOT']]
    
    for crop in CROPS:
        crop_data = hotspot_data[hotspot_data['CROP'] == crop]
        total_area = crop_data['Value'].sum()
        num_counties = len(crop_data)
        
        statistics.append({
            'Region': 'Emerging Hotspots',
            'Crop': crop,
            'Total_Area_Acres': total_area,
            'Number_of_Counties': num_counties,
            'Average_Area_per_County': total_area / num_counties if num_counties > 0 else 0
        })

    hotspot_total = hotspot_data['Value'].sum()
    hotspot_counties = len(hotspot_data.groupby(['STATE_ABBREV', 'COUNTY_CLEAN']))
    statistics.append({
        'Region': 'Emerging Hotspots',
        'Crop': 'TOTAL',
        'Total_Area_Acres': hotspot_total,
        'Number_of_Counties': hotspot_counties,
        'Average_Area_per_County': hotspot_total / hotspot_counties if hotspot_counties > 0 else 0
    })
    baseline_data = combined_df[combined_df['IS_BASELINE']]
    
    for crop in CROPS:
        crop_data = baseline_data[baseline_data['CROP'] == crop]
        total_area = crop_data['Value'].sum()
        num_counties = len(crop_data)
        
        statistics.append({
            'Region': 'Baseline',
            'Crop': crop,
            'Total_Area_Acres': total_area,
            'Number_of_Counties': num_counties,
            'Average_Area_per_County': total_area / num_counties if num_counties > 0 else 0
        })
    
    baseline_total = baseline_data['Value'].sum()
    baseline_counties = len(baseline_data.groupby(['STATE_ABBREV', 'COUNTY_CLEAN']))
    statistics.append({
        'Region': 'baseline',
        'Crop': 'TOTAL',
        'Total_Area_Acres': baseline_total,
        'Number_of_Counties': baseline_counties,
        'Average_Area_per_County': baseline_total / baseline_counties if baseline_counties > 0 else 0
    })
    stats_df = pd.DataFrame(statistics)
    
    for crop in CROPS:
        crop_hotspot = hotspot_data[hotspot_data['CROP'] == crop]['Value'].sum()
        crop_all = combined_df[combined_df['CROP'] == crop]['Value'].sum()
        crop_baseline = baseline_data[baseline_data['CROP'] == crop]['Value'].sum()
        percentage = crop_hotspot / crop_baseline * 100 if crop_baseline > 0 else 0
        #percentage = crop_hotspot / crop_all * 100 if crop_all > 0 else 0
        print(f"  {crop}: {crop_hotspot:,.0f} acres ({percentage:.1f}% of national total)")
    
    print("=" * 60)
    
    return stats_df

def main():
    try:
        crop_dir = "../../data/crop"
        hotspot_file = "../../data/february_emerging_hotspots.xlsx"
        baseline_file = "../../data/february_baseline.xlsx"
        output_dir = "./figure"
        all_crop_data = load_crop_data(crop_dir)
        
        if len(all_crop_data) == 0:
            return

        dominant_crop = find_dominant_crop(all_crop_data)

        counties_gdf, states_gdf = download_geographic_data()

        hotspots_df = load_emerging_hotspots(hotspot_file)
        baseline_df = load_emerging_hotspots(baseline_file)
 
        merged_gdf = merge_crop_with_geography(dominant_crop, counties_gdf)
            

        map_file = create_crop_map(merged_gdf, states_gdf, hotspots_df, output_dir)
  
        stats_df = calculate_hotspot_crop_statistics(all_crop_data, baseline_df, hotspots_df, merged_gdf, output_dir)
        

        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()