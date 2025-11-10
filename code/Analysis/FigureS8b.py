import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
import xarray as xr
from shapely.geometry import Point, Polygon
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

HIGH_VEG_TYPES = {
    0: 'No high vegetation',
    3: 'Evergreen needleleaf trees',
    4: 'Deciduous needleleaf trees',
    5: 'Deciduous broadleaf trees',
    6: 'Evergreen broadleaf trees',
    18: 'Mixed forest/woodland',
    19: 'Interrupted forest'
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

def load_vegetation_type_data(nc_file="../../data/LAI_1996_2025_Feb.nc"):

    ds = xr.open_dataset(nc_file)

    tvh_2025 = ds['tvh'].isel(valid_time=-1)
    tvl_2025 = ds['tvl'].isel(valid_time=-1)

    lons = ds.longitude.values
    lats = ds.latitude.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    unique_hveg = np.unique(tvh_2025.values[~np.isnan(tvh_2025.values)])
    for vtype in unique_hveg:
        count = np.sum(tvh_2025.values == vtype)
        type_name = HIGH_VEG_TYPES.get(int(vtype), f'Unknown ({int(vtype)})')
    
    return tvh_2025, tvl_2025, lon_grid, lat_grid, ds

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

def create_vegetation_type_map(veg_data, veg_type, mask, states_gdf, counties_gdf, 
                                hotspots_df=None, 
                                output_dir="./figure"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_size = (6, 4)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    veg_masked = np.ma.masked_where(~mask, veg_data.values)

    lons = veg_data.longitude.values
    lats = veg_data.latitude.values
    if veg_type == 'high':
        colors = [
            '#CCCCCC',  
            '#FFFFFF', 
            '#FFFFFF', 
            'darkgreen',  
            'orange', 
            'brown',  
            'green', 
            '#FFFFFF', 
            '#FFFFFF',
            '#FFFFFF',
            '#FFFFFF',
            '#FFFFFF',
            '#FFFFFF',
            '#FFFFFF',
            '#FFFFFF',
            '#FFFFFF',
            '#FFFFFF',
            '#FFFFFF',
            'lightgreen',  
            'cyan',  
            '#FFFFFF'  
        ]
        type_dict = HIGH_VEG_TYPES
        unique_types = np.unique(veg_masked[~veg_masked.mask]).astype(int)
        boundaries = np.arange(-0.5, 21, 1) 

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)
    
    im = ax.pcolormesh(lons, lats, veg_masked, 
                       cmap=cmap,
                       norm=norm,
                       shading='auto')

    
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color='darkgreen', label='Evergreen Needleleaf'),
        mpatches.Patch(color='orange', label='Deciduous Needleleaf'),
        mpatches.Patch(color='brown', label='Deciduous Broadleaf'),
        mpatches.Patch(color='green', label='Evergreen Broadleaf'),
        mpatches.Patch(color='lightgreen', label='Mixed forest/woodland'),
        mpatches.Patch(color='cyan', label='Interrupted forest')
    ]
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.2), loc='upper center',
         ncol=3, fontsize=8, title='Vegetation Type', title_fontsize=9)
    states_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
    
    counties_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.2)
 
    if hotspots_df is not None and len(hotspots_df) > 0:
        hotspot_counties = counties_gdf.merge(
            hotspots_df,
            left_on=['STUSPS', 'COUNTY_CLEAN'],
            right_on=['STATE_ABBREV', 'COUNTY_NAME'],
            how='inner'
        )
        
        if len(hotspot_counties) > 0:
            hotspot_counties.boundary.plot(ax=ax, color='gold', linewidth=0.3)
    ax.set_xlim(-125, -65)
    ax.set_ylim(25, 50)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.tight_layout()

    output_file = output_path / f'us_{veg_type}_vegetation_type_2025.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def main():
    """
    主函数
    """
    try:
        nc_file = "../../data/LAI_1996_2025_Feb.nc"
        hotspot_file = "../../data/february_emerging_hotspots.xlsx"
        output_dir = "./figure"

        tvh_2025, tvl_2025, lon_grid, lat_grid, ds = load_vegetation_type_data(nc_file)

        counties_gdf, states_gdf = download_geographic_data()

        hotspots_df = load_emerging_hotspots(hotspot_file)

        mask = create_us_mask(lat_grid, lon_grid, states_gdf)

        map_hveg = create_vegetation_type_map(tvh_2025, 'high', mask, states_gdf, counties_gdf, 
                                             hotspots_df, output_dir)
        
        ds.close()
  
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()