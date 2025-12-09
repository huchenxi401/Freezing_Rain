import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from matplotlib.patches import Rectangle
import pandas as pd
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

def download_coastline_data():
    try:
        coastline_url = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip"
        coastline_gdf = gpd.read_file(coastline_url)
        return coastline_gdf
    except Exception as e:
        try:
            coastline_url_backup = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/physical/ne_50m_coastline.zip"
            coastline_gdf = gpd.read_file(coastline_url_backup)
            return coastline_gdf
        except Exception as e2:
            return None
def load_emerging_hotspots(hotspot_file):
    try:
        hotspots_df = pd.read_excel(hotspot_file)
        return hotspots_df
    except FileNotFoundError:
        return None
    except Exception as e:
        return None
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

def load_and_process_data(file_path):
    ds = xr.open_dataset(file_path)

    ds['valid_time'] = xr.decode_cf(ds).valid_time

    u = ds['u'].squeeze()
    v = ds['v'].squeeze()

    wind_speed = np.sqrt(u**2 + v**2)

    return ds, u, v, wind_speed

def extract_monthly_variability_data(ds, u, v, wind_speed, period_start, period_end):

    time_values = ds.valid_time.values
    time_pd = pd.to_datetime(time_values)
    years = time_pd.year.values
    months = time_pd.month.values
    
    monthly_data = {}
    
    for month in [2]:
        mask = [(years[i] >= period_start and years[i] <= period_end and months[i] == month) 
                for i in range(len(years))]
        
        if sum(mask) > 0:
            u_month_all = u[mask]
            v_month_all = v[mask]
            ws_month_all = wind_speed[mask]

            u_mean = u_month_all.mean(dim='valid_time')
            v_mean = v_month_all.mean(dim='valid_time')
            ws_mean = ws_month_all.mean(dim='valid_time')

            v_std = v_month_all.std(dim='valid_time')
  
            ws_std = ws_month_all.std(dim='valid_time')
            
            v_p5 = v_month_all.quantile(0.05, dim='valid_time')

            v_p90 = v_month_all.quantile(0.9, dim='valid_time')
            
            monthly_data[month] = {
                'u_mean': u_mean,
                'v_mean': v_mean,
                'ws_mean': ws_mean,
                'v_std': v_std,
                'ws_std': ws_std,
                'v_p5': v_p5,
                'v_p90': v_p90,
                'n_samples': sum(mask)
            }
             
    return monthly_data


def calculate_variance_significance(ds, period1_start, period1_end, 
                                   period2_start, period2_end, month):   
    time_values = ds.valid_time.values
    time_pd = pd.to_datetime(time_values)
    years = time_pd.year.values
    months = time_pd.month.values
    v = ds['v'].squeeze()

    mask1 = [(years[i] >= period1_start and years[i] <= period1_end and months[i] == month) 
             for i in range(len(years))]
    mask2 = [(years[i] >= period2_start and years[i] <= period2_end and months[i] == month) 
             for i in range(len(years))]
    
    v_period1 = v[mask1] 
    v_period2 = v[mask2] 
    
    n_lat, n_lon = v_period1.shape[1], v_period1.shape[2]
    p_values = np.ones((n_lat, n_lon))

    for i in range(n_lat):
        for j in range(n_lon):
            series1 = v_period1[:, i, j].values
            series2 = v_period2[:, i, j].values

            series1_clean = series1[~np.isnan(series1)]
            series2_clean = series2[~np.isnan(series2)]
            
            if len(series1_clean) > 2 and len(series2_clean) > 2:
                var1 = np.var(series1_clean, ddof=1)
                var2 = np.var(series2_clean, ddof=1)
                
                if var1 > 0 and var2 > 0:
                    F = var2 / var1 if var2 > var1 else var1 / var2
                    df1 = len(series2_clean) - 1 if var2 > var1 else len(series1_clean) - 1
                    df2 = len(series1_clean) - 1 if var2 > var1 else len(series2_clean) - 1

                    p_val = 2 * min(sp_stats.f.cdf(F, df1, df2), 
                                   1 - sp_stats.f.cdf(F, df1, df2))
                    p_values[i, j] = p_val
        

    
    significant_mask = p_values < 0.05
    sig_count = np.sum(significant_mask)
    total_count = n_lat * n_lon
    
    return p_values, significant_mask


def plot_difference(ds, monthly_data_early, monthly_data_late, month,
                   counties_gdf, states_gdf, coastline_gdf, output_file, hotspots_df=None):

    fig_size = (6, 4)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    
    lons = ds.longitude.values
    lats = ds.latitude.values
    
    data_early = monthly_data_early[month]
    data_late = monthly_data_late[month]
    
    v_std_early = data_early['v_std'].values
    v_std_late = data_late['v_std'].values
    delta_v_std = v_std_late - v_std_early
    
    ws_early = data_early['v_mean'].values
    ws_late = data_late['v_mean'].values
    delta_ws = ws_late - ws_early
    
    v_p5_early = data_early['v_p5'].values
    v_p5_late = data_late['v_p5'].values
    delta_v_p5 = v_p5_late - v_p5_early

    p_values, significant_mask = calculate_variance_significance(
        ds, 1996, 2010, 2011, 2025, month
    )

    levels_std = np.linspace(-3, 3, 21)
    cf = ax.contourf(lons, lats, delta_v_std, levels=levels_std,
                    cmap='coolwarm',
                    extend='both')
    
    if significant_mask is not None:
        sig_count = np.sum(significant_mask)
        
        if sig_count > 0:
            step = 2 
            sig_indices = np.where(significant_mask)
            
            for idx in range(0, len(sig_indices[0]), step):
                i = sig_indices[0][idx]
                j = sig_indices[1][idx]
                ax.plot(lons[j], lats[i], 'k.', markersize=2, 
                       markerfacecolor='black', markeredgewidth=0)
            
    levels_cold = np.arange(0, 8, 4)
    if np.nanmax(-delta_v_p5) > 1:
        from scipy import ndimage
        from scipy.ndimage import binary_dilation, binary_erosion

        filtered_data = -delta_v_p5.copy()
        pos_mask = filtered_data > 0
        neg_mask = filtered_data < 0

        labeled_pos, num_pos = ndimage.label(pos_mask)

        labeled_neg, num_neg = ndimage.label(neg_mask)

        for region_id in range(1, num_pos + 1):
            region_mask = (labeled_pos == region_id)
            region_size = np.sum(region_mask)

            touches_boundary = (
                np.any(region_mask[0, :]) or   
                np.any(region_mask[-1, :]) or  
                np.any(region_mask[:, 0]) or 
                np.any(region_mask[:, -1])   
            )
            if region_size < 80000 and not touches_boundary:  
                filtered_data[region_mask] = -1.0

        for region_id in range(1, num_neg + 1):
            region_mask = (labeled_neg == region_id)
            region_size = np.sum(region_mask)
            
            touches_boundary = (
                np.any(region_mask[0, :]) or
                np.any(region_mask[-1, :]) or
                np.any(region_mask[:, 0]) or
                np.any(region_mask[:, -1])
            )

            if region_size < 80000 and not touches_boundary:
                filtered_data[region_mask] = 1.0

        cs_cold = ax.contour(lons, lats, filtered_data, levels=levels_cold,
                        colors='green', linestyles='solid', linewidths=1)
        cs_zero = ax.contour(lons, lats, filtered_data, levels=0,
                        colors='cyan', linestyles='solid', linewidths=1)
        ax.clabel(cs_cold, inline=True, fontsize=9, fmt='%d')
        ax.clabel(cs_zero, inline=True, fontsize=9, fmt='%d')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='cyan', linewidth=1, label='Δ 95th north V (0 m/s)'),
        Line2D([0], [0], color='green', linewidth=1, label='Δ 95th north V (4 m/s)')
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
    
    if coastline_gdf is not None:
        coastline_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
    
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
    ax.set_xlim(-130, -60)
    ax.set_ylim(25, 55)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.8, pad=0.15, aspect=30)
    cbar.set_label('Difference in February 250hPa V-wind Std Dev (m/s)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    data_file = "../../data/250hPa_monthly_wind_DJF_1996_2025.nc"
    output_dir = "./"

    counties_gdf, states_gdf = download_boundary_data()
    coastline_gdf = download_coastline_data()

    global ds
    ds, u, v, wind_speed = load_and_process_data(data_file)

    monthly_data_early = extract_monthly_variability_data(ds, u, v, wind_speed, 1996, 2010)

    monthly_data_late = extract_monthly_variability_data(ds, u, v, wind_speed, 2011, 2025)
   
    month_names = {2: 'February'}

    hotspot_file = "../../data/february_emerging_hotspots.xlsx"
    hotspots_df = load_emerging_hotspots(hotspot_file)
    for month in [2]:
        month_name = month_names[month]

        output_file = f"figure4a.png"
        plot_difference(ds, monthly_data_early, monthly_data_late, month,
                       counties_gdf, states_gdf, coastline_gdf, output_file, hotspots_df=hotspots_df)

    ds.close()

if __name__ == "__main__":

    main()
