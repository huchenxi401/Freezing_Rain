import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.stats import bootstrap
from sklearn.linear_model import LinearRegression
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

def calculate_confidence_interval(x, y, confidence=0.95):
    n = len(x)
    if n < 3:
        return None, None, None, None

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    y_pred = slope * x + intercept
    residuals = y - y_pred

    mse = np.sum(residuals**2) / (n - 2) 

    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean)**2)

    slope_std_err = np.sqrt(mse / sxx)

    alpha = 1 - confidence
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    
    slope_ci_lower = slope - t_val * slope_std_err
    slope_ci_upper = slope + t_val * slope_std_err
    
    return slope, slope_ci_lower, slope_ci_upper, p_value

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

def download_county_geographic_data():

    try:
        counties_gdf = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip")

        exclude_states = ['15', '60', '66', '69', '72', '78']  
        counties_gdf = counties_gdf[~counties_gdf['STATEFP'].isin(exclude_states)]
        
        counties_gdf['COUNTY_CLEAN'] = counties_gdf['NAME'].apply(standardize_county_name)

        counties_gdf['CENTROID'] = counties_gdf.geometry.centroid
        counties_gdf['LONGITUDE'] = counties_gdf['CENTROID'].x
        counties_gdf['LATITUDE'] = counties_gdf['CENTROID'].y

        coord_lookup = {}
        for _, row in counties_gdf.iterrows():
            key = f"{row['STUSPS']}_{row['COUNTY_CLEAN']}"
            coord_lookup[key] = {
                'longitude': row['LONGITUDE'],
                'latitude': row['LATITUDE']
            }

        return coord_lookup
    
    except Exception as e:
        return None

def load_and_process_event_data(excel_file="../../data/freezing_rain_events_county_llm.xlsx"):

    df = pd.read_excel(excel_file, usecols=['STATE', 'COUNTY', 'BEGIN_YEAR', 'DURATION_HOURS'])
    
    df['STATE_CLEAN'] = df['STATE'].str.strip().str.upper()
    df['COUNTY_CLEAN'] = df['COUNTY'].apply(standardize_county_name)
    df['STATE_ABBREV'] = df['STATE_CLEAN'].map(STATE_ABBREV)

    df['STATE_COUNTY'] = df['STATE_ABBREV'] + '_' + df['COUNTY_CLEAN']

    df_clean = df.dropna(subset=['STATE_ABBREV', 'COUNTY_CLEAN', 'BEGIN_YEAR'])
    df_clean = df_clean[df_clean['DURATION_HOURS'] <= 144]

    df_clean = df_clean[(df_clean['BEGIN_YEAR'] >= 1996) & (df_clean['BEGIN_YEAR'] <= 2025)]

    
    return df_clean

def add_coordinates_to_events(df, coord_lookup):

    df['LONGITUDE'] = df['STATE_COUNTY'].map(lambda x: coord_lookup.get(x, {}).get('longitude', None))
    df['LATITUDE'] = df['STATE_COUNTY'].map(lambda x: coord_lookup.get(x, {}).get('latitude', None))

    df_with_coords = df.dropna(subset=['LONGITUDE', 'LATITUDE'])

    return df_with_coords

def calculate_yearly_location_averages(df):

    yearly_locations = df.groupby('BEGIN_YEAR').agg({
        'LONGITUDE': 'mean',
        'LATITUDE': 'mean',
        'STATE_COUNTY': 'count'  
    }).reset_index()
    
    yearly_locations.columns = ['Year', 'Avg_Longitude', 'Avg_Latitude', 'Event_Count']
    
    return yearly_locations

def calculate_bootstrap_ci(x_vals, y_vals, unique_years, n_bootstrap=10000, confidence=0.95):
    
    def trend_func(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        return slope * unique_years + intercept
    data = (x_vals, y_vals)
    
    rng = np.random.default_rng(42) 
    res = bootstrap(data, trend_func, n_resamples=n_bootstrap, 
                   confidence_level=confidence, random_state=rng,
                   paired=True)
    
    return res.confidence_interval.low, res.confidence_interval.high

def calculate_period_comparison(yearly_locations):
    from geopy.distance import geodesic

    period1 = yearly_locations[(yearly_locations['Year'] >= 1996) & (yearly_locations['Year'] <= 2010)]
    period2 = yearly_locations[(yearly_locations['Year'] >= 2011) & (yearly_locations['Year'] <= 2025)]
    
    if len(period1) == 0 or len(period2) == 0:
        return

    period1_avg_lon = period1['Avg_Longitude'].mean()
    period1_avg_lat = period1['Avg_Latitude'].mean()
    period2_avg_lon = period2['Avg_Longitude'].mean()
    period2_avg_lat = period2['Avg_Latitude'].mean()

    point1 = (period1_avg_lat, period1_avg_lon)
    point2 = (period2_avg_lat, period2_avg_lon)
    distance = geodesic(point1, point2).kilometers

def moving_average(data, window=5):
    if len(data) < window:
        return data
    
    smoothed = np.full_like(data, np.nan)
    half_window = window // 2
    
    for i in range(len(data)):
        if i <= half_window - 1: 
            start_idx = 0
            end_idx = min(i + half_window + 1, len(data))
        elif i >= len(data) - half_window: 
            start_idx = max(i - half_window, 0)
            end_idx = len(data)
        else: 
            start_idx = i - half_window
            end_idx = i + half_window + 1

        window_data = data[start_idx:end_idx]
        if len(window_data) > 0:
            smoothed[i] = np.nanmean(window_data)
        else:
            smoothed[i] = np.nan
    
    return smoothed

def create_combined_location_trend_plot(yearly_locations, output_dir="./figure"):
    
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    x_vals = yearly_locations['Year'].values
    longitude_vals = yearly_locations['Avg_Longitude'].values
    latitude_vals = yearly_locations['Avg_Latitude'].values
    longitude_vals = moving_average(longitude_vals)
    latitude_vals = moving_average(latitude_vals)

    longitude_vals_abs = np.abs(longitude_vals)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()  
    from scipy import stats
    model_lon = LinearRegression()
    model_lon.fit(x_vals.reshape(-1, 1), longitude_vals_abs)
    trend_line_lon = model_lon.predict(x_vals.reshape(-1, 1))
    slope_lon, intercept_lon, r_lon, p_lon, _ = stats.linregress(x_vals, longitude_vals_abs)
    slope, ci_lower_lon, ci_upper_lon, p_value_lon = calculate_confidence_interval(x_vals, longitude_vals_abs)
    print(f"lon trend: {slope:.3f} h/yr (95% CI: {ci_lower_lon:.3f} to {ci_upper_lon:.3f}, P={p_value_lon:.4f})")  
    ci_low_lon, ci_high_lon = calculate_bootstrap_ci(x_vals, longitude_vals_abs, x_vals, 
                                                   n_bootstrap=10000, confidence=0.95)

    model_lat = LinearRegression()
    model_lat.fit(x_vals.reshape(-1, 1), latitude_vals)
    trend_line_lat = model_lat.predict(x_vals.reshape(-1, 1))
    slope_lat, intercept_lat, r_lat, p_lat, _ = stats.linregress(x_vals, latitude_vals)
    slope, ci_lower_lat, ci_upper_lat, p_value_lat = calculate_confidence_interval(x_vals, latitude_vals)
    print(f"lat trend: {slope:.3f} h/yr (95% CI: {ci_lower_lat:.3f} to {ci_upper_lat:.3f}, P={p_value_lat:.4f})")
    ci_low_lat, ci_high_lat = calculate_bootstrap_ci(x_vals, latitude_vals, x_vals, 
                                                   n_bootstrap=10000, confidence=0.95)
    
    line1 = ax1.plot(x_vals, longitude_vals_abs, '-', color='black', 
                     markersize=6, alpha=0.8,  
                     label=f'Longitude R:{r_lon:.2f}**')
    
    line2 = ax1.plot(x_vals, trend_line_lon, '--', color='black')
    
    ax1.plot(x_vals, ci_low_lon, '--', color='black', alpha=0.5)
    ax1.plot(x_vals, ci_high_lon, '--', color='black', alpha=0.5)
    
    line3 = ax2.plot(x_vals, latitude_vals, '-', color='.7', 
                     markersize=6, alpha=0.8,  
                     label=f'Latitude R:{r_lat:.2f}**')
    
    line4 = ax2.plot(x_vals, trend_line_lat, '--', color='.7')
    
    ax2.plot(x_vals, ci_low_lat, '--', color='.7', alpha=0.5)
    ax2.plot(x_vals, ci_high_lat, '--', color='.7', alpha=0.5)

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Longitude (°W)')
    ax2.set_ylabel('Latitude (°N)')
    
    ax1.set_xticks(range(1996, 2026, 5))

    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc='best')


    output_file = output_path / 'longitude_latitude_combined_trend.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    
    plt.close()

    from scipy import stats
    correlation_lon, p_value_lon = stats.pearsonr(x_vals, longitude_vals_abs)

    total_change_lon = (longitude_vals_abs[-1] - longitude_vals_abs[0])
    years_span = x_vals[-1] - x_vals[0]

    km_per_degree_lon = 85
    total_distance_change_lon = abs(total_change_lon) * km_per_degree_lon
    annual_distance_change_lon = abs(model_lon.coef_[0]) * km_per_degree_lon

    correlation_lat, p_value_lat = stats.pearsonr(x_vals, latitude_vals)
    
    total_change_lat = (latitude_vals[-1] - latitude_vals[0])

    km_per_degree_lat = 111
    total_distance_change_lat = abs(total_change_lat) * km_per_degree_lat
    annual_distance_change_lat = abs(model_lat.coef_[0]) * km_per_degree_lat

def main():

    try:
        excel_file = "../../data/freezing_rain_events_county_llm.xlsx"
        output_dir = "./figure"

        coord_lookup = download_county_geographic_data()
        if coord_lookup is None:
            return

        df = load_and_process_event_data(excel_file)

        df_with_coords = add_coordinates_to_events(df, coord_lookup)
        
        if len(df_with_coords) == 0:
            return

        yearly_locations = calculate_yearly_location_averages(df_with_coords)

        create_combined_location_trend_plot(yearly_locations, output_dir)
             

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()