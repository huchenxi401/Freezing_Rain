import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
from scipy import stats
from shapely.geometry import Point, box
import shapely.ops
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
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

def load_and_process_data(excel_file):
    df = pd.read_excel(excel_file)

    df['STATE_CLEAN'] = df['STATE'].str.strip().str.upper()
    df['COUNTY_CLEAN'] = df['COUNTY'].apply(standardize_county_name)
    df['STATE_ABBREV'] = df['STATE_CLEAN'].map(STATE_ABBREV)

    df['STATE_COUNTY'] = df['STATE_ABBREV'] + '_' + df['COUNTY_CLEAN']

    df_clean = df.dropna(subset=['STATE_ABBREV', 'COUNTY_CLEAN'])
    df_clean = df_clean[df_clean['DURATION_HOURS'] < 144]

    df_clean = df_clean[df_clean['BEGIN_MONTH'] == 2]

    df_clean['TIME_PERIOD'] = df_clean['BEGIN_YEAR'].apply(
        lambda year: '1996-2010' if 1996 <= year <= 2010 else '2011-2025'
    )

    return df_clean

def calculate_county_statistics_by_year(df, time_period=None):
    if time_period:
        df_filtered = df[df['TIME_PERIOD'] == time_period]
    else:
        df_filtered = df
    
    yearly_stats = df_filtered.groupby(['STATE_ABBREV', 'COUNTY_CLEAN', 'BEGIN_YEAR']).agg({
        'EVENT_ID': 'count',
    }).reset_index()

    yearly_stats.columns = ['STATE_ABBREV', 'COUNTY_NAME', 'YEAR', 'EVENT_COUNT']
    
    period_text = f"({time_period}) " if time_period else ""
    
    return yearly_stats

def perform_ttest_by_county(yearly_data_dict):
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
            t_stat, p_value = stats.ttest_ind(data2, data1, equal_var=False)
            mean_diff = np.mean(data2) - np.mean(data1)
            
            ttest_results.append({
                'STATE_ABBREV': state_abbrev,
                'COUNTY_NAME': county_name,
                't_statistic': t_stat,
                'p_value': p_value,
                'mean_difference': mean_diff,
                'significant': p_value < 0.05,
                'period1_mean': np.mean(data1),
                'period2_mean': np.mean(data2),
                'period1_total': np.sum(data1),
                'period2_total': np.sum(data2)
            })
        except:
            ttest_results.append({
                'STATE_ABBREV': state_abbrev,
                'COUNTY_NAME': county_name,
                't_statistic': np.nan,
                'p_value': 1.0,
                'mean_difference': np.mean(data2) - np.mean(data1),
                'significant': False,
                'period1_mean': np.mean(data1),
                'period2_mean': np.mean(data2),
                'period1_total': np.sum(data1),
                'period2_total': np.sum(data2)
            })
    
    ttest_df = pd.DataFrame(ttest_results)
    
    significant_counties = ttest_df[ttest_df['significant']].shape[0]
    total_counties = len(ttest_df)
    return ttest_df


def field_significance_test(yearly_data_dict, ttest_results, n_permutations=10000, alpha=0.05, title_suffix=""):    
    period1_data = yearly_data_dict['1996-2010']
    period2_data = yearly_data_dict['2011-2025']

    counties_to_test = set(
        ttest_results.apply(
            lambda x: f"{x['STATE_ABBREV']}_{x['COUNTY_NAME']}", 
            axis=1
        )
    )
    
    common_counties = list(counties_to_test)
    n_counties = len(counties_to_test)

    county_data = {}
    for county_key in common_counties:
        state_abbrev, county_name = county_key.split('_', 1)
 
        mask1 = (period1_data['STATE_ABBREV'] == state_abbrev) & \
                (period1_data['COUNTY_NAME'] == county_name)
        data1 = period1_data[mask1]['EVENT_COUNT'].values
        years1 = period1_data[mask1]['YEAR'].values
        
        full_data1 = np.zeros(15)
        if len(data1) > 0:
            for i, year in enumerate(range(1996, 2011)):
                if year in years1:
                    idx = np.where(years1 == year)[0][0]
                    full_data1[i] = data1[idx]
 
        mask2 = (period2_data['STATE_ABBREV'] == state_abbrev) & \
                (period2_data['COUNTY_NAME'] == county_name)
        data2 = period2_data[mask2]['EVENT_COUNT'].values
        years2 = period2_data[mask2]['YEAR'].values
        
        full_data2 = np.zeros(15)
        if len(data2) > 0:
            for i, year in enumerate(range(2011, 2026)):
                if year in years2:
                    idx = np.where(years2 == year)[0][0]
                    full_data2[i] = data2[idx]
        
        county_data[county_key] = {
            'period1': full_data1,
            'period2': full_data2,
            'combined': np.concatenate([full_data1, full_data2]),
            'state': state_abbrev,
            'county': county_name
        }

    observed_significant = 0
    observed_results = []
    
    for county_key, data in county_data.items():
        data1 = data['period1']
        data2 = data['period2']
        
        try:
            t_stat, p_value = stats.ttest_ind(data2, data1, equal_var=False)
            is_significant = p_value < alpha
            if is_significant:
                observed_significant += 1
            
            observed_results.append({
                'county': county_key,
                'state': data['state'],
                'county_name': data['county'],
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': is_significant,
                'mean_diff': np.mean(data2) - np.mean(data1)
            })
        except:
            observed_results.append({
                'county': county_key,
                'state': data['state'],
                'county_name': data['county'],
                't_statistic': np.nan,
                'p_value': 1.0,
                'significant': False,
                'mean_diff': np.mean(data2) - np.mean(data1)
            })
    
    np.random.seed(42)
    null_significant_counts = []
    
    for perm in tqdm(range(n_permutations), desc="Permutations"):
        permuted_significant = 0
        
        for county_key, data in county_data.items():
            combined = data['combined'].copy()
            np.random.shuffle(combined)
            
            perm_data1 = combined[:15]
            perm_data2 = combined[15:]

            t_stat, p_value = stats.ttest_ind(perm_data2, perm_data1, equal_var=False)
            if p_value < alpha:
                permuted_significant += 1
         
        
        null_significant_counts.append(permuted_significant)
    
    null_significant_counts = np.array(null_significant_counts)

    field_p_value = (null_significant_counts >= observed_significant).sum() / n_permutations
    null_mean = null_significant_counts.mean()
    null_std = null_significant_counts.std()
    ci_lower = np.percentile(null_significant_counts, 2.5)
    ci_upper = np.percentile(null_significant_counts, 97.5)
    
    return {
        'n_counties': n_counties,
        'observed_significant': observed_significant,
        'expected_significant': null_mean,
        'null_mean': null_mean,
        'null_std': null_std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'field_p_value': field_p_value,
        'is_field_significant': field_p_value < 0.05,
        'null_distribution': null_significant_counts,
        'observed_results': pd.DataFrame(observed_results)
    }
    
    return results

def plot_field_significance(field_sig_results, output_path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(field_sig_results['null_distribution'], bins=10, alpha=0.7, 
            color='gray', edgecolor='black', label='Null distribution (random)')

    ax.axvline(field_sig_results['expected_significant'], color='blue', 
              linestyle='--', 
               label=f"Expected: {field_sig_results['expected_significant']:.1f} counties")

    ax.axvline(field_sig_results['observed_significant'], color='red', 
                linestyle='--', 
               label=f"Observed: {field_sig_results['observed_significant']} counties \n P = {field_sig_results['field_p_value']:.2f}")

    ax.axvline(field_sig_results['ci_lower'], color='gray', 
               linewidth=1, linestyle=':', alpha=0.5)
    ax.axvline(field_sig_results['ci_upper'], color='gray', 
               linewidth=1, linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Number of Significant Counties (p<0.05)')
    ax.set_ylabel('Frequency')

    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    try:
        excel_file = "../../data/freezing_rain_events_county_llm.xlsx"
        output_dir = "./"

        df = load_and_process_data(excel_file)
        hotspot_file = "../../data/february_emerging_hotspots.xlsx"
        hotspot_df = pd.read_excel(hotspot_file)

        hotspot_counties = set(
            hotspot_df.apply(lambda x: f"{x['STATE_ABBREV']}_{x['COUNTY_NAME']}", axis=1)
        )
        time_periods = ['1996-2010', '2011-2025']
        yearly_data_dict = {}
        
        for period in time_periods:
            yearly_stats = calculate_county_statistics_by_year(df, period)
            yearly_data_dict[period] = yearly_stats

        ttest_results_all = perform_ttest_by_county(yearly_data_dict)

        total_events_30yr = df.groupby(['STATE_ABBREV', 'COUNTY_CLEAN']).size().reset_index(name='total_events_30yr')
        total_events_30yr.columns = ['STATE_ABBREV', 'COUNTY_NAME', 'total_events_30yr']

        ttest_results_all = ttest_results_all.merge(
            total_events_30yr,
            on=['STATE_ABBREV', 'COUNTY_NAME'],
            how='left'
        )
        ttest_results_all['total_events_30yr'] = ttest_results_all['total_events_30yr'].fillna(0)

        ttest_v1 = ttest_results_all[ttest_results_all['total_events_30yr'] > 1].copy()

        field_sig_v1 = field_significance_test(yearly_data_dict, ttest_v1, n_permutations=10000, 
                                                title_suffix=" (30yr events > 0)")
        
        output_v1 = Path(output_dir) / 'figureS5a.png'
        plot_field_significance(field_sig_v1, output_v1, 
                                title_suffix="\n(Counties with >0 events in 30 years)")
        

        ttest_v2 = ttest_results_all[ttest_results_all['period1_total'] > 5].copy()

        field_sig_v2 = field_significance_test(yearly_data_dict, ttest_v2, n_permutations=10000,
                                                title_suffix=" (1996-2010 avg > 1/yr)")

        output_v2 = Path(output_dir) / 'figureS5b.png'
        plot_field_significance(field_sig_v2, output_v2,
                                title_suffix="\n(Counties with >1 event/year in 1996-2010)")
        

        target_states = ['TX', 'LA', 'TN', 'AR','OK']
        ttest_v3 = ttest_results_all[ttest_results_all['STATE_ABBREV'].isin(target_states)].copy()
        for state in target_states:
            n_counties = len(ttest_v3[ttest_v3['STATE_ABBREV'] == state])

        field_sig_v3 = field_significance_test(yearly_data_dict, ttest_v3, n_permutations=10000,
                                                title_suffix=" (TX, LA, TN, AR)")

        output_v3 = Path(output_dir) / 'figureS5c.png'
        plot_field_significance(field_sig_v3, output_v3,
                                title_suffix="\n(Texas, Louisiana, Tennessee, Arkansas)")

        ttest_v4 = ttest_results_all[
            ttest_results_all.apply(
                lambda x: f"{x['STATE_ABBREV']}_{x['COUNTY_NAME']}" in hotspot_counties, 
                axis=1
            )
        ].copy()

        state_counts = ttest_v4['STATE_ABBREV'].value_counts().sort_index()

        field_sig_v4 = field_significance_test(yearly_data_dict, ttest_v4, n_permutations=10000,
                                                title_suffix=" (February Hotspots)")

        output_v4 = Path(output_dir) / 'figureS5d.png'
        plot_field_significance(field_sig_v4, output_v4,
                                title_suffix="\n(February Emerging Hotspot Counties)")

    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()