import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data(excel_file="../../data/freezing_rain_events_county_llm.xlsx"):

    df = pd.read_excel(excel_file, usecols=['EPISODE_ID', 'BEGIN_YEAR', 'BEGIN_MONTH', 'DURATION_HOURS'])
    df_clean = df.dropna(subset=['EPISODE_ID', 'BEGIN_MONTH', 'DURATION_HOURS'])
    df_clean = df_clean[df_clean['DURATION_HOURS'] <= 144] 

    df_clean = df_clean[(df_clean['BEGIN_YEAR'] >= 1996) & (df_clean['BEGIN_YEAR'] <= 2025)]

    df_clean['TIME_PERIOD'] = df_clean['BEGIN_YEAR'].apply(
        lambda year: '1996-2010' if 1996 <= year <= 2010 else '2011-2025'
    )

    
    return df_clean

def get_episode_first_event(df):
    df_sorted = df.sort_values(['EPISODE_ID', 'BEGIN_YEAR', 'BEGIN_MONTH'])
    first_events = df_sorted.groupby('EPISODE_ID').first().reset_index()
    
    return first_events

def calculate_yearly_episode_counts(df):

    winter_months = [10, 11, 12, 1, 2, 3, 4]

    winter_data = df[df['BEGIN_MONTH'].isin(winter_months)]

    yearly_counts = winter_data.groupby(['TIME_PERIOD', 'BEGIN_YEAR', 'BEGIN_MONTH']).size().reset_index(name='EPISODE_COUNT')

    periods = [('1996-2010', range(1996, 2011)), ('2011-2025', range(2011, 2026))]
    
    full_data = []
    for period_name, years in periods:
        for year in years:
            for month in winter_months:
                full_data.append({
                    'TIME_PERIOD': period_name,
                    'BEGIN_YEAR': year,
                    'BEGIN_MONTH': month,
                    'EPISODE_COUNT': 0
                })
    
    full_df = pd.DataFrame(full_data)

    result = full_df.set_index(['TIME_PERIOD', 'BEGIN_YEAR', 'BEGIN_MONTH'])
    yearly_counts_indexed = yearly_counts.set_index(['TIME_PERIOD', 'BEGIN_YEAR', 'BEGIN_MONTH'])

    result.loc[yearly_counts_indexed.index, 'EPISODE_COUNT'] = yearly_counts_indexed['EPISODE_COUNT']
    result = result.reset_index()
    
    return result

def calculate_monthly_statistics(yearly_data):
    
    winter_months = [10, 11, 12, 1, 2, 3, 4]
    results = []
    
    for month in winter_months:
        period1_data = yearly_data[
            (yearly_data['TIME_PERIOD'] == '1996-2010') & 
            (yearly_data['BEGIN_MONTH'] == month)
        ]['EPISODE_COUNT'].values
        
        period2_data = yearly_data[
            (yearly_data['TIME_PERIOD'] == '2011-2025') & 
            (yearly_data['BEGIN_MONTH'] == month)
        ]['EPISODE_COUNT'].values
        period1_mean = np.mean(period1_data)
        period1_std = np.std(period1_data, ddof=1) 
        period1_se = period1_std / np.sqrt(len(period1_data))  
        
        period2_mean = np.mean(period2_data)
        period2_std = np.std(period2_data, ddof=1)
        period2_se = period2_std / np.sqrt(len(period2_data))

        t_stat, p_value = stats.ttest_ind(period2_data, period1_data, equal_var=False)
        
        results.append({
            'month': month,
            'period1_mean': period1_mean,
            'period1_se': period1_se,
            'period2_mean': period2_mean,
            'period2_se': period2_se,
            't_statistic': t_stat
        })
        
    
    return pd.DataFrame(results)

def create_monthly_comparison_chart(monthly_stats, output_dir="D:/py/freezing rain/freezing rain analysis/figure"):
    plt.figure(figsize=(6, 4))
    
    months = [10, 11, 12, 1, 2, 3, 4]
    month_names = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    

    legend_color_light = 'brown'  
    legend_color_dark = 'green'   
    month_colors = ['#b8ddf0', '#7fb3d3', '#4575b4', '#762a83', '#af8dc3', '#f7e6f7', '#F5F5DC']
    
    x_positions = np.arange(len(months))
    bar_width = 0.35
    
    period1_means = []
    period1_errors = []
    period2_means = []
    period2_errors = []
    
    for month in months:
        month_data = monthly_stats[monthly_stats['month'] == month].iloc[0]
        period1_means.append(month_data['period1_mean'])
        period1_errors.append(month_data['period1_se'])
        period2_means.append(month_data['period2_mean'])
        period2_errors.append(month_data['period2_se'])

    bars1 = plt.bar(x_positions - bar_width/2, period1_means, bar_width, 
                   color='brown', alpha=0.7, 
                   label='1996-2010', edgecolor='black', linewidth=0.5,
                   yerr=period1_errors, capsize=3, error_kw={'color': 'black', 'alpha': 0.7})
    
    bars2 = plt.bar(x_positions + bar_width/2, period2_means, bar_width, 
                   color='green', alpha=0.7, 
                   label='2011-2025', edgecolor='black', linewidth=0.5,
                   yerr=period2_errors, capsize=3, error_kw={'color': 'black', 'alpha': 0.7})

    max_heights = [max(period1_means[i] + period1_errors[i], 
                      period2_means[i] + period2_errors[i]) for i in range(len(months))]
    
    for i, (month) in enumerate(zip(months)):
        if i==2 or i==4:
            star_height = max_heights[i] + max(max_heights) * 0.05
            plt.text(x_positions[i], star_height, '*', 
                    ha='center', va='bottom', fontsize=16, fontweight='bold')
            
    def add_value_labels(bars, values, errors):
        for bar, value, error in zip(bars, values, errors):
            if value > 0:  
                height = bar.get_height() + error
                plt.text(bar.get_x() + bar.get_width()/2., height + max(max_heights) * 0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1, period1_means, period1_errors)
    add_value_labels(bars2, period2_means, period2_errors)

    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Freezing Rain Events Yr⁻¹', fontsize=12)

    plt.xticks(x_positions, month_names)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=legend_color_light, edgecolor='black', alpha=0.7, label='1996-2010'),
        Patch(facecolor=legend_color_dark, edgecolor='black', alpha=0.7, label='2011-2025')
    ]
    plt.legend(handles=legend_elements, fontsize=10)

    max_y = max(max_heights) * 1.15  
    plt.ylim(0, max_y)

    plt.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()


    output_file = "./figure/monthly_episode_comparison_with_stats.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    

def main():

    try:
        excel_file = "../../data/freezing_rain_events_county_llm.xlsx"
        output_dir = "./figure"
        
        df = load_and_process_data(excel_file)

        first_events = get_episode_first_event(df)

        yearly_counts = calculate_yearly_episode_counts(first_events)

        monthly_stats = calculate_monthly_statistics(yearly_counts)
        
        create_monthly_comparison_chart(monthly_stats, output_dir)
        
        
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()