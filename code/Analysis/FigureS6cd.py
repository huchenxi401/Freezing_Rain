import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy import stats

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

def standardize_damage_severity(severity):
    if pd.isna(severity):
        return None
    
    severity = str(severity).strip().upper()
    if severity in ['LOW', 'L']:
        return 'Low'
    elif severity in ['MEDIUM', 'MED', 'M']:
        return 'Medium'
    elif severity in ['HIGH', 'H']:
        return 'High'
    else:
        return None

def process_monthly_damage_severity_data(df, target_month, severity_level):
    monthly_df = df[df['BEGIN_MONTH'] == target_month].copy()
    severity_df = monthly_df[monthly_df['DAMAGE_SEVERITY_CLEAN'] == severity_level].copy()

    if len(severity_df) == 0:
        return None

    episode_counts = severity_df.groupby(['EPISODE_ID', 'BEGIN_YEAR']).size().reset_index(name='Event_Count')
    episode_counts.columns = ['EPISODE_ID', 'Year', 'Event_Count']

    yearly_episode_count = episode_counts.groupby('Year').size().reset_index(name='Count_Episodes')

    all_years = pd.DataFrame({'Year': range(1996, 2026)})
    yearly_episode_count = all_years.merge(yearly_episode_count, on='Year', how='left')
    yearly_episode_count['Count_Episodes'] = yearly_episode_count['Count_Episodes'].fillna(0)
    
    return yearly_episode_count, episode_counts, severity_df

def create_monthly_damage_severity_trend_plot(yearly_episode_count, target_month, severity_level, output_dir="./figure"):
    if yearly_episode_count is None or len(yearly_episode_count) < 3:
        return None

    x_vals = yearly_episode_count['Year'].values
    y_vals = yearly_episode_count['Count_Episodes'].values
    unique_years = x_vals
    y_vals_smoothed = moving_average(y_vals)

    model = LinearRegression()
    model.fit(x_vals.reshape(-1, 1), y_vals_smoothed)
    trend_line = model.predict(x_vals.reshape(-1, 1))
    slope_weak, intercept_weak, r_weak, p_weak, _ = stats.linregress(x_vals, y_vals_smoothed)
    slope, ci_lower_weak, ci_upper_weak, p_value_weak = calculate_confidence_interval(x_vals, y_vals_smoothed)

    print(f"trend: {slope:.3f} events/yr (95% CI: {ci_lower_weak:.3f} to {ci_upper_weak:.3f}, P={p_value_weak:.4f})")

    
    ci_low, ci_high = calculate_bootstrap_ci(x_vals, y_vals_smoothed, unique_years, 
                                               n_bootstrap=10000, confidence=0.95)

    plt.figure(figsize=(6, 4))
    
    plt.plot(x_vals, y_vals_smoothed, '-', color='black',
             markersize=6, alpha=0.8)
    
    if severity_level == 'High':
        plt.plot(x_vals, trend_line, '--', color='black',  
             label=f'R: {r_weak:.2f}')
    else:
        plt.plot(x_vals, trend_line, '--', color='black',  
             label=f'R: {r_weak:.2f}**')

    plt.plot(x_vals, ci_low, '--', color='.7', alpha=0.8)
    plt.plot(x_vals, ci_high, '--', color='.7', alpha=0.8)

    plt.xlabel('Year',fontsize=12)
    if target_month==2:
        plt.ylabel(f'February Events Yr⁻² ({severity_level})', fontsize=12)
    plt.legend(fontsize=12)  

    plt.xticks(range(1996, 2026, 5))
    
    plt.tight_layout()

    output_file = f"{output_dir}/{severity_level.lower()}_damage_trend_month{target_month:02d}.png"
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        return output_file
    except Exception as e:
        plt.close()
        return None



def main():
    try:
        df = pd.read_excel('../../data/freezing_rain_events_county_llm.xlsx', 
                          usecols=['EPISODE_ID', 'BEGIN_YEAR', 'BEGIN_MONTH', 'DURATION_HOURS', 'DAMAGE_SEVERITY'])
    except FileNotFoundError:
        return
    except Exception as e:
        return
    
    df = df[(df['BEGIN_YEAR'] >= 1996) & (df['BEGIN_YEAR'] <= 2025)]

    df['DAMAGE_SEVERITY_CLEAN'] = df['DAMAGE_SEVERITY'].apply(standardize_damage_severity)

    df = df.dropna(subset=['DAMAGE_SEVERITY_CLEAN'])

    df = df[(df['DURATION_HOURS'] >= 0) & (df['DURATION_HOURS'] <= 144)]

    df = df[df['BEGIN_MONTH'].isin([2])]
    target_months = [2]
    severity_levels = ['Medium', 'High'] 
    output_dir = "./figure"
    
    all_output_files = []
    
    for severity_level in severity_levels:
        for target_month in target_months:

            result = process_monthly_damage_severity_data(df, target_month, severity_level)
            
            if result is None:
                continue
                
            yearly_episode_count, episode_counts, severity_df = result

            output_file = create_monthly_damage_severity_trend_plot(yearly_episode_count, target_month, severity_level, output_dir)
            if output_file:
                all_output_files.append(output_file)
            
if __name__ == "__main__":
    main()