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

def main():
    try:
        df = pd.read_excel('../../data/freezing_rain_events_county_llm.xlsx', 
                          usecols=['EPISODE_ID', 'BEGIN_YEAR', 'BEGIN_MONTH', 'DURATION_HOURS'])
    except FileNotFoundError:
        return
    except Exception as e:
        return

    df = df[(df['BEGIN_YEAR'] >= 1996) & (df['BEGIN_YEAR'] <= 2025)]
    df = df[(df['DURATION_HOURS'] >= 0) & (df['DURATION_HOURS'] <= 144)]
    yearly_episodes = df.groupby('BEGIN_YEAR')['EPISODE_ID'].nunique().reset_index()
    yearly_episodes.columns = ['Year', 'Episode_Count']

    all_years = pd.DataFrame({'Year': range(1996, 2026)})
    yearly_episodes = all_years.merge(yearly_episodes, on='Year', how='left')
    yearly_episodes['Episode_Count'] = yearly_episodes['Episode_Count'].fillna(0)

    x_vals = yearly_episodes['Year'].values
    y_vals = yearly_episodes['Episode_Count'].values
    unique_years = x_vals
    y_vals = moving_average(y_vals)
    model = LinearRegression()
    model.fit(x_vals.reshape(-1, 1), y_vals)
    trend_line = model.predict(x_vals.reshape(-1, 1))
    slope_weak, intercept_weak, r_weak, p_weak, _ = stats.linregress(x_vals, y_vals)
    slope, ci_lower_weak, ci_upper_weak, p_value_weak = calculate_confidence_interval(x_vals, y_vals)
    total_events = yearly_episodes['Episode_Count'].sum()
    period1_events = yearly_episodes[(yearly_episodes['Year'] >= 1996) & (yearly_episodes['Year'] <= 2010)]['Episode_Count'].sum()
    period2_events = yearly_episodes[(yearly_episodes['Year'] >= 2011) & (yearly_episodes['Year'] <= 2025)]['Episode_Count'].sum()

    ci_low, ci_high = calculate_bootstrap_ci(x_vals, y_vals, unique_years, 
                                           n_bootstrap=10000, confidence=0.95)
    plt.figure(figsize=(6, 4))

    plt.plot(x_vals, y_vals, '-', color='black',
             markersize=6, alpha=0.8)
    
    plt.plot(x_vals, trend_line, '--', color='black',  
             label=f'R:{r_weak:.2f}')

    plt.plot(x_vals, ci_low, '--', color='.7', alpha=0.8)
    plt.plot(x_vals, ci_high, '--', color='.7', alpha=0.8)
    plt.xlabel('Year')
    plt.ylabel('Freezing Rain Events Yr⁻¹')
    plt.legend(fontsize=10)  
    plt.xticks(range(1996, 2026, 5))


    plt.savefig('./figure/freezing_rain_yearly_trend.png', 
                   dpi=300)


if __name__ == "__main__":
    main()