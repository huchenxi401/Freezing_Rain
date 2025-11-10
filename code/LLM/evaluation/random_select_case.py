import pandas as pd
import os
from datetime import datetime
import numpy as np

def random_sample_freezing_rain_data():
    input_file = "../../../data/freezing_rain_events_county_llm.xlsx"
    output_dir = "./"
    output_file = os.path.join(output_dir, "freezing_rain_random_sample_500.xlsx")

    try:
        df = pd.read_excel(input_file)

        expected_columns = [
            'EPISODE_ID', 'EVENT_ID', 'STATE', 'COUNTY', 
            'BEGIN_DATE_TIME_UTC', 'END_DATE_TIME_UTC',
            'EPISODE_NARRATIVE', 'EVENT_NARRATIVE', 
            'ICE_THICKNESS_INCHES', 'DAMAGE_SEVERITY', 
            'CONFIDENCE', 'ICE_SOURCE', 'DAMAGE_SOURCE', 'API_SUCCESS'
        ]
        
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if 'DAMAGE_SEVERITY' in df.columns:
            damage_counts = df['DAMAGE_SEVERITY'].value_counts()
 

        sample_size = 500
        if len(df) < sample_size:
            df_sample = df.copy()
        else:
            np.random.seed(42)
            df_sample = df.sample(n=sample_size, random_state=42).copy()

        df_sample = df_sample.reset_index(drop=True)

        os.makedirs(output_dir, exist_ok=True)

        df_sample.to_excel(output_file, index=False)
        
      
        
        return output_file
        
    except FileNotFoundError:
        return None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def main():
    result_file = random_sample_freezing_rain_data()

if __name__ == "__main__":
    main()