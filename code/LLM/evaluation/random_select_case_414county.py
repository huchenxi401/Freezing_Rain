import pandas as pd
import os
from datetime import datetime
import numpy as np

def stratified_sample_by_county():
    input_file = "../../../data/freezing_rain_events_county_llm.xlsx"
    hotspot_file = "../../../data/february_emerging_hotspots.xlsx"
    output_dir = "./"
    output_file = os.path.join(output_dir, "freezing_rain_stratified_sample_414_counties.xlsx")
    
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
    
    ABBREV_TO_STATE = {v: k for k, v in STATE_ABBREV.items()}
    try:
        df_hotspots = pd.read_excel(hotspot_file)
        if 'STATE_ABBREV' not in df_hotspots.columns or 'COUNTY_NAME' not in df_hotspots.columns:
            df_hotspots.columns = ['STATE_ABBREV', 'COUNTY_NAME'] + list(df_hotspots.columns[2:])

        df_complete = pd.read_excel(input_file)
        def standardize_county_name(name):
            if pd.isna(name):
                return ''
            name = str(name).upper().strip()
            if name.endswith(' COUNTY'):
                name = name.replace(' COUNTY', '')
            return name

        df_hotspots['COUNTY_CLEAN'] = df_hotspots['COUNTY_NAME'].apply(standardize_county_name)
        df_hotspots['STATE_FULL'] = df_hotspots['STATE_ABBREV'].map(ABBREV_TO_STATE)
        
        df_complete['STATE_UPPER'] = df_complete['STATE'].str.upper().str.strip()
        df_complete['COUNTY_CLEAN'] = df_complete['COUNTY'].apply(standardize_county_name)

        np.random.seed(42)
        
        sampled_records = []
        counties_with_data = []
        counties_without_data = []
        county_data_counts = {}
        
        for idx, row in df_hotspots.iterrows():
            state_abbrev = row['STATE_ABBREV']
            county_name = row['COUNTY_CLEAN']
            state_full = row['STATE_FULL']
            
            if pd.isna(state_full):
                counties_without_data.append(f"{state_abbrev}-{county_name}")
                continue

            mask = (df_complete['STATE_UPPER'] == state_full) & \
                   (df_complete['COUNTY_CLEAN'] == county_name)
            
            county_data = df_complete[mask]
            county_data_counts[f"{state_abbrev}-{county_name}"] = len(county_data)
            
            if len(county_data) > 0:
                sampled_record = county_data.sample(n=1, random_state=42+idx)
                sampled_records.append(sampled_record)
                counties_with_data.append(f"{state_abbrev}-{county_name}")
                
            else:
                counties_without_data.append(f"{state_abbrev}-{county_name}")

        if len(sampled_records) > 0:
            df_sample = pd.concat(sampled_records, ignore_index=True)
        else:
            return None

        
        os.makedirs(output_dir, exist_ok=True)
        df_sample.to_excel(output_file, index=False)

        
        if 'API_SUCCESS' in df_sample.columns:
            success_rate = df_sample['API_SUCCESS'].mean() * 100

        if 'ICE_THICKNESS_INCHES' in df_sample.columns:
            ice_data = df_sample['ICE_THICKNESS_INCHES'].dropna()

        if 'DAMAGE_SEVERITY' in df_sample.columns:
            sample_damage_counts = df_sample['DAMAGE_SEVERITY'].value_counts()
            for severity, count in sample_damage_counts.items():
                percentage = count / len(df_sample) * 100
       
        
        return output_file
        
    except FileNotFoundError as e:
        return None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def main():
   
    result_file = stratified_sample_by_county()
    


if __name__ == "__main__":
    main()