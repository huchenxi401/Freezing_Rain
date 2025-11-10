import pandas as pd
import requests
import json
import time
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any
import concurrent.futures
import threading
from pathlib import Path

class FreezingRainAnalyzer:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Freezing Rain Analyzer"
        }
        self.total_tokens = 0
        self.total_cost = 0.0
        self.lock = threading.Lock() 

        self.system_prompt = """You are an expert meteorologist analyzing freezing rain events. You will be given text descriptions of freezing rain events and must extract two key pieces of information:

1. ICE THICKNESS (in inches)
2. DAMAGE SEVERITY (Low/Medium/High)

ICE THICKNESS RULES:
Step 1: Identify all ice thickness mentioned in text above
- Look for ice thickness, ice accumulation, freezing rain amount, ice accretion measurements in INCHES only in the above text
- If "freezing rain", "Ice accretions", "Ice accumulations", "accumulation of ice", "ice" amount is given, that equals ice thickness
- Read the ENTIRE text for ALL ice thickness/freezing rain amount mentioned. Do not stop after finding the first one.
- Give higher priority to ice thickness values found in the EVENT NARRATIVE, even it is smaller than values in the EPISODE NARRATIVE, because it contains event-specific details
- Ignore: general "rainfall", "precipitation", "snow" unless explicitly stated as freezing rain
Step 2: For each mention, find values
- If single value (e.g., "0.5 inch") → use that value
- If a range of value is given, calculate the AVERAGE.
Range indicators include:
   • "X to Y": "0.50 to 1.00 inch" → 0.75
   • "X-Y": "0.25-0.40 inch" → 0.325  
   • "between X and Y": "between one quarter and one third of an inch" → 0.29
   • "from X to Y": "from 0.5 to 0.8 inches" → 0.65
   • "ranging X to Y": "ranging 0.3 to 0.5 inch" → 0.4
- A range of value is only one measurement. 
- WRONG: "0.5 to 1.0 inch" → use 1.0
STEP 3: Select final value
- If multiple SEPARATE measurements for different locations or times are mentioned, use the MAXIMUM value among all measurements:
   Example: "0.25 inches in locations near Interstate to between 1.0 to 1.5 inches near the southern Missouri border"
   → Calculate average of each range first: [0.25, 1.25]
   → Then use MAXIMUM: 1.25
- If only ONE measurement (even if it's a range) → use the value from Step 2

Key points:
- Common phrases: "quarter inch", "half inch", "1/4 inch", "0.25 inches", etc.
- "Less than/Up tp/More than 0.1 inch" = 0.1
- If no specific value is given, give the ice thickness based on the adjective or hazard levels
- If text says "light glaze", "coating", "trace ice", "little", "thin ice", "light ice", etc. = 0.1 inches
- If text says “heavy”, "substantial", “thick”, "significant" etc. = 0.5 to 1 inches, depends on you
- If indeed NO ice thickness found = 0.01 inches (default minimum)

DAMAGE SEVERITY RULES:
1. Start by assuming LOW unless clear evidence of higher severity
2. When in doubt between two levels, choose the LOWER level
LOW (default - inconvenient but not life-threatening):
- Power outages: <1,000 affected (people/customers/households/residents)
- Casualties: Minor injuries only, NO fatalities
- Tree damage: Scattered branches or limbs down
- Traffic: Icy roads, several accidents, no fatalities
- Closures: Schools/delays, minor transit disruption
- Scale words: "scattered", "several", "isolated", "some", "sporadic"

MEDIUM (significant losses and safety risks):
- Power outages: 1,000-10,000 affected
- Casualties: 1-2 fatalities OR multiple injuries (without mass casualties)
- Tree damage: Extensive damage or downed power lines (regional scale)
- Infrastructure: Major road closures, public service interruption
- Response: Emergency response activated (no state emergency declared)
- Structures: Damage to buildings mentioned

HIGH (catastrophic - disaster-level impacts):
- Power outages: >10,000 affected (regardless of duration)
- Casualties: ≥3 deaths OR ≥20 total injuries OR (deaths + injuries ≥15)
- Declarations: State of emergency, disaster declaration
- Infrastructure: Systemic failure (power grid, water, communications)
- Tree damage: Devastating forest destruction
- Structures: Large-scale building damage
- Response: External rescue forces required
- Economic: Disaster-level losses explicitly stated

- "scattered/several/sporadic" impacts = LOW
- "widespread/extensive/numerous/significant" impacts = MEDIUM or HIGH
- "downed power lines" (without scale) = MEDIUM
- "thousands affected/impacted" = MEDIUM or HIGH
- If text ONLY describes weather phenomena (rain, ice, snow distribution) without mentioning ANY impacts (outages, accidents, damage, closures) → classify as LOW by default
  Low: "widespread freezing rain" - not an impact indicator
  Low: "mixed precipitation across the state" - not an impact indicator
  Medium or High: "widespread outages" - valid impact indicator

Examples:
- "quarter inch of ice caused power outages to 5000 customers" → thickness: 0.25, damage: Medium
- "light glaze made roads slippery" → thickness: 0.01, damage: Low  
- "half inch of ice brought down power lines across the region, leaving 50,000 without power" → thickness: 0.5, damage: High
- "ice accumulations of 0.1 to 0.3 inches with scattered outages" → thickness: 0.2, damage: Low
-  "ice ranged from 0.25 inches in northern counties to between 0.5 and 0.75 inches in southern areas, causing widespread outages affecting 8,000 customers" → thickness: 0.625 (max of [0.25, mean(0.5,0.75)=0.625]), damage: Medium

You MUST return ONLY a valid JSON object in this exact format:
{"ice_thickness_inches": 0.25, "damage_severity": "Medium", "confidence": "high", "ice_source": "quarter inch of ice", "damage_source": "power outages to 5000 customers"}

Where:
- ice_thickness_inches: number (the thickness in inches)
- damage_severity: "Low", "Medium", or "High" 
- confidence: "high", "medium", or "low"
- ice_source: brief quote of text mentioning ice thickness
- damage_source: brief quote of text mentioning damage/impacts"""

    def call_api(self, event_data: Dict[str, Any], max_retries: int = 5) -> Optional[Dict]:
        
        user_message = f"""Analyze this freezing rain event for ice thickness and damage severity:

EVENT_ID: {event_data['EVENT_ID']}
EPISODE_ID: {event_data['EPISODE_ID']}
LOCATION: {event_data['STATE']}, {event_data['COUNTY']}
TIME: {event_data['BEGIN_DATE_TIME_UTC']} to {event_data['END_DATE_TIME_UTC']}

EPISODE NARRATIVE:
{event_data.get('EPISODE_NARRATIVE', 'No episode narrative available')}

EVENT NARRATIVE:
{event_data.get('EVENT_NARRATIVE', 'No event narrative available')}

Extract ice thickness in inches and assess damage severity (Low/Medium/High) from the above text."""
        
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 200,
            "temperature": 0,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    usage = result.get('usage', {})
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
                    
                    with self.lock:
                        self.total_tokens += total_tokens
                        input_cost = (prompt_tokens / 1_000_000) * 0.15
                        output_cost = (completion_tokens / 1_000_000) * 0.6
                        self.total_cost += (input_cost + output_cost)

                    parsed_data = self.parse_json_response(content)
                    if parsed_data:
                        return parsed_data
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                    
                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                    
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return None

    def parse_json_response(self, content: str) -> Optional[Dict]:
        try:
            json_data = json.loads(content)
            return self.validate_json_data(json_data)
        except json.JSONDecodeError:
            pass
        
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                json_data = json.loads(json_str)
                return self.validate_json_data(json_data)
        except (json.JSONDecodeError, ValueError):
            pass
        
        try:
            thickness_match = re.search(r'"ice_thickness_inches":\s*([0-9.]+)', content)
            damage_match = re.search(r'"damage_severity":\s*"(Low|Medium|High)"', content, re.IGNORECASE)
            confidence_match = re.search(r'"confidence":\s*"(high|medium|low)"', content, re.IGNORECASE)
            
            if thickness_match and damage_match:
                return {
                    'ice_thickness_inches': float(thickness_match.group(1)),
                    'damage_severity': damage_match.group(1).capitalize(),
                    'confidence': confidence_match.group(1).lower() if confidence_match else 'medium',
                    'ice_source': 'Extracted from partial response',
                    'damage_source': 'Extracted from partial response'
                }
        except (ValueError, AttributeError):
            pass
        
        return None

    def validate_json_data(self, data: Dict) -> Optional[Dict]:
        try:
            if 'ice_thickness_inches' not in data or 'damage_severity' not in data:
                return None
            
            validated_data = {
                'ice_thickness_inches': float(data['ice_thickness_inches']),
                'damage_severity': str(data['damage_severity']).capitalize(),
                'confidence': str(data.get('confidence', 'medium')).lower(),
                'ice_source': str(data.get('ice_source', '')),
                'damage_source': str(data.get('damage_source', ''))
            }
            
            if validated_data['ice_thickness_inches'] < 0 or validated_data['ice_thickness_inches'] > 10:
                validated_data['ice_thickness_inches'] = max(0.01, min(10.0, validated_data['ice_thickness_inches']))
            
            if validated_data['damage_severity'] not in ['Low', 'Medium', 'High']:
                validated_data['damage_severity'] = 'Low'
            
            if validated_data['confidence'] not in ['high', 'medium', 'low']:
                validated_data['confidence'] = 'medium'
            
            return validated_data
            
        except (ValueError, TypeError, KeyError):
            return None

    def process_batch(self, df_batch: pd.DataFrame, batch_id: int, output_dir: str) -> Dict:

        results = []
        success_count = 0
        
        for index, row in df_batch.iterrows():
            event_data = {
                'EVENT_ID': row['EVENT_ID'],
                'EPISODE_ID': row['EPISODE_ID'],
                'STATE': row['STATE'],
                'COUNTY': row['COUNTY'],
                'BEGIN_DATE_TIME_UTC': row['BEGIN_DATE_TIME_UTC'],
                'END_DATE_TIME_UTC': row['END_DATE_TIME_UTC'],
                'EPISODE_NARRATIVE': str(row['EPISODE_NARRATIVE']) if pd.notna(row['EPISODE_NARRATIVE']) else '',
                'EVENT_NARRATIVE': str(row['EVENT_NARRATIVE']) if pd.notna(row['EVENT_NARRATIVE']) else ''
            }

            analysis_result = self.call_api(event_data)

            result_row = {
                'ORIGINAL_INDEX': row.name,
                'EPISODE_ID': row['EPISODE_ID'],
                'EVENT_ID': row['EVENT_ID'],
                'STATE': row['STATE'],
                'COUNTY': row['COUNTY'],
                'BEGIN_DATE_TIME_UTC': row['BEGIN_DATE_TIME_UTC'],
                'END_DATE_TIME_UTC': row['END_DATE_TIME_UTC'],
                'EPISODE_NARRATIVE': event_data['EPISODE_NARRATIVE'],
                'EVENT_NARRATIVE': event_data['EVENT_NARRATIVE']
            }
            
            if analysis_result:
                result_row.update({
                    'ICE_THICKNESS_INCHES': analysis_result.get('ice_thickness_inches', 0.01),
                    'DAMAGE_SEVERITY': analysis_result.get('damage_severity', 'Low'),
                    'CONFIDENCE': analysis_result.get('confidence', 'medium'),
                    'ICE_SOURCE': analysis_result.get('ice_source', ''),
                    'DAMAGE_SOURCE': analysis_result.get('damage_source', ''),
                    'API_SUCCESS': True
                })
                success_count += 1
            else:
                result_row.update({
                    'ICE_THICKNESS_INCHES': 0.01,
                    'DAMAGE_SEVERITY': 'Low',
                    'CONFIDENCE': 'failed',
                    'ICE_SOURCE': 'API调用失败',
                    'DAMAGE_SOURCE': 'API调用失败',
                    'API_SUCCESS': False
                })
            
            results.append(result_row)

            time.sleep(0.3)
            
        df_results = pd.DataFrame(results)
        output_file = Path(output_dir) / f"batch_{batch_id:03d}_results.xlsx"
        df_results.to_excel(output_file, index=False)
        
        return {
            'batch_id': batch_id,
            'total_events': len(df_batch),
            'success_count': success_count,
            'output_file': str(output_file)
        }


def parallel_process_all_events(input_file: str, output_dir: str, api_key: str, 
                               batch_size: int = 500, max_workers: int = 8):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        return

    required_columns = ['EPISODE_ID', 'EVENT_ID', 'STATE', 'COUNTY', 
                       'BEGIN_DATE_TIME_UTC', 'END_DATE_TIME_UTC', 
                       'EPISODE_NARRATIVE', 'EVENT_NARRATIVE']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return
    
    total_batches = (len(df) + batch_size - 1) // batch_size

    analyzer = FreezingRainAnalyzer(api_key)

    batches = []
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()
        batches.append((batch_df, i, output_dir))

    start_time = datetime.now()
    
    batch_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(analyzer.process_batch, batch_df, batch_id, output_dir): batch_id 
            for batch_df, batch_id, output_dir in batches
        }

        for future in concurrent.futures.as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                result = future.result()
                batch_results.append(result)
            except Exception as exc:
                print(f"{batch_id} failed: {exc}")
    
    end_time = datetime.now()
    duration = end_time - start_time

    total_events = sum(r['total_events'] for r in batch_results)
    total_success = sum(r['success_count'] for r in batch_results)
    
    merge_all_results(output_dir, input_file)


def merge_all_results(output_dir: str, original_file: str):

    
    output_path = Path(output_dir)
    result_files = sorted(list(output_path.glob("batch_*_results.xlsx")))
    
    if not result_files:
        return
    
    all_data = []
    for file in result_files:
        try:
            df = pd.read_excel(file)
            all_data.append(df)
        except Exception as e:
            print(f"{file.name} failed: {e}")
    
    if not all_data:
        return

    final_df = pd.concat(all_data, ignore_index=True)
    
    final_df = final_df.sort_values('ORIGINAL_INDEX').reset_index(drop=True)

    if 'ORIGINAL_INDEX' in final_df.columns:
        final_df = final_df.drop('ORIGINAL_INDEX', axis=1)

    final_output = output_path / "freezing_rain_complete_analysis.xlsx"
    final_df.to_excel(final_output, index=False)
    

def main():
    API_KEY = "myapi"  # replaced by your api
    INPUT_FILE = "../data/freezing_rain_events.xlsx"
    OUTPUT_DIR = "../../data/"
    BATCH_SIZE = 500      
    MAX_WORKERS = 8       

    if API_KEY == "your_api_key_here":
        print("please set your OpenRouter API")
        API_KEY = input("input your OpenRouter API: ").strip()
        if not API_KEY:
            print("quit")
            return
    
    if not os.path.exists(INPUT_FILE):
        print(f"wrong: {INPUT_FILE}")
        return

    parallel_process_all_events(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        api_key=API_KEY,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS
    )


if __name__ == "__main__":
    main()