import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import aiofiles
import logging

@dataclass
class ClimateStats:
    collection_date: str
    spatial_coverage: Dict[str, float]
    temporal_coverage: Dict[str, str]
    parameters: Dict[str, Dict[str, float]]
    resolution: Dict[str, str]
    source_details: Dict[str, str]

class OpenMeteoCollector:
    def __init__(self):
        """Initialize the OpenMeteo data collector"""
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.params = [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "windspeed_10m_max",
            "windgusts_10m_max",
            "winddirection_10m_dominant",
            "shortwave_radiation_sum"
        ]

    def get_grid_points(self, bbox: Tuple[float, float, float, float], spacing: float = 0.5) -> List[Tuple[float, float]]:
        """Create a grid of points within the bounding box"""
        lats = np.arange(bbox[1], bbox[3], spacing)
        lons = np.arange(bbox[0], bbox[2], spacing)
        return [(round(lat, 3), round(lon, 3)) for lat in lats for lon in lons]

    def validate_dates(self, start_date: datetime, end_date: datetime) -> Tuple[datetime, datetime]:
        """Validate and adjust dates to ensure they're within valid ranges"""
        today = datetime.now()
        min_date = datetime(1940, 1, 1)
        
        if end_date >= today:
            end_date = today - timedelta(days=1)
            self.logger.warning(f"Adjusted end date to yesterday: {end_date.strftime('%Y-%m-%d')}")
        
        if start_date < min_date:
            start_date = min_date
            self.logger.warning(f"Adjusted start date to minimum available: {start_date.strftime('%Y-%m-%d')}")
        
        if start_date > end_date:
            start_date = end_date - timedelta(days=30)
            self.logger.warning(f"Start date was after end date. Adjusted to 30 days before end date: {start_date.strftime('%Y-%m-%d')}")
        
        return start_date, end_date

    async def fetch_point_data(self, 
                             session: aiohttp.ClientSession, 
                             lat: float, 
                             lon: float, 
                             start_date: datetime, 
                             end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data for a single point with retries"""
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'daily': ','.join(self.params),
            'timezone': 'GMT'
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 429:  # Rate limit
                        wait_time = int(response.headers.get('Retry-After', 5))
                        self.logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    response.raise_for_status()
                    data = await response.json()
                    
                    if 'daily' in data:
                        df = pd.DataFrame(data['daily'])
                        df['latitude'] = lat
                        df['longitude'] = lon
                        return df
                    
                    return None
                    
            except aiohttp.ClientError as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to fetch data for ({lat}, {lon}): {str(e)}")
                    return None
                await asyncio.sleep(2 ** attempt) 
                
            except Exception as e:
                self.logger.error(f"Unexpected error for ({lat}, {lon}): {str(e)}")
                return None
        
        return None

    async def collect_climate_data(self, 
                                 bbox: Tuple[float, float, float, float],
                                 start_date: datetime,
                                 end_date: datetime,
                                 spacing: float = 0.5) -> Optional[pd.DataFrame]:
        """Collect climate data for the entire region"""
        start_date, end_date = self.validate_dates(start_date, end_date)
        
        try:
            grid_points = self.get_grid_points(bbox, spacing)
            self.logger.info(f"Collecting data for {len(grid_points)} grid points...")
            
            async with aiohttp.ClientSession() as session:
                all_data = []
                chunk_size = 10 
                
                for i in range(0, len(grid_points), chunk_size):
                    chunk = grid_points[i:i + chunk_size]
                    tasks = []
                    
                    for lat, lon in chunk:
                        task = self.fetch_point_data(session, lat, lon, start_date, end_date)
                        tasks.append(task)
                    
                    results = []
                    for f in tqdm(asyncio.as_completed(tasks), 
                                total=len(tasks),
                                desc=f"Processing chunk {i//chunk_size + 1}/{(len(grid_points)-1)//chunk_size + 1}"):
                        df = await f
                        if df is not None:
                            results.append(df)
                    
                    all_data.extend(results)
                    await asyncio.sleep(1)  
            
            if all_data:
                # Combine all data
                combined_df = pd.concat(all_data, ignore_index=True)
                
                column_mapping = {
                    'temperature_2m_max': 'TMAX',
                    'temperature_2m_min': 'TMIN',
                    'temperature_2m_mean': 'TMEAN',
                    'precipitation_sum': 'PRCP',
                    'rain_sum': 'RAIN',
                    'snowfall_sum': 'SNOW',
                    'windspeed_10m_max': 'WMAX',
                    'windgusts_10m_max': 'GUST',
                    'winddirection_10m_dominant': 'WDIR',
                    'shortwave_radiation_sum': 'SRAD'
                }
                
                combined_df = combined_df.rename(columns=column_mapping)
                return combined_df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error collecting climate data: {str(e)}")
            return None

    async def save_climate_data(self, 
                              df: pd.DataFrame, 
                              output_dir: Path, 
                              bbox: Tuple[float, float, float, float],
                              start_date: datetime,
                              end_date: datetime) -> bool:
        """Save collected climate data and metadata"""
        try:
            if df is None or df.empty:
                self.logger.error("No data to save")
                return False

            stats = ClimateStats(
                collection_date=datetime.now().isoformat(),
                spatial_coverage={
                    'west': bbox[0],
                    'south': bbox[1],
                    'east': bbox[2],
                    'north': bbox[3]
                },
                temporal_coverage={
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                parameters={},
                resolution={
                    'spatial': '0.5 degrees',
                    'temporal': 'daily'
                },
                source_details={
                    'name': 'OpenMeteo Historical Weather API',
                    'version': 'v1',
                    'url': 'https://open-meteo.com/',
                    'citation': 'Data provided by OpenMeteo.com'
                }
            )

            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col not in ['latitude', 'longitude']:
                    stats.parameters[col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std())
                    }

            output_dir.mkdir(parents=True, exist_ok=True)
            
            data_file = output_dir / "weather_data.csv.gz"
            df.to_csv(data_file, index=False, compression='gzip')
            
            stats_file = output_dir / "weather_stats.json"
            async with aiofiles.open(stats_file, 'w') as f:
                await f.write(json.dumps(asdict(stats), indent=2))
            
            monthly_df = df.copy()
            monthly_df['month'] = pd.to_datetime(monthly_df['time']).dt.to_period('M')
            
            monthly_summary = monthly_df.groupby('month').agg({
                'TMAX': ['mean', 'max'],
                'TMIN': ['mean', 'min'],
                'PRCP': 'sum',
                'WMAX': 'mean',
                'SRAD': 'mean'
            }).reset_index()
            
            summary_file = output_dir / "monthly_summary.csv"
            monthly_summary.to_csv(summary_file, index=False)
            
            self.logger.info(f"Successfully saved climate data to {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving climate data: {str(e)}")
            return False

async def main():
    test_areas = {
        "California": (-124.5, 36.0, -119.0, 42.0)
    }

    base_dir = Path("collected_data")
    base_dir.mkdir(exist_ok=True)

    end_date = datetime.now() - timedelta(days=1) 
    start_date = end_date - timedelta(days=365)    

    collector = OpenMeteoCollector()
    
    for state_name, bbox in test_areas.items():
        try:
            state_dir = base_dir / state_name.lower()
            state_dir.mkdir(exist_ok=True)
            weather_dir = state_dir / "weather_data"
            weather_dir.mkdir(exist_ok=True)
            
            collector.logger.info(f"\nProcessing historical climate data for {state_name}")
            collector.logger.info(f"Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            df = await collector.collect_climate_data(bbox, start_date, end_date)
            
            if df is not None and not df.empty:
                await collector.save_climate_data(df, weather_dir, bbox, start_date, end_date)
                
                collector.logger.info("\nData Collection Summary:")
                collector.logger.info(f"Total records: {len(df)}")
                collector.logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
                collector.logger.info(f"Grid points: {df[['latitude', 'longitude']].drop_duplicates().shape[0]}")
                
                print("\nData sample (first few records):")
                print(df.head())
                
            else:
                collector.logger.warning(f"No data collected for {state_name}")
                
        except Exception as e:
            collector.logger.error(f"Error processing {state_name}: {str(e)}")
            continue

if __name__ == "__main__":
    asyncio.run(main())