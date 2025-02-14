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

class NASAPowerCollector:
    def __init__(self):
        """Initialize the NASA POWER Climate Data collector"""
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.session = None
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.parameters = {
            'T2M_MAX': 'Maximum Temperature at 2 Meters (°C)',
            'T2M_MIN': 'Minimum Temperature at 2 Meters (°C)',
            'T2M': 'Temperature at 2 Meters (°C)',
            'PRECTOT': 'Precipitation (mm/day)',
            'RH2M': 'Relative Humidity at 2 Meters (%)',
            'WS2M': 'Wind Speed at 2 Meters (m/s)',
            'ALLSKY_SFC_SW_DWN': 'All Sky Surface Shortwave Downward Irradiance (W/m^2)',
            'ALLSKY_SFC_PAR_TOT': 'All Sky Surface PAR Total (W/m^2)'
        }
        
        self.column_mapping = {
            'T2M_MAX': 'TMAX',
            'T2M_MIN': 'TMIN',
            'T2M': 'TAVG',
            'PRECTOT': 'PRCP',
            'RH2M': 'RH',
            'WS2M': 'WIND',
            'ALLSKY_SFC_SW_DWN': 'SRAD',
            'ALLSKY_SFC_PAR_TOT': 'PAR'
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def get_grid_points(self, bbox: Tuple[float, float, float, float], spacing: float = 0.5) -> List[Tuple[float, float]]:
        """Create a grid of points within the bounding box"""
        west, south, east, north = bbox
        lats = np.arange(south, north, spacing)
        lons = np.arange(west, east, spacing)
        return [(round(lat, 3), round(lon, 3)) for lat in lats for lon in lons]

    async def fetch_point_data(self, 
                             lat: float, 
                             lon: float, 
                             start_date: datetime, 
                             end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data for a single point"""
        params = {
            'parameters': ','.join(self.parameters.keys()),
            'community': 'AG',
            'longitude': lon,
            'latitude': lat,
            'start': start_date.strftime('%Y%m%d'),
            'end': end_date.strftime('%Y%m%d'),
            'format': 'JSON'
        }
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if 'properties' in data and 'parameter' in data['properties']:
                    df = pd.DataFrame(data['properties']['parameter'])
                    df['latitude'] = lat
                    df['longitude'] = lon
                    
                    df.index = pd.to_datetime(df.index)
                    
                    return df
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching data for point ({lat}, {lon}): {str(e)}")
            return None

    async def collect_climate_data(self, 
                                 bbox: Tuple[float, float, float, float],
                                 start_date: datetime,
                                 end_date: datetime,
                                 spacing: float = 0.5) -> Optional[pd.DataFrame]:
        """Collect climate data for the entire region"""
        try:
            grid_points = self.get_grid_points(bbox, spacing)
            self.logger.info(f"Collecting data for {len(grid_points)} grid points...")
            
            all_data = []
            chunk_size = 5  
            
            for i in range(0, len(grid_points), chunk_size):
                chunk = grid_points[i:i + chunk_size]
                tasks = []
                
                for lat, lon in chunk:
                    task = self.fetch_point_data(lat, lon, start_date, end_date)
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
                combined_df = pd.concat(all_data, ignore_index=True)
                
                combined_df = combined_df.rename(columns=self.column_mapping)
                
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
                    'name': 'NASA POWER (Prediction Of Worldwide Energy Resources)',
                    'version': 'Daily 0.5° x 0.5° Version 2.0',
                    'url': 'https://power.larc.nasa.gov/',
                    'citation': 'NASA POWER Project Team. (2020). NASA POWER Project'
                }
            )

            for original_name, new_name in self.column_mapping.items():
                if new_name in df.columns:
                    stats.parameters[new_name] = {
                        'description': self.parameters[original_name],
                        'min': float(df[new_name].min()),
                        'max': float(df[new_name].max()),
                        'mean': float(df[new_name].mean()),
                        'std': float(df[new_name].std())
                    }

            output_dir.mkdir(parents=True, exist_ok=True)

            data_file = output_dir / "historical_climate.csv.gz"
            df.to_csv(data_file, index=True, compression='gzip')

            stats_file = output_dir / "climate_stats.json"
            async with aiofiles.open(stats_file, 'w') as f:
                await f.write(json.dumps(asdict(stats), indent=2))

            monthly_df = df.copy()
            monthly_df.index = pd.to_datetime(monthly_df.index)
            monthly_df['month'] = monthly_df.index.to_period('M')
            
            monthly_summary = monthly_df.groupby('month').agg({
                'TMAX': ['mean', 'max'],
                'TMIN': ['mean', 'min'],
                'PRCP': 'sum',
                'WIND': 'mean',
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

    end_date = datetime.now()
    start_date = end_date.replace(year=end_date.year - 10)

    async with NASAPowerCollector() as collector:
        for state_name, bbox in test_areas.items():
            try:
                state_dir = base_dir / state_name.lower()
                state_dir.mkdir(exist_ok=True)
                climate_dir = state_dir / "historical_climate"
                climate_dir.mkdir(exist_ok=True)
                
                collector.logger.info(f"\nProcessing historical climate data for {state_name}")
                collector.logger.info("="*80)
                collector.logger.info(f"Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                
                df = await collector.collect_climate_data(bbox, start_date, end_date)
                
                if df is not None and not df.empty:
                    await collector.save_climate_data(df, climate_dir, bbox, start_date, end_date)
                    
                    collector.logger.info("\nData Collection Summary:")
                    collector.logger.info(f"Total records: {len(df)}")
                    collector.logger.info(f"Variables collected: {', '.join(collector.column_mapping.values())}")
                    collector.logger.info(f"Grid resolution: 0.5 degrees")
                    
                    print("\nData sample (first few records):")
                    print(df.head())
                    
                else:
                    collector.logger.warning(f"No data collected for {state_name}")
                
            except Exception as e:
                collector.logger.error(f"Error processing {state_name}: {str(e)}")
                continue

if __name__ == "__main__":
    asyncio.run(main())