#Neat chalat nahie, dont use for feature extraction and fusion with other data

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple
import logging

class SoilCollector:
    def __init__(self):
        self.rest_url = "https://rest.isric.org"
        self.prop_query_url = f"{self.rest_url}/soilgrids/v2.0/properties/query"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_grid_points(self, bbox: Tuple[float, float, float, float], spacing: float = 0.05) -> List[Tuple[float, float]]:
        """Create a grid of points within the bounding box"""
        west, south, east, north = bbox
        lats = np.arange(south, north, spacing)
        lons = np.arange(west, east, spacing)
        return [(lat, lon) for lat in lats for lon in lons]

    def get_soil_data(self, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
        """Get soil data for all points in the bounding box"""
        points = self.get_grid_points(bbox)
        self.logger.info(f"Processing {len(points)} points...")
        
        data = []
        properties = ['clay', 'sand', 'silt', 'phh2o', 'soc']
        
        for i, (lat, lon) in enumerate(points, 1):
            point_data = {'latitude': lat, 'longitude': lon}
            
            for prop in properties:
                try:
                    point = {"lat": lat, "lon": lon}
                    props = {"property": prop, "depth": "0-5cm", "value": "mean"}
                    
                    response = requests.get(self.prop_query_url, params={**point, **props})
                    response.raise_for_status()
                    
                    value = response.json()['properties']['layers'][0]['depths'][0]['values'].get('mean')
                    
                    if value is not None:
                        if prop in ['clay', 'sand', 'silt']:
                            value = value / 10  
                        elif prop == 'phh2o':
                            value = value / 10  
                        elif prop == 'soc':
                            value = value / 10  
                        
                        point_data[prop] = round(value, 2)
                    else:
                        point_data[prop] = None
                        
                except Exception as e:
                    self.logger.warning(f"Error getting {prop} for point ({lat}, {lon}): {str(e)}")
                    point_data[prop] = None
                
                time.sleep(0.5) 
            
            data.append(point_data)
            

            if i % 5 == 0:
                self.logger.info(f"Processed {i}/{len(points)} points")
        
        return pd.DataFrame(data)

    def save_data(self, df: pd.DataFrame, output_dir: Path, bbox: Tuple[float, float, float, float]) -> None:
        """Save the collected data"""
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_dir / "soil_data.csv", index=False)
        stats = {
            'bbox': {
                'west': bbox[0],
                'south': bbox[1],
                'east': bbox[2],
                'north': bbox[3]
            },
            'points_collected': len(df),
            'stats': {}
        }
        
        for col in ['clay', 'sand', 'silt', 'phh2o', 'soc']:
            if col in df.columns:
                valid_values = df[col].dropna()
                if not valid_values.empty:
                    stats['stats'][col] = {
                        'min': float(valid_values.min()),
                        'max': float(valid_values.max()),
                        'mean': float(valid_values.mean()),
                        'std': float(valid_values.std())
                    }
        
        with open(output_dir / "soil_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

def main():
    bbox = (-121.8, 36.4, -121.3, 36.9)
    area_name = "salinas_valley"
    
    base_dir = Path("collected_data")
    area_dir = base_dir / area_name
    soil_dir = area_dir / "soil_data"
    
    collector = SoilCollector()
    
    print(f"Collecting soil data for {area_name}")
    print("="*80)
    
    df = collector.get_soil_data(bbox)
    
    if not df.empty:
        collector.save_data(df, soil_dir, bbox)
        
        print("\nData Collection Summary:")
        print(f"Total points collected: {len(df)}")
        print("\nSample of collected data:")
        print(df.head().to_string(index=False))
        
        print("\nData saved to:", soil_dir)
    else:
        print("No data collected!")

if __name__ == "__main__":
    main()