import aiohttp
import asyncio
import rasterio
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil
import logging
import sys
from datetime import datetime, timedelta
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Set
import aiofiles
import os
from rasterio.enums import Resampling

class OptimizedSentinelCollector:
    def __init__(self):
        self.copernicus_user = "bhushananokar72@gmail.com"
        self.copernicus_password = "t:7kj_nDM9trMc:"
        self.copernicus_base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
        self.token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.copernicus_token = None
        self.token_expiry = None
        self.session = None
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self.band_config = {
            'B04': {'resolution': ['R20m'], 'name': 'red'},    
            'B08': {'resolution': ['R10m'], 'name': 'nir'},     
            'B11': {'resolution': ['R20m'], 'name': 'swir1'}  
        }

        self.band_locations = {
            'R10m': ['AOT', 'B02', 'B08', 'TCI'],
            'R20m': ['AOT', 'B01', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12'],
            'R60m': ['B01', 'B02', 'B03', 'B06', 'B07', 'B8A', 'B09', 'SCL', 'WVP']
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_token(self) -> str:
        """Get authentication token asynchronously"""
        if self.copernicus_token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.copernicus_token

        data = {
            'grant_type': 'password',
            'username': self.copernicus_user,
            'password': self.copernicus_password,
            'client_id': 'cdse-public'
        }

        try:
            async with self.session.post(self.token_url, data=data) as response:
                response.raise_for_status()
                token_data = await response.json()
                self.copernicus_token = token_data['access_token']
                self.token_expiry = datetime.now() + timedelta(seconds=token_data['expires_in'] - 300)
                return self.copernicus_token
        except Exception as e:
            self.logger.error(f"Token error: {str(e)}")
            return None

    def _bbox_to_wkt(self, bbox: tuple) -> str:
        """Convert bbox to WKT format"""
        min_lon, min_lat, max_lon, max_lat = bbox
        return f"POLYGON(({min_lon} {min_lat}, {min_lon} {max_lat}, {max_lon} {max_lat}, {max_lon} {min_lat}, {min_lon} {min_lat}))"

    async def query_products(self, bbox: tuple, date_range: tuple, max_cloud_cover: float = 15.0) -> List[Dict]:
        """Query Sentinel products asynchronously"""
        token = await self.get_token()
        if not token:
            return []

        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json'
        }

        start_date, end_date = date_range
        area_filter = f"OData.CSC.Intersects(area=geography'SRID=4326;{self._bbox_to_wkt(bbox)}')"
        date_filter = f"ContentDate/Start gt {start_date.strftime('%Y-%m-%dT00:00:00.000Z')} and ContentDate/Start lt {end_date.strftime('%Y-%m-%dT23:59:59.999Z')}"
        cloud_filter = f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/Value lt {max_cloud_cover})"
        
        params = {
            '$filter': f"Collection/Name eq 'SENTINEL-2' and {area_filter} and {date_filter} and {cloud_filter}",
            '$orderby': "ContentDate/Start desc",
            '$top': 1  
        }

        try:
            async with self.session.get(
                f"{self.copernicus_base_url}/Products",
                params=params,
                headers=headers
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('value', [])
        except Exception as e:
            self.logger.error(f"Query error: {str(e)}")
            return []

    async def verify_zip_file(self, file_path: str) -> bool:
        """Verify if the file is a valid ZIP file"""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File does not exist: {file_path}")
                return False
                
            if os.path.getsize(file_path) < 100:  
                self.logger.error(f"File is too small to be a valid ZIP: {file_path}")
                return False
                
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                contents = zip_ref.namelist()
                if not contents:
                    self.logger.error("ZIP file is empty")
                    return False

                test_result = zip_ref.testzip()
                if test_result is not None:
                    self.logger.error(f"Corrupt ZIP file, first bad file: {test_result}")
                    return False
                    
            return True
        except zipfile.BadZipFile as e:
            self.logger.error(f"Invalid ZIP file: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error verifying ZIP file: {str(e)}")
            return False

    async def download_product(self, product_id: str, output_path: str) -> bool:
        """Download product with simple streaming approach"""
        token = await self.get_token()
        if not token:
            return False

        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/octet-stream'
        }

        download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

        try:
            self.logger.info(f"Starting download of product {product_id}")

            temp_path = output_path + ".tmp"
            
            async with self.session.get(download_url, headers=headers) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('Content-Length', 0))
                
                if total_size == 0:
                    self.logger.error("Received empty file")
                    return False

                self.logger.info(f"Downloading {total_size / (1024*1024):.2f} MB")
                
                async with aiofiles.open(temp_path, 'wb') as f:
                    downloaded = 0
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (1024*1024*10) == 0: 
                            self.logger.info(f"Downloaded: {downloaded / (1024*1024):.2f} MB")

            if await self.verify_zip_file(temp_path):
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_path, output_path)
                self.logger.info("Download completed and verified successfully")
                return True
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                self.logger.error("Downloaded file verification failed")
                return False

        except Exception as e:
            self.logger.error(f"Download error: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    async def process_bands(self, safe_dir: Path, output_dir: Path) -> Dict:
        """Process bands with different resolutions"""
        features = {}
        
        try:
            subdirs = [d for d in safe_dir.iterdir() if d.is_dir()]
            if not subdirs:
                raise ValueError("No subdirectories found in extraction directory")
            safe_dir = subdirs[0]
            self.logger.info(f"Using SAFE directory: {safe_dir}")

            granule_folders = list(safe_dir.glob('GRANULE/*'))
            if not granule_folders:
                self.logger.error(f"Available folders in {safe_dir}:")
                for path in safe_dir.rglob("*"):
                    if path.is_dir():
                        self.logger.error(f"- {path.relative_to(safe_dir)}")
                raise ValueError("No GRANULE folder found")
            
            granule_path = granule_folders[0]
            img_data_path = granule_path / 'IMG_DATA'

            ref_file = next((img_data_path / 'R10m').glob('*B08_*.jp2'))
            with rasterio.open(ref_file) as src:
                ref_metadata = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': src.width,
                    'height': src.height
                }
            self.logger.info("Got reference resolution from B08 band in R10m")

            for band_code, config in self.band_config.items():
                try:
                    resolution_folder = img_data_path / config['resolution'][0]  
                    if not resolution_folder.exists():
                        raise ValueError(f"Resolution folder {config['resolution'][0]} not found")
                            
                    self.logger.info(f"Looking for {band_code} in {resolution_folder}:")
                    for file in resolution_folder.glob('*.jp2'):
                        self.logger.info(f"- {file.name}")
                    
                    band_files = list(resolution_folder.glob(f'*{band_code}_*.jp2'))
                    if not band_files:
                        self.logger.error(f"Available bands in each resolution:")
                        for res, bands in self.band_locations.items():
                            self.logger.error(f"{res}: {', '.join(bands)}")
                        raise ValueError(f"Band file for {band_code} not found in {resolution_folder.name}")
                    
                    band_file = band_files[0]
                    self.logger.info(f"Processing band {band_code} from {band_file.name}")
                    original_resolution = resolution_folder.name

                    with rasterio.open(band_file) as src:
                        band_data = src.read(1)

                        if config['resolution'][0] == 'R20m':
                            self.logger.info(f"Resampling {band_code} from 20m to 10m resolution")
                            band_data = src.read(
                                1,
                                out_shape=(ref_metadata['height'], ref_metadata['width']),
                                resampling=Resampling.bilinear
                            )

                        band_data = band_data.astype('float32') / 10000.0

                        features[config['name']] = band_data
                        np.save(output_dir / f"{config['name']}.npy", band_data)

                except Exception as e:
                    self.logger.error(f"Error processing band {band_code}: {str(e)}")
                    raise

            metadata = {
                'acquisition_date': safe_dir.name.split('_')[2][:8],
                'bands_processed': [config['name'] for config in self.band_config.values()],
                'spatial_reference': ref_metadata['crs'].to_string(),
                'transform': ref_metadata['transform'].to_gdal(),
                'width': ref_metadata['width'],
                'height': ref_metadata['height']
            }
            
            with open(output_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            return features

        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return {}

async def validate_outputs(sat_dir: Path) -> bool:
    """Validate the generated outputs"""
    try:
        required_files = ['red.npy', 'nir.npy', 'swir1.npy', 'metadata.json']
        for file in required_files:
            if not (sat_dir / file).exists():
                return False

        shapes = []
        for band in ['red.npy', 'nir.npy', 'swir1.npy']:
            arr = np.load(sat_dir / band)
            shapes.append(arr.shape)
            
        if not all(shape == shapes[0] for shape in shapes):
            return False
            
        return True
    except Exception:
        return False

def setup_error_handling():
    """Setup global error handling and logging"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        logging.error("An unhandled exception occurred:", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception

async def main():
    test_areas = {
        "California": (-124.5, 36.0, -119.0, 42.0)
    }

    base_dir = Path("collected_data")
    base_dir.mkdir(exist_ok=True)

    async with OptimizedSentinelCollector() as collector:
        for state_name, bbox in test_areas.items():
            try:
                state_dir = base_dir / state_name.lower()
                state_dir.mkdir(exist_ok=True)
                
                sat_dir = state_dir / "satellite_data"
                sat_dir.mkdir(exist_ok=True)
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)  
                
                products = await collector.query_products(bbox, (start_date, end_date))
                
                if products:
                    product = products[0]  
                    collector.logger.info(f"Found product: {product['Name']}")
                    
                    
                    image_path = sat_dir / f"{product['Name']}.zip"
                    if await collector.download_product(product['Id'], str(image_path)):
                        collector.logger.info("Starting extraction...")
                        
                        try:
                            safe_dir = sat_dir / Path(product['Name']).stem
                            if safe_dir.exists():
                                shutil.rmtree(safe_dir)
                            safe_dir.mkdir(exist_ok=True)

                            with zipfile.ZipFile(image_path) as zip_ref:
                                
                                collector.logger.info("ZIP contents:")
                                for file in zip_ref.namelist()[:10]:
                                    collector.logger.info(f"- {file}")
                                
                                
                                collector.logger.info(f"Extracting to {safe_dir}")
                                zip_ref.extractall(safe_dir)
                            
                            
                            features = await collector.process_bands(safe_dir, sat_dir)
                            
                            
                            if await validate_outputs(sat_dir):
                                collector.logger.info(f"Successfully processed {state_name}")
                                collector.logger.info("Generated files:")
                                for file in sat_dir.glob('*.npy'):
                                    collector.logger.info(f"- {file.name}")
                                collector.logger.info("- metadata.json")
                                
                                with open(sat_dir / "metadata.json", 'r') as f:
                                    metadata = json.load(f)
                                    collector.logger.info(f"Acquisition date: {metadata['acquisition_date']}")
                                    collector.logger.info(f"Processed bands: {', '.join(metadata['bands_processed'])}")
                                    collector.logger.info(f"Image dimensions: {metadata['width']}x{metadata['height']}")
                            else:
                                collector.logger.error("Output validation failed")
                                raise ValueError("Generated outputs are invalid or incomplete")
                            
                            shutil.rmtree(safe_dir)
                            image_path.unlink()
                            
                        except zipfile.BadZipFile as e:
                            collector.logger.error(f"ZIP extraction failed: {str(e)}")
                            if os.path.exists(image_path):
                                os.remove(image_path)
                        except Exception as e:
                            collector.logger.error(f"Processing failed: {str(e)}")
                            if os.path.exists(image_path):
                                os.remove(image_path)
                            if os.path.exists(safe_dir):
                                shutil.rmtree(safe_dir)
                    else:
                        collector.logger.error("Product download failed")
                        
                else:
                    collector.logger.warning(f"No suitable images found for {state_name}")
                    
            except Exception as e:
                collector.logger.error(f"Error processing {state_name}: {str(e)}")
                continue

if __name__ == "__main__":
    setup_error_handling()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)    