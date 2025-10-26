import os
import requests
import flickrapi
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils import load_config, load_environment, ProgressTracker, save_results

class FlickrDataCollector:
    """Collect images and metadata from Flickr API."""
    
    def __init__(self):
        load_environment()
        self.config = load_config()
        
        # Initialize Flickr API
        api_key = os.getenv('FLICKR_API_KEY')
        api_secret = os.getenv('FLICKR_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("Flickr API credentials not found in environment variables")
        
        self.flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
        
        # Setup directories
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.raw_data_dir / "flickr_metadata.json"
        self.images_dir = self.raw_data_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def search_photos(self, tags: List[str], min_records: int = 2000) -> List[Dict[str, Any]]:
        """Search for photos on Flickr with specified tags."""
        all_photos = []
        photos_per_tag = min_records // len(tags) + 100  # Add buffer
        
        for tag in tags:
            self.logger.info(f"Searching for photos with tag: {tag}")
            
            try:
                page = 1
                photos_collected = 0
                
                while photos_collected < photos_per_tag:
                    response = self.flickr.photos.search(
                        tags=tag,
                        tag_mode='any',
                        media='photos',
                        content_type=1,  # Photos only
                        safe_search=1,   # Safe content
                        extras='url_o,url_l,url_m,owner_name,date_taken,views,tags,description',
                        per_page=self.config['data_collection']['per_page'],
                        page=page
                    )
                    
                    if response['stat'] != 'ok':
                        self.logger.error(f"API error: {response.get('message', 'Unknown error')}")
                        break
                    
                    photos = response['photos']['photo']
                    if not photos:
                        self.logger.info(f"No more photos found for tag: {tag}")
                        break
                    
                    for photo in photos:
                        # Add search tag to photo metadata
                        photo['search_tag'] = tag
                        all_photos.append(photo)
                        photos_collected += 1
                        
                        if photos_collected >= photos_per_tag:
                            break
                    
                    page += 1
                    time.sleep(0.1)  # Rate limiting
                    
            except Exception as e:
                self.logger.error(f"Error searching for tag {tag}: {str(e)}")
                continue
        
        # Remove duplicates based on photo ID
        unique_photos = {photo['id']: photo for photo in all_photos}
        final_photos = list(unique_photos.values())
        
        self.logger.info(f"Found {len(final_photos)} unique photos")
        return final_photos[:min_records] if len(final_photos) > min_records else final_photos
    
    def download_image(self, photo: Dict[str, Any]) -> Optional[str]:
        """Download a single image from Flickr."""
        photo_id = photo['id']
        
        # Try different URL sizes in order of preference
        url_keys = ['url_o', 'url_l', 'url_m']
        download_url = None
        
        for key in url_keys:
            if key in photo:
                download_url = photo[key]
                break
        
        if not download_url:
            self.logger.warning(f"No download URL found for photo {photo_id}")
            return None
        
        try:
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            
            # Determine file extension
            parsed_url = urlparse(download_url)
            file_extension = Path(parsed_url.path).suffix
            if not file_extension:
                file_extension = '.jpg'
            
            # Save image
            filename = f"{photo_id}{file_extension}"
            filepath = self.images_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error downloading photo {photo_id}: {str(e)}")
            return None
    
    def download_images_parallel(self, photos: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
        """Download images in parallel."""
        successful_downloads = []
        
        with ProgressTracker(len(photos), "Downloading images") as progress:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_photo = {
                    executor.submit(self.download_image, photo): photo 
                    for photo in photos
                }
                
                for future in as_completed(future_to_photo):
                    photo = future_to_photo[future]
                    try:
                        filepath = future.result()
                        if filepath:
                            photo['local_path'] = filepath
                            photo['download_status'] = 'success'
                            successful_downloads.append(photo)
                        else:
                            photo['download_status'] = 'failed'
                    except Exception as e:
                        self.logger.error(f"Error processing photo {photo['id']}: {str(e)}")
                        photo['download_status'] = 'error'
                    
                    progress.update(1)
        
        self.logger.info(f"Successfully downloaded {len(successful_downloads)} images")
        return successful_downloads
    
    def save_metadata(self, photos: List[Dict[str, Any]]) -> None:
        """Save photo metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(photos, f, indent=2, default=str)
        
        self.logger.info(f"Metadata saved to {self.metadata_file}")
    
    def collect_data(self) -> List[Dict[str, Any]]:
        """Main method to collect Flickr data."""
        config = self.config['data_collection']
        
        self.logger.info("Starting Flickr data collection...")
        
        # Search for photos
        photos = self.search_photos(
            tags=config['search_tags'],
            min_records=config['min_records']
        )
        
        if len(photos) < config['min_records']:
            self.logger.warning(f"Only found {len(photos)} photos, less than minimum {config['min_records']}")
        
        # Download images
        successful_photos = self.download_images_parallel(
            photos, 
            max_workers=config.get('max_workers', 4)
        )
        
        # Save metadata
        self.save_metadata(successful_photos)
        
        # Create summary
        summary = {
            'total_photos_found': len(photos),
            'successful_downloads': len(successful_photos),
            'download_success_rate': len(successful_photos) / len(photos) if photos else 0,
            'tags_searched': config['search_tags']
        }
        
        save_results(summary, "data_collection_summary", "json")
        
        self.logger.info("Data collection completed!")
        return successful_photos

def main():
    """Main function to run data collection."""
    try:
        collector = FlickrDataCollector()
        photos = collector.collect_data()
        print(f"Successfully collected {len(photos)} photos with metadata")
        return photos
    except Exception as e:
        logging.error(f"Data collection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()