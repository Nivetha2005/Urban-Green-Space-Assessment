# === FILE: utils/maps.py ===
import requests
from PIL import Image
import io
import time
from typing import Dict, Tuple, Optional

def geocode_location(location_name: str, api_key: str) -> Tuple[float, float, str]:
    """
    Convert location name to coordinates using Google Geocoding API
    
    Args:
        location_name: Name of the location (e.g., "New York City, NY")
        api_key: Google Maps API key
    
    Returns:
        Tuple of (latitude, longitude, formatted_address)
    """
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': location_name,
            'key': api_key
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'OK' and len(data['results']) > 0:
            location = data['results'][0]['geometry']['location']
            lat = location['lat']
            lng = location['lng']
            formatted_address = data['results'][0].get('formatted_address', location_name)
            return lat, lng, formatted_address
        elif data['status'] == 'ZERO_RESULTS':
            raise ValueError(f"No location found for: {location_name}")
        elif data['status'] == 'REQUEST_DENIED':
            raise ValueError(f"API key error: {data.get('error_message', 'Invalid API key')}")
        else:
            raise ValueError(f"Geocoding failed with status: {data['status']}")
    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error during geocoding: {e}")
    except Exception as e:
        raise RuntimeError(f"Geocoding failed: {e}")

def fetch_satellite_image(lat: float, lng: float, api_key: str, zoom: int = 18, size: str = "640x640") -> Image.Image:
    """
    Fetch satellite image from Google Maps Static API
    
    Args:
        lat: Latitude
        lng: Longitude
        api_key: Google Maps API key
        zoom: Zoom level (0-21, higher = more detail)
        size: Image size (e.g., "640x640")
    
    Returns:
        PIL Image of the satellite view
    """
    try:
        url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            'center': f"{lat},{lng}",
            'zoom': zoom,
            'size': size,
            'maptype': 'satellite',
            'key': api_key
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        return image
    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch satellite image: {e}")
    except Exception as e:
        raise RuntimeError(f"Satellite image error: {e}")

def fetch_street_view(lat: float, lng: float, api_key: str, size: str = "640x640", 
                      heading: int = 0, pitch: int = 0, fov: int = 90) -> Optional[Image.Image]:
    """
    Fetch street view image from Google Street View API
    
    Args:
        lat: Latitude
        lng: Longitude
        api_key: Google Maps API key
        size: Image size (e.g., "640x640")
        heading: Camera heading in degrees (0 = north)
        pitch: Camera pitch in degrees (-90 to 90)
        fov: Field of view (0-120)
    
    Returns:
        PIL Image of the street view, or None if not available
    """
    try:
        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            'size': size,
            'location': f"{lat},{lng}",
            'heading': heading,
            'pitch': pitch,
            'fov': fov,
            'key': api_key
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        # Check if we got a valid image (Google returns a "no image" placeholder for unavailable locations)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Check if it's a "no street view" image (approximate check by size or content)
        # If image is very small or has specific characteristics, return None
        if image.size[0] < 100 or image.size[1] < 100:
            return None
        
        return image
    
    except requests.exceptions.RequestException as e:
        # Street view might not be available
        return None
    except Exception as e:
        return None

def fetch_mapillary_fallback(lat: float, lng: float, mapillary_token: str) -> Optional[Image.Image]:
    """
    Fallback to Mapillary API if Google Street View fails
    
    Args:
        lat: Latitude
        lng: Longitude
        mapillary_token: Mapillary access token
    
    Returns:
        PIL Image or None
    """
    if not mapillary_token:
        return None
    
    try:
        # Search for images near the location
        bbox = f"{lng-0.001},{lat-0.001},{lng+0.001},{lat+0.001}"
        url = "https://graph.mapillary.com/images"
        
        headers = {'Authorization': f'OAuth {mapillary_token}'}
        params = {
            'bbox': bbox,
            'fields': 'id,thumb_2048_url',
            'limit': 1
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if 'data' not in data or len(data['data']) == 0:
            return None
        
        image_id = data['data'][0]['id']
        thumb_url = data['data'][0].get('thumb_2048_url')
        
        if not thumb_url:
            return None
        
        # Fetch the actual image
        img_response = requests.get(thumb_url, timeout=15)
        img_response.raise_for_status()
        
        return Image.open(io.BytesIO(img_response.content)).convert('RGB')
    
    except Exception:
        return None

def fetch_images_for_location(location_name: str, google_api_key: str, 
                              mapillary_token: str = None) -> Dict:
    """
    Main function to fetch both satellite and street images for a location
    
    Args:
        location_name: Name of the location (e.g., "Central Park, New York")
        google_api_key: Google Maps API key
        mapillary_token: Optional Mapillary token as fallback for street view
    
    Returns:
        Dictionary with keys: 'satellite', 'street', 'lat', 'lng', 'address'
    """
    try:
        # Step 1: Geocode the location
        lat, lng, address = geocode_location(location_name, google_api_key)
        
        # Step 2: Fetch satellite image
        satellite_img = fetch_satellite_image(lat, lng, google_api_key, zoom=18)
        
        # Step 3: Fetch street view image (try Google first)
        street_img = fetch_street_view(lat, lng, google_api_key)
        
        # Step 4: Fallback to Mapillary if Google Street View failed
        if street_img is None and mapillary_token:
            street_img = fetch_mapillary_fallback(lat, lng, mapillary_token)
        
        # Step 5: Prepare result
        result = {
            'satellite': satellite_img,
            'street': street_img,  # May be None if no street view available
            'lat': lat,
            'lng': lng,
            'address': address
        }
        
        return result
    
    except Exception as e:
        raise RuntimeError(f"Failed to fetch images for '{location_name}': {e}")

def fetch_images_for_location_osm_fallback(location_name: str, mapillary_token: str = None) -> Dict:
    """
    Fallback function using OpenStreetMap/Nominatim for geocoding and Esri for satellite
    Use this if Google API key is not available
    
    Args:
        location_name: Name of the location
        mapillary_token: Mapillary token for street view
    
    Returns:
        Dictionary with keys: 'satellite', 'street', 'lat', 'lng', 'address'
    """
    try:
        # Geocode using Nominatim (OSM)
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        resp = requests.get(nominatim_url, params={
            'q': location_name,
            'format': 'json',
            'limit': 1
        }, headers={'User-Agent': 'UrbanGVI/1.0 (urban green space research)'}, timeout=15)
        resp.raise_for_status()
        
        data = resp.json()
        if not data:
            raise ValueError('No geocoding result found')
        
        loc = data[0]
        lat = float(loc['lat'])
        lon = float(loc['lon'])
        address = loc.get('display_name', location_name)
        
        # Fetch satellite from Esri
        satellite_img = _fetch_esri_satellite(lat, lon)
        
        # Fetch street from Mapillary
        street_img = None
        if mapillary_token:
            street_img = fetch_mapillary_fallback(lat, lon, mapillary_token)
        
        return {
            'satellite': satellite_img,
            'street': street_img,
            'lat': lat,
            'lng': lon,
            'address': address
        }
    
    except Exception as e:
        raise RuntimeError(f"OSM fallback failed: {e}")

def _fetch_esri_satellite(lat, lon, zoom=18):
    """Helper function to fetch Esri satellite tiles"""
    import math
    
    def lat_lon_to_tile(lat, lon, zoom):
        n = 2 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        lat_r = math.radians(lat)
        y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
        return x, y
    
    x, y = lat_lon_to_tile(lat, lon, zoom)
    tiles = []
    
    for dy in [-1, 0, 1]:
        row = []
        for dx in [-1, 0, 1]:
            tx, ty = x + dx, y + dy
            url = f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            row.append(Image.open(io.BytesIO(resp.content)).convert('RGB'))
        tiles.append(row)
    
    width, height = tiles[0][0].size
    mosaic = Image.new('RGB', (width * 3, height * 3))
    for i in range(3):
        for j in range(3):
            mosaic.paste(tiles[i][j], (j * width, i * height))
    
    return mosaic