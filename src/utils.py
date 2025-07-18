import base64
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()


def get_location_from_gps(
    latitude: float,
    longitude: float,
    language: str = "en",
    timeout: int = 10,
    user_agent: str = "keyframe_extraction_app",
) -> str:
    """
    Get location information from GPS coordinates using Nominatim API (OpenStreetMap).

    Args:
        latitude (float): Latitude coordinate (-90 to 90)
        longitude (float): Longitude coordinate (-180 to 180)
        language (str): Language for results (default: 'en')
        timeout (int): Request timeout in seconds (default: 10)
        user_agent (str): User agent string for API requests

    Returns:
        str: Location address string, or "Unknown location" if failed

    Example:
        >>> location = get_location_from_gps(40.7128, -74.0060)
        >>> print(location)
        'New York, NY, United States'
    """

    # Validate coordinates
    if not (-90 <= latitude <= 90):
        return f"Invalid latitude: {latitude}"

    if not (-180 <= longitude <= 180):
        return f"Invalid longitude: {longitude}"

    # Nominatim API endpoint
    base_url = "https://nominatim.openstreetmap.org/reverse"

    # Parameters for the API request
    params = {
        "lat": latitude,
        "lon": longitude,
        "format": "json",
        "addressdetails": 1,
        "accept-language": language,
        "zoom": 18,  # Detail level (18 = building level)
    }

    # Headers (User-Agent is required by Nominatim)
    headers = {"User-Agent": user_agent}

    try:
        # Make the API request
        response = requests.get(
            base_url, params=params, headers=headers, timeout=timeout
        )

        # Check if request was successful
        response.raise_for_status()

        # Parse JSON response
        data = response.json()

        if "error" in data:
            return "Unknown location"

        # Return the display name (full address)
        return data.get("display_name", "Unknown location")

    except:
        return "Unknown location"


def get_gps_from_location(
    location: str,
    language: str = "en",
    timeout: int = 10,
    user_agent: str = "keyframe_extraction_app",
) -> Tuple[Optional[float], Optional[float]]:
    """
    Get GPS coordinates from a location string using Nominatim API (OpenStreetMap).

    Args:
        location (str): Location string (address, place name, city, etc.)
        language (str): Language for search (default: 'en')
        timeout (int): Request timeout in seconds (default: 10)
        user_agent (str): User agent string for API requests

    Returns:
        Tuple[float, float]: (latitude, longitude) coordinates, or (None, None) if failed

    Example:
        >>> lat, lon = get_gps_from_location("New York, NY, USA")
        >>> print(f"NYC coordinates: {lat}, {lon}")
        'NYC coordinates: 40.7127281, -74.0060152'
    """

    # Validate input
    if not location or not isinstance(location, str):
        return (None, None)

    location = location.strip()
    if not location:
        return (None, None)

    # Nominatim API endpoint for forward geocoding
    base_url = "https://nominatim.openstreetmap.org/search"

    # Parameters for the API request
    params = {
        "q": location,
        "format": "json",
        "addressdetails": 1,
        "accept-language": language,
        "limit": 1,  # Only return the best match
    }

    # Headers (User-Agent is required by Nominatim)
    headers = {"User-Agent": user_agent}

    try:
        # Make the API request
        response = requests.get(
            base_url, params=params, headers=headers, timeout=timeout
        )

        # Check if request was successful
        response.raise_for_status()

        # Parse JSON response
        data = response.json()

        if not data:
            return (None, None)

        # Get the first (best) result
        result = data[0]

        # Extract latitude and longitude
        lat = float(result.get("lat"))
        lon = float(result.get("lon"))

        return (lat, lon)

    except (requests.RequestException, ValueError, KeyError, TypeError):
        return (None, None)
