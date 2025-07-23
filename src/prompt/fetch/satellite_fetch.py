import logging

import httpx
from geopy import Point
from geopy.distance import distance

logger = logging.getLogger("uvicorn.error")


def meter_offsets(lat: float, lon: float, extend: float) -> tuple[float, float]:
    """
    Returns (lat_offset, lon_offset) in degrees for a given
    center point (lat, lon) and radial distance in meters (extend).
    """
    origin = Point(lat, lon)
    # Move north (bearing=0°) and east (bearing=90°)
    north = distance(meters=extend).destination(origin, bearing=0)
    east = distance(meters=extend).destination(origin, bearing=90)
    return north.latitude - lat, east.longitude - lon


def fetch_satellite_image(
    lat: float, lon: float, extend: float, output_path: str = "esri_sat.png"
) -> None:
    """
    Fetches a satellite PNG from Esri's World Imagery service.

    Parameters:
    - lat: Latitude of the center point (decimal degrees).
    - lon: Longitude of the center point (decimal degrees).
    - extend: Buffer distance from center in meters (radius).
    - output_path: File path to save the resulting PNG.

    Attempts the highest resolution (1024x1024) first,
    halving the dimensions on failure until success.
    Retries up to 3 times if all size attempts fail.
    """
    # Compute lat/lon degree offsets using geopy
    lat_offset, lon_offset = meter_offsets(lat, lon, extend)

    # Compute bounding box in lon/lat
    minx = lon - lon_offset
    miny = lat - lat_offset
    maxx = lon + lon_offset
    maxy = lat + lat_offset

    base_url = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/export"
    )

    # Retry up to 3 times
    for attempt in range(3):
        logger.info(f"Attempt {attempt + 1}/3 to fetch satellite image...")

        # Try descending sizes until success
        size = 1024
        while size >= 128:
            params = {
                "bbox": f"{minx},{miny},{maxx},{maxy}",
                "bboxSR": "4326",
                "size": f"{size},{size}",
                "format": "png",
                "f": "image",
            }
            try:
                response = httpx.get(base_url, params=params, timeout=30.0)
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"Saved Esri image to {output_path} ({size}x{size})")
                    return
                else:
                    logger.info(
                        f"Failed at size {size} (status {response.status_code}), trying {size // 2}"
                    )
                    size //= 2
            except Exception as e:
                logger.error(f"Network error at size {size}: {e}, trying {size // 2}")
                size //= 2

        # If this attempt failed for all sizes, log and continue to next attempt
        if attempt < 2:  # Don't print this message on the last attempt
            logger.info(f"Attempt {attempt + 1} failed for all sizes, retrying...")

    # If all attempts fail
    logger.warning("Unable to fetch Esri imagery: all retry attempts failed.")
