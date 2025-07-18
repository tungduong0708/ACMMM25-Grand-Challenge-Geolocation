
import torch
import torch.nn as nn
import requests
import numpy as np
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple, Union
import asyncio

load_dotenv()

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

def calculate_similarity_scores(
        model: nn.Module,
        device: torch.device,
        predicted_coords: List[Tuple[float, float]],
        image_dir: Union[str, Path] = "images"
    ) -> np.ndarray:
        """
        Calculate similarity scores between images and predicted coordinates.

        Args:
            rgb_images: List of PIL Images
            predicted_coords: List of (lat, lon) tuples

        Returns:
            np.ndarray: Average similarity scores across all images for each coordinate
        """
        all_similarities = []
        image_dir = Path(image_dir)

        if not image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")

        for image_file in image_dir.glob("image_*.*"):
            # Load image as PIL Image first
            pil_image = Image.open(image_file).convert("RGB")
            
            # Process the PIL image
            image = model.vision_processor(images=pil_image, return_tensors="pt")["pixel_values"].reshape(-1, 224, 224)
            image = image.unsqueeze(0).to(device)

            with torch.no_grad():
                vision_output = model.vision_model(image)[1]

                image_embeds = model.vision_projection_else_2(
                    model.vision_projection(vision_output)
                )
                image_embeds = image_embeds / image_embeds.norm(
                    p=2, dim=-1, keepdim=True
                )  # b, 768

                # Process coordinates
                gps_batch = torch.tensor(predicted_coords, dtype=torch.float32).to(device)
                gps_input = gps_batch.clone().detach().unsqueeze(0)  # Add batch dimension
                b, c, _ = gps_input.shape
                gps_input = gps_input.reshape(b * c, 2)
                location_embeds = model.location_encoder(gps_input)
                location_embeds = model.location_projection_else(
                    location_embeds.reshape(b * c, -1)
                )
                location_embeds = location_embeds / location_embeds.norm(
                    p=2, dim=-1, keepdim=True
                )
                location_embeds = location_embeds.reshape(b, c, -1)  # b, c, 768

                similarity = torch.matmul(
                    image_embeds.unsqueeze(1), location_embeds.permute(0, 2, 1)
                )  # b, 1, c
                similarity = similarity.squeeze(1).cpu().detach().numpy()
                all_similarities.append(similarity[0])  # Remove batch dimension

        # Calculate average similarity across all images
        avg_similarities = np.mean(all_similarities, axis=0)
        return avg_similarities

def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable (server errors, connection issues, etc.)

    Args:
        error (Exception): The exception to check

    Returns:
        bool: True if the error should be retried, False otherwise
    """
    error_str = str(error).lower()

    # Check for various retryable error patterns
    retryable_patterns = [
        # Server errors
        "503",
        "500",
        "502",
        "504",
        "overloaded",
        "unavailable",
        "internal",
        # Connection errors
        "disconnected",
        "connection",
        "timeout",
        "remoteprotocolerror",
        "remote protocol error",
        # Network errors
        "network",
        "socket",
        "ssl",
        "tls",
        # Rate limiting
        "rate limit",
        "too many requests",
        "429",
        # Service unavailable
        "service unavailable",
        "temporarily unavailable",
    ]

    # Check if any retryable pattern is found in the error string
    for pattern in retryable_patterns:
        if pattern in error_str:
            return True

    # Additional checks for specific error types
    error_type = type(error).__name__.lower()
    retryable_error_types = [
        "connectionerror",
        "timeout",
        "httperror",
        "remoteclosederror",
        "remoteprotocolerror",
        "sslerror",
        "tlserror",
    ]

    return error_type in retryable_error_types

async def handle_async_api_call_with_retry(
    api_call_func,
    max_retries: int = 10,
    base_delay: float = 2.0,
    fallback_result: Optional[dict] = None,
    error_context: str = "API call",
) -> dict:
    """
    Centralized async API call with retry logic for Google API errors.

    Args:
        api_call_func: Async function to call (should return the API response)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        fallback_result: Result to return if all retries fail
        error_context: Description of the operation for logging

    Returns:
        API response or fallback_result if all attempts fail
    """
    for attempt in range(max_retries):
        try:
            return await api_call_func()

        except Exception as e:
            error_str = str(e).lower()
            print(
                f"{error_context} error (attempt {attempt + 1}/{max_retries}): {e}"
            )

            # Check if error is retryable using our centralized function
            if is_retryable_error(e):
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    print(f"🔄 Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"❌ Max retries ({max_retries}) exceeded. Giving up.")
                    break
            else:
                print(f"❌ Non-retryable error: {e}")
                break

    # All attempts failed
    if fallback_result is not None:
        print(f"⚠️ Returning fallback result for {error_context}")
        return fallback_result
    else:
        print(f"❌ No fallback available for {error_context}")
        return {}