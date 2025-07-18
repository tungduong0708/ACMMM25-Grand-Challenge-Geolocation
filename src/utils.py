
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
        device: str,
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