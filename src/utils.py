import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image

# Set up logger
logger = logging.getLogger("uvicorn.error")

T = TypeVar("T")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_USER_AGENT = "keyframe_extraction_app"


def get_gps_from_location(
    location: str,
    language: str = "en",
    timeout: int = 10,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Get GPS coordinates from a location string using Nominatim (OpenStreetMap).

    Args:
        location (str): Location string (e.g., city, address)
        language (str): Language for results (default: 'en')
        timeout (int): Request timeout in seconds (default: 10)
        user_agent (str): User-Agent header (required by Nominatim)

    Returns:
        Tuple[Optional[float], Optional[float]]: (latitude, longitude), or (None, None) on failure
    """
    if not isinstance(location, str) or not location.strip():
        logger.warning("Invalid or empty location string provided.")
        return (None, None)

    params = {
        "q": location.strip(),
        "format": "json",
        "addressdetails": 1,
        "accept-language": language,
        "limit": 1,
    }

    headers = {
        "User-Agent": user_agent,
    }

    try:
        response = requests.get(
            NOMINATIM_URL, params=params, headers=headers, timeout=timeout
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            logger.info(f"No results found for location: '{location}'")
            return (None, None)

        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return (lat, lon)

    except requests.RequestException as req_err:
        logger.error(f"Request error while geocoding '{location}': {req_err}")
    except (ValueError, KeyError, TypeError) as parse_err:
        logger.error(
            f"Failed to parse geocoding response for '{location}': {parse_err}"
        )

    return (None, None)


def calculate_similarity_scores(
    model: nn.Module,
    device: torch.device,
    predicted_coords: List[Tuple[float, float]],
    image_dir: Union[str, Path] = "images",
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
        image = model.vision_processor(images=pil_image, return_tensors="pt")[
            "pixel_values"
        ].reshape(-1, 224, 224)
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
    Determines if the given exception is retryable based on known patterns
    and exception types.

    Args:
        error (Exception): The exception to evaluate.

    Returns:
        bool: True if the error is considered retryable.
    """
    error_str = str(error).lower()

    # Known substrings that indicate retryable errors
    retryable_patterns = [
        "503",
        "500",
        "502",
        "504",
        "overloaded",
        "unavailable",
        "internal",
        "disconnected",
        "connection",
        "timeout",
        "remoteprotocolerror",
        "remote protocol error",
        "network",
        "socket",
        "ssl",
        "tls",
        "rate limit",
        "too many requests",
        "429",
        "service unavailable",
        "temporarily unavailable",
    ]

    for pattern in retryable_patterns:
        if pattern in error_str:
            return True

    # Retryable exception types
    retryable_types = {
        "connectionerror",
        "timeout",
        "httperror",
        "remoteclosederror",
        "remoteprotocolerror",
        "sslerror",
        "tlserror",
        "valueerror",
    }

    error_type = type(error).__name__.lower()
    return error_type in retryable_types


async def handle_async_api_call_with_retry(
    api_call_func: Callable[[], Any],
    max_retries: int = 10,
    base_delay: float = 2.0,
    fallback_result: Optional[T] = None,
    error_context: str = "API call",
) -> T:
    """
    Executes an asynchronous API call with retry logic and exponential backoff.

    Args:
        api_call_func (Callable): An async function that returns any type (T).
        max_retries (int): Maximum retry attempts.
        base_delay (float): Initial delay for backoff (doubles each retry).
        fallback_result (Optional[T]): Optional result to return on failure.
        error_context (str): Contextual info for logging.

    Returns:
        T: Result from the API call or fallback.
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = await api_call_func()
            return result

        except Exception as error:
            is_last_attempt = attempt == max_retries
            retryable = is_retryable_error(error)

            logger.warning(
                f"{error_context} failed (attempt {attempt}/{max_retries}): {error}"
            )

            if retryable and not is_last_attempt:
                delay = base_delay * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                continue

            if not retryable:
                logger.error(f"Non-retryable error encountered: {error}")
            elif is_last_attempt:
                logger.error(f"Max retries reached for {error_context}. Giving up.")

            break

    if fallback_result is not None:
        logger.warning(f"Returning fallback result for {error_context}")
        return fallback_result

    logger.error(f"No fallback result provided for {error_context}.")
    raise RuntimeError(f"{error_context} failed with no result.")


def extract_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Extract and parse the first JSON object found in raw_text.
    Only returns a dict; falls back to {} on failure or if parsed value isn't a dict.

    Args:
        raw_text (str): Raw text (e.g., from an LLM response)

    Returns:
        Dict[str, Any]: Parsed JSON dict, or {} if none valid is found.
    """
    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        logger.error("⚠️ No JSON object found. Snippet:", raw_text[:200])
        return {}

    snippet = raw_text[start : end + 1]

    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
        logger.error("⚠️ JSON parsed but not a dict—got type:", type(parsed).__name__)
    except json.JSONDecodeError as e:
        logger.error("⚠️ JSON decoding error:", e)

    return {}


def image_to_base64(image_path: Path) -> str:
    if not image_path.is_file():
        raise FileNotFoundError(f"No such image: {image_path}")
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")
