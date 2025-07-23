import json
import logging
import os
import time
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("uvicorn.error")


def retry_request(func, max_retries=3, base_delay=2.0):
    """
    Retry a function with exponential backoff for timeout and connection errors.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff

    Returns:
        Result of the function call

    Raises:
        Last exception if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"⚠️ Timeout error (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s..."
                )
                time.sleep(delay)
                continue
            else:
                logger.error(
                    f"❌ Max retries ({max_retries}) exceeded for timeout error."
                )
                raise e
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [500, 502, 503, 504]:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"⚠️ Server error {e.response.status_code} (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
            logger.error(f"❌ HTTP error {e.response.status_code}: {e}")
            raise e
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            raise e

    # Should never reach here
    raise RuntimeError("Retry logic failed")


def extension_from_content_type(content_type: str) -> str:
    # Define allowed image types
    allowed_types = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/heic": "heic",
        "image/heif": "heif",
    }

    # Normalize content type (remove charset, etc.)
    content_type = content_type.split(";")[0].strip().lower()

    if content_type in allowed_types:
        return allowed_types[content_type]
    else:
        raise ValueError(
            f"Content type '{content_type}' is not supported. Allowed types: {list(allowed_types.keys())}"
        )


def text_search_image(
    query: str,
    num_images: int = 5,
    api_key: str | None = None,
    cx: str | None = None,
    output_dir: str = "g3/data/prompt_data/images",
    start_index: int = 0,
) -> list[str]:
    if not api_key or not cx:
        raise ValueError("GOOGLE_CLOUD_API_KEY or GOOGLE_CSE_CX not set.")

    os.makedirs(output_dir, exist_ok=True)
    downloaded_files: list[str] = []
    start: int = 1

    idx = start_index
    while len(downloaded_files) < num_images:
        params = {
            "q": query,
            "searchType": "image",
            "cx": cx,
            "key": api_key,
            "num": min(10, num_images - len(downloaded_files)),
            "start": start,
        }

        # Use retry logic for the API request
        try:
            response = retry_request(
                lambda: httpx.get(
                    "https://customsearch.googleapis.com/customsearch/v1",
                    params=params,
                    timeout=30.0,  # Increased timeout
                )
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"❌ Failed to search for images after retries: {e}")
            break

        results = response.json().get("items", [])

        if not results:
            logger.info("No more results from API")
            break

        for item in results:
            img_url: str | None = item.get("link")
            if not img_url:
                continue
            try:
                # Use retry logic for image download
                r = retry_request(lambda url=img_url: httpx.get(url, timeout=15.0))
                r.raise_for_status()
                content_type = r.headers.get("Content-Type", "")

                # Check if content type is supported before processing
                try:
                    ext = extension_from_content_type(content_type)
                except ValueError as e:
                    logger.info(f"Skipping {img_url}: {e}")
                    continue

                filename = os.path.join(output_dir, f"image_{idx:03d}.{ext}")
                with open(filename, "wb") as f:
                    f.write(r.content)
                downloaded_files.append(filename)
                idx += 1
                if len(downloaded_files) >= num_images:
                    break
            except httpx.HTTPError as e:
                logger.error(f"HTTP error downloading {img_url}: {e}")
            except Exception as e:
                logger.error(f"Failed to download {img_url}: {e}")

        start += 10

    return downloaded_files


def text_search_link(
    query: str,
    output_dir: str = "g3/data/prompt_data",
    filename: str = "text_search.json",
    num_results: int = 10,
    api_key: Optional[str] = None,
    cx: Optional[str] = None,
) -> str:
    """
    Search for web links using Google Custom Search API and save results to JSON file.

    Args:
        query (str): Search query string
        output_dir (str): Directory to save the results file
        filename (str): Name of the JSON file to save results
        num_results (int): Number of search results to retrieve (max 100)
        api_key (Optional[str]): Google API key, defaults to environment variable
        cx (Optional[str]): Custom Search Engine ID, defaults to environment variable

    Returns:
        str: Path to the saved JSON file

    Raises:
        ValueError: If API key or CX not provided
        httpx.HTTPError: If API request fails
    """
    if not api_key:
        api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
    if not cx:
        cx = os.getenv("GOOGLE_CSE_CX")

    if not api_key or not cx:
        raise ValueError("GOOGLE_CLOUD_API_KEY or GOOGLE_CSE_CX not set.")

    os.makedirs(output_dir, exist_ok=True)

    links = []
    start = 1

    # Google Custom Search API allows max 10 results per request
    while len(links) < num_results:
        remaining = num_results - len(links)
        current_num = min(10, remaining)

        params = {
            "q": query,
            "cx": cx,
            "key": api_key,
            "num": current_num,
            "start": start,
        }

        try:
            response = retry_request(
                lambda: httpx.get(
                    "https://customsearch.googleapis.com/customsearch/v1",
                    params=params,
                    timeout=30.0,
                )
            )
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])
            if not items:
                logger.info(
                    f"No more results available. Retrieved {len(links)} results."
                )
                break

            links.extend([item.get("link", "") for item in items if "link" in item])

            if len(links) >= num_results:
                break

        except httpx.HTTPError as e:
            logger.error(f"HTTP error during search: {e}")
            break
        except Exception as e:
            logger.error(f"Error during search: {e}")
            break

        start += 10

    # Ensure we only take the first num_results links
    links = links[:num_results]

    # Prepare final results with metadata
    search_results = {"query": query, "links": links}

    # Save results to JSON file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(search_results, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Saved {len(links)} search results to: {output_path}")
    return output_path
